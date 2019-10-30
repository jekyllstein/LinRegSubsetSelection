using DelimitedFiles
using Plots
using Statistics

function getFileList(dir = pwd())
	if try isdir(dir) catch; false end
		mapreduce(d -> getFileList(joinpath(dir, d)), vcat, readdir(dir))
	else
		dir
	end
end

function getParams(paramString, param = "Seed"; paramList::Vector = [])
	s = split(paramString, param)
	if param == "Seed" 
		getParams(s[2], "Delt", paramList = [paramList; parse(Int64, s[1])])
	elseif param == "Delt"
		getParams(s[2], "Steps", paramList = [paramList; parse(Float64, s[1])])
	elseif param == "Steps"
		getParams(s[2], "CHI", paramList = [paramList; parse(Int64, s[1])])
	else
		[paramList; parse(Float64, s[1])]
	end
end

function extractLine(line)
	(changeStr, cols, trainErr, testErr) = split(line, '\t')
	colSplit = split(cols, ',')
	numCols = length(colSplit)
	cols = [parse(Int64, colSplit[1][2:end]); parse.(Int64, colSplit[2:end-1]); parse(Int64, colSplit[end][1:end-1])]
	(numCols, trainErr, testErr, cols)
end

function getFileParams(fpath)
	fname = splitpath(fpath)[end]
	paramString = split(fname, '_')[2]
	(seed, delt, steps, chi) = getParams(paramString)
end

makeColVec(cols, N) = BitArray([in(a, cols) for a in 1:N])


function extractRecord(fpath, N)
	(seed, delt, steps, chi) = getFileParams(fpath)
	lines = readlines(fpath)[2:end]
	bestLine = lines[1]
	recordLines = lines[2:end]
	l = length(recordLines)
	bestLineInd = findfirst(recordLines .== bestLine)
	lineExtracts = extractLine.(recordLines)
	testErrs = [a[3] for a in lineExtracts]
	cols = [a[4] for a in lineExtracts]
	colVecs = makeColVec.(cols, N)

	(bestNumCols, bestTrainErr, bestTestErr) = extractLine(bestLine)
	(seed, delt, steps, chi, l, bestLineInd, bestNumCols, parse(Float64, bestTrainErr), parse(Float64, bestTestErr), testErrs, colVecs)
end

function extractTempSteps(fpath)
	(seed, delt, steps, chi) = getFileParams(fpath)
	data = readdlm(fpath, ',')
	temps = Float64.(data[2:end, 1])
	costs = Float64.(data[2:end, 2])
	
	((seed, delt, steps, chi), length(temps))
end

function countQuantiles(errDict, t)
	(m, cutoff) = quantile(values(errDict), [0, t])
	subDict = filter(a -> a[2] < cutoff, errDict)
	bestColEntry = collect(filter(a -> a[2] == m, errDict))[1]
	c = length(subDict)
	(c, cutoff, subDict, bestColEntry)
end
	

function makeRecordSummary(name, prefix = "QuasistaticAnnealLinReg")
	cd(@__DIR__)
	# prefix = "QuasistaticAnnealLinReg"
	loadFlist = getFileList()
	fnameList = [splitpath(a)[end] for a in loadFlist]

	subInds = findall(n -> (occursin(prefix, n) && occursin(name, n)), fnameList)
	recordInds = findall(n -> occursin("_Record_", n), fnameList[subInds])
	tempInds = findall(n -> occursin("_TemperatureSteps_", n), fnameList[subInds])
	usedColInds = findall(n -> occursin("_BestUsedCols_", n), fnameList[subInds])
	tempLoadPaths = loadFlist[subInds[tempInds]]
	recordLoadPaths = loadFlist[subInds[recordInds]]
	usedColLoadPaths = loadFlist[subInds[usedColInds]]
	recordNames = fnameList[subInds[recordInds]]
	savepath = split(recordLoadPaths[1], recordNames[1])[1]

	colNamesBody = readdlm(usedColLoadPaths[1], '\t')
	colNames = colNamesBody[:, 2]
	numCols = length(colNames)
	recordData = extractRecord.(recordLoadPaths, numCols)
	fullTestErrs = mapreduce(a -> parse.(Float64, a[10]), vcat, recordData)
	fullColVecs = mapreduce(a -> a[11], vcat, recordData)
	errDict = Dict(zip(fullColVecs, fullTestErrs))
	(c, cutoff, subDict, bestColEntry) = countQuantiles(errDict, 0.01)
	bestColAvgs = mean(collect(keys(subDict)))
	bestColVec = bestColEntry[1]
	bestColXs = [a ? "x" : "" for a in bestColVec]
	colNamesHeader = ["Col Name" "Top 1% Occurance" "Best Col Use"]
	writedlm(joinpath(savepath, "$(prefix)_BestColsSummary_$(name).csv"), [colNamesHeader; [colNames bestColAvgs bestColXs]], ',')
	bestErr = bestColEntry[2]
	tempData = extractTempSteps.(tempLoadPaths)
	tempParams = [a[1] for a in tempData]
	recordParams = [(a[1], a[2], a[3], a[4]) for a in recordData]
	recordTestErrs = [a[9] for a in recordData]
	commonInds = [findfirst(b -> b == a, recordParams) for a in tempParams]
	sortInd = sortperm(recordTestErrs[commonInds])
	header = ("Seed", "Delt", "Eq Steps", "CHI", "Temp Cycles", "Record Length", "Best Position", "Best Num Cols", "Best Train Err", "Best Test Err")
	body = map((a, b) -> (a[1], a[2], a[3], a[4], b[2], a[5], a[6], a[7], a[8], a[9]), recordData[commonInds][sortInd], tempData[sortInd])
	writedlm(joinpath(savepath, "$(prefix)_ResultsSummary_$(name).csv"), [header; body], ',')
end

if length(ARGS) > 1
	makeRecordSummary(ARGS[1], ARGS[2])
elseif length(ARGS) == 1
	makeRecordSummary(ARGS[1])
elseif isempty(ARGS)
	makeRecordSummary("ecdfInputRandomMcapScaled1QFMDV", "QuasistaticAnnealReheatLinReg")
end

