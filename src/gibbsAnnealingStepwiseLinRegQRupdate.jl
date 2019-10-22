using QRupdate
using LinearAlgebra
using Statistics
using DelimitedFiles
using Random
using StatsBase
using DataStructures

recordType{T} = OrderedDict{Array{Bool, 1}, Tuple{Tuple{T, T}, Matrix{T}, Vector{Int64}}}

function purgeRecord!(colsRecord::recordType{T}, p = 0.1) where T <: Real
    l = length(colsRecord)
    if l > 1
	    ps = LinRange(2*p, 0.0, l)
	    for (i, k) in enumerate(keys(colsRecord))
	        if rand() < ps[i]       		
	       		# push!(colsRecord, k => ((T(Inf), T(Inf)), Matrix{T}(undef, 0, 0), Vector{Int64}(undef, 0)))
	       		delete!(colsRecord, k)
	       	end
		end
		GC.gc()
	end
	return length(colsRecord)
end

function purgeRecord!(colsRecord::Dict, p = 0.1)
    for (i, k) in enumerate(keys(colsRecord))
        if rand() < p     		
       		# push!(colsRecord, k => ((T(Inf), T(Inf)), Matrix{T}(undef, 0, 0), Vector{Int64}(undef, 0)))
       		delete!(colsRecord, k)
       	end
	end
	GC.gc()
return length(colsRecord)
end


function printColsVec(origColsVec, colsVec, switchInd, acc)
	# cmax = 100 #max number of characters per line
	nmax = length(digits(length(colsVec))) #maximum number of digits for indicator column
	# emax = floor(Int64, cmax/(nmax+5)) #maximum number of entries per line
	emax = 20
	c = 0
	l = 1
	print(repeat(" ", nmax+1)) #add nmax+1 spaces of padding for the row labels
	for i in 1:emax
		print(string(lpad(i, 2), " "))
	end
	println()
	print(repeat(" ", nmax+1))
	for i in eachindex(colsVec)
    	#highlight cells with attempted changes
    	bracketColor = if i == switchInd
    		#if an attempted change is accepted make the brackets blue else red
    		acc ? :blue : :red
    	else
    		#if no change leave green
    		:green
    	end

    	fillState = ((i == switchInd) && acc) ? !colsVec[i] : colsVec[i]
    	#fill cell with X if being used and O if not
    	fillChar = fillState ? 'X' : ' '
    	#if current state differs from original highlight in yellow
    	fillColor = fillState == origColsVec[i] ? :default : :reverse

    	printstyled(IOContext(stdout, :color => true), "[", color = bracketColor)
    	printstyled(IOContext(stdout, :color => true), fillChar, color = fillColor)
    	printstyled(IOContext(stdout, :color => true), "]", color = bracketColor)
        c += 1
        if i != length(colsVec)
	        if c == emax
	        	println()
	        	print(string(lpad(emax*l, nmax), " "))
	        	# print(string(lpad(emax*l+1, nmax), " "))
	        	c = 0
	        	l += 1
	        end
	    else
	    	newColChange = if acc
	    		colsVec[switchInd] ?  -1 : 1
	    	else
	    		0
	    	end
	    	print(string(" ", sum(colsVec) + newColChange, "/", length(colsVec)))
	    end
    end
    return l
end

function getNewTemps(tempRecord, cutoff = 0)
	l = length(tempRecord)
	high = 0.5
	Ts = [a[1] for a in tempRecord]
	Ys = [a[2] for a in tempRecord]
	Ybar = [mean(Ys[i:i+10]) for i in 1+floor(Int64, l*cutoff):max(1, length(Ys)-10)]
	Tbar = [mean(Ts[i:i+10]) for i in 1+floor(Int64, l*cutoff):max(1, length(Ys)-10)]
	(Ymin, Ymax) = extrema(Ybar)
	convY(y) = (y - Ymin)/(Ymax-Ymin)
	inds = [findfirst(a -> a <= b, convY.(Ybar)) for b in LinRange(high, high/5, 5)]
	Ts = [Tbar[inds]; 0.0]
end


function calcLinRegErr(R::AbstractMatrix{T}, X::AbstractMatrix{T}, Y::AbstractVector{T}, Xtest::AbstractMatrix{T}, Ytest::AbstractVector{T}) where T <: AbstractFloat
#calculates lin reg error with X input matrix and Y output but with the 
#R matrix provided which is the upper triangular portion of the QR factorization
#of X
	betas, r = csne(R, X, Y)
	linRegTrainErr = sum(abs2, r)/length(Y)
	linRegTestErr = sum(abs2, (Xtest*betas .- Ytest))/length(Ytest)
	(linRegTrainErr, linRegTestErr)
end

function calcLinRegErr(R::AbstractMatrix{T}, X::AbstractMatrix{T}, Y::AbstractVector{T}) where T <: AbstractFloat
#calculates lin reg error with X input matrix and Y output but with the 
#R matrix provided which is the upper triangular portion of the QR factorization
#of X
	betas, r = csne(R, X, Y)
	err = sum(abs2, r)/length(Y)
	BIC = length(Y)*log(err) + log(length(Y))*size(X, 2)
	(err, BIC)
end

function calcLinRegErr(X::AbstractMatrix{T}, Y::AbstractVector{T}, Xtest::AbstractMatrix{T}, Ytest::AbstractVector{T}) where T <: AbstractFloat
	# Xtrain_lin = [ones(Float32, size(Y, 1)) X]
	# Xtest_lin = [ones(Float32, size(Ytest, 1)) Xtest]
	# betas = pinv(Xtrain_lin'*Xtrain_lin)*Xtrain_lin'*Y
	# betas = inv(collect(X')*collect(X))*collect(X')*Y
	betas = X \ Y
	# betas = pinv(collect(X))*Y
	out = X*betas
	outTest = Xtest*betas
	linRegTrainErr = sum(abs2, (out .- Y))/length(Y)
	linRegTestErr = sum(abs2, (outTest .- Ytest))/length(Ytest)
	(linRegTrainErr, linRegTestErr) #, logLikelihoodTest)
end

function gibbsStep(xCols, X, Y, lastR, currentCols, lastBICs, colsRecord, colsVec, T, iNum, Tsteps, startTime, acceptRate, dictHitRate, firstColsVec, printIter = true)
	switchInd = ceil(Int64, rand()*length(colsVec))
	changeString = if colsVec[switchInd]
		string("- ", switchInd)
	else
		string("+ ", switchInd)
	end

	# println(string("Converted Cols Vec = ", find(colsVec)))

	newColsVec = [i == switchInd ? !colsVec[i] : colsVec[i] for i in 1:length(colsVec)]
	# candidateCols = find(newColsVec)
	# currentCols = find(colsVec)
	
	errTimeStr = []

	t = time()
	(BICs, Rnew, newCols) = if haskey(colsRecord, newColsVec) # in(newColsVec, map(a -> a[1], colsRecord))
		# colsRecord[find(a -> a == newColsVec, map(a -> a[1], colsRecord))][1][2]
		#update newCols with entry from dictionary to ensure column order is preserved
		for i in 1:3 
			# println()
			push!(errTimeStr, "\n")
		end
		push!(errTimeStr, "Got errors from dictionary\n")
		colsRecord[newColsVec]
		colsRecord[newColsVec]
	else
		#otherwise form it from the previous value and the switching index
		newCols = if !colsVec[switchInd]
			[currentCols; switchInd]
		else
			setdiff(currentCols, switchInd)
		end
		# println(string("Current Cols = ", currentCols))
		# println(string("New Cols = ", newCols))
		# println(string("Switch Index = ", switchInd, " and has a value of ", colsVec[switchInd]))
		t2 = time()
		Rnew = if !colsVec[switchInd]
			t3 = time()
			# Xold = ones(eltype(xCols[1]), length(Y), length(currentCols)+1)
			for j in eachindex(currentCols)	
				X[:, j+1] = xCols[currentCols[j]]	
			end
			# Xold = view(X, :, [1; currentCols+1])
			Xold = view(X, :, 1:length(currentCols)+1)
			tfill = time() - t3
			# println(string("Old X fill time = ", tfill))
			push!(errTimeStr, string("Old X fill time = ", tfill, "\n"))
			qraddcol(Xold, lastR, xCols[switchInd])
		else
			push!(errTimeStr, "\n")
			k = findfirst(a -> a == switchInd, currentCols)
			qrdelcol(lastR, k+1) #index of column being removed
		end
		tR = time() - t2
		# println(string("R update time = ", tR))
		push!(errTimeStr, string("R update time = ", tR, "\n"))

		t2 = time()
		# Xtmp = ones(eltype(xCols[1]), length(Y), length(newCols)+1)
		# Xtmp2 = ones(eltype(xCols[1]), length(Ytest), length(newCols)+1)
		#fill matrices with new cols, repeatedly using X and Xtest to avoid allocating new memory
		for j in eachindex(newCols)
			if colsVec[switchInd]
				X[:, j+1] = xCols[newCols[j]]
			elseif j == length(newCols) #if a column was added then only change X in the final added column since it was already formed above
				X[:, j+1] = xCols[newCols[j]]
			end
		end
		# Xtmp = view(X, :, [1; newCols+1])
		# Xtmp2 = view(Xtest, :, [1; newCols+1])
		Xtmp = view(X, :, 1:length(newCols)+1)
		tfill = time()-t2
		# println(string("New X fill time = ", tfill))
		push!(errTimeStr, string("New X fill time = ", tfill, "\n"))
		t2 = time()
		BICs = calcLinRegErr(Rnew, Xtmp, Y)
		tErr = time()-t2
		# println(string("Err update time = ", tErr))
		push!(errTimeStr, string("Err update time = ", tErr, "\n"))
		(BICs, Rnew, newCols)
	end
	tErr = time() - t
	# println(string("Total New Err time = ", tErr))
	push!(errTimeStr, string("Total New Err time = ", tErr, "\n"))
	alpha = if BICs[2] < lastBICs[2]
		1
	else
		#base acceptance probability on the distance to the naive logLikelihoodError so if the
		#new columns would lower the log likelihood to the naive case the acceptance probability is 0
		# exp((BICs[3] - lastBICs[3])/T) # 1 - 2*(log(lastBICs[3]) - log(BICs[3]))/(T*nTest) 
		exp(-(BICs[2] - lastBICs[2])/T)
	end

	acc = (alpha > rand())
	
	if printIter #only print if true
		#delete lines for printing new step update only after calculations are complete
		lines = ceil(Int64, size(X, 2)/20) + 13
		for i in 1:lines-1
			print("\r\u1b[K\u1b[A")
		end
		
		# elapsedT = time() - startTime
		# avgIterTime = elapsedT / iNum
		avgIterTime = rpad(round(startTime, digits = 4), 6, '0')
		ETA = round(startTime*(length(Tsteps) - iNum), digits =4)
		ETAmins = floor(Int64, ETA/60)
		ETAsecs = round(ETA - ETAmins*60, digits = 2)
		ETAstr = string(ETAmins, ":", lpad(ETAsecs, 5, '0'))
		println(string("Completed step ", iNum, " of ", length(Tsteps), ", Repeat Step Rate - ", rpad(round(dictHitRate, digits = 4), 6, '0'), ", Avg Step Time - ", avgIterTime, ", ETA - ", ETAstr))

		for s in errTimeStr
			print(s)
		end

		println("Done evaluating candidate columns:")
		printColsVec(firstColsVec, colsVec, switchInd, acc)
		println()
		println(string("Got error and BIC of ", (BICs[1], BICs[2])))
		# println(string("New test log likelihood is ", errs[3], " compared to ", lastErrs[3], " for the previous set"))
		
		alphaStr = rpad(round(alpha, digits = 10), 12, '0')
		if acc
			println(string("Accepting new candidate columns with an alpha of ", alphaStr, ", recent acceptance rate = ", round(acceptRate, digits = 5)))
			# println()
		else
			println(string("Rejecting new candidate columns with an alpha of ", alphaStr, ", recent acceptance rate = ", round(acceptRate, digits = 5)))
			# println()
		end
		println(string("Current state error and BIC: ", (lastBICs[1], lastBICs[2])))
		println()
	end
	(BICs, newColsVec, changeString, acc, Rnew, newCols)

end

function gibbsStep(xCols, X, Y, xTestCols, Xtest, Ytest, lastR, currentCols, lastErrs, colsRecord, colsVec, T, iNum, Tsteps, startTime, acceptRate, dictHitRate, firstColsVec, printIter = true; xFillCol = 0)
	switchInd = ceil(Int64, rand()*length(colsVec))
	changeString = if colsVec[switchInd]
		string("- ", switchInd)
	else
		string("+ ", switchInd)
	end

	# println(string("Converted Cols Vec = ", find(colsVec)))

	newColsVec = [i == switchInd ? !colsVec[i] : colsVec[i] for i in 1:length(colsVec)]
	# candidateCols = find(newColsVec)
	# currentCols = find(colsVec)
	
	errTimeStr = []

	t = time()
	(errs, Rnew, newCols) = if haskey(colsRecord, newColsVec) # in(newColsVec, map(a -> a[1], colsRecord))
		# colsRecord[find(a -> a == newColsVec, map(a -> a[1], colsRecord))][1][2]
		#update newCols with entry from dictionary to ensure column order is preserved
		for i in 1:3 
			# println()
			push!(errTimeStr, "\n")
		end
		push!(errTimeStr, "Got errors from dictionary\n")
		colsRecord[newColsVec]
	else
		#otherwise form it from the previous value and the switching index
		newCols = if !colsVec[switchInd]
			[currentCols; switchInd]
		else
			setdiff(currentCols, switchInd)
		end
		# println(string("Current Cols = ", currentCols))
		# println(string("New Cols = ", newCols))
		# println(string("Switch Index = ", switchInd, " and has a value of ", colsVec[switchInd]))
		t2 = time()
		Rnew = if !colsVec[switchInd]
			t3 = time()
			# Xold = ones(eltype(xCols[1]), length(Y), length(currentCols)+1)
			for j in xFillCol+1:length(currentCols)	
				X[:, j+1] = xCols[currentCols[j]]	
			end
			# Xold = view(X, :, [1; currentCols+1])
			Xold = view(X, :, 1:length(currentCols)+1)
			tfill = time() - t3
			# println(string("Old X fill time = ", tfill))
			push!(errTimeStr, string("Old X fill time = ", tfill, "\n"))
			qraddcol(Xold, lastR, xCols[switchInd])
		else
			# println()
			push!(errTimeStr, "\n")
			k = findfirst(a -> a == switchInd, currentCols)
			qrdelcol(lastR, k+1) #index of column being removed
		end
		tR = time() - t2
		# println(string("R update time = ", tR))
		push!(errTimeStr, string("R update time = ", tR, "\n"))

		t2 = time()
		# Xtmp = ones(eltype(xCols[1]), length(Y), length(newCols)+1)
		# Xtmp2 = ones(eltype(xCols[1]), length(Ytest), length(newCols)+1)
		#fill matrices with new cols, repeatedly using X and Xtest to avoid allocating new memory
		if !colsVec[switchInd]
			#if a column was added then only change X in the final added column since it was already formed above
			X[:, length(newCols)+1] = xCols[switchInd]
			for j in xFillCol+1:length(newCols)
				Xtest[:, j+1] = xTestCols[newCols[j]]
			end
		else
			for j in min(xFillCol+1, k):length(newCols)
				X[:, j+1] = xCols[newCols[j]]
				Xtest[:, j+1] = xTestCols[newCols[j]]
			end
		end


		# for j in eachindex(newCols)
		# 	if colsVec[switchInd]
		# 		X[:, j+1] = xCols[newCols[j]]
		# 	elseif j == length(newCols) #if a column was added then only change X in the final added column since it was already formed above
		# 		X[:, j+1] = xCols[newCols[j]]
		# 	end

		# 	Xtest[:, j+1] = xTestCols[newCols[j]]
		# end
		# Xtmp = view(X, :, [1; newCols+1])
		# Xtmp2 = view(Xtest, :, [1; newCols+1])
		Xtmp = view(X, :, 1:length(newCols)+1)
		Xtmp2 = view(Xtest, :, 1:length(newCols)+1)
		tfill = time()-t2
		# println(string("New X fill time = ", tfill))
		push!(errTimeStr, string("New X fill time = ", tfill, "\n"))
		t2 = time()
		errs = calcLinRegErr(Rnew, Xtmp, Y, Xtmp2, Ytest)
		tErr = time()-t2
		# println(string("Err update time = ", tErr))
		push!(errTimeStr, string("Err update time = ", tErr, "\n"))
		(errs, Rnew, newCols)
	end
	tErr = time() - t
	# println(string("Total New Err time = ", tErr))
	push!(errTimeStr, string("Total New Err time = ", tErr, "\n"))
	alpha = if errs[2] < lastErrs[2]
		1
	else
		#base acceptance probability on the distance to the naive logLikelihoodError so if the
		#new columns would lower the log likelihood to the naive case the acceptance probability is 0
		# exp((errs[3] - lastErrs[3])/T) # 1 - 2*(log(lastErrs[3]) - log(errs[3]))/(T*nTest) 
		exp(-(errs[2] - lastErrs[2])/T)
	end
	acc = (alpha > rand())

	if printIter #only print if true
		#delete lines for printing new step update only after calculations are complete
		lines = ceil(Int64, length(xCols)/20) + 14

		for i in 1:lines-1
			print("\r\u1b[K\u1b[A")
		end

		
		# elapsedT = time() - startTime
		# avgIterTime = elapsedT / iNum
		avgIterTime = rpad(round(startTime, digits = 4), 6, '0')
		ETA = round(startTime*(length(Tsteps) - iNum), digits =4)
		ETAmins = floor(Int64, ETA/60)
		ETAsecs = round(ETA - ETAmins*60, digits = 2)
		ETAstr = string(ETAmins, ":", lpad(ETAsecs, 5, '0'))
		println(string("Completed step ", iNum, " of ", length(Tsteps), ", Repeat Step Rate - ", rpad(round(dictHitRate, digits = 4), 6, '0'), ", Avg Step Time - ", avgIterTime, ", ETA - ", ETAstr))

		for s in errTimeStr
			print(s)
		end

		println("Done evaluating candidate columns:")
		printColsVec(firstColsVec, colsVec, switchInd, acc)
		println()
		println(string("Got training and test set errors of ", (errs[1], errs[2])))
		# println(string("New test log likelihood is ", errs[3], " compared to ", lastErrs[3], " for the previous set"))
		
		alphaStr = rpad(round(alpha, digits = 10), 12, '0')
		if acc
			println(string("Accepting new candidate columns with an alpha of ", alphaStr, ", recent acceptance rate = ", round(acceptRate, digits = 5)))
			# println()
		else
			println(string("Rejecting new candidate columns with an alpha of ", alphaStr, ", recent acceptance rate = ", round(acceptRate, digits = 5)))
			# println()
		end
		println(string("Current state training and test errors: ", (lastErrs[1], lastErrs[2])))
		println("Current temperature = $T")
		println()
	end
	# println(string("Converted New Cols Vec = ", find(newColsVec)))
	# sleep(0.5)

	#number of columns which X and xTest are still appropriately filled to considering the currently accepted
	#state of the column selection ignoring the initial column of ones
	xFillCol = if haskey(colsRecord, newColsVec)
		if acc
			0
		else
			xFillCol
		end
	else
		if acc
			length(newCols)
		elseif !colsVec[switchInd]
			length(currentCols)
		else
			findfirst(currentCols .== switchInd) - 1
		end
	end

	(errs, newColsVec, changeString, acc, Rnew, newCols, xFillCol)

end

function runGibbsStep(regData, Rnow, currentCols, errsRecord, colsRecord::recordType, Tsteps; printStep = true, updateInterval = 2.0, acceptRate = 0.0, dictHitRate = 0.0, iterTime = 0.0)
	
	f = 0.99 #controls how long running average is for iter time and accept and repeat rate
	# Rnew = copy(Rnow)
	# newCols = copy(currentCols)
	#on first iteration define update variables and print
	stepStartTime = time()
	# iterTime = 0.0
	firstColsVec = [in(i, currentCols) ? true : false for i in 1:length(regData[1])]
	
	l = length(regData[1][1])
	N = length(regData[1])
	M = length(Tsteps)
	memBuffer = if Sys.isapple()
		N*N*min(M, 1000)*8
	else
		1e9 + (N*l*4 + N*N*1000)*8 #number of bytes to make sure are available for dictionary addition
	end

	memCheck = (Sys.free_memory() > memBuffer) #make sure enough free memory to add to dictionary
	costSequence = Array{Float64, 1}(undef, 0)

	colsVec = errsRecord[end][2]
	errs = errsRecord[end][3]
	push!(costSequence, errs[2])
	(newErrs, newColsVec, changeString, acc, Rnew, newCols, xFillCol) = gibbsStep(regData..., Rnow, currentCols, errs, colsRecord, colsVec, Tsteps[1], 1, Tsteps, iterTime, acceptRate, dictHitRate, firstColsVec, false)

	keyCheck = haskey(colsRecord, newColsVec)
	dictHitRate = dictHitRate*f + (1-f)*keyCheck
	if keyCheck
		delete!(colsRecord, newColsVec)
	elseif !memCheck #if not enough memory purge 10% of colsRecord weighed more towards earlier entries
		while !memCheck
			purgeRecord!(colsRecord)
			memCheck = (Sys.free_memory() > memBuffer) #make sure enough free memory to add to dictionary
		end
	end

	push!(colsRecord, newColsVec => (newErrs, Rnew, newCols))
	acceptRate = f*acceptRate + (1-f)*acc
	
	if acc
		push!(errsRecord, (changeString, newColsVec, newErrs))
		Rnow = Rnew
		currentCols = newCols
		colsVec = newColsVec
		errs = newErrs
	end
	push!(costSequence, errs[2])

	iterTime = if iterTime == 0
		(time() - stepStartTime)
	else
		f*iterTime + (1-f)*(time() - stepStartTime)
	end
	lastPrintTime = time()


	for i in 2:length(Tsteps)
		if i%1000 == 0 #update memCheck every 1000 steps
			memCheck = (Sys.free_memory() > memBuffer) #make sure enough free memory to add to dictionary
		end

		# readline(stdin)
		#print second iteration and subsequent once every 1 second by default
		printIter = if printStep
			if i == 2
				true
			elseif (time() - lastPrintTime) > updateInterval
				true
			else
				false
			end
		else
			false
		end
		
		stepStartTime = time()
		#on first iteration define update variables and print
		(newErrs, newColsVec, changeString, acc, Rnew, newCols, xFillCol) = gibbsStep(regData..., Rnow, currentCols, errs, colsRecord, colsVec, Tsteps[i], i, Tsteps, iterTime, acceptRate, dictHitRate, firstColsVec, printIter, xFillCol = xFillCol)

		keyCheck = haskey(colsRecord, newColsVec)
		dictHitRate = dictHitRate*f + (1-f)*keyCheck
		if keyCheck
			delete!(colsRecord, newColsVec)
		elseif !memCheck #if not enough memory purge 10% of colsRecord prioritizing early entries
			while !memCheck
				purgeRecord!(colsRecord)
				memCheck = (Sys.free_memory() > memBuffer) #make sure enough free memory to add to dictionary
			end
		end
		
		push!(colsRecord, newColsVec => (newErrs, Rnew, newCols))
		acceptRate = f*acceptRate + (1-f)*acc
		
		if acc
			push!(errsRecord, (changeString, newColsVec, newErrs))
			Rnow = Rnew
			currentCols = newCols
			colsVec = newColsVec
			errs = newErrs
		end

		push!(costSequence, errs[2])

		iterTime = f*iterTime + (1-f)*(time() - stepStartTime)

		lastPrintTime = printIter ? time() : lastPrintTime
	end
	return (errsRecord, colsRecord, costSequence, acceptRate, dictHitRate, currentCols, Rnow, iterTime)
end

function calibrateGibbsTemp(regData, Rnow, currentCols, errsRecord, colsRecord::recordType; printStep = true, updateInterval = 1.0, chi = 0.9)
	
	# Rnew = copy(Rnow)
	# newCols = copy(currentCols)
	#on first iteration define update variables and print
	stepStartTime = time()
	iterTime = 0.0
	acceptRate = 0.0
	dictHitRate = 0.0
	firstColsVec = [in(i, currentCols) ? true : false for i in 1:length(regData[1])]
	N = length(firstColsVec)
	M = 2*round(Int64, N*log(N))
	l = length(regData[1][1])
	Tsteps = ones(M+1)
	accs = zeros(Int64, M)
	testErrs = Vector{Float64}(undef, M)
	deltCbar = 0.0
	m1 = 0
	m2 = 0
	f = 0.99 #controls how long running average is for iter time and accept and repeat rate
	memBuffer = if Sys.isapple()
		N*N*min(M, 1000)*8
	else
		1e9 + (N*l*4 + N*N*1000)*8 #number of bytes to make sure are available for dictionary addition
	end

	memCheck = (Sys.free_memory() > memBuffer) #make sure enough free memory to add to dictionary

	
	colsVec = errsRecord[end][2]
	errs = errsRecord[end][3]
	(newErrs, newColsVec, changeString, acc, Rnew, newCols, xFillCol) = gibbsStep(regData..., Rnow, currentCols, errs, colsRecord, colsVec, Tsteps[1], 1, Tsteps, iterTime, acceptRate, dictHitRate, firstColsVec, false)

	keyCheck = haskey(colsRecord, newColsVec)
	if keyCheck
		delete!(colsRecord, newColsVec)
	elseif !memCheck #if not enough memory purge 10% of colsRecord weighted towards earlier elements
		while !memCheck
			purgeRecord!(colsRecord)
			memCheck = (Sys.free_memory() > memBuffer) #make sure enough free memory to add to dictionary
		end
	end

	dictHitRate = Float64(keyCheck)
	push!(colsRecord, newColsVec => (newErrs, Rnew, newCols))
	acceptRate = Float64(acc)
	deltC = newErrs[2] - errs[2]
	if deltC > 0
		deltCbar += deltC
		m2 += 1
	else
		m1 += 1
	end

	Tsteps[2] = deltCbar / log(m2 / (m2*chi - (1-chi)*m1)) / m2

	if acc
		accs[1] = 1
		push!(errsRecord, (changeString, newColsVec, newErrs))
		Rnow = Rnew
		currentCols = newCols
		colsVec = newColsVec
		errs = newErrs
	end
	testErrs[1] = errs[2]


	iterTime = (time() - stepStartTime)
	lastPrintTime = time()


	for i in 2:length(Tsteps)-1
		if i%1000 == 0 #update memCheck every 1000 steps
			memCheck = (Sys.free_memory() > memBuffer) #make sure enough free memory to add to dictionary
		end
		#print second iteration and subsequent once every 1 second by default
		printIter = if printStep
			if i == 2
				true
			elseif (time() - lastPrintTime) > updateInterval
				true
			else
				false
			end
		else
			false
		end
		
		stepStartTime = time()
		#on first iteration define update variables and print
		(newErrs, newColsVec, changeString, acc, Rnew, newCols, xFillCol) = gibbsStep(regData..., Rnow, currentCols, errs, colsRecord, colsVec, Tsteps[i], i, Tsteps, iterTime, acceptRate, dictHitRate, firstColsVec, printIter, xFillCol = xFillCol)

		
		keyCheck = haskey(colsRecord, newColsVec)
		if keyCheck
			delete!(colsRecord, newColsVec)
		elseif !memCheck #if not enough memory purge 10% of colsRecord weighted towards earlier elements
			while !memCheck
				purgeRecord!(colsRecord)
				memCheck = (Sys.free_memory() > memBuffer) #make sure enough free memory to add to dictionary
			end
		end

		dictHitRate = dictHitRate*f + (1-f)*keyCheck
		
		push!(colsRecord, newColsVec => (newErrs, Rnew, newCols))
		acceptRate = f*acceptRate + (1-f)*acc

		deltC = newErrs[2] - errs[2]
		if deltC > 0
			deltCbar += deltC
			m2 += 1
		else
			m1 += 1
		end

		c = m2 / (m2*chi - (1-chi)*m1)
		Tsteps[i+1] = if c <= 0
			0.0
		elseif c <= 1
			Inf 
		else
			deltCbar / log(c) / m2
		end

		#in second half try resetting values to get more accurate calibration
		if i == round(Int64, length(Tsteps)/2) #((i < round(Int64, length(Tsteps)*0.75)) && ((i % N) == 0))
			m1 = 0
			m2 = 0
			deltCbar = 0.0
		end
		
		if acc
			accs[i] = 1
			push!(errsRecord, (changeString, newColsVec, newErrs))
			Rnow = Rnew
			currentCols = newCols
			colsVec = newColsVec
			errs = newErrs
		end
		testErrs[i] = errs[2]

		iterTime = f*iterTime + (1-f)*(time() - stepStartTime)

		lastPrintTime = printIter ? time() : lastPrintTime
	end
	return (errsRecord, colsRecord, Tsteps, accs, testErrs)
end

function runThermodynamicAnneal(regData, Rnow, currentCols, errsRecord, colsRecord, Tsteps; printStep = true, updateInterval = 1.0, kA = 1.0, calibrate = true)
	
	firstColsVec = [in(i, currentCols) ? true : false for i in 1:length(xCols)]
	# Rnew = copy(Rnow)
	# newCols = copy(currentCols)
	#on first iteration define update variables and print
	stepStartTime = time()
	iterTime = 0.0
	acceptRate = 0.0
	dictHitRate = 0.0
	deltCT = 0.0
	deltST = 0.0
	deltCAbsSum = 0.0
	tempRecord = []
	T0 = calibrate ? Inf : Tsteps[1]

	colsVec = errsRecord[end][2]
	errs = errsRecord[end][3]
	(newErrs, newColsVec, changeString, acc, Rnew, newCols) = gibbsStep(regData..., Rnow, currentCols, errs, colsRecord, colsVec, T0, 1, Tsteps, iterTime, acceptRate, dictHitRate, firstColsVec, false)

	dictHitRate = Float64(haskey(colsRecord, newColsVec))
	push!(colsRecord, newColsVec => (newErrs, Rnew, newCols))
	acceptRate = Float64(acc)
	deltCk = newErrs[2] - errs[2]
	deltCAbsSum += abs(deltCk)

	if acc
		push!(errsRecord, (changeString, newColsVec, newErrs))
		Rnow = Rnew
		currentCols = newCols
		colsVec = newColsVec
		errs = newErrs
		deltCT = deltCT + deltCk
	end

	iterTime = (time() - stepStartTime)
	lastPrintTime = time()

	# T0 = calibrate ? Inf : 1e-10
	# T0 = Inf
	currentTemp = T0 #for initial calibration accept all changes
	calSteps = calibrate ? 5*length(colsVec) : 0 #number of calibration steps to determine initial temperature
	# calSteps = 5*length(colsVec)
	for i in 2:length(Tsteps)
		# sleep(1.0)
		#print second iteration and subsequent once every 1 second by default
		printIter = if printStep
			if i == 2
				true
			elseif (time() - lastPrintTime) > updateInterval
				true
			else
				false
			end
		else
			false
		end
		
		stepStartTime = time()
		#on first iteration define update variables and print
		(newErrs, newColsVec, changeString, acc, Rnew, newCols) = gibbsStep(regData..., Rnow, currentCols, errs, colsRecord, colsVec, currentTemp, i, Tsteps, iterTime, acceptRate, dictHitRate, firstColsVec, printIter)
		dictHitRate = dictHitRate*0.95 + 0.05*haskey(colsRecord, newColsVec)
		
		
		push!(colsRecord, newColsVec => (newErrs, Rnew, newCols))
		acceptRate = 0.95*acceptRate + 0.05*acc
		deltCk = newErrs[2] - errs[2]
		deltCAbsSum += abs(deltCk)

		#after calibration steps are done calculate T0 based on average cost variation of steps 
		if i == calSteps
			T0 = -deltCAbsSum/(log(0.8)*calSteps)
			deltCT = 0.0 #reset deltCT
			currentTemp = T0
		end

		if acc
			push!(errsRecord, (changeString, newColsVec, newErrs))
			Rnow = Rnew
			currentCols = newCols
			colsVec = newColsVec
			errs = newErrs
			deltCT = deltCT + deltCk
		end

		if i > calSteps
			if printIter
				println("Current temperature = $currentTemp, T0 = $T0 after $calSteps steps of calibration")
				println()
			end
			#update entropy variation
			if deltCk > 0
				deltST = deltST - deltCk/currentTemp
			end

			#update current temperature resetting to T0 if net cost has increased or there is no variation
			# if (deltCT >= 0) | (deltST == 0)
			# 	currentTemp = T0
			# else
			# 	currentTemp = kA * (deltCT/deltST) #constant thermodynamic temperature rate
			# end

			# currentTemp = T0/log(i) #boltzman slow guaranteed convergence rate

			alpha = if acceptRate > 0.96
				0.5
			elseif acceptRate > 0.8
				0.9
			elseif acceptRate > 0.15
				0.95
			else
				0.8
			end

			if acc
				push!(tempRecord, (currentTemp, newErrs[2]))
			else
				push!(tempRecord, (currentTemp, errs[2]))
			end
			
			if calibrate
				currentTemp = currentTemp * 0.95^(1000/(length(Tsteps)-calSteps))
				# currentTemp = T0 - T0*(i-calSteps)/(length(Tsteps)-calSteps)
			# elseif i%calSteps == 0
				# currentTemp = alpha*currentTemp #VPR annealing schedule
			else
				currentTemp = Tsteps[i]
			end

		elseif printIter
			println("Current temperature = $currentTemp, waiting $calSteps steps for T0 calibration")
			println()
		end


		iterTime = 0.95*iterTime + 0.05*(time() - stepStartTime)

		lastPrintTime = printIter ? time() : lastPrintTime
	end
	return (errsRecord, colsRecord, tempRecord)
end

function runStepwiseAnneal(X::Array{T, 2}, Y::Array{T, 1}; initialColsVec = fill(false, size(X, 2)), initialColsRecord = Dict{Array{Bool, 1}, Tuple{Tuple{T, T}, Matrix{T}, Vector{Int64}}}(), iter = 0, printIter = true, updateInterval = 1.0, errDelt = 0.0) where T <: Real
	# T = eltype(X)
	newX = [ones(T, length(Y)) X]
	println("Beginning gibbs annealing stepwise regression")
	initialCols = findall(initialColsVec)
	(err, BIC, R) = if sum(initialColsVec) == 0
		# initialColsVec = fill(false, size(X, 2))
		println("Calculating bias term error as a baseline")
		meanY = mean(Y)
		err = sum(abs2, Y .- meanY)/length(Y)
		BIC = length(Y)*log(err)+log(length(Y))
		_, R = qr([ones(T, length(Y)) view(X, :, initialCols)])
		(err, BIC, R)
	else
		if haskey(initialColsRecord, initialColsVec)
			(errs, R, cols) = initialColsRecord[initialColsVec]
			println("Using dictionary to get errors for initial columns ", cols)
			(errs[1], errs[2], R)
		else
			println("Finding errors for initial columns ", initialCols)
			_, R = qr([ones(T, length(Y)) view(X, :, initialCols)])
			errs = calcLinRegErr(R, view(newX, :, [1; initialCols .+ 1]), Y)
			(errs[1], errs[2], R)
		end
	end
		
	println(string("Got err and BIC of : ", (err, BIC), " using ", sum(initialColsVec), "/", length(initialColsVec), " columns"))
	println("--------------------------------------------------------------------")
	println("Setting up $(size(X, 2)) columns for step updates")
	println()

	# Tmax = abs(BIC)/2000 #corresponds to a ~.03% increase in error having a ~50% chance of acceptance
	Tmax = -errDelt/log(0.5) #corresponds to a typical delta seen at the end of the last iteration having a ~50% chance of acceptance
	numSteps = 6*length(initialColsVec) #max(100, round(Int64, length(initialColsVec)*sqrt(length(initialColsVec))/2))
	# Tsteps = initialT*exp.(-7*(0:numSteps-1)/numSteps)
	Tsteps = if iter == 1
		zeros(T, 5*numSteps)
	else
		#with this schedule Tmax is reduced to 0.0067*Tmax
		# [Tmax*exp.(-5*(0:3*numSteps-1)/(3*(numSteps-1))); zeros(T, 2*numSteps)]
		[Tmax*LinRange(1, 0, 3*numSteps); zeros(T, 5*numSteps)]
	end
	xCols = [X[:, i] for i in 1:size(X, 2)]

	#add line padding for print update if print is turned on
	if printIter
		lines = ceil(Int64, size(X, 2)/20) + 13
		for i in 1:lines-1 
			println()
		end
		println("Waiting for first step to complete")
	else
		println("Starting $(length(Tsteps)) steps of annealing process without printing updates")
	end

	(BICsRecord, colsRecord) = runGibbsStep((xCols, newX, Y), R, initialCols, [("", initialColsVec, (err, BIC))], push!(initialColsRecord, initialColsVec => ((err, BIC), R, initialCols)), Tsteps, printStep=printIter, updateInterval = updateInterval)
	BICsOut = [begin
	    (a[1], findall(a[2]), a[3][1], a[3][2])
	end
	for a in BICsRecord]
	(BICsOut, colsRecord)
end


function runStepwiseAnneal(X::Array{T, 2}, Y::Array{T, 1}, Xtest::Array{T, 2}, Ytest::Array{T, 1}; initialColsVec = fill(false, size(X, 2)), initialColsRecord = Dict{Array{Bool, 1}, Tuple{Tuple{T, T}, Matrix{T}, Vector{Int64}}}(), iter = 0, printIter = true, updateInterval = 1.0, errDelt = 0.0, Tpoints = []) where T <: Real
	# T = eltype(X)
	newX = [ones(T, length(Y)) X]
	newXtest = [ones(T, length(Ytest)) Xtest]
	println("Beginning gibbs annealing stepwise regression")
	# initialCols = findall(initialColsVec)
	(trainErr, testErr, R, initialCols) = if sum(initialColsVec) == 0
		# initialColsVec = fill(false, size(X, 2))
		initialCols = findall(initialColsVec)
		println("Calculating bias term error as a baseline")
		meanY = mean(Y)
		trainErr = sum(abs2, Y .- meanY)/length(Y)
		testErr = sum(abs2, Ytest .- meanY)/length(Ytest)
		_, R = qr([ones(T, length(Y)) view(X, :, initialCols)])
		(trainErr, testErr, R, initialCols)
	else
		if haskey(initialColsRecord, initialColsVec)
			(errs, R, cols) = initialColsRecord[initialColsVec]
			println("Using dictionary to get errors for initial columns ", cols)
			(errs[1], errs[2], R, cols)
		else
			initialCols = findall(initialColsVec)
			println("Finding errors for initial columns ", initialCols)
			_, R = qr([ones(T, length(Y)) view(X, :, initialCols)])
			errs = calcLinRegErr(R, view(newX, :, [1; initialCols .+ 1]), Y, view(newXtest, :, [1; initialCols .+ 1]), Ytest)
			(errs[1], errs[2], R, initialCols)
		end
	end
		
	println(string("Got training and test set errors of : ", (trainErr, testErr), " using ", sum(initialColsVec), "/", length(initialColsVec), " columns"))
	println("--------------------------------------------------------------------")
	println("Setting up $(size(X, 2)) columns for step updates")
	# Tmax = testErr/460 #corresponds to a ~1% increase in error having a ~1% chance of acceptance
	Tmax = max(testErr/3000, -errDelt/log(0.5)) #corresponds to a typical delta seen at the end of the last iteration having a ~50% chance of acceptance
	numSteps = 6*length(initialColsVec) #max(100, round(Int64, length(initialColsVec)*sqrt(length(initialColsVec))/2))
	# Tsteps = initialT*exp.(-7*(0:numSteps-1)/numSteps)
	Tsteps = if iter == 1
		zeros(T, 5*numSteps)
	else
		#with this schedule Tmax is reduced to 0.0067*Tmax
		# [Tmax*exp.(-5*(0:3*numSteps-1)/(3*(numSteps-1))); zeros(T, 2*numSteps)]
		[Tmax*LinRange(1, 0, 3*numSteps); zeros(T, 5*numSteps)]
	end
	
	# Tsteps = zeros(T, 8*numSteps)
	Tsteps = if isempty(Tpoints)
		zeros(T, 5*numSteps)
	else
		reduce(vcat, [LinRange(Tpoints[i], Tpoints[i+1], numSteps) for i in 1:length(Tpoints)-1])
	end

	xCols = [X[:, i] for i in 1:size(X, 2)]
	xTestCols = [Xtest[:, i] for i in 1:size(Xtest, 2)]

	#add line padding for print update if print is turned on
	if printIter
		lines = ceil(Int64, size(X, 2)/20) + 13
		for i in 1:lines-1 
			println()
		end
		println("Waiting for first step to complete")
	else
		println("Starting $(length(Tsteps)) steps of annealing process without printing updates")
	end

	# (errsRecord, colsRecord) = runGibbsStep((xCols, newX, Y, xTestCols, newXtest, Ytest), R, initialCols, [("", initialColsVec, (trainErr, testErr))], push!(initialColsRecord, initialColsVec => ((trainErr, testErr), R, initialCols)), Tsteps, firstColsVec = initialColsVec, printStep=printIter, updateInterval = updateInterval)
	(errsRecord, colsRecord, tempRecord) = runThermodynamicAnneal((xCols, newX, Y, xTestCols, newXtest, Ytest), R, initialCols, [("", initialColsVec, (trainErr, testErr))], push!(initialColsRecord, initialColsVec => ((trainErr, testErr), R, initialCols)), Tsteps, printStep=printIter, updateInterval = updateInterval, calibrate = (iter == 1))
	errsOut = [begin
	    (a[1], findall(a[2]), a[3][1], a[3][2])
	end
	for a in errsRecord]
	(errsOut, colsRecord, tempRecord)
end

function sampleColStats(name, X::Array{T, 2}, Y::Array{T, 1}, Xtest::Array{T, 2}, Ytest::Array{T, 1}; seed = 1, initialColsRecord::recordType{T} = recordType{T}(), printIter = true, updateInterval = 2.0) where T <: Real
	# T = eltype(X)
	Random.seed!(seed)
	N = size(X, 2) #number of columns in configuration
	M = round(Int64, N*log(N)) #number of steps to sample nearest neighbors
	# M = 100
	initialColsVec = rand(Bool, size(X, 2))

	newX = [ones(T, length(Y)) X]
	newXtest = [ones(T, length(Ytest)) Xtest]
	println("Beginning regression column sampling")
	# initialCols = findall(initialColsVec)
	(trainErr, testErr, R, initialCols) = if sum(initialColsVec) == 0
		# initialColsVec = fill(false, size(X, 2))
		initialCols = findall(initialColsVec)
		println("Calculating bias term error as a baseline")
		meanY = mean(Y)
		trainErr = sum(abs2, Y .- meanY)/length(Y)
		testErr = sum(abs2, Ytest .- meanY)/length(Ytest)
		_, R = qr([ones(T, length(Y)) view(X, :, initialCols)])
		(trainErr, testErr, R, initialCols)
	else
		if haskey(initialColsRecord, initialColsVec)
			(errs, R, cols) = initialColsRecord[initialColsVec]
			println("Using dictionary to get errors for initial columns ", cols)
			(errs[1], errs[2], R, cols)
		else
			initialCols = findall(initialColsVec)
			println("Finding errors for initial columns ", initialCols)
			_, R = qr([ones(T, length(Y)) view(X, :, initialCols)])
			errs = calcLinRegErr(R, view(newX, :, [1; initialCols .+ 1]), Y, view(newXtest, :, [1; initialCols .+ 1]), Ytest)
			(errs[1], errs[2], R, initialCols)
		end
	end
		
	println(string("Got training and test set errors of : ", (trainErr, testErr), " using ", sum(initialColsVec), "/", length(initialColsVec), " columns"))
	println("--------------------------------------------------------------------")
	println("Setting up $(size(X, 2)) columns for step updates")
	
	Tsteps = fill(Inf, 10*M)

	xCols = [X[:, i] for i in 1:size(X, 2)]
	xTestCols = [Xtest[:, i] for i in 1:size(Xtest, 2)]

	#add line padding for print update if print is turned on
	if printIter
		lines = ceil(Int64, size(X, 2)/20) + 13
		for i in 1:lines-1 
			println()
		end
		println("Waiting for first step to complete")
	else
		println("Starting $(length(Tsteps)) steps of sampling process without printing updates")
	end

	(errsRecord, colsRecord) = runGibbsStep((xCols, newX, Y, xTestCols, newXtest, Ytest), R, initialCols, [("", initialColsVec, (trainErr, testErr))], push!(initialColsRecord, initialColsVec => ((trainErr, testErr), R, initialCols)), Tsteps, printStep=printIter, updateInterval = updateInterval)
	sampleCols = [a[2] for a in errsRecord][5*M+1:end]
	sampleErrs = [a[3] for a in errsRecord][5*M+1:end]

	colNums = [sum(a) for a in sampleCols]
	testErrs = [a[2] for a in sampleErrs]
	left = ["", "Mean", "Median", "Std"]
	row1 = ["% Cols" "Test Err"]
	row2 = [mean(colNums ./ N) mean(testErrs)]
	row3 = [median(colNums ./ N) median(testErrs)]
	row4 = [std(colNums ./ N) std(testErrs)]
	fileOut = [left [row1; row2; row3; row4]]
	writedlm("$(name)_$(seed)Seed_randomConfigSampleStats.csv", fileOut, ',')
	(errsRecord, colsRecord, sampleCols, sampleErrs)
end

function runTempCalibrate(X::Array{T, 2}, Y::Array{T, 1}, Xtest::Array{T, 2}, Ytest::Array{T, 1}; seed = 1, initialColsRecord::recordType{T} = recordType{T}(), printIter = true, updateInterval = 2.0, chi = 0.9) where T <: Real
	# T = eltype(X)
	Random.seed!(seed)
	initialColsVec = rand(Bool, size(X, 2))

	newX = [ones(T, length(Y)) X]
	newXtest = [ones(T, length(Ytest)) Xtest]
	println("Beginning temperature calibration with a target select rate of $chi")
	# initialCols = findall(initialColsVec)
	(trainErr, testErr, R, initialCols) = if sum(initialColsVec) == 0
		# initialColsVec = fill(false, size(X, 2))
		initialCols = findall(initialColsVec)
		println("Calculating bias term error as a baseline")
		meanY = mean(Y)
		trainErr = sum(abs2, Y .- meanY)/length(Y)
		testErr = sum(abs2, Ytest .- meanY)/length(Ytest)
		_, R = qr([ones(T, length(Y)) view(X, :, initialCols)])
		(trainErr, testErr, R, initialCols)
	else
		if haskey(initialColsRecord, initialColsVec)
			(errs, R, cols) = initialColsRecord[initialColsVec]
			println("Using dictionary to get errors for initial columns ", cols)
			(errs[1], errs[2], R, cols)
		else
			initialCols = findall(initialColsVec)
			println("Finding errors for initial columns ", initialCols)
			_, R = qr([ones(T, length(Y)) view(X, :, initialCols)])
			errs = calcLinRegErr(R, view(newX, :, [1; initialCols .+ 1]), Y, view(newXtest, :, [1; initialCols .+ 1]), Ytest)
			(errs[1], errs[2], R, initialCols)
		end
	end
		
	println(string("Got training and test set errors of : ", (trainErr, testErr), " using ", sum(initialColsVec), "/", length(initialColsVec), " columns"))
	println("--------------------------------------------------------------------")
	println("Setting up $(size(X, 2)) columns for step updates")

	xCols = [X[:, i] for i in 1:size(X, 2)]
	xTestCols = [Xtest[:, i] for i in 1:size(Xtest, 2)]

	#add line padding for print update if print is turned on
	if printIter
		lines = ceil(Int64, size(X, 2)/20) + 13
		for i in 1:lines-1 
			println()
		end
		println("Waiting for first step to complete")
	else
		println("Starting $(length(Tsteps)) steps of sampling process without printing updates")
	end

	calibrateGibbsTemp((xCols, newX, Y, xTestCols, newXtest, Ytest), R, initialCols, [("", initialColsVec, (trainErr, testErr))], push!(initialColsRecord, initialColsVec => ((trainErr, testErr), R, initialCols)); printStep = printIter, updateInterval = updateInterval, chi = chi)
end

function runStationarySteps(X::Array{T, 2}, Y::Array{T, 1}, Xtest::Array{T, 2}, Ytest::Array{T, 1}, initialColsVec::Array{Bool, 1}, initialColsRecord::recordType{T}, initialTemp::Float64, C0::Float64; seed = 1, printIter = true, updateInterval = 2.0, delt = 0.001, M = round(Int64, size(X, 2)*log(size(X, 2)))) where T <: Real
	# T = eltype(X)
	Random.seed!(seed)

	newX = [ones(T, length(Y)) X]
	newXtest = [ones(T, length(Ytest)) Xtest]
	println()
	println("-------------------------------------------------------------------------------------------------------------------")
	println("Beginning quasistatic steps at temperature of $initialTemp with a delta of $delt and equilibrium plateau of $M steps")
	println("-------------------------------------------------------------------------------------------------------------------")
	println()
	# initialCols = findall(initialColsVec)
	(trainErr, testErr, R, initialCols) = if sum(initialColsVec) == 0
		# initialColsVec = fill(false, size(X, 2))
		initialCols = findall(initialColsVec)
		println("Calculating bias term error as a baseline")
		meanY = mean(Y)
		trainErr = sum(abs2, Y .- meanY)/length(Y)
		testErr = sum(abs2, Ytest .- meanY)/length(Ytest)
		_, R = qr([ones(T, length(Y)) view(X, :, initialCols)])
		(trainErr, testErr, R, initialCols)
	else
		if haskey(initialColsRecord, initialColsVec)
			(errs, R, cols) = initialColsRecord[initialColsVec]
			println("Using dictionary to get errors for initial columns ", cols)
			(errs[1], errs[2], R, cols)
		else
			initialCols = findall(initialColsVec)
			println("Finding errors for initial columns ", initialCols)
			_, R = qr([ones(T, length(Y)) view(X, :, initialCols)])
			errs = calcLinRegErr(R, view(newX, :, [1; initialCols .+ 1]), Y, view(newXtest, :, [1; initialCols .+ 1]), Ytest)
			(errs[1], errs[2], R, initialCols)
		end
	end
		
	println(string("Got training and test set errors of : ", (trainErr, testErr), " using ", sum(initialColsVec), "/", length(initialColsVec), " columns"))
	println("--------------------------------------------------------------------")
	println("Setting up $(size(X, 2)) columns for step updates")

	xCols = [X[:, i] for i in 1:size(X, 2)]
	xTestCols = [Xtest[:, i] for i in 1:size(Xtest, 2)]

	N = length(xCols)
	# M = round(Int64, N*log(N))
	Tsteps = fill(initialTemp, M)
	

	#add line padding for print update if print is turned on
	lines = ceil(Int64, size(X, 2)/20) + 13
	if printIter
		println("On initial step using starting temperature of $initialTemp")
		for j in 1:lines-1
			println()
		end
	else
		println("Starting $(length(Tsteps)) steps of sampling process without printing updates")
	end

	(errsRecord, colsRecord, costSequence, acceptRate, dictHitRate, currentCols, R, iterTime) = runGibbsStep((xCols, newX, Y, xTestCols, newXtest, Ytest), R, initialCols, [("", initialColsVec, (trainErr, testErr))], push!(initialColsRecord, initialColsVec => ((trainErr, testErr), R, initialCols)), Tsteps; printStep = printIter, updateInterval = updateInterval)
	Cs = mean(costSequence)

	function calcAcceptRate(C)
		l = length(C)
		numAcc = 0
		for i in 1:(l-1)
			if C[i] != C[i+1]
				numAcc += 1
			end
		end
		numAcc/(l-1)
	end

	Ar = calcAcceptRate(costSequence)

	tempRecord = [(initialTemp, Cs, Ar)]
	
	sig = std(costSequence)
	newTemp = initialTemp/(1+(log(1+delt)*initialTemp/(3*sig)))
	dT = newTemp - initialTemp
	dC = Cs - C0
	thresh = abs(dC/dT * newTemp/C0)

	i = 1
	# while !isnan(newTemp)
	while thresh > eps(Cs)
		if printIter
			print("\u001b[$(lines+1)F") #move cursor to beginning of lines lines+1 lines up
			print("\u001b[2K") #clear entire line
			println("Reducing temperature from #$(i-1):$initialTemp to #$i:$newTemp with thresh:$thresh")
			print("\u001b[$(lines+1)E") #move cursor to beginning of lines lines+1 lines down
		end
		Tsteps = fill(newTemp, M)
		initialTemp = newTemp
		# currentCols = colsRecord[errsRecord[end][2]][3]
		# R = colsRecord[errsRecord[end][2]][2]
		(errsRecord, colsRecord, costSequence, acceptRate, dictHitRate, currentCols, R, iterTime) = runGibbsStep((xCols, newX, Y, xTestCols, newXtest, Ytest), R, currentCols, errsRecord, colsRecord, Tsteps; printStep = printIter, updateInterval = updateInterval, acceptRate = acceptRate, dictHitRate = dictHitRate, iterTime = iterTime)
		Ar = calcAcceptRate(costSequence)
		push!(tempRecord, (newTemp, mean(costSequence), Ar))

		sig = std(costSequence)
		newTemp = initialTemp/(1+(log(1+delt)*initialTemp/(3*sig)))
		dC = mean(costSequence) - Cs
		Cs = mean(costSequence)
		dT = newTemp - initialTemp
		thresh = abs(dC/dT * newTemp/C0)
		i += 1
	end	

	colsCheck = true
	while colsCheck
		if printIter
			print("\u001b[$(lines+1)F") #move cursor to beginning of lines lines+1 lines up
			print("\u001b[2K") #clear entire line
			println("Confirming local minimum at a temperature of 0.0")
			print("\u001b[$(lines+1)E") #move cursor to beginning of lines lines+1 lines down
		end
		Tsteps = fill(0.0, round(Int64, N*log(N)))
		currentCols = colsRecord[errsRecord[end][2]][3]
		R = colsRecord[errsRecord[end][2]][2]
		(errsRecord, colsRecord, costSequence, acceptRate, dictHitRate, currentCols, R, iterTime) = runGibbsStep((xCols, newX, Y, xTestCols, newXtest, Ytest), R, currentCols, errsRecord, colsRecord, Tsteps; printStep = printIter, updateInterval = updateInterval, acceptRate = acceptRate, dictHitRate = dictHitRate, iterTime = iterTime)
		Ar = calcAcceptRate(costSequence)
		push!(tempRecord, (0.0, mean(costSequence), Ar))
		colsCheck = (length(unique(costSequence)) > 1) #verify that sequence is still fluctuating
	end

	(errsRecord, colsRecord, tempRecord)
end

function runGreedyStepwise(X::Array{T, 2}, Y::Array{T, 1}, Xtest::Array{T, 2}, Ytest::Array{T, 1}; seed = 1, initialColsRecord::recordType{T} = recordType{T}(), printIter = true, updateInterval = 2.0) where T <: Real
	# T = eltype(X)
	Random.seed!(seed)
	N = size(X, 2) #number of columns in configuration
	M = round(Int64, N*log(N)) #number of steps to sample nearest neighbors
	initialColsVec = rand(Bool, N)

	newX = [ones(T, length(Y)) X]
	newXtest = [ones(T, length(Ytest)) Xtest]
	println("Beginning greedy stepwise column selection")
	# initialCols = findall(initialColsVec)
	(trainErr, testErr, R, initialCols) = if sum(initialColsVec) == 0
		# initialColsVec = fill(false, size(X, 2))
		println("Calculating bias term error as a baseline")
		initialCols = findall(initialColsVec)
		meanY = mean(Y)
		trainErr = sum(abs2, Y .- meanY)/length(Y)
		testErr = sum(abs2, Ytest .- meanY)/length(Ytest)
		_, R = qr(view(newX, :, [1; 1 .+ initialCols]))
		(trainErr, testErr, R, initialCols)
	else
		if haskey(initialColsRecord, initialColsVec)
			(errs, R, cols) = initialColsRecord[initialColsVec]
			println("Using dictionary to get errors for initial columns ", cols)
			(errs[1], errs[2], R, cols)
		else
			initialCols = findall(initialColsVec)
			println("Finding errors for initial columns ", initialCols)
			_, R = qr(view(newX, :, [1; initialCols .+ 1]))
			errs = calcLinRegErr(R, view(newX, :, [1; initialCols .+ 1]), Y, view(newXtest, :, [1; initialCols .+ 1]), Ytest)
			(errs[1], errs[2], R, initialCols)
		end
	end
	GC.gc()

	println(string("Got training and test set errors of : ", (trainErr, testErr), " using ", sum(initialColsVec), "/", length(initialColsVec), " columns"))
	println("--------------------------------------------------------------------")
	println("Setting up $N columns for step updates")
	
	Tsteps = fill(0.0, M)

	xCols = [X[:, i] for i in 1:N]
	xTestCols = [Xtest[:, i] for i in 1:N]

	#add line padding for print update if print is turned on
	if printIter
		lines = ceil(Int64, N/20) + 13
		for i in 1:lines-1 
			println()
		end
		println("Waiting for first step to complete")
	else
		println("Starting $(length(Tsteps)) steps of sampling process without printing updates")
	end

	(errsRecord, colsRecord, costSequence, acceptRate, dictHitRate) = runGibbsStep((xCols, newX, Y, xTestCols, newXtest, Ytest), R, initialCols, [("", initialColsVec, (trainErr, testErr))], push!(initialColsRecord, initialColsVec => ((trainErr, testErr), R, initialCols)), Tsteps, printStep=printIter, updateInterval = updateInterval)
	currentColsVec = errsRecord[end][2]
	currentR = colsRecord[currentColsVec][2]
	currentCols = colsRecord[currentColsVec][3]

	while (costSequence[end] != costSequence[1])
		Tsteps = fill(0.0, M)
		while (costSequence[end] != costSequence[1])
			(errsRecord, colsRecord, costSequence, acceptRate, dictHitRate) = runGibbsStep((xCols, newX, Y, xTestCols, newXtest, Ytest), currentR, currentCols, errsRecord, colsRecord, Tsteps, printStep=printIter, updateInterval = updateInterval, acceptRate = acceptRate, dictHitRate = dictHitRate)
			currentColsVec = errsRecord[end][2]
			currentR = colsRecord[currentColsVec][2]
			currentCols = colsRecord[currentColsVec][3]
		end

		#make final confirmation of local minimum
		Tsteps = fill(0.0, 2*M)
		(errsRecord, colsRecord, costSequence, acceptRate, dictHitRate) = runGibbsStep((xCols, newX, Y, xTestCols, newXtest, Ytest), currentR, currentCols, errsRecord, colsRecord, Tsteps, printStep=printIter, updateInterval = updateInterval, acceptRate = acceptRate, dictHitRate = dictHitRate)
		currentColsVec = errsRecord[end][2]
		currentR = colsRecord[currentColsVec][2]
		currentCols = colsRecord[currentColsVec][3]
	end

	(errsRecord, colsRecord)

end


function runGreedyStepwiseProcess(name::String, regData; seed = 1, colNames = map(a -> string("Col ", a), 1:size(regData[1], 2)), initialColsRecord::recordType{Float64} = recordType{Float64}(), printIter=true, updateInterval = 2.0, N = 100)

	#check input format of regData
	@assert (typeof(regData) <: Tuple) & ((length(regData) == 2) | (length(regData) == 4)) "regData must be a tuple of length two or four"

	@assert typeof(regData[1]) <: AbstractMatrix{T} where T <: AbstractFloat "regData must be a tuple with a matrix of floating point values as the first element"
	@assert typeof(regData[2]) <: AbstractVector{T} where T <: AbstractFloat "regData must be a tuple with a vector of floating point values as the second element"

	
	if length(regData) == 4
		@assert typeof(regData[3]) <: AbstractMatrix{T} where T <: AbstractFloat "regData must be a tuple with a matrix of floating point values as the first element"
		@assert typeof(regData[4]) <: AbstractVector{T} where T <: AbstractFloat "regData must be a tuple with a vector of floating point values as the second element"
		@assert size(regData[3], 1) == length(regData[4]) "Xtest and Ytest must have the same number of rows"
	end

	X = regData[1]
	Y = regData[2]
	Xtest = regData[3]
	Ytest = regData[4]
	@assert size(X, 1) == length(Y) "X and Y must have the same number of rows"

	badCols = findall([sum(X[:, c]) for c in 1:size(X, 2)] == 0)
	@assert isempty(badCols) "Cannot continue with columns $badCols being zero valued"

	n = size(X, 2)
	namePrefix = length(regData) == 4 ? "gibbsStepwiseAnnealQRupdate" : "gibbsStepwiseBICAnnealQRupdate"
	numExamples = size(X, 1)

	# newX = [ones(T, length(Y)) X]
	# newXtest = [ones(T, length(Ytest)) Xtest]
	# xCols = [X[:, i] for i in 1:size(X, 2)]
	# xTestCols = [Xtest[:, i] for i in 1:size(Xtest, 2)]
	
	# newRegData = (xCols, newX, Y, xTestCols, newXtest, Ytest)

	Random.seed!(seed)
	println(string("Generating ", N, " seeds from starting seed ", seed))
	seeds = rand(UInt32, N)

	errResults = []
	for i in 1:N
		startTime = time()
		# (errsRecord, _) = runGreedyStepwise(regData..., seed = seeds[i], initialColsRecord = initialColsRecord, printIter = printIter, updateInterval = updateInterval)
		(errsRecord, _) = runGreedyStepwise(regData..., seed = seeds[i], printIter = printIter, updateInterval = updateInterval)
		push!(errResults, errsRecord)

		iterSeconds = time() - startTime
		iterMins = floor(Int64, iterSeconds/60)
		iterSecs = iterSeconds - 60*iterMins
		iterTimeStr = string(iterMins, ":", round(iterSecs, digits = 2))
		println(string("Iteration $i complete after ", iterTimeStr, " with best columns: ", findall(errsRecord[end][2]), "\nwith training and test errors of ", errsRecord[end][3]))
		println("--------------------------------------------------------------------")
		println()
	end

	bestErrs = [a[end][3][2] for a in errResults]
	trainErrs = [a[end][3][1] for a in errResults]
	bestColList = [a[end][2] for a in errResults]
	numSteps = [length(a) for a in errResults]
	numCols = [sum(a) for a in bestColList]	

	(bestErr, bestInd) = findmin(bestErrs)
	
	bestCols = bestColList[bestInd]
	commonCols = mode(bestColList)
	commonCount = count(a -> a == commonCols, bestColList)
	commonFrac = round(100*commonCount/N, digits = 1)

	commonInd = findfirst(a -> a == commonCols, bestColList)

	usedColsCheck = [a ? "x" : "" for a in bestCols]
	commonColsCheck = [a ? "x" : "" for a in commonCols]
	writedlm("GreedyStepwiseLinReg_$(seed)Seed$(N)Samples_BestUsedCols_$name.txt", [usedColsCheck colNames])
	writedlm("GreedyStepwiseLinReg_$(seed)Seed$(N)Samples_$(commonFrac)%ModeUsedCols_$name.txt", [commonColsCheck colNames])
	writedlm("GreedyStepwiseLinReg_$(seed)Seed$(N)Samples_BestErrsRecord_$name.txt", [(a[1], findall(a[2]), a[3]) for a in errResults[bestInd]])
	left = ["Mean", "Std", "Median", "Mode", "Best"]
	col1 = [mean(bestErrs), std(bestErrs), median(bestErrs), mode(bestErrs), bestErr]
	col2 = [mean(trainErrs), std(trainErrs), trainErrs[findfirst(bestErrs .== median(bestErr))], trainErrs[findfirst(bestErrs .== mode(bestErrs))], trainErrs[findfirst(bestErrs .== bestErr)]]
	col3 = round.(Int64, [mean(numSteps), std(numSteps), median(numSteps), numSteps[commonInd], numSteps[bestInd]])
	col4 = round.(Int64, [mean(numCols), std(numCols), median(numCols), numCols[commonInd], numCols[bestInd]])
	top = ["" "Test Err" "Train Err" "Num Steps" "Num Cols"]
	fileOut = [top; [left col1 col2 col3 col4]]
	writedlm("GreedyStepwiseLinReg_$(seed)Seed$(N)Samples_summaryStats_$name.csv", fileOut, ',') 
	

	header = ["Test Err" "Train Err" "Num Steps" "Num Cols"]
	body = [bestErrs trainErrs numSteps numCols]
	writedlm("GreedyStepwiseLinReg_$(seed)Seed$(N)Samples_FullData_$name.csv", [header; body], ',') 
	(errResults, bestCols, errResults[bestInd][end][3], initialColsRecord)
end

function runQuasistaticAnnealProcess(name::String, regData; seed = 1, colNames = map(a -> string("Col ", a), 1:size(regData[1], 2)), initialColsRecord::recordType{Float64} = recordType{Float64}(), printIter=true, updateInterval = 2.0, chi = 0.9, previousRun = (), delt = 0.001, M = round(Int64, size(regData[1], 2)*log(size(regData[1], 2))))

	#check input format of regData
	@assert (typeof(regData) <: Tuple) & ((length(regData) == 2) | (length(regData) == 4)) "regData must be a tuple of length two or four"

	@assert typeof(regData[1]) <: AbstractMatrix{T} where T <: AbstractFloat "regData must be a tuple with a matrix of floating point values as the first element"
	@assert typeof(regData[2]) <: AbstractVector{T} where T <: AbstractFloat "regData must be a tuple with a vector of floating point values as the second element"

	
	if length(regData) == 4
		@assert typeof(regData[3]) <: AbstractMatrix{T} where T <: AbstractFloat "regData must be a tuple with a matrix of floating point values as the first element"
		@assert typeof(regData[4]) <: AbstractVector{T} where T <: AbstractFloat "regData must be a tuple with a vector of floating point values as the second element"
		@assert size(regData[3], 1) == length(regData[4]) "Xtest and Ytest must have the same number of rows"
	end

	X = regData[1]
	Y = regData[2]
	@assert size(X, 1) == length(Y) "X and Y must have the same number of rows"

	badCols = findall([sum(X[:, c]) for c in 1:size(X, 2)] == 0)
	@assert isempty(badCols) "Cannot continue with columns $badCols being zero valued"

	n = size(X, 2)
	namePrefix = length(regData) == 4 ? "quasistaticGibbsAnnealQRupdate" : "quasistaticGibbsBICAnnealQRupdate"
	numExamples = size(X, 1)
	
	startTime = time()
	if isempty(previousRun)
		(errsRecord1, colsRecord, Tsteps, accs, testErrs) = runTempCalibrate(regData..., seed = seed, initialColsRecord = initialColsRecord, printIter = true, updateInterval = 2.0, chi = chi)
		l = length(accs)
		accRate = mean(accs[round(Int64, l/2):end])
		newTemp = Tsteps[end]
		println("Achieved an acceptance rate of $accRate compared to a target of $chi with a final temperature of $newTemp")
	else
		errsRecord1 = previousRun[1]
		colsRecord = previousRun[2]
		newTemp = previousRun[3]
		println("Starting with a previous run with a temperature of $newTemp")
	end
	startingColsVec = errsRecord1[end][2]

	C0 = mean(testErrs[round(Int64, l/2):end])
	(errsRecord2, colsRecord, tempRecord) = runStationarySteps(regData..., startingColsVec, colsRecord, newTemp, C0, seed = seed, delt = delt, M = M)
	
	iterSeconds = time() - startTime
	iterMins = floor(Int64, iterSeconds/60)
	iterSecs = iterSeconds - 60*iterMins
	iterTimeStr = string(iterMins, ":", round(iterSecs, digits = 2))

	errsRecord = [errsRecord1; errsRecord2[2:end]]
	testErrs = [a[3][2] for a in errsRecord]
	(bestTestErr, bestInd) = findmin(testErrs)
	bestRecord = errsRecord[bestInd]
	bestCols = bestRecord[2]

	recordFileOut = [("Col Changes", "Columns", "Train Err", "Test Err"); reduce(vcat, [(bestRecord[1], findall(bestCols), bestRecord[3][1], bestRecord[3][2]); [(a[1], findall(a[2]), a[3][1], a[3][2]) for a in errsRecord]])]

	tempFileOut = [["Temperature" "Cost Avg" "Accept Rate"]; reduce(vcat, [[a[1] a[2] a[3]] for a in tempRecord])]

	println(string("Quasitatic annealing complete after ", iterTimeStr, " with best columns: ", findall(bestCols), "\nwith training and test errors of ", bestRecord[3]))

	usedColsCheck = [a ? "x" : "" for a in bestCols]
	writedlm("QuasistaticAnnealLinReg_$(seed)Seed$(delt)Delt$(M)Steps$(chi)CHI_BestUsedCols_$name.txt", [usedColsCheck colNames])
	
	writedlm("QuasistaticAnnealLinReg_$(seed)Seed$(delt)Delt$(M)Steps$(chi)CHI_Record_$name.txt", recordFileOut, '\t')
	
	writedlm("QuasistaticAnnealLinReg_$(seed)Seed$(delt)Delt$(M)Steps$(chi)CHI_TemperatureSteps_$name.csv", tempFileOut, ',')

	(errsRecord, colsRecord, tempRecord, (bestTestErr, bestInd))
end

function runQuasistaticAnnealProcessV2(name::String, regData; seed = 1, colNames = map(a -> string("Col ", a), 1:size(regData[1], 2)), initialColsRecord::recordType{Float64} = recordType{Float64}(), printIter=true, updateInterval = 2.0, chi = 0.9, delt = 0.001, M = round(Int64, size(regData[1], 2)*log(size(regData[1], 2))))

	#check input format of regData
	@assert (typeof(regData) <: Tuple) & ((length(regData) == 2) | (length(regData) == 4)) "regData must be a tuple of length two or four"

	@assert typeof(regData[1]) <: AbstractMatrix{T} where T <: AbstractFloat "regData must be a tuple with a matrix of floating point values as the first element"
	@assert typeof(regData[2]) <: AbstractVector{T} where T <: AbstractFloat "regData must be a tuple with a vector of floating point values as the second element"

	
	if length(regData) == 4
		@assert typeof(regData[3]) <: AbstractMatrix{T} where T <: AbstractFloat "regData must be a tuple with a matrix of floating point values as the first element"
		@assert typeof(regData[4]) <: AbstractVector{T} where T <: AbstractFloat "regData must be a tuple with a vector of floating point values as the second element"
		@assert size(regData[3], 1) == length(regData[4]) "Xtest and Ytest must have the same number of rows"
	end

	X = regData[1]
	Y = regData[2]
	@assert size(X, 1) == length(Y) "X and Y must have the same number of rows"

	badCols = findall([sum(X[:, c]) for c in 1:size(X, 2)] == 0)
	@assert isempty(badCols) "Cannot continue with columns $badCols being zero valued"

	n = size(X, 2)
	namePrefix = length(regData) == 4 ? "quasistaticGibbsAnnealQRupdate" : "quasistaticGibbsBICAnnealQRupdate"
	numExamples = size(X, 1)
	
	namePrefix = "QuasistaticAnnealReheatLinReg"
	
	(errsRecord1, colsRecord, Tsteps, accs, testErrs) = runTempCalibrate(regData..., seed = seed, initialColsRecord = initialColsRecord, printIter = true, updateInterval = 2.0, chi = chi)
	l = length(accs)
	accRate = mean(accs[round(Int64, l/2):end])
	newTemp = Tsteps[end]
	println("Achieved an acceptance rate of $accRate compared to a target of $chi with a final temperature of $newTemp")


	# (newBestTestErr, bestInd) = findmin([a[3][2] for a in errsRecord1])
	newBestInd = length(errsRecord1)
	newBestRecord = errsRecord1[end]
	newBestTestErr = newBestRecord[3][2]
	newBestCols = newBestRecord[2]
	startingColsVec = newBestCols

	C0 = mean(testErrs[round(Int64, l/2):end])
	errsRecord = deepcopy(errsRecord1)
	bestTestErr = Inf
	fullTempRecord = []
	origSeed = seed
	while newBestTestErr < bestTestErr
		bestTestErr = newBestTestErr
		startTime = time()
		(newErrsRecord, colsRecord, tempRecord) = runStationarySteps(regData..., startingColsVec, colsRecord, newTemp, C0, seed = seed, delt = delt, M = M)

		seed = rand(UInt32)

		startingColsVec = newErrsRecord[end][2]
		temps = [a[1] for a in tempRecord]
		ARs = [a[3] for a in tempRecord]
		avgErrs = [a[2] for a in tempRecord]
		ind = findfirst(a -> a < 0.25, ARs)

		if ind > 1
			newTemp = (temps[ind] + temps[ind-1])/2
			C0 = (avgErrs[ind] + avgErrs[ind-1])/2
		else
			newTemp = temps[ind]
			C0 = avgErrs[ind]
		end

		iterSeconds = time() - startTime
		iterMins = floor(Int64, iterSeconds/60)
		iterSecs = iterSeconds - 60*iterMins
		iterTimeStr = string(iterMins, ":", round(iterSecs, digits = 2))

		errsRecord = [errsRecord; newErrsRecord[2:end]]
		fullTempRecord = [fullTempRecord; tempRecord]
		testErrs = [a[3][2] for a in errsRecord]
		(newBestTestErr, bestInd) = findmin(testErrs)
		newBestRecord = errsRecord[bestInd]
		newBestCols = newBestRecord[2]

		println(string("Quasitatic annealing complete after ", iterTimeStr, " with best columns: ", findall(newBestCols), "\nwith training and test errors of ", newBestRecord[3]))
		if newBestTestErr < bestTestErr
			println(string("Resetting temperature to ", newTemp, " to try to find a better configuration"))
		else
			println("No improvement found after reheating so terminating process")
		end
	end

	recordFileOut = [("Col Changes", "Columns", "Train Err", "Test Err"); reduce(vcat, [(newBestRecord[1], findall(newBestCols), newBestRecord[3][1], newBestRecord[3][2]); [(a[1], findall(a[2]), a[3][1], a[3][2]) for a in errsRecord]])]

	tempFileOut = [["Temperature" "Cost Avg" "Accept Rate"]; reduce(vcat, [[a[1] a[2] a[3]] for a in fullTempRecord])]


	usedColsCheck = [a ? "x" : "" for a in newBestCols]
	writedlm("$(namePrefix)_$(origSeed)Seed$(delt)Delt$(M)Steps$(chi)CHI_BestUsedCols_$name.txt", [usedColsCheck colNames])
	
	writedlm("$(namePrefix)_$(origSeed)Seed$(delt)Delt$(M)Steps$(chi)CHI_Record_$name.txt", recordFileOut, '\t')
	
	writedlm("$(namePrefix)_$(origSeed)Seed$(delt)Delt$(M)Steps$(chi)CHI_TemperatureSteps_$name.csv", tempFileOut, ',')

	(errsRecord, colsRecord, fullTempRecord, (newBestTestErr, newBestInd))
end



function runStepwiseAnnealProcess(name::String, regData; seed = 1, colNames = map(a -> string("Col ", a), 1:size(X, 2)), initialColsRecord = Dict{Array{Bool, 1}, Tuple{Tuple{T, T}, Matrix{T}, Vector{Int64}}}(), printIter=true, updateInterval = 1.0)

	#check input format of regData
	@assert (typeof(regData) <: Tuple) & ((length(regData) == 2) | (length(regData) == 4)) "regData must be a tuple of length two or four"

	@assert typeof(regData[1]) <: AbstractMatrix{T} where T <: AbstractFloat "regData must be a tuple with a matrix of floating point values as the first element"
	@assert typeof(regData[2]) <: AbstractVector{T} where T <: AbstractFloat "regData must be a tuple with a vector of floating point values as the second element"

	
	if length(regData) == 4
		@assert typeof(regData[3]) <: AbstractMatrix{T} where T <: AbstractFloat "regData must be a tuple with a matrix of floating point values as the first element"
		@assert typeof(regData[4]) <: AbstractVector{T} where T <: AbstractFloat "regData must be a tuple with a vector of floating point values as the second element"
		@assert size(regData[3], 1) == length(regData[4]) "Xtest and Ytest must have the same number of rows"
	end

	X = regData[1]
	Y = regData[2]
	@assert size(X, 1) == length(Y) "X and Y must have the same number of rows"

	badCols = findall([sum(X[:, c]) for c in 1:size(X, 2)] == 0)
	@assert isempty(badCols) "Cannot continue with columns $badCols being zero valued"

	n = size(X, 2)
	namePrefix = length(regData) == 4 ? "gibbsStepwiseAnnealQRupdate" : "gibbsStepwiseBICAnnealQRupdate"
	numExamples = size(X, 1)

	Random.seed!(seed)
	println(string("Starting with seed ", seed))
	initialColsVec = if seed == 1
		fill(false, size(X, 2))
	elseif seed == 2
		fill(true, size(X, 2))
	else
		rand(Bool, size(X, 2))
	end

	startTime = time()
	(errsRecord1, colsRecord1, tempRecord1) = runStepwiseAnneal(regData..., iter = 1, initialColsRecord = initialColsRecord, initialColsVec = initialColsVec, printIter=printIter, updateInterval = updateInterval) #, initialColsVec=initialColsVec)
	iterSeconds = time() - startTime
	iterMins = floor(Int64, iterSeconds/60)
	iterSecs = iterSeconds - 60*iterMins
	iterTimeStr = string(iterMins, ":", round(iterSecs, digits = 2))
	errs = [a[4] for a in errsRecord1]
	sortInd = sortperm(errs)
	errDelt = quantile(abs.(errs[2:end] .- errs[1:end-1]), 0.25)

	writedlm(string(namePrefix, "_", T, "_", name, "_", seed, "Seed_tempRecord1.csv"), [[a[1] for a in tempRecord1] [a[2] for a in tempRecord1]], ',')
	writedlm(string(namePrefix, "_", T, "_", name, "_", seed, "Seed_record1.txt"), [errsRecord1[sortInd[1]]; errsRecord1])
	bestCols1 = errsRecord1[sortInd[1]][2]
	bestTrainErr1 = errsRecord1[sortInd[1]][3]
	bestTestErr1 = errsRecord1[sortInd[1]][4]
	println(string("Iteration 1 complete after ", iterTimeStr, " with best columns: ", bestCols1, "\nwith training and test errors of ", (bestTrainErr1, bestTestErr1)))
	println("--------------------------------------------------------------------")
	println()
	# return (bestCols1, [in(i, bestCols1) ? "x" : "" for i in 1:length(colNames)], bestTrainErr1, bestTestErr1, colsRecord1, tempRecord1)
	newTemps = getNewTemps(tempRecord1)

	println("Starting second iteration from this starting point")
	colsVec2 = [in(i, bestCols1) ? true : false for i in 1:size(X, 2)]
	startTime = time()
	(errsRecord2, colsRecord2, tempRecord2) = runStepwiseAnneal(regData..., iter = 2, initialColsVec = colsVec2, initialColsRecord = colsRecord1, printIter=printIter, updateInterval = updateInterval, errDelt = errDelt, Tpoints = newTemps)
	iterSeconds = time() - startTime
	iterMins = floor(Int64, iterSeconds/60)
	iterSecs = iterSeconds - 60*iterMins
	iterTimeStr = string(iterMins, ":", round(iterSecs, digits = 2))
	errs = [a[4] for a in errsRecord2]
	sortInd = sortperm(errs)
	errDelt = quantile(abs.(errs[2:end] .- errs[1:end-1]), 0.25)
	writedlm(string(namePrefix, "_", T, "_", name, "_", seed, "Seed_tempRecord2.csv"), [[a[1] for a in tempRecord2] [a[2] for a in tempRecord2]], ',')
	writedlm(string(namePrefix, "_", T, "_", name, "_", seed, "Seed_record2.txt"), [errsRecord2[sortInd[1]]; errsRecord2])
	bestCols2 = errsRecord2[sortInd[1]][2]
	bestTrainErr2 = errsRecord2[sortInd[1]][3]
	bestTestErr2 = errsRecord2[sortInd[1]][4]
	newTemps = getNewTemps(tempRecord2, 0.5) #only use second half to record to determine new temp range
	println(string("Iteration 2 complete after ", iterTimeStr, " with best columns: ", bestCols2, "\nwith training and test errors of ", (bestTrainErr2, bestTestErr2)))
	println("--------------------------------------------------------------------")
	println()

	if bestCols2 == bestCols1
		println("Skipping further iterations because best columns are still from iteration 1")
		println()
		println()
		usedCols = bestCols2
		usedColsCheck = [in(i, usedCols) ? "x" : "" for i in 1:length(colNames)]
		writedlm(string(namePrefix, "LinReg_", seed, "Seed_UsedCols_", name, ".txt"), [usedColsCheck colNames])
		(usedCols, usedColsCheck, bestTrainErr2, bestTestErr2, colsRecord2, tempRecord2)
	else
		iter = 3
		newBestCols = bestCols2
		oldBestCols = bestCols1
		newColsRecord = colsRecord2
		newTempRecord = tempRecord2
		newBestTrainErr = Inf
		newBestTestErr = Inf
		while newBestCols != oldBestCols
			oldBestCols = newBestCols 
			println(string("Starting iteration ", iter, " from previous best columns"))
			newColsVec = [in(i, oldBestCols) ? true : false for i in 1:size(X, 2)]
			startTime = time()
			(newErrsRecord, newColsRecord, newTempRecord) = runStepwiseAnneal(regData..., iter = iter, initialColsVec = newColsVec, initialColsRecord = newColsRecord, printIter=printIter, updateInterval = updateInterval, errDelt = errDelt, Tpoints = newTemps)
			newTemps = getNewTemps(newTempRecord, 0.5) #only use second half to record to determine new temp range
			iterSeconds = time() - startTime
			iterMins = floor(Int64, iterSeconds/60)
			iterSecs = iterSeconds - 60*iterMins
			iterTimeStr = string(iterMins, ":", round(iterSecs, digits = 2))			
			errs = [a[4] for a in newErrsRecord]
			sortInd = sortperm(errs)
			errDelt = quantile(abs.(errs[2:end] .- errs[1:end-1]), 0.25)
			writedlm(string(namePrefix, "_", T, "_", name, "_", seed, "Seed_tempRecord$iter.csv"), [[a[1] for a in newTempRecord] [a[2] for a in newTempRecord]], ',')
			writedlm(string(namePrefix, "_", T, "_", name, "_", seed, "Seed_record", iter, ".txt"), [newErrsRecord[sortInd[1]]; newErrsRecord])
			newBestCols = newErrsRecord[sortInd[1]][2]
			newBestTrainErr = newErrsRecord[sortInd[1]][3]
			newBestTestErr = newErrsRecord[sortInd[1]][4]
			println(string("Iteration ", iter, " complete after ", iterTimeStr, " with best columns: ", newBestCols, "\nwith training and test errors of ", (newBestTrainErr, newBestTestErr)))
			println("--------------------------------------------------------------------")
			println()
			iter += 1
		end
		println(string("Skipping further iterations because best columns are still from iteration ", iter - 2))
		println()
		println()
		usedCols = newBestCols
		usedColsCheck = [in(i, usedCols) ? "x" : "" for i in 1:length(colNames)]
		writedlm(string(namePrefix, "LinReg_", seed, "Seed_UsedCols_", name, ".txt"), [usedColsCheck colNames])
		(usedCols, usedColsCheck, newBestTrainErr, newBestTestErr, newColsRecord, newTempRecord)
	end
end


