##########################---Find Best Forward Step---#####################
function find_best_add_col!(traincols::Vector{Vector{T}}, y::Vector{T}, colsubset::AbstractVector{W}, R::Matrix{T}, Xtmp::Matrix{T}, besterr::T, bestbic::T) where T <: AbstractFloat where W <: Integer
	l = length(colsubset)
	@assert size(Xtmp) == (length(y), length(traincols)+1)
	candidateCols = setdiff(eachindex(traincols), colsubset)
	println()
	println()
	println()

	tstart = time()
	bestR = R
	bestcol = 0
	newbest = false
	Rnew = Matrix{T}(undef, l+2, l+2)
	r = Vector{T}(undef, length(y))
	for (i, c) in enumerate(candidateCols)
		t = time()
		#update Rnew in place
		qraddcol!(view(Xtmp, :, 1:l+1), R, traincols[c], Rnew, r)
		tR = time() - t

		t = time()
		Xtmp[:, l+2] .= traincols[c]
		tFill = time() - t

		t = time()
		newerr, newbeta = calc_linreg_error(Rnew, view(Xtmp, :, 1:l+2), y)
		newbic = calc_linreg_bic(newerr, view(Xtmp, :, 1:l+2), y)
		tErr = time() - t

		printupdate = (i == 1) || ((time() - tstart) >= 2) 
		
		if printupdate
			print("\r\u1b[K\u1b[A")
			print("\r\u1b[K\u1b[A")
			print("\r\u1b[K\u1b[A")
			println("R update time = $tR, Fill time = $tFill, Fit time = $tErr")
			println(string("Done with column ", c, ": number ", i, " out of ", length(candidateCols), " total candidates"))
			tstart = time()
		end

		if newbic < bestbic
			besterr = newerr
			bestbic = newbic
			bestR = copy(Rnew)
			bestcol = c
			newbest = true
			printupdate && println("Got new best error and BIC of : $(round(newerr, sigdigits = 3)), $(round(newbic, sigdigits = 3))")

		else
			printupdate && println("Got worse error and BIC of :  $(round(newerr, sigdigits = 3)), $(round(newbic, sigdigits = 3))")
		end

	end
	
	if newbest && (bestcol != candidateCols[end])
		#leave Xtmp with best subset
		view(Xtmp, :, l+2) .= traincols[bestcol]
	end

	(bestbic, besterr, bestR, bestcol, newbest)
end

function find_best_add_col!(traincols::Vector{Vector{T}}, y::Vector{T}, testcols::Vector{Vector{T}}, ytest::Vector{T}, colsubset::AbstractVector{W}, R::Matrix{T}, Xtmp::Matrix{T}, Xtmp2::Matrix{T}, besttrainerr::T, besttesterr::T) where T <: AbstractFloat where W <: Integer
	l = length(colsubset)
	@assert size(Xtmp) == (length(y), length(traincols)+1)
	@assert size(Xtmp2) == (length(ytest), length(testcols)+1)
	candidateCols = setdiff(eachindex(traincols), colsubset)
	println()
	println()
	println()
	
	tstart = time()
	bestR = R
	bestcol = 0
	newbest = false
	Rnew = Matrix{T}(undef, l+2, l+2)
	r = Vector{T}(undef, length(y))
	for (i, c) in enumerate(candidateCols)
		t = time()
		#update Rnew in place
		qraddcol!(view(Xtmp, :, 1:l+1), R, traincols[c], Rnew, r)
		tR = time() - t

		t = time()
		Xtmp[:, l+2] .= traincols[c]
		Xtmp2[:, l+2] .= testcols[c]
		tFill = time() - t

		t = time()
		newtrainerr, newbeta = calc_linreg_error(Rnew, view(Xtmp, :, 1:l+2), y)
		newtesterr = calc_linreg_error(newbeta, view(Xtmp2, :, 1:l+2), ytest)
		tErr = time() - t

		printupdate = (i == 1) || ((time() - tstart) >= 2) 
		if printupdate
			print("\r\u1b[K\u1b[A")
			print("\r\u1b[K\u1b[A")
			print("\r\u1b[K\u1b[A")
			println("R update time = $tR, Fill time = $tFill, Fit time = $tErr")
			println(string("Done with column ", c, ": number ", i, " out of ", length(candidateCols), " total candidates"))
			tstart = time()
		end

		if newtesterr < besttesterr
			besttrainerr = newtrainerr
			besttesterr = newtesterr
			bestR = copy(Rnew)
			bestcol = c
			newbest = true
			printupdate && println("Got new best train and test errors of: $(round(newtrainerr, sigdigits = 3)),  $(round(newtesterr, sigdigits = 3))")

		else
			printupdate && println("Got worse train and test error of : $(round(newtrainerr, sigdigits = 3)),  $(round(newtesterr, sigdigits = 3))")
		end

	end

	#leave Xtmp and Xtmp2 in best column state
	if newbest && (bestcol != candidateCols[end])
		view(Xtmp, :, l+2) .= traincols[bestcol]
		view(Xtmp2, :, l+2) .= testcols[bestcol]
	end

	(besttesterr, besttrainerr, bestR, bestcol, newbest)
end

##########################---Find Best Backward Step---#####################
function find_best_remove_col!(traincols::Vector{Vector{T}}, y::Vector{T}, colsubset::AbstractVector{W}, R::Matrix{T}, Xtmp::Matrix{T}, besterr::T, bestbic::T) where T <: AbstractFloat where W <: Integer
	l = length(colsubset)
	@assert size(Xtmp) == (length(y), length(traincols)+1)
	println()
	println()
	println()

	tstart = time()
	bestR = R
	bestcol = 0
	besti = 0
	newbest = false
	#initially have 0, 1, 2, 3, 4, 5
	#iteration 1 leave unchanged and cut off last column: 0, 1, 2, 3, 4
	#iteration 2 want to use: 0, 1, 2, 3, 5
	#iteration 3 want to use: 0, 1, 2, 4, 5
	#iteration 4 want to use: 0, 1, 3, 4, 5
	#iteration 5 want to use: 0, 2, 3, 4, 5
	#iterate through current columns in reverse where at each step column c is being removed
	for (i, c) in Iterators.reverse(enumerate(colsubset))
		t = time()
		Rnew = qrdelcol(R, i+1)
		tR = time() - t

		t = time()
		newerr, newbeta = calc_linreg_error(Rnew, view(Xtmp, :, 1:l), y)
		newbic = calc_linreg_bic(newerr, view(Xtmp, :, 1:l), y)
		tErr = time() - t

		#replace column that was just removed and overwrite the next column to be removed
		t = time()
		if i > 1
			view(Xtmp, :, i) .= traincols[c]
		end
		tFill = time() - t

		printupdate = (i == length(colsubset)) || ((time() - tstart) >= 2)

		if printupdate
			print("\r\u1b[K\u1b[A")
			print("\r\u1b[K\u1b[A")
			print("\r\u1b[K\u1b[A")
			println("R update time = $tR, Fill time = $tFill, Fit time = $tErr")
			println(string("Done with column ", c, ": number ", i, " out of ", length(colsubset), " total candidates"))
			tstart = time()
		end
		if newbic < bestbic
			besterr = newerr
			bestbic = newbic
			bestR = copy(Rnew)
			bestcol = c
			besti = i
			newbest = true
			printupdate && println("Got new best error and BIC of : $(round(newerr, sigdigits = 3)), $(round(newbic, sigdigits = 3))")

		else
			printupdate && println("Got worse error and BIC of : $(round(newerr, sigdigits = 3)), $(round(newbic, sigdigits = 3))")
		end
	end

	if newbest
		#undo column removal back to desired column
		for i in 1:besti-1
			view(Xtmp, :, i+1) .= traincols[colsubset[i]]
		end
	else
		#restore Xtmp to original state
		for i in 1:l
			view(Xtmp, :, i+1) .= traincols[colsubset[i]]
		end
	end
	(bestbic, besterr, bestR, bestcol, newbest)
end

function find_best_remove_col!(traincols::Vector{Vector{T}}, y::Vector{T}, testcols::Vector{Vector{T}}, ytest::Vector{T},colsubset::AbstractVector{W}, R::Matrix{T}, Xtmp::Matrix{T}, Xtmp2::Matrix{T}, besttrainerr::T, besttesterr::T) where T <: AbstractFloat where W <: Integer
	l = length(colsubset)
	@assert size(Xtmp) == (length(y), length(traincols)+1)
	@assert size(Xtmp2) == (length(ytest), length(testcols)+1)

	println()
	println()
	println()

	tstart = time()
	bestR = R
	bestcol = 0
	besti = 0
	newbest = false
	#initially have 0, 1, 2, 3, 4, 5
	#iteration 1 leave unchanged and cut off last column: 0, 1, 2, 3, 4
	#iteration 2 want to use: 0, 1, 2, 3, 5
	#iteration 3 want to use: 0, 1, 2, 4, 5
	#iteration 4 want to use: 0, 1, 3, 4, 5
	#iteration 5 want to use: 0, 2, 3, 4, 5
	#iterate through current columns in reverse where at each step column c is being removed
	for (i, c) in Iterators.reverse(enumerate(colsubset))
		t = time()
		Rnew = qrdelcol(R, i+1)
		tR = time() - t

		t = time()
		newtrainerr, newbeta = calc_linreg_error(Rnew, view(Xtmp, :, 1:l), y)
		newtesterr = calc_linreg_error(newbeta, view(Xtmp2, :, 1:l), ytest)
		tErr = time() - t

		#replace column that was just removed and overwrite the next column to be removed
		t = time()
		if i > 1
			Xtmp[:, i] .= traincols[c]
			Xtmp2[:, i] .= testcols[c]
		end
		tFill = time() - t

		printupdate = (i == length(colsubset)) || ((time() - tstart) >= 2) 
		if printupdate
			print("\r\u1b[K\u1b[A")
			print("\r\u1b[K\u1b[A")
			print("\r\u1b[K\u1b[A")
			println("R update time = $tR, Fill time = $tFill, Fit time = $tErr")
			println(string("Done with column ", c, ": number ", i, " out of ", length(colsubset), " total candidates"))
			tstart = time()
		end

		if newtesterr < besttesterr
			besttrainerr = newtrainerr
			besttesterr = newtesterr
			bestR = copy(Rnew)
			bestcol = c
			besti = i
			newbest = true
			printupdate && println("Got new best train and test error of: $(round(newtrainerr, sigdigits = 3)), $(round(newtesterr, sigdigits = 3))r")
		else
			printupdate && println("Got worse train and test error of: $(round(newtrainerr, sigdigits = 3)), $(round(newtesterr, sigdigits = 3))")
		end
	end

	if newbest
		#undo column removal back to desired column
		for i in 1:besti-1
			view(Xtmp, :, i+1) .= traincols[colsubset[i]]
			view(Xtmp2, :, i+1) .= testcols[colsubset[i]]
		end
	else
		#restore both Xtmps to original state
		for i in 1:l
			view(Xtmp, :, i+1) .= traincols[colsubset[i]]
			view(Xtmp2, :, i+1) .= testcols[colsubset[i]]
		end
	end
	(besttesterr, besttrainerr, bestR, bestcol, newbest)
end

########################----Iterate Steps---#############################
InOutPairCols{T} = Tuple{Vector{U}, U} where U <: Vector{T} where T <: AbstractFloat
function stepwise_forward_init(data::InOutPairCols{T}...) where T <: AbstractFloat
	testerr = (length(data) > 1)
	println("Initializing forward stepwise selection from zero columns")
	println("Calculating bias term error as a baseline")

	y = data[1][2]
	traincols = data[1][1]

	err1 = sum(abs2, (y .- mean(y)))/length(y)
	err2 = testerr ? sum(abs2, (data[2][2] .- mean(y)))/length(data[2][2]) : length(y)*log(err1)+log(length(y))

	str = testerr ? "train and test error" : "error and bic"
	println("Using 0 columns out of a possible $(length(traincols)), $str is: $(round(err1, sigdigits = 3)), $(round(err2, sigdigits = 3))")
	println("----------------------------------------------------------")

	#get initial R vector for bias terms
	_, R = qr(ones(T, length(y), 1))
	#initialize column subset as an empty vector
	colsubset = Vector{Integer}()
	#initialize record for bias terms
	record = [("", copy(sort(colsubset)), err1, err2)]
	#initialize Xtmp
	Xtmp = ones(T, length(y), length(traincols)+1)

	tmpX = if testerr
		Xtmp2 = ones(T, length(data[2][2]), length(data[1][1])+1)
		(Xtmp, Xtmp2)
	else
		(Xtmp,)
	end

	(err2, err1, colsubset, R, record, tmpX...)
end

function stepwise_iterate!(data::NTuple{N, InOutPairCols{T}}, err2::T, err1::T, colsubset::AbstractVector{W}, R::Matrix{T}, record::Vector{U}, tmpX::Matrix{T}...; numsteps::Integer = 0, direction::Bool = true) where T <: AbstractFloat where W <:Integer where U <: Tuple where N

	#direction is forward if true and backward if false, acts as a toggle
	testerr = (length(data) > 1)
	dirstr = direction ? "forward" : "backward"
	dirstr2 = direction ? "adding" : "removing"
	recstr = direction ? "+" : "-"
	func! = direction ? find_best_add_col! : find_best_remove_col!

	
	traincols = data[1][1]
	if direction ? length(colsubset) == length(traincols) : isempty(colsubset)
		println()
		println("Ending $dirstr iteration after $dirstr2 all $(length(traincols)) available columns.")
		println("-----------------------------------------------------------------------")
		return (err2, err1, colsubset, R, record, tmpX..., numsteps)
	end

	if numsteps == 0
		println("Starting $dirstr selection from $(length(colsubset)) columns")
		println()
		println()
		println()
		println()
	end

	print("\r\u1b[K\u1b[A")
	print("\r\u1b[K\u1b[A")
	print("\r\u1b[K\u1b[A")
	print("\r\u1b[K\u1b[A")

	str = testerr ? "train and test error" : "error and bic"
	println("On $dirstr step $numsteps using $(length(colsubset)) columns out of a possible $(length(traincols)) with best $str of: $(round(err1, sigdigits = 3)), $(round(err2, sigdigits = 3))")
	(besterr2, besterr1, bestR, bestcol, newbest) = func!(reduce((a, b) -> (a..., b...), data)..., colsubset, R, tmpX..., err1, err2)

	if !newbest
		println()
		println("Ending $dirstr iteration using $(length(colsubset)) out of $(length(traincols)) available columns and $numsteps $dirstr steps")
		println("-----------------------------------------------------------------------")
		(err2, err1, colsubset, R, record, tmpX..., numsteps)
	else
		#update colsubset
		direction ? push!(colsubset, bestcol) : setdiff!(colsubset, bestcol)
		#update record
		push!(record, ("$recstr$bestcol", copy(sort(colsubset)), besterr1, besterr2))
		stepwise_iterate!(data, besterr2, besterr1, colsubset, bestR, record, tmpX..., numsteps = numsteps+1, direction = direction)
	end
end

###############################----Run Stepwise----#######################################
InOutPair{T} = Tuple{Matrix{T}, Vector{T}} where T <: AbstractFloat
# function run_stepwise_reg(data::T...; colnames = ["Col $a" for a in 1:size(data[1][1], 2)]) where T <: NTuple{N, InOutPair{U}} where U <: AbstractFloat where N
function run_stepwise_reg(data::InOutPair{T}...; colnames = ["Col $a" for a in 1:size(data[1][1], 2)]) where T <: AbstractFloat
	println()
	println("==================================================================")
	println("Starting stepwise selection with $(size(data[1][1], 2)) possible columns")
	println("==================================================================")
	#convert X's in data to vectors of columns
	datainput = map(a -> (collect.(eachcol(a[1])), a[2]), data)

	direction = true #start in forward direction
	output = stepwise_iterate!(datainput, stepwise_forward_init(datainput...)...)

	while output[end] > 0 #check that the previous direction took successful steps
		direction = !direction #change direction
		#pass previous output into iteration leaving out the number of steps
		output = stepwise_iterate!(datainput, output[1:end-1]..., direction = direction)
	end

	colsubset = output[3]
	record = output[5]
	println("=================================================================")
	println("Finished stepwise selection with $(length(record[end][2])) columns and final errors: $(record[end][end-1:end])")
	println("=================================================================")
	println()
	usedcols = sort(colsubset)
	usedcolscheck = [in(i, usedcols) ? "x\t$n" : " \t$n" for (i, n) in enumerate(colnames)]
	(colsubset, usedcolscheck, record)
end


