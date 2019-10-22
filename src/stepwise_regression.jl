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
	for (i, c) in enumerate(candidateCols)
		t = time()
		Rnew = qraddcol(view(Xtmp, :, 1:l+1), R, traincols[c])
		tR = time() - t

		t = time()
		Xtmp[:, l+2] .= traincols[c]
		tFill = time() - t

		t = time()
		newerr, newbeta = calc_linreg_error(Rnew, view(Xtmp, :, 1:l+2), y)
		newbic = calc_linreg_bic(newerr, view(Xtmp, :, 1:l+2), y)
		tErr = time() - t

		printupdate = (i == 1) || (i == length(candidateCols)) || ((time() - tstart) >= 2) 
		
		if printupdate
			print("\r\u1b[K\u1b[A")
			print("\r\u1b[K\u1b[A")
			print("\r\u1b[K\u1b[A")
			println("R update time = $tR, Fill time = $tFill, Fit time = $tErr")
			println(string("Done with column ", c, ": number ", i, " out of ", length(candidateCols), " total candidates"))
		end

		if newbic < bestbic
			besterr = newerr
			bestbic = newbic
			bestR = Rnew
			bestcol = c
			newbest = true
			printupdate && println("Got new best error and BIC of : $(round(newerr, sigdigits = 3)), $(round(newbic, sigdigits = 3))")

		else
			printupdate && println("Got worse error and BIC of :  $(round(newerr, sigdigits = 3)), $(round(newbic, sigdigits = 3))")
		end

	end
	
	if newbest && (bestcol != candidateCols[end])
		#leave Xtmp with best subset
		Xtmp[:, l+2] .= traincols[bestcol]
	end

	(bestbic, besterr, bestR, bestcol, newbest)
end

function find_best_add_col!(traincols::Vector{Vector{T}}, y::Vector{T}, testcols::Vector{Vector{T}}, ytest::Vector{T}, colsubset::AbstractVector{W}, R::Matrix{T}, Xtmp::Matrix{T}, Xtmp2::Matrix{T}, besttrainerr::T, besttesterr::T) where T <: AbstractFloat where W <: Integer
	l = length(colsubset)
	@assert size(Xtmp) == (length(y), length(traincols)+1)
	@assert size(Xtmp2) == (length(ytest), length(testcols)+1)
	candidateCols = setdiff(eachindex(traincols), colsubset)
	println(string("Preparing to evaluate ", length(candidateCols), " remaining columns for addition.  Currently using $(length(colsubset)) columns out of a possible $(length(traincols)) with best train and test error of: ", (besttrainerr, besttesterr)))
	println()
	println()
	println()
	
	bestR = R
	bestcol = 0
	newbest = false
	for (i, c) in enumerate(candidateCols)
		t = time()
		Rnew = qraddcol(view(Xtmp, :, 1:l+1), R, traincols[c])
		tR = time() - t

		t = time()
		Xtmp[:, l+2] .= traincols[c]
		Xtmp2[:, l+2] .= testcols[c]
		tFill = time() - t

		t = time()
		newtrainerr, newbeta = calc_linreg_error(Rnew, view(Xtmp, :, 1:l+2), y)
		newtesterr = calc_linreg_err(newbeta, view(Xtmp2, :, 1:l+2), ytest)
		tErr = time() - t

		print("\r\u1b[K\u1b[A")
		print("\r\u1b[K\u1b[A")
		print("\r\u1b[K\u1b[A")
		println("R update time = $tR, Fill time = $tFill, Fit time = $tErr")
		println(string("Done with column ", c, ": number ", i, " out of ", length(candidateCols), " total candidates"))

		if newtesterr < besttesterr
			besttrainerr = newtrainerr
			besttesterr = newtesterr
			bestR = Rnew
			bestcol = c
			newbest = true
			println("Got new best train and test errors of: $newtrainerr, $newtesterr")

		else
			println("Got worse train and test error of : $newtrainerr, $newtesterr")
		end

	end

	#leave Xtmp and Xtmp2 in best column state
	if newbest && (bestcol != candidateCols[end])
		Xtmp[:, l+2] .= traincols[bestcol]
		Xtmp2[:, l+2] .= testcols[bestcol]
	end

	(besttesterr, besttrainerr, bestR, bestcol, newbest)
end

########################----Iterate Forward---#############################
function stepwise_forward_init(traincols::Vector{Vector{T}}, y::Vector{T}) where T <: AbstractFloat
	println("Initializing forward stepwise selection from zero columns")
	println("Calculating bias term error as a baseline")

	err = sum(abs2, (y .- mean(y)))/length(y)
	bic = length(y)*log(err)+log(length(y))

	println("Using 0 columns out of a possible $(length(traincols)), the error and bic is: $(round(err, sigdigits = 3)), $(round(bic, sigdigits = 3))")
	println("----------------------------------------------------------")

	#get initial R vector for bias terms
	_, R = qr(ones(T, length(y), 1))
	#initialize column subset as an empty vector
	colsubset = Vector{Integer}()
	#initialize record for bias terms
	record = [("", copy(sort(colsubset)), err, bic)]
	#initialize Xtmp
	Xtmp = ones(T, length(y), length(traincols)+1)

	(bic, err, colsubset, R, record, Xtmp, 0)
end

function stepwise_forward!(traincols::Vector{Vector{T}}, y::Vector{T}, bic::T, err::T, colsubset::AbstractVector{W}, R::Matrix{T}, record::Vector{U}, Xtmp::Matrix{T}, numsteps::Integer = 0) where T <: AbstractFloat where U <: Tuple where W <: Integer
	if length(colsubset) == length(traincols)
		println()
		println("Ending forward iteration after adding all $(length(traincols)) available columns.")
		println("-----------------------------------------------------------------------")
		return (bic, err, colsubset, record, Xtmp, numsteps, R)
	end

	if numsteps == 0
		println("Starting forward selection from $(length(colsubset)) columns")
		println()
		println()
		println()
		println()
	end

	print("\r\u1b[K\u1b[A")
	print("\r\u1b[K\u1b[A")
	print("\r\u1b[K\u1b[A")
	print("\r\u1b[K\u1b[A")

	println("On forward step $numsteps using $(length(colsubset)) columns out of a possible $(length(traincols)) with best error and bic of: $(round(err, sigdigits = 3)), $(round(bic, sigdigits = 3))")
	(bestbic, besterr, bestR, bestcol, newbest) = find_best_add_col!(traincols, y, colsubset, R, Xtmp, err, bic)

	if !newbest
		println()
		println("Ending forward iteration using $(length(colsubset)) out of $(length(traincols)) available columns and $numsteps forward steps")
		println("-----------------------------------------------------------------------")
		(bic, err, colsubset, record, Xtmp, numsteps, R)
	else
		#update colsubset
		push!(colsubset, bestcol)
		#update record
		push!(record, ("+$bestcol", copy(sort(colsubset)), besterr, bestbic))
		stepwise_forward!(traincols, y, bestbic, besterr, colsubset, bestR, record, Xtmp, numsteps+1)
	end
end

function stepwise_forward!(traincols::Vector{Vector{T}}, y::Vector{T}, testcols::Vector{Vector{T}}, ytest::Vector{T}, testerr::T, trainerr::T, colsubset::AbstractVector{W}, R::Matrix{T}, record::Vector{U}, Xtmp::Matrix{T}, Xtmp2::Matrix{T}, numsteps::Integer = 0) where T <: AbstractFloat where U <: Tuple where W <: Integer
	if length(colsubset) == length(traincols)
		println("Ending forward iteration after adding all $(length(traincols)) available columns.")
		return (testerr, trainerr, colsubset, record, Xtmp, Xtmp2, numsteps, R)
	end

	print("\r\u1b[K\u1b[A")
	print("\r\u1b[K\u1b[A")
	print("\r\u1b[K\u1b[A")
	print("\r\u1b[K\u1b[A")
	print("\r\u1b[K\u1b[A")
	print("\r\u1b[K\u1b[A")
	(besttesterr, besttrainerr, bestR, bestcol, newbest) = find_best_add_col!(traincols, y, testcols, ytest, colsubset, R, Xtmp, Xtmp2, trainerr, testerr)
	if !newbest
		println("Ending forward iteration using $(length(colsubset)) out of $(length(traincols)) available columns and $numsteps forward steps")
		(testerr, trainerr, colsubset, record, Xtmp, Xtmp2, numsteps, R)
	else
		println("On forward step $numsteps adding column $bestcol with a test error of $besttesterr.")
		#update colsubset
		push!(colsubset, bestcol)
		#update record
		push!(record, ("+$bestcol", colsubset, besttrainerr, besttesterr))
		stepwise_forward!(traincols, y, testcols, ytest, besttesterr, besttrainerr, colsubset, bestR, record, Xtmp, Xtmp2, numsteps+1)
	end
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
			Xtmp[:, i] .= traincols[c]
		end
		tFill = time() - t

		print("\r\u1b[K\u1b[A")
		print("\r\u1b[K\u1b[A")
		print("\r\u1b[K\u1b[A")
		println("R update time = $tR, Fill time = $tFill, Fit time = $tErr")
		println(string("Done with column ", c, ": number ", i, " out of ", length(colsubset), " total candidates"))
		if newbic < bestbic
			besterr = newerr
			bestbic = newbic
			bestR = Rnew
			bestcol = c
			besti = i
			newbest = true
			println("Got new best error and BIC of : $newerr, $newbic")

		else
			println("Got worse error and BIC of : $newerr, $newbic")
		end
	end

	if newbest && (besti != 1)
		#undo column removal back to desired column
		for i in 1:besti-1
			Xtmp[:, i+1] .= traincols[colsubset[i]]
		end
	end
	(bestbic, besterr, bestR, bestcol, newbest)
end

function find_best_remove_col(traincols::Vector{Vector{T}}, y::Vector{T}, testcols::Vector{Vector{T}}, ytest::Vector{T},colsubset::AbstractVector{W}, R::Matrix{T}, Xtmp::Matrix{T}, Xtmp2::Matrix{T}, besttrainerr::T, besttesterr::T) where T <: AbstractFloat where W <: Integer
	l = length(colsubset)
	@assert size(Xtmp) = (length(y), length(traincols)+1)
	println(string("Preparing to evaluate ", length(colsubset), " columns for removal."))
	println("Currently using $(length(colsubset)) columns out of a possible $(length(traincols)) with best error and BIC of $besttrainerr, $besttesterr")
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
		newtesterr = calc_linreg_bic(newbeta, view(Xtmp2, :, 1:l), ytest)
		tErr = time() - t

		#replace column that was just removed and overwrite the next column to be removed
		t = time()
		if i > 1
			Xtmp[:, i] .= traincols[c]
			Xtmp2[:, i] .= testcols[c]
		end
		tFill = time() - t

		printupdate = (i == 1) || (i == length(candidateCols)) || ((time() - tstart) >= 2) 
		if printupdate
			print("\r\u1b[K\u1b[A")
			print("\r\u1b[K\u1b[A")
			print("\r\u1b[K\u1b[A")
			println("R update time = $tR, Fill time = $tFill, Fit time = $tErr")
			println(string("Done with column ", c, ": number ", i, " out of ", length(candidateCols), " total candidates"))
		end

		if newtesterr < besttesterr
			besttrainerr = newtrainerr
			besttesterr = newtesterr
			bestR = Rnew
			bestcol = c
			besti = i
			newbest = true
			printupdate && println("Got new best train and test error of: $newtrainerr, $newtesterr")
		else
			printupdate && println("Got worse train and test error of: $newtrainerr, $newtesterr")
		end
	end

	if newbest && (besti != 1)
		#undo column removal back to desired column
		for i in 1:besti-1
			Xtmp[:, i+1] .= traincols[colsubset[i]]
			Xtmp2[:, i+1] .= testcols[colsubset[i]]
		end
	end
	(besttesterr, besttrainerr, bestR, bestcol, newbest)
end

########################----Iterate Backwards---#############################
function stepwise_backward!(traincols::Vector{Vector{T}}, y::Vector{T}, bic::T, err::T, colsubset::AbstractVector{W}, R::Matrix{T}, record::Vector{U}, Xtmp::Matrix{T}, numsteps::Integer = 0) where T <: AbstractFloat where U <: Tuple where W <: Integer
	if isempty(colsubset)
		println("Ending backward iteration after removing all $(length(traincols)) available columns.")
		println("----------------------------------------------------------------")
		return (bic, err, colsubset, record, Xtmp, numsteps, R)
	end

	if numsteps == 0
		println("Starting backward selection from $(length(colsubset)) columns")
		println()
		println()
		println()
		println()
	end

	print("\r\u1b[K\u1b[A")
	print("\r\u1b[K\u1b[A")
	print("\r\u1b[K\u1b[A")
	print("\r\u1b[K\u1b[A")

	println("On backward step $numsteps using $(length(colsubset)) columns out of a possible $(length(traincols)) with best error and bic of: $(round(err, sigdigits = 3)), $(round(bic, sigdigits = 3))")
	(bestbic, besterr, bestR, bestcol, newbest) = find_best_remove_col!(traincols, y, colsubset, R, Xtmp, err, bic)

	if !newbest
		println()
		println("Ending backward iteration using $(length(colsubset)) out of $(length(traincols)) available columns and $numsteps backward steps")
		println("-----------------------------------------------------------------------")

		(bic, err, colsubset, record, Xtmp, numsteps, R)
	else
		#update colsubset
		setdiff!(colsubset, bestcol)
		#update record
		push!(record, ("-$bestcol", colsubset, besterr, bestbic))
		stepwise_backward!(traincols, y, bestbic, besterr, colsubset, bestR, record, Xtmp, numsteps+1)
	end
end

function stepwise_backward!(traincols::Vector{Vector{T}}, y::Vector{T}, testcols::Vector{Vector{T}}, ytest::Vector{T}, testerr::T, trainerr::T, colsubset::AbstractVector{W}, R::Matrix{T}, record::Vector{U}, Xtmp::Matrix{T}, Xtmp2::Matrix{T}, numsteps::Integer = 0) where T <: AbstractFloat where U <: Tuple where W <: Integer
	if isempty(colsubset)
		println("Ending backward iteration after removing all $(length(traincols)) available columns.")
		return (testerr, trainerr, colsubset, record, Xtmp, Xtmp2, numsteps, R)
	end

	print("\r\u1b[K\u1b[A")
	print("\r\u1b[K\u1b[A")
	print("\r\u1b[K\u1b[A")
	print("\r\u1b[K\u1b[A")
	print("\r\u1b[K\u1b[A")
	print("\r\u1b[K\u1b[A")
	(besttrainerr, besttesterr, bestR, bestcol, newbest) = find_best_remove_col!(traincols, y, testcols, ytest, colsubset, R, Xtmp, Xtmp2, trainerr, testerr)

	if !newbest
		println("Ending backward iteration using $(length(colsubset)) out of $(length(traincols)) available columns and $numsteps backward steps")
		(testerr, trainerr, colsubset, record, Xtmp, Xtmp2, numsteps, R)
	else
		println("On backward step $numsteps removing column $bestcol with a test error of $besttesterr.")
		#update colsubset
		setdiff!(colsubset, bestcol)
		#update record
		push!(record, ("-$bestcol", colsubset, besttrainerr, besttesterr))
		stepwise_backward!(traincols, y, testcols, ytest, besttesterr, besttrainerr, colsubset, bestR, record, Xtmp, Xtmp2, numsteps+1)
	end
end

###############################----Run Stepwise----#######################################
function run_stepwise_reg(X::Matrix{T}, y::Vector{T}; colnames = ["Col $a" for a in 1:size(X, 2)]) where T <: AbstractFloat

	traincols = collect.(eachcol(X))

	(bic, err, colsubset, record, Xtmp, numsteps, R) = stepwise_forward!(traincols, y, stepwise_forward_init(traincols, y)...)

	while numsteps > 0

		(bic, err, colsubset, record, Xtmp, numsteps, R) = stepwise_backward!(traincols, y, bic, err, colsubset, R, record, Xtmp)

		if numsteps > 0
			(bic, err, colsubset, record, Xtmp, numsteps, R) = stepwise_forward!(traincols, y, bic, err, colsubset, R, record, Xtmp)
		end
	end

	usedcols = sort(colsubset)
	usedcolscheck = [in(i, usedcols) ? "x\t$n" : " \t$n" for (i, n) in enumerate(colnames)]
	(colsubset, usedcolscheck, record)
end



