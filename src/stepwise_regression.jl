##########################---Find Best Forward Step---#####################
function find_best_add_col!(traincols::Vector{Vector{T}}, y::Vector{T}, colsubset::AbstractVector{Int64}, R::Matrix{T}, Xtmp::Matrix{T}, besterr::T, bestbic::T) where T <: AbstractFloat
	l = length(colsubset)
	@assert size(Xtmp) == (length(y), length(traincols)+1)
	candidateCols = setdiff(eachindex(traincols), colsubset)
	println(string("Preparing to evaluate ", length(candidateCols), " remaining columns for addition."))
	println("Currently using $(length(colsubset)) columns out of a possible $(length(traincols)) with best error and BIC of $besterr, $bestbic")
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
			printupdate && println("Got new best error and BIC of : $newerr, $newbic")

		else
			printupdate && println("Got worse error and BIC of : $newerr, $newbic")
		end

	end
	
	if newbest && (bestcol != candidateCols[end])
		#leave Xtmp with best subset
		Xtmp[:, l+2] .= traincols[bestcol]
	end

	(bestbic, besterr, bestR, bestcol, newbest)
end

function find_best_add_col!(traincols::Vector{Vector{T}}, y::Vector{T}, testcols::Vector{Vector{T}}, ytest::Vector{T}, colsubset::AbstractVector{Int64}, R::Matrix{T}, Xtmp::Matrix{T}, Xtmp2::Matrix{T}, besttrainerr::T, besttesterr::T) where T <: AbstractFloat
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
	println("Calculating bias term error as a baseline")
	err = sum(abs2, (y .- mean(y)))/length(y)
	bic = length(y)*log(err)+log(length(y))

	#get initial R vector for bias terms
	_, R = qr(ones(T, length(y), 1))
	#initialize column subset as an empty vector
	colsubset = Vector{Int64}()
	#initialize record for bias terms
	record = [("", colsubset, err, bic)]
	#initialize Xtmp
	Xtmp = ones(T, length(y), length(traincols)+1)

	(bic, err, colsubset, R, record, Xtmp, 0)
end

function stepwise_forward!(traincols::Vector{Vector{T}}, y::Vector{T}, bic::T, err::T, colsubset::AbstractVector{Int64}, R::Matrix{T}, record::Vector{U}, Xtmp::Matrix{T}, numsteps::Int64) where T <: AbstractFloat where U <: Tuple
	if length(colsubset) == length(traincols)
		println("Ending forward iteration after adding all $(length(traincols)) available columns.")
		return (bic, err, colsubset, record, Xtmp, numsteps, R)
	end

	print("\r\u1b[K\u1b[A")
	print("\r\u1b[K\u1b[A")
	print("\r\u1b[K\u1b[A")
	print("\r\u1b[K\u1b[A")
	print("\r\u1b[K\u1b[A")
	print("\r\u1b[K\u1b[A")
	(bestbic, besterr, bestR, bestcol, newbest) = find_best_add_col!(traincols, y, colsubset, R, Xtmp, err, bic)

	if !newbest
		println("Ending forward iteration using $(length(colsubset)) out of $(length(traincols)) available columns and $numsteps forward steps")
		(bic, err, colsubset, record, Xtmp, numsteps, R)
	else
		println("On forward step $numsteps adding column $bestcol with a BIC of $bestbic.")
		#update colsubset
		push!(colsubset, bestcol)
		#update record
		push!(record, ("+$bestcol", colsubset, besterr, bestbic))
		stepwise_forward!(traincols, y, bestbic, besterr, colsubset, bestR, record, Xtmp, numsteps+1)
	end
end

function stepwise_forward!(traincols::Vector{Vector{T}}, y::Vector{T}, testcols::Vector{Vector{T}}, ytest::Vector{T}, testerr::T, trainerr::T, colsubset::AbstractVector{Int64}, R::Matrix{T}, record::Vector{U}, Xtmp::Matrix{T}, Xtmp2::Matrix{T}, numsteps::Int64) where T <: AbstractFloat where U <: Tuple
	if length(colsubset) == length(traincols)
		println("Ending forward iteration after adding all $(length(traincols)) available columns.")
		return (testerr, trainerr, colsubset, record, Xtmp, Xtmp2, numsteps, R)
	end

	(besttesterr, besttrainerr, bestR, bestcol, newbest) = find_best_add_col!(traincols, y, testcols, ytest, colsubset, R, Xtmp, Xtmp2, trainerr, testerr)
	if !newbest
		println("Ending forward iteration using $(length(colsubset)) out of $(length(traincols)) available columns and $numsteps forward steps")
		(testerr, trainerr, colsubset, record, Xtmp, Xtmp2, numsteps, R)
	else
		println("Best column to add is $bestcol with a test error of $besttesterr")
		#update colsubset
		push!(colsubset, bestcol)
		#update record
		push!(record, ("+$bestcol", colsubset, besttrainerr, besttesterr))
		stepwise_forward!(traincols, y, testcols, ytest, besttesterr, besttrainerr, colsubset, bestR, record, Xtmp, Xtmp2, numsteps+1)
	end
end

########################----Iterate Backwards---#############################







# function find_best_remove_col(traincols::Vector{Vector{T}}, y::Vector{T}, colsubset::AbstractVector{Int64}, R::Matrix{T}, Xtmp::Matrix{T}, besterr::T, bestbic::T) where T <: AbstractFloat
# 	l = length(colsubset)
# 	@assert size(Xtmp) = (length(y), length(traincols)+1)
# 	println(string("Preparing to evaluate ", colsubset, " columns for removal with best error and bic of: ", (besterr, bestbic)))
# 	println()
# 	println()
# 	println()

# 	bestR = R
# 	bestcol = 0
# 	besti = 0
# 	newbest = false
# 	#initially have 0, 1, 2, 3, 4, 5
# 	#iteration 1 leave unchanged and cut off last column: 0, 1, 2, 3, 4
# 	#iteration 2 want to use: 0, 1, 2, 3, 5
# 	#iteration 3 want to use: 0, 1, 2, 4, 5
# 	#iteration 4 want to use: 0, 1, 3, 4, 5
# 	#iteration 5 want to use: 0, 2, 3, 4, 5
# 	#iterate through current columns in reverse where at each step column c is being removed
# 	for (i, c) in Iterators.reverse(enumerate(colsubset))
# 		t = time()
# 		Rnew = qrdelcol(R, i+1)
# 		tR = time() - t

# 		t = time()
# 		newerr, newbeta = calc_linreg_error(Rnew, view(Xtmp, :, 1:l), y)
# 		newbic = calc_linreg_bic(newerr, view(Xtmp, :, 1:l), y)
# 		tErr = time() - t

# 		#replace column that was just removed and overwrite the next column to be removed
# 		t = time()
# 		if i > 1
# 			Xtmp[:, i] .= traincols[c]
# 		end
# 		tFill = time() - t

# 		print("\r\u1b[K\u1b[A")
# 		print("\r\u1b[K\u1b[A")
# 		print("\r\u1b[K\u1b[A")
# 		println("R update time = $tR, Fill time = $tFill, Fit time = $tErr")
# 		println(string("Done with column ", c, ": number ", i, " out of ", length(candidateCols), " total candidates"))
# 		println("Got worse error and BIC of : $newerr, $newbic")

# 		if newbic < bestbic
# 			besterr = newerr
# 			bestbic = newbic
# 			bestR = Rnew
# 			bestcol = c
# 			besti = i
# 			newbest = true
# 			println("Got new best error and BIC of : $newerr, $newbic")

# 		else
# 			println("Got worse error and BIC of : $newerr, $newbic")
# 		end
# ]	end

# 	if newbest && (besti != 1)
# 		#undo column removal back to desired column
# 		for i in 1:besti-1
# 			Xtmp[:, i+1] .= traincols[colsubset[i]]
# 		end
# 	end
# 	(bestbic, besterr, bestR, bestcol, newbest)
# end

# function find_best_remove_col(traincols::Vector{Vector{T}}, y::Vector{T}, testcols::Vector{Vector{T}}, ytest::Vector{T},colsubset::AbstractVector{Int64}, R::Matrix{T}, Xtmp::Matrix{T}, Xtmp2::Matrix{T}, besttrainerr::T, besttesterr::T) where T <: AbstractFloat
# 	l = length(colsubset)
# 	@assert size(Xtmp) = (length(y), length(traincols)+1)
# 	println(string("Preparing to evaluate ", colsubset, " columns for removal with best test and train error of: ", (besttrainerr, besttesterr)))
# 	println()
# 	println()
# 	println()

# 	bestR = R
# 	bestcol = 0
# 	besti = 0
# 	newbest = false
# 	#initially have 0, 1, 2, 3, 4, 5
# 	#iteration 1 leave unchanged and cut off last column: 0, 1, 2, 3, 4
# 	#iteration 2 want to use: 0, 1, 2, 3, 5
# 	#iteration 3 want to use: 0, 1, 2, 4, 5
# 	#iteration 4 want to use: 0, 1, 3, 4, 5
# 	#iteration 5 want to use: 0, 2, 3, 4, 5
# 	#iterate through current columns in reverse where at each step column c is being removed
# 	for (i, c) in Iterators.reverse(enumerate(colsubset))
# 		t = time()
# 		Rnew = qrdelcol(R, i+1)
# 		tR = time() - t

# 		t = time()
# 		newtrainerr, newbeta = calc_linreg_error(Rnew, view(Xtmp, :, 1:l), y)
# 		newtestbic = calc_linreg_bic(newbeta, view(Xtmp2, :, 1:l), ytest)
# 		tErr = time() - t

# 		#replace column that was just removed and overwrite the next column to be removed
# 		t = time()
# 		if i > 1
# 			Xtmp[:, i] .= traincols[c]
# 			Xtmp2[:, i] .= testcols[c]
# 		end
# 		tFill = time() - t

# 		print("\r\u1b[K\u1b[A")
# 		print("\r\u1b[K\u1b[A")
# 		print("\r\u1b[K\u1b[A")
# 		println("R update time = $tR, Fill time = $tFill, Fit time = $tErr")
# 		println(string("Done with column ", c, ": number ", i, " out of ", length(candidateCols), " total candidates"))
# 		println("Got worse train and test error of: $newtesterr, $newtrainerr")

# 		if newbic < bestbic
# 			besttrainerr = newtrainerr
# 			besttesterr = newtesterr
# 			bestR = Rnew
# 			bestcol = c
# 			besti = i
# 			newbest = true
# 			println("Got new best train and test error of: $newtrainerr, $newtesterr")

# 		else
# 			println("Got new best train and test error of: $newtrainerr, $newtesterr")
# 		end
# 	end

# 	if newbest && (besti != 1)
# 		#undo column removal back to desired column
# 		for i in 1:besti-1
# 			Xtmp[:, i+1] .= traincols[colsubset[i]]
# 			Xtmp2[:, i+1] .= testcols[colsubset[i]]
# 		end
# 	end
# 	(besttesterr, besttrainerr, bestR, bestcol, newbest)
# end
