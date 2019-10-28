##########################--Get Update Paths-##############################
function updatecols(colsubset::ColSubset, newcolsvec::ColVec)
	l = length(newcolsvec)
	# println("l = $l")
	colsvec = subset_to_bin(colsubset, l)
	diffs = newcolsvec .- colsvec
	# println("colsvec true = $(sum(colsvec))")
	# println("colsubset = $colsubset")
	# println("newcolsvec true = $(sum(newcolsvec))")
	# println("Num diffs = $(sum(abs.(diffs)))")
	if sum(abs.(diffs)) > sum(newcolsvec)
		remcols = Vector{Integer}()
		addcols = findall(newcolsvec)
		startvec = Vector{Integer}()
		newvec = addcols
	else
		remcols = findall(diffs .== -1)
		addcols = findall(diffs .== 1)
		startvec = colsubset 
		newvec = vcat(setdiff(colsubset, remcols), addcols)
	end
	# println("colsubset = $colsubset")
	# println("newcolsvec = $newcolsvec")
	# println("startvec = $startvec")
	# println("remcols = $remcols")
	# println("addcols = $addcols")
	# println("newvec = $newvec")
	(startvec, remcols, addcols, newvec)
end

function removecols!(traincols::Vector{Vector{T}}, X::Matrix{T}, colsubset::ColSubset, ks::ColSubset) where T <: AbstractFloat
	l = length(colsubset)
	inds = filter(!isnothing, [findfirst(colsubset .== k) for k in ks])
	if !isempty(inds)
		starti = minimum(inds)
		subset2 = setdiff(colsubset, ks)
		l = length(subset2)
		for i in starti:l
			view(X, :, i+1) .= traincols[subset2[i]]
		end
	end
	return (inds, l)
end 


function update_errs!(traincols::Vector{Vector{T}}, y::Vector{T}, colsubset::ColSubset, newcolsvec::ColVec, R::Matrix{T}, Xtmp::Matrix{T}, Rorig::Matrix{T}) where T <: AbstractFloat

	Rnew = Matrix{T}(undef, 0, 0)
	(startvec, remcols, addcols, newvec) = updatecols(colsubset, newcolsvec)
	# println("colsubset = $colsubset")
	# println("newvec = $newvec")
	#build R from scratch starting with the orignal based on the bias column
	if isempty(startvec)
		Rnew = qraddcol(view(Xtmp, :, 1:1), Rorig, traincols[addcols[1]])
		view(Xtmp, :, 2) .= traincols[addcols[1]]
		if length(addcols) > 1
			for (i, c) in enumerate(addcols[2:end])
				Rnew = qraddcol(view(Xtmp, :, 1:i+1), Rnew, traincols[c])
				view(Xtmp, :, i+2) .= traincols[c]
			end
		end
	else
		#start Rnew where R is currently
		Rnew = copy(R)
		#get removal indicies and update Xtmp
		(inds, l) = removecols!(traincols, Xtmp, colsubset, remcols)
		# println("colsubset = $colsubset")
		# println("remcols = $remcols")
		# println("reminds = $inds")
		#remove columns in reverse so indicies don't get screwed up
		for i in Iterators.reverse(sort(inds))
			Rnew = qrdelcol(Rnew, i+1)
		end

		#update Rnew and add columns to Xtmp one at a time
		# println("addcols = $addcols")
		# println("l = $l")
		for (i, c) in enumerate(addcols)
			Rnew = qraddcol(view(Xtmp, :, 1:l+1), Rnew, traincols[c])
			view(Xtmp, :, i+l+1) .= traincols[c]
		end
	end

	l2 = sum(newcolsvec)
	@assert l2 == length(newvec) "$l2 is not equal to the new vector length $(length(newvec))"
	newerr, newbeta = calc_linreg_error(Rnew, view(Xtmp, :, 1:l2+1), y)
	newbic = calc_linreg_bic(newerr, view(Xtmp, :, 1:l2+1), y)
	(newerr, newbic, newvec, Rnew)
end

function update_errs!(traincols::Vector{Vector{T}}, y::Vector{T}, testcols::Vector{Vector{T}}, ytest::Vector{T}, colsubset::ColSubset, newcolsvec::ColVec, R::Matrix{T}, Xtmp::Matrix{T}, Xtmp2::Matrix{T}, Rorig::Matrix{T}) where T <: AbstractFloat

	Rnew = Matrix{T}(undef, 0, 0)
	(startvec, remcols, addcols, newvec) = updatecols(colsubset, newcolsvec)
	
	#build R from scratch starting with the orignal based on the bias column
	if isempty(startvec)
		Rnew = qraddcol(view(Xtmp, :, 1:1), Rorig, traincols[addcols[1]])
		view(Xtmp, :, 2) .= traincols[addcols[1]]
		view(Xtmp2, :, 2) .= testcols[addcols[1]]
		if length(addcols) > 1
			for (i, c) in enumerate(addcols[2:end])
				Rnew = qraddcol(view(Xtmp, :, 1:i+1), Rnew, traincols[c])
				view(Xtmp, :, i+2) .= traincols[c]
				view(Xtmp2, :, i+2) .= testcols[c]
			end
		end
	else
		#get removal indicies and update Xtmp
		Rnew = copy(R)
		(inds, l) = removecols!(traincols, Xtmp, colsubset, remcols)
		removecols!(testcols, Xtmp2, colsubset, remcols)
		#remove columns in reverse so indicies don't get screwed up
		for i in Iterators.reverse(sort(inds))
			Rnew = qrdelcol(Rnew, i+1)
		end

		#update Rnew and add columns to Xtmp one at a time
		for (i, c) in enumerate(addcols)
			Rnew = qraddcol(view(Xtmp, :, 1:l+1), Rnew, traincols[c])
			view(Xtmp, :, i+l+1) .= traincols[c]
			view(Xtmp2, :, i+l+1) .= testcols[c]
		end
	end

	l2 = sum(newcolsvec)
	newerr, newbeta = calc_linreg_error(Rnew, view(Xtmp, :, 1:l2+1), y)
	newtesterr = calc_linreg_error(newbeta, view(Xtmp2, :, 1:l2+1), ytest)
	(newerr, newtesterr, newvec, Rnew)
end



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

function update_error!(traincols::Vector{Vector{T}}, y::Vector{T}, colsubset::AbstractVector{W}, newcols::AbstractVector{W}, R::Matrix{T}, Xtmp::Matrix{T}, r::Vector{T}) where T <: AbstractFloat where W <: Integer
	l1 = length(colsubset)
	l2 = length(newcols)
	@assert size(Xtmp) == (length(y), length(traincols)+1)

	addcols = setdiff(newcols, colsubset)
	delind = findall(a -> !in(a, newcols), colsubset)
	
	if !isempty(delind)
		for i in Iterators.reverse(delind)
			t = time()
			Rnew = qrdelcol(R, i+1)
			tR = time() - t

			
			tFill = time() - t
		end

		for i in minimum(delind):l1-delind
			view(Xtmp, :, i+1) .= traincols[newcols[i]]
		end
	end

	for (i, c) in enumerate(addcols)
		qraddcol!(view(Xtmp, :, 1:l1-length(delind)+1), R, traincols[c], Rnew, r)
		Xtmp[:, l1-length(delind)+2] .= traincols[c]
	end


	


	newerr, newbeta = calc_linreg_error(Rnew, view(Xtmp, :, 1:l), y)
	newbic = calc_linreg_bic(newerr, view(Xtmp, :, 1:l), y)
		
	(newbic, newerr, Rnew)
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

function stepwise_iterate!(data::NTuple{N, InOutPairCols{T}}, err2::T, err1::T, colsubset::AbstractVector{W}, R::Matrix{T}, record::Vector{U}, tmpX::Matrix{T}...; numsteps::Integer = 0) where T <: AbstractFloat where W <:Integer where U <: Tuple where N

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

function run_subset_reg(data::InOutPair{T}...; colnames = ["Col $a" for a in 1:size(data[1][1], 2)]) where T <: AbstractFloat
	
	maxrecordl = 1000 #maximum number of records to store
	y = data[1][2]
	n = size(data[1][1], 2)
	m = length(y)

	testerr = length(data) > 1

	colvecs = formvecs(n, 1) #start with first non empty vector
	numsets = length(colvecs)+1
	Xtmp = ones(T, m, n+1)
	tmpX = if testerr
		m2 = length(data[2][2])
		Xtmp2 = ones(T, m2, n+1)
		(Xtmp, Xtmp2)
	else
		(Xtmp,)
	end


	println()
	println("==================================================================")
	println("Starting full subset selection with $n possible columns and $numsets possible subeets")
	println("==================================================================")
	#convert X's in data to vectors of columns
	datainput = mapreduce(a -> (collect.(eachcol(a[1])), a[2]), (a, b) -> (a...,b...), data)


	#start with initial empty subset
	colsubset = Vector{Integer}()
	bestsubset = copy(colsubset)
	bestcolvec = BitVector(undef, n)

	_, Rorig = qr(view(Xtmp, :, 1))
	R = copy(Rorig)
	bestR = copy(R)

	err1 = sum(abs2, (y .- mean(y)))/length(y)
	err2 = testerr ? sum(abs2, (data[2][2] .- mean(y)))/length(data[2][2]) : length(y)*log(err1)+log(length(y))

	besterr1 = err1
	besterr2 = err2
	bestnum = BigInt(0)

	record = [(besterr1, besterr2, bestcolvec, BigInt(0))]
	str = testerr ? "train and test error" : "error and BIC"
	println()
	println()
	println()
	println()

	tstart = time()
	lastreport = time()
	iteravg = 0.0

	for (i, newcolsvec) in enumerate(colvecs)
		t = time()
		(err1, err2, colsubset, R) = update_errs!(datainput..., colsubset, newcolsvec, R, tmpX..., Rorig)
		if err2 < besterr2
			bestR = copy(R)
			bestsubset = copy(colsubset)
			besterr1 = err1
			besterr2 = err2
			bestnum = i
			push!(record, (besterr1, besterr2, newcolsvec, i))
			if length(record) > maxrecordl
				popfirst!(record)
			end
		end
		stept = time() - t
		iteravg = (i == 1) ? stept : 0.99*iteravg + 0.01*stept

		if time() - lastreport > 2.0
			print("\r\u1b[K\u1b[A")
			print("\r\u1b[K\u1b[A")
			print("\r\u1b[K\u1b[A")
			print("\r\u1b[K\u1b[A")
			println("On subset $(i+1) of $numsets using $(length(colsubset)) columns")
			println("Current results are $str of: $((round(err1, sigdigits = 3), round(err2, sigdigits = 3)))")
			println("Best results from subset $bestnum using $(length(bestsubset)) columns with $str of: $((round(besterr1, sigdigits = 3), round(besterr2, sigdigits = 3)))")

			stepsleft = numsets - (i+1)
			timeleft = iteravg*stepsleft
			println("ETA = $(maketimestr(timeleft))")
			lastreport = time()
		end
	end 

	println("=================================================================")
	println("Finished subset iteration with the best result for $(length(bestsubset)) columns and final $str of: $((besterr1, besterr2))")
	println("=================================================================")
	println()
	usedcols = sort(bestsubset)
	usedcolscheck = [in(i, usedcols) ? "x\t$n" : " \t$n" for (i, n) in enumerate(colnames)]
	(bestsubset, usedcolscheck, record)
end

