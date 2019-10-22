function removecol(Rold, Xnew, Y, k)
#Rold is the R matrix of the previous fit, Xnew is the updated 
#input matrix, Y is the desired output, k is the column being removed
#from the original data set 
	Rnew = qrdelcol(Rold, k)
	betas, r = csne(Rnew, Xnew, Y)
	linRegTrainErr = sum(abs2, r)/length(Y)
	linRegTestErr = sum(abs2, (Xtest*betas .- Ytest))/length(Ytest)
	(linRegTrainErr, linRegTestErr, Rnew)
end

function addcol(Rold, Xold, Xnew, Y, newCol)
#Rold is the R matrix of the previous fit, Xnew is the updated 
#input matrix, Y is the desired output, Xold is the previous input matrix,
#newCol is the column being added
	Rnew = qraddcol(Xold, Rold, newCol)
	betas, r = csne(Rnew, Xnew, Y)
	linRegTrainErr = sum(abs2, r)/length(Y)
	linRegTestErr = sum(abs2, (Xtest*betas .- Ytest))/length(Ytest)
	(linRegTrainErr, linRegTestErr, Rnew)
end

function getfitbetas(X::Matrix{T}, y::Vector{T}) where T <: AbstractFloat
	betas = X \ y
	r = y .- X*betas
	(betas, r)
end

calc_linreg_bic(err::T, X::Matrix{T}, y::Vector{T}) where T <: AbstractFloat = length(y)*log(err) + log(length(y))*size(X, 2)

calc_linreg_error(betas::Vector{T}, X::Matrix{T}, y::Vector{T}) where T <: AbstractFloat = sum(abs2, y .- X*betas)/length(y)

function calc_linreg_error(X::Matrix{T}, y::Vector{T}) where T <: AbstractFloat
#calculates lin reg error with X input matrix and Y output but with the 
	betas, r = getfitbetas(X, y)
	err = sum(abs2, r)/length(r)
	(err, betas)
end

function calc_linreg_error(Rin::Matrix{T}, X::Matrix{T}, y::Vector{T}) where T <: AbstractFloat
#calculates lin reg error with X input matrix and Y output but with the 
#R matrix provided which is the upper triangular portion of the QR factorization
#of X
	betas, r = csne(Rin, X, y)
	err = sum(abs2, r)/length(r)
	(err, betas)
end

function find_best_add_col(traincols::Vector{Vector{T}}, y::Vector{T}, colsubset::AbstractVector{Int64}, R::Matrix{T}) where T <: AbstractFloat
	candidateCols = setdiff(eachindex(traincols), colsubset)
	println(string("Preparing to evaluate ", candidateCols, " remaining columns for addition"))
	println()
	println()
	println()
	counter = 0
	Xtmp = ones(eltype(xCols[1]), length(Y), length(colsubset)+2)
	Xold = ones(eltype(xCols[1]), length(Y), length(colsubset)+1)
	for j in 1:length(colsubset)
		Xtmp[:, j+1] = traincols[colsubset[j]]
		Xold[:, j+1] = traincols[colsubset[j]]
	end
	
	newRegErrs = [begin
		Xtmp[:, end] = traincols[c]
		t = time()
		# Rnew = qraddcol(view(X, :, [1; cols+1]), R, xCols[c])
		Rnew = qraddcol(Xold, R, traincols[c])
		tR = time() - t
		t = time()
		# errs = calcLinRegErr(Rnew, view(X, :, [1; [cols;c]+1]), Y, view(Xtest, :, [1; [cols;c]+1]), Ytest) 
		newerr, newbeta = calc_linreg_error(Rnew, Xtmp, y) 
		newbic = calc_linreg_bic(newerr, Xtmp, y)
		tErr = time() - t
		counter += 1
		print("\r\u1b[K\u1b[A")
		print("\r\u1b[K\u1b[A")
		print("\r\u1b[K\u1b[A")
		println(string("R update time = ", tR, ", Fit time = ", tErr))
		println(string("Done with column ", c, ": number ", counter, " out of ", length(candidateCols), " total candidates"))
		println("Got error and BIC of : $newerr, $newbic")
		(c, newerr, newbic, Rnew)
	end
	for c in candidateCols]
	print("\r\u1b[K\u1b[A")
	print("\r\u1b[K\u1b[A")
	# (bestTrainErr, bestTestErr, bestR, bestCols)
	(newRegErrs, candidateCols)
end

