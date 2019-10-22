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

function getfitbetas(X::AbstractMatrix{T}, y::Vector{T}) where T <: AbstractFloat
	betas = X \ y
	r = y .- X*betas
	(betas, r)
end

calc_linreg_bic(err::T, X::AbstractMatrix{T}, y::Vector{T}) where T <: AbstractFloat = length(y)*log(err) + log(length(y))*size(X, 2)

calc_linreg_error(betas::Vector{T}, X::AbstractMatrix{T}, y::Vector{T}) where T <: AbstractFloat = sum(abs2, y .- X*betas)/length(y)

function calc_linreg_error(X::AbstractMatrix{T}, y::Vector{T}) where T <: AbstractFloat
#calculates lin reg error with X input matrix and Y output but with the 
	betas, r = getfitbetas(X, y)
	err = sum(abs2, r)/length(r)
	(err, betas)
end

function calc_linreg_error(Rin::Matrix{T}, X::AbstractMatrix{T}, y::Vector{T}) where T <: AbstractFloat
#calculates lin reg error with X input matrix and Y output but with the 
#R matrix provided which is the upper triangular portion of the QR factorization
#of X
	betas, r = csne(Rin, X, y)
	err = sum(abs2, r)/length(r)
	(err, betas)
end

