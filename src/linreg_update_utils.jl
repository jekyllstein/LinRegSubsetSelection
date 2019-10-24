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

function printcolsvec(origcolsvec, colsvec, switchind, acc)
	nmax = length(digits(length(colsvec))) #maximum number of digits for indicator column
	emax = 20
	c = 0
	l = 1
	print(repeat(" ", nmax+1)) #add nmax+1 spaces of padding for the row labels
	for i in 1:emax print(string(lpad(i, 2), " ")) end
	println()
	print(repeat(" ", nmax+1))
	for i in eachindex(colsvec)
    	#highlight cells with attempted changes
    	bracketcolor = if i == switchind
    		#if an attempted change is accepted make the brackets blue else red
    		acc ? :blue : :red
    	else
    		#if no change leave green
    		:green
    	end

    	fillstate = ((i == switchind) && acc) ? !colsvec[i] : colsvec[i]
    	#fill cell with X if being used and nothing if not
    	fillchar = fillstate ? 'X' : ' '
    	#if current state differs from original highlight in yellow
    	fillcolor = fillstate == origcolsvec[i] ? :default : :reverse

    	printstyled(IOContext(stdout, :color => true), "[", color = bracketcolor)
    	printstyled(IOContext(stdout, :color => true), fillchar, color = fillcolor)
    	printstyled(IOContext(stdout, :color => true), "]", color = bracketcolor)
        c += 1
        if i != length(colsvec)
	        if c == emax
	        	println()
	        	print(string(lpad(emax*l, nmax), " "))
	        	c = 0
	        	l += 1
	        end
	    else
	    	newcolchange = if acc
	    		colsvec[switchind] ?  -1 : 1
	    	else
	    		0
	    	end
	    	print(string(" ", sum(colsvec) + newcolchange, "/", length(colsvec)))
	    end
    end
    return l
end

RecordType{T} = OrderedDict{Array{Bool, 1}, Tuple{T, T, Matrix{T}, Vector{Int64}}}
function purge_record!(colsrecord::RecordType{T}, p = 0.1) where T <: Real
    l = length(colsrecord)
    if l > 1
	    ps = LinRange(2*p, 0.0, l)
	    for (i, k) in enumerate(keys(colsrecord))
	        if rand() < ps[i]       		
	       		delete!(colsrecord, k)
	       	end
		end
		GC.gc()
	end
	return length(colsrecord)
end