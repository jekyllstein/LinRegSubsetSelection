###################################--Types--###################################
ColVec{T} = AbstractVector{T} where T <: Bool
ColSubset{T} = AbstractVector{T} where T <: Integer
InOutPairCols{T} = Tuple{Vector{U}, U} where U <: Vector{T} where T <: AbstractFloat
InOutPair{T} = Tuple{Matrix{T}, Vector{T}} where T <: AbstractFloat
###############################################################################

function subset_to_bin(colset::ColSubset, l::Integer)
	binvec = BitVector(zeros(Bool, l))
	for c in colset
		binvec[c] = true
	end
	return binvec
end

function get_neighbors(binvec::ColVec; flag = :all)
	binvec2 = copy(binvec)
	neighbors = Vector{BitVector}()
	for i in eachindex(binvec)
		check1 = (flag == :all)
		check2 = (flag == :add) && (!binvec[i])
		check3 = (flag == :remove) && (binvec[i])
		if (check1 || check2 || check3)
			binvec2[i] = !binvec2[i] #switch index 1
			push!(neighbors, copy(binvec2)) #add to neighbor list
			binvec2[i] = !binvec2[i] #undo switch
		end
	end
	return neighbors
end

function iterate_subsets(binvec::ColVec, acc::ColVec; flag = :add)
	neighbors = filter!(a -> !in(a, acc), get_neighbors(binvec, flag = flag))
	if isempty(neighbors)
		(binvec, acc)
	else
		#find the neighbor with the most subsequent neighbors to iterate through
		(l, i) = findmax(length.(get_neighbors.(neighbors, flag = flag)))
		iterate_subsets(neighbors[i], [acc; neighbors], flag = flag)
	end
end

function iterate_subsets(l::Integer)
	binvec = subset_to_bin(Vector{Integer}(), l)
	acc = [binvec]
	(binvec, acc1) = iterate_subsets(binvec, acc)
	(binvec, acc2) = iterate_subsets(binvec, acc1, flag = :remove)
	while length(acc2) != length(acc1)
		(binvec, acc1) = iterate_subsets(binvec, acc2)
		(binvec, acc2) = iterate_subsets(binvec, acc1, flag = :remove)
	end
	return acc2
end

function formvec(n, acc = [[0], [1]])
	if n == 0
		return acc
	else
		formvec(n-1, [[[0; a] for a in acc]; [[1; a] for a in acc]])
	end
end

function getvec(n::Integer, N::Integer)
	pad = ndigits(2^BigInt(N) - 1, base=2)
	BitVector(digits(Bool, n, base=2, pad=pad))
end

#creates a generator that contains all subset vectors for N features
formvecs(N::Integer, start::Integer = 0) = (getvec(n, N) for n in start:2^BigInt(N)-1)

function maketimestr(t::AbstractFloat)
	hours = floor(Integer, t / 60 / 60)
	minutes = floor(Integer, t / 60) - hours*60
	seconds = t - (minutes*60) - (hours*60*60)
	secstr = string(round(seconds, digits = 2))
	"$hours:$minutes:$(secstr[1:min(4, length(secstr))])"
end


