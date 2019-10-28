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

function get_add_neighbor(binvec::ColVec, i::Integer)
	neighbor = copy(binvec)
	if !binvec[i]
		neighbor[i] = !neighbor[i] #switch index i
		return (neighbor, true)
	else
		return (neighbor, false)
	end
end

function get_add_neighbors(binvec::ColVec) where T <: Bool
	ind = findall(.!binvec)
	(get_add_neighbor(binvec, i)[1] for i in ind)
end


function iterate_subsets!(binvec::T, acc::Vector{T}) where T <: ColVec
	push!(acc, binvec)
	neighbors = filter!(a -> !in(a, acc), get_neighbors(binvec))	
	for n in neighbors
		iterate_subsets!(n, acc)
	end
end

function iterate_subsets(N::Integer)
	global acc = Vector{BitVector}()
	iterate_subsets!(subset_to_bin(Vector{Integer}(), N), acc)
	return acc
end

iterate_subsets!(N::Integer, acc) = iterate_subsets!(subset_to_bin(Vector{Integer}(), N), acc)

function formvec(n, acc = [BitVector([false]), BitVector([true])])
	if n == 1
		return acc
	else
		formvec(n-1, vcat([vcat(false, a) for a in acc], [vcat(true, a) for a in acc]))
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

