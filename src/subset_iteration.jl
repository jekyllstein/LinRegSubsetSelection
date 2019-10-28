function subset_to_bin(colset::AbstractVector{T}, l::Integer) where T <: Integer
	binvec = BitVector(undef, l)
	for c in colset
		binvec[c] = true
	end
	return binvec
end

function get_neighbors(binvec::AbstractVector{T}; flag = :all) where T <: Bool
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

function get_add_neighbor(binvec::AbstractVector{T}, i::Integer) where T <: Bool
	neighbor = copy(binvec)
	if !binvec[i]
		neighbor[i] = !neighbor[i] #switch index i
		return (neighbor, true)
	else
		return (neighbor, false)
	end
end

function get_add_neighbors(binvec::AbstractVector{T}) where T <: Bool
	ind = findall(.!binvec)
	(get_add_neighbor(binvec, i)[1] for i in ind)
end

BinVec = AbstractVector{T} where T <: Bool

function iterate_subsets!(binvec::T, acc::Vector{T}) where T <: BinVec
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

function iterate_subsets(binvec::T, acc::Vector{T}; flag = :add) where T <: BinVec
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

function formvec(n, acc = [BitVector([false]), BitVector([true])])
	if n == 1
		return acc
	else
		formvec(n-1, vcat([vcat(false, a) for a in acc], [vcat(true, a) for a in acc]))
	end
end

function getvec(n, N)
	pad = ndigits(2^BigInt(N) - 1, base = 2)
	BitVector(digits(Bool, n, base = 2, pad = pad))
end

#creates a generator that contains all subset vectors for N features
formvecs(N) = (getvec(n, N) for n in 0:2^BigInt(N) - 1)

function bitvec_to_num(v::BitVector)
	acc = BigInt(0)
	for (i, a) in enumerate(v)
		if a
			acc += 2^BigInt(i-1)
		end
	end
	return acc
end
