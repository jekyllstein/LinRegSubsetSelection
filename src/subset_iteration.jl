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

BinVec = AbstractVector{T} where T <: Bool

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
