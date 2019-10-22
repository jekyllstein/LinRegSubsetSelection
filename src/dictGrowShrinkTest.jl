using InteractiveUtils
versioninfo()

println()
if isinteractive()
	println("Julia running in interactive mode")
else
	println("Julia not running in interactive mode")
end
println()
totMem = Int64(Sys.total_memory())

@noinline function createTestDict(N)
	testDict = Dict{Int64, Matrix{Float64}}()
	for i in 1:N
		push!(testDict, i => ones(Float64, 100, 100))
	end
	(testDict, Int64(Sys.free_memory()))
end

@noinline function growTestDict!(testDict, N1, N2)
	for i in N1+1:N2
		push!(testDict, i => ones(Float64, 100, 100))
	end
	Int64(Sys.free_memory())
end

@noinline function shrinkTestDict!(testDict, N)
	ks = collect(keys(testDict))[1:N]
	for k in ks
		delete!(testDict, k)
	end
	Int64(Sys.free_memory())
end

function modifyDict!(testDict, N)
	for i in 1:10
		mem = growTestDict!(testDict, i*N, (i+1)*N)
		println("Memory use after growing dictionary is $((mem1 - mem)/(10^9)) gigabytes")
		mem = shrinkTestDict!(testDict, N)
		GC.gc()
		println("Memory use after shrinking dictionary is $((mem1 - mem)/(10^9)) gigabytes")
	end
end

function runTest(mem1)

	N = 10000
	(testDict, mem2) = createTestDict(N)

	println("Memory added from dictionary is $((mem1 - mem2)/(10^9)) gigabytes")

	modifyDict!(testDict, N)
end

mem1 = Int64(Sys.free_memory())
runTest(mem1)
GC.gc()
println("Final memory use is $((mem1 - Int64(Sys.free_memory()))/(10^9)) gigabytes")