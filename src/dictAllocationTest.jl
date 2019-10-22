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

println()
if isempty(ARGS)
	println("Testing in global space")
else
	println("Testing inside a function")
end
println()

function createTestDict(totMem)
	testDict = Dict{Int64, Matrix{Float64}}()
	for i in 1:100000
		push!(testDict, i => ones(Float64, 100, 100))
	end
	mem2 = totMem - Int64(Sys.free_memory())
	testDict = ()
	return mem2
end

mem1 = totMem - Int64(Sys.free_memory())

if isempty(ARGS)
	testDict = Dict{Int64, Matrix{Float64}}()
	for i in 1:100000
		push!(testDict, i => ones(Float64, 100, 100))
	end
	mem2 = totMem - Int64(Sys.free_memory())
	testDict = ()
else
	mem2 = createTestDict(totMem)
end

GC.gc()

mem3 = totMem - Int64(Sys.free_memory())

println("Memory added from dictionary is $((mem2 - mem1)/(10^9)) gigabytes")
println("Memory freed from clearing dictionary is $((mem2 - mem3)/(10^9)) gigabytes")
println("Final memory usage is $(100*mem3/mem1) % of the original")

