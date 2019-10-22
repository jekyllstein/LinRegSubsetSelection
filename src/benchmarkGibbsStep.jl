include("gibbsAnnealingStepwiseLinRegQRupdate.jl")

#dataset parameters
L = 100000
M = 500


#set up random data
X = [ones(L) rand(L, M)]
Y = rand(L)
Xtest = [ones(L) rand(L, M)]
Ytest = rand(L)


initialColsVec = fill(false, M)
(trainErr, testErr, R, initialCols) = begin
	# initialColsVec = fill(false, size(X, 2))
	initialCols = Vector{Int64}(undef, 0)
	println("Calculating bias term error as a baseline")
	meanY = mean(Y)
	trainErr = sum(abs2, Y .- meanY)/length(Y)
	testErr = sum(abs2, Ytest .- meanY)/length(Ytest)
	_, R = qr(ones(length(Y)))
	(trainErr, testErr, R, initialCols)
end

Tsteps = fill(Inf, 1000)
xCols = [X[:, i] for i in 2:M+1]
xTestCols = [Xtest[:, i] for i in 2:M+1]


lines = ceil(Int64, M/20) + 13
for i in 1:lines-1 
	println()
end
println("Waiting for first step to complete")

t = time()
(errsRecord, colsRecord, _, acceptRate, dictHitRate) = runGibbsStep((xCols, X, Y, xTestCols, Xtest, Ytest), R, initialCols, [("", initialColsVec, (trainErr, testErr))], recordType{Float64}(initialColsVec => ((trainErr, testErr), R, initialCols)), Tsteps)
runTime = time() - t

avgStepTime = runTime / length(Tsteps)

println("For $L examples and $M features, average step time = $avgStepTime seconds with a repeat rate of $dictHitRate")


