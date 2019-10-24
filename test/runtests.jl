using LinRegSubsetSelection
using Random
using Test

M = 10000
N = 100
l = rand(10:20)
X = randn(M, N)
e = 0.1

#have only a subset of size l from a possible N to be useful
shuffleind = shuffle(1:N)
usedindtrain = shuffleind[1:l]
testgap = rand(1:l-1)

betas = rand([-2.0, -1.0, 1.0, 2.0], l)
y = X[:, usedindtrain]*betas .+ 1.0 .+ e.*randn(M)

#have test set overlap only partially and contain no noise
usedindtest = shuffleind[testgap+1:l+testgap]
Xtest = randn(M, N)
testbetas = [betas[testgap+1:end]; rand([-2.0, -1.0, 1.0, 2.0], testgap)]
ytest = Xtest[:, usedindtest]*testbetas .+ 1.0

commoncols = shuffleind[testgap+1:l]
println("$l columns with $(length(commoncols)) shared between test and train")
println("---------------------------------------------------------------------")

(colsubset, err, bic, colsrecord) = run_stepwise_anneal_process((X, y))
missingcols = setdiff(usedindtrain, colsubset)
addedcols = setdiff(colsubset, usedindtrain)
println("Missing gibbs anneal columns: $missingcols")
println("Added gibbs anneal columns: $addedcols")
println("----------------------------------------------")

@test isempty(missingcols)

(colsubset, usedcolscheck, record) = run_stepwise_reg((X, y))

missingcols = setdiff(usedindtrain, colsubset)
addedcols = setdiff(colsubset, usedindtrain)
println("Missing stepwise columns: $missingcols")
println("Added stepwise columns: $addedcols")
println("----------------------------------------------")

@test isempty(missingcols)

(colsubset, usedcolscheck, record) = run_stepwise_reg((X, y), (Xtest, ytest))
missingtestcols = setdiff(usedindtest, colsubset)
addedtestcols = setdiff(colsubset, usedindtest)
# missingtraincols = setdiff(usedindtrain, colsubset)
# addedtraincols = setdiff(colsubset, usedindtrain)
missingcommoncols = setdiff(commoncols, colsubset)
addedcommoncols = setdiff(colsubset, commoncols)

println("Added test cols: $(shuffleind[l+1:l+testgap])")
println("Removed test cols: $(shuffleind[1:testgap])")

println("Missing stepwise test columns: $missingtestcols")
println("Added stepwise test columns: $addedtestcols")
# println("Missing stepwise train columns: $missingtraincols")
# println("Added stepwise train columns: $addedtraincols")
println("Missing stepwise common columns: $missingcommoncols")
println("Added stepwise common columns: $addedcommoncols")
@test isempty(missingcommoncols)

