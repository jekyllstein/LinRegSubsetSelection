using LinRegSubsetSelection
using Random
using Test

Random.seed!(1)
M = 1000
N = 15
l = rand(5:min(N, 20))
X = randn(M, N)
e = 0.1

#have only a subset of size l from a possible N to be useful
shuffleind = shuffle(1:N)
usedindtrain = shuffleind[1:l]
testgap = min(N-l, rand(min(l-1, round(Int64, l/2)):l-1))

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

################################Stepwise Quasistatic Anneal Test##################################################
(colsubset, colscheck, err, bic, colsrecord) = run_quasistatic_anneal_process((X, y), chi = 0.5, delt = 1.0)
missingcols = setdiff(usedindtrain, colsubset)
addedcols = setdiff(colsubset, usedindtrain)
println("Missing quasistatic anneal columns: $missingcols")
println("Added quasistatic anneal columns: $addedcols")
println("----------------------------------------------")
@test isempty(missingcols)

(colsubset, colscheck, err1, err2, colsrecord) = run_quasistatic_anneal_process((X, y), (Xtest, ytest), chi = 0.5,delt = 1.0)
missingtraincols = setdiff(usedindtrain, colsubset)
addedtraincols = setdiff(colsubset, usedindtrain)
missingtestcols = setdiff(usedindtest, colsubset)
addedtestcols = setdiff(colsubset, usedindtest)
missingcommoncols = setdiff(commoncols, colsubset)
addedcommoncols = setdiff(colsubset, commoncols)
println("Missing quasistatic anneal test columns: $missingtestcols")
println("Added quasistatic anneal test columns: $addedtestcols")
println("Missing quasistatic anneal columns: $missingcommoncols")
println("Added quasistatic anneal columns: $addedcommoncols")
println("----------------------------------------------")
@test isempty(missingcommoncols)


################################Full Subset Selection Test########################################################
(colsubset, usedcolscheck, record) = run_subset_reg((X, y))
missingcols = setdiff(usedindtrain, colsubset)
addedcols = setdiff(colsubset, usedindtrain)
println("Missing subset search columns: $missingcols")
println("Added subset search columns: $addedcols")
println("----------------------------------------------")

@test isempty(missingcols)
@test isempty(addedcols)

(colsubset, usedcolscheck, record) = run_subset_reg((X, y), (Xtest, ytest))
missingtraincols = setdiff(usedindtrain, colsubset)
addedtraincols = setdiff(colsubset, usedindtrain)
missingtestcols = setdiff(usedindtest, colsubset)
addedtestcols = setdiff(colsubset, usedindtest)
missingcommoncols = setdiff(commoncols, colsubset)
addedcommoncols = setdiff(colsubset, commoncols)
println("Missing subset search test columns: $missingtestcols")
println("Added subset search test columns: $addedtestcols")
println("Missing subset search columns: $missingcommoncols")
println("Added subset search columns: $addedcommoncols")
println("----------------------------------------------")
@test isempty(missingcommoncols)

################################Stepwise Anneal Test########################################################
(colsubset, colscheck, err, bic, colsrecord) = run_stepwise_anneal_process((X, y))
missingcols = setdiff(usedindtrain, colsubset)
addedcols = setdiff(colsubset, usedindtrain)
println("Missing gibbs anneal columns: $missingcols")
println("Added gibbs anneal columns: $addedcols")
println("----------------------------------------------")
@test isempty(missingcols)

(colsubset, colscheck, err, bic, colsrecord) = run_stepwise_anneal_process((X, y), (Xtest, ytest))
missingtraincols = setdiff(usedindtrain, colsubset)
addedtraincols = setdiff(colsubset, usedindtrain)
missingtestcols = setdiff(usedindtest, colsubset)
addedtestcols = setdiff(colsubset, usedindtest)
missingcommoncols = setdiff(commoncols, colsubset)
addedcommoncols = setdiff(colsubset, commoncols)
println("Missing gibbs anneal test columns: $missingtestcols")
println("Added gibbs anneal test columns: $addedtestcols")
println("Missing gibbs anneal columns: $missingcommoncols")
println("Added gibbs anneal columns: $addedcommoncols")
println("----------------------------------------------")
@test isempty(missingcommoncols)


################################Stepwise Regression Test########################################################
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

