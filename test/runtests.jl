using LinRegSubsetSelection
using Random
using Test

M = 10000
N = 100
l = 10
X = randn(M, N)
e = 0.1

#have only a subset of size l from a possible N to be useful
usedind = shuffle(1:N)[1:l]

betas = rand([-2.0, -1.0, 1.0, 2.0], l)
y = X[:, usedind]*betas .+ 1.0 .+ e.*randn(M)

Xtest = randn(M, N)
ytest = Xtest[:, usedind]*betas .+ 1.0

# traincols = collect.(eachcol(X))
# testcols = collect.(eachcol(Xtest))

# (bic, err, colsubset, record, Xtmp, forwardsteps, R) = LinRegSubsetSelection.stepwise_forward!(traincols, y, LinRegSubsetSelection.stepwise_forward_init(traincols, y)...)

# (bic, err, colsubset, record, Xtmp, backwardsteps, R) = LinRegSubsetSelection.stepwise_backward!(traincols, y, bic, err, colsubset, R, record, Xtmp, 0)

(colsubset, usedcolscheck, record) = run_stepwise_reg(X, y)

println("Record is : $record")
# println("Used columns for model: $(sort(usedind))")
# println("Stepwise selected columns: $(sort(colsubset))")
missingcols = setdiff(usedind, colsubset)
addedcols = setdiff(colsubset, usedind)
println("Missing stepwise columns: $(setdiff(usedind, colsubset))")
println("Added stepwise columns: $(setdiff(colsubset, usedind))")

@test isempty(missingcols)