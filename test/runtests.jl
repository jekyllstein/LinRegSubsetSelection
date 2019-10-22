using LinRegSubsetSelection
using Random

M = 1000
N = 1000
l = 100
X = rand(M, N)

#have only a subset of size l from a possible N to be useful
usedind = shuffle(1:N)[1:l]

betas = rand(l)
y = X[:, usedind]*betas .+ 1.0

traincols = collect.(eachcol(X))

println()
println()
println()
println()
println()
(bic, err, colsubset, record, Xtmp, numsteps, R) = LinRegSubsetSelection.stepwise_forward!(traincols, y, LinRegSubsetSelection.stepwise_forward_init(traincols, y)...)
