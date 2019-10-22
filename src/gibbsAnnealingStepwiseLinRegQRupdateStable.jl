using QRupdate
using LinearAlgebra
using Statistics
using DelimitedFiles
using Random

function printColsVec(origColsVec, colsVec, switchInd, acc)
	# cmax = 100 #max number of characters per line
	nmax = length(digits(length(colsVec))) #maximum number of digits for indicator column
	# emax = floor(Int64, cmax/(nmax+5)) #maximum number of entries per line
	emax = 20
	c = 0
	l = 1
	print(repeat(" ", nmax+1)) #add nmax+1 spaces of padding for the row labels
	for i in 1:emax
		print(string(lpad(i, 2), " "))
	end
	println()
	print(repeat(" ", nmax+1))
	for i in eachindex(colsVec)
    	#highlight cells with attempted changes
    	bracketColor = if i == switchInd
    		#if an attempted change is accepted make the brackets blue else red
    		acc ? :blue : :red
    	else
    		#if no change leave green
    		:green
    	end

    	fillState = ((i == switchInd) && acc) ? !colsVec[i] : colsVec[i]
    	#fill cell with X if being used and O if not
    	fillChar = fillState ? 'X' : ' '
    	#if current state differs from original highlight in yellow
    	fillColor = fillState == origColsVec[i] ? :default : :reverse

    	printstyled("[", color = bracketColor)
    	printstyled(fillChar, color = fillColor)
    	printstyled("]", color = bracketColor)
        c += 1
        if i != length(colsVec)
	        if c == emax
	        	println()
	        	print(string(lpad(emax*l, nmax), " "))
	        	# print(string(lpad(emax*l+1, nmax), " "))
	        	c = 0
	        	l += 1
	        end
	    else
	    	newColChange = if acc
	    		colsVec[switchInd] ?  -1 : 1
	    	else
	    		0
	    	end
	    	print(string(" ", sum(colsVec) + newColChange, "/", length(colsVec)))
	    end
    end
    return l
end

function calcLinRegErr(R, X, Y, Xtest, Ytest)
#calculates lin reg error with X input matrix and Y output but with the 
#R matrix provided which is the upper triangular portion of the QR factorization
#of X
	betas, r = csne(R, X, Y)
	linRegTrainErr = sum(abs2, r)/length(Y)
	linRegTestErr = sum(abs2, (Xtest*betas .- Ytest))/length(Ytest)
	(linRegTrainErr, linRegTestErr)
end

function calcLinRegErr(R, X, Y)
#calculates lin reg error with X input matrix and Y output but with the 
#R matrix provided which is the upper triangular portion of the QR factorization
#of X
	betas, r = csne(R, X, Y)
	err = sum(abs2, r)/length(Y)
	BIC = length(Y)*log(err) + log(length(Y))*size(X, 2)
	(err, BIC)
end

function calcLinRegErr(X, Y, Xtest, Ytest)
	# Xtrain_lin = [ones(Float32, size(Y, 1)) X]
	# Xtest_lin = [ones(Float32, size(Ytest, 1)) Xtest]
	# betas = pinv(Xtrain_lin'*Xtrain_lin)*Xtrain_lin'*Y
	# betas = inv(collect(X')*collect(X))*collect(X')*Y
	betas = X \ Y
	# betas = pinv(collect(X))*Y
	out = X*betas
	outTest = Xtest*betas
	linRegTrainErr = sum(abs2, (out .- Y))/length(Y)
	linRegTestErr = sum(abs2, (outTest .- Ytest))/length(Ytest)
	(linRegTrainErr, linRegTestErr) #, logLikelihoodTest)
end

function gibbsStep(xCols, X, Y, lastR, currentCols, lastBICs, colsRecord, colsVec, T, iNum, Tsteps, startTime, acceptRate, dictHitRate, firstColsVec, printIter = true)
	switchInd = ceil(Int64, rand()*length(colsVec))
	changeString = if colsVec[switchInd]
		string("- ", switchInd)
	else
		string("+ ", switchInd)
	end

	# println(string("Converted Cols Vec = ", find(colsVec)))

	newColsVec = [i == switchInd ? !colsVec[i] : colsVec[i] for i in 1:length(colsVec)]
	# candidateCols = find(newColsVec)
	# currentCols = find(colsVec)
	
	errTimeStr = []

	t = time()
	(BICs, Rnew, newCols) = if haskey(colsRecord, newColsVec) # in(newColsVec, map(a -> a[1], colsRecord))
		# colsRecord[find(a -> a == newColsVec, map(a -> a[1], colsRecord))][1][2]
		#update newCols with entry from dictionary to ensure column order is preserved
		for i in 1:3 
			# println()
			push!(errTimeStr, "\n")
		end
		push!(errTimeStr, "Got errors from dictionary\n")
		colsRecord[newColsVec]
		colsRecord[newColsVec]
	else
		#otherwise form it from the previous value and the switching index
		newCols = if !colsVec[switchInd]
			[currentCols; switchInd]
		else
			setdiff(currentCols, switchInd)
		end
		# println(string("Current Cols = ", currentCols))
		# println(string("New Cols = ", newCols))
		# println(string("Switch Index = ", switchInd, " and has a value of ", colsVec[switchInd]))
		t2 = time()
		Rnew = if !colsVec[switchInd]
			t3 = time()
			# Xold = ones(eltype(xCols[1]), length(Y), length(currentCols)+1)
			for j in eachindex(currentCols)	
				X[:, j+1] = xCols[currentCols[j]]	
			end
			# Xold = view(X, :, [1; currentCols+1])
			Xold = view(X, :, 1:length(currentCols)+1)
			tfill = time() - t3
			# println(string("Old X fill time = ", tfill))
			push!(errTimeStr, string("Old X fill time = ", tfill, "\n"))
			qraddcol(Xold, lastR, xCols[switchInd])
		else
			push!(errTimeStr, "\n")
			k = findfirst(a -> a == switchInd, currentCols)
			qrdelcol(lastR, k+1) #index of column being removed
		end
		tR = time() - t2
		# println(string("R update time = ", tR))
		push!(errTimeStr, string("R update time = ", tR, "\n"))

		t2 = time()
		# Xtmp = ones(eltype(xCols[1]), length(Y), length(newCols)+1)
		# Xtmp2 = ones(eltype(xCols[1]), length(Ytest), length(newCols)+1)
		#fill matrices with new cols, repeatedly using X and Xtest to avoid allocating new memory
		for j in eachindex(newCols)
			if colsVec[switchInd]
				X[:, j+1] = xCols[newCols[j]]
			elseif j == length(newCols) #if a column was added then only change X in the final added column since it was already formed above
				X[:, j+1] = xCols[newCols[j]]
			end
		end
		# Xtmp = view(X, :, [1; newCols+1])
		# Xtmp2 = view(Xtest, :, [1; newCols+1])
		Xtmp = view(X, :, 1:length(newCols)+1)
		tfill = time()-t2
		# println(string("New X fill time = ", tfill))
		push!(errTimeStr, string("New X fill time = ", tfill, "\n"))
		t2 = time()
		BICs = calcLinRegErr(Rnew, Xtmp, Y)
		tErr = time()-t2
		# println(string("Err update time = ", tErr))
		push!(errTimeStr, string("Err update time = ", tErr, "\n"))
		(BICs, Rnew, newCols)
	end
	tErr = time() - t
	# println(string("Total New Err time = ", tErr))
	push!(errTimeStr, string("Total New Err time = ", tErr, "\n"))
	alpha = if BICs[2] < lastBICs[2]
		1
	else
		#base acceptance probability on the distance to the naive logLikelihoodError so if the
		#new columns would lower the log likelihood to the naive case the acceptance probability is 0
		# exp((BICs[3] - lastBICs[3])/T) # 1 - 2*(log(lastBICs[3]) - log(BICs[3]))/(T*nTest) 
		exp(-(BICs[2] - lastBICs[2])/T)
	end

	acc = (alpha > rand())
	
	if printIter #only print if true
		#delete lines for printing new step update only after calculations are complete
		lines = ceil(Int64, size(X, 2)/20) + 13
		for i in 1:lines-1
			print("\r\u1b[K\u1b[A")
		end
		
		# elapsedT = time() - startTime
		# avgIterTime = elapsedT / iNum
		avgIterTime = rpad(round(startTime, digits = 4), 6, '0')
		ETA = round(startTime*(length(Tsteps) - iNum), digits =4)
		ETAmins = floor(Int64, ETA/60)
		ETAsecs = round(ETA - ETAmins*60, digits = 2)
		ETAstr = string(ETAmins, ":", lpad(ETAsecs, 5, '0'))
		println(string("Completed step ", iNum, " of ", length(Tsteps), ", Repeat Step Rate - ", rpad(round(dictHitRate, digits = 4), 6, '0'), ", Avg Step Time - ", avgIterTime, ", ETA - ", ETAstr))

		for s in errTimeStr
			print(s)
		end

		println("Done evaluating candidate columns:")
		printColsVec(firstColsVec, colsVec, switchInd, acc)
		println()
		println(string("Got error and BIC of ", (BICs[1], BICs[2])))
		# println(string("New test log likelihood is ", errs[3], " compared to ", lastErrs[3], " for the previous set"))
		
		alphaStr = rpad(round(alpha, digits = 10), 12, '0')
		if acc
			println(string("Accepting new candidate columns with an alpha of ", alphaStr, ", recent acceptance rate = ", round(acceptRate, digits = 5)))
			# println()
		else
			println(string("Rejecting new candidate columns with an alpha of ", alphaStr, ", recent acceptance rate = ", round(acceptRate, digits = 5)))
			# println()
		end
		println(string("Current state error and BIC: ", (lastBICs[1], lastBICs[2])))
		println()
	end
	(BICs, newColsVec, changeString, acc, Rnew, newCols)

end

function gibbsStep(xCols, X, Y, xTestCols, Xtest, Ytest, lastR, currentCols, lastErrs, colsRecord, colsVec, T, iNum, Tsteps, startTime, acceptRate, dictHitRate, firstColsVec, printIter = true)
	switchInd = ceil(Int64, rand()*length(colsVec))
	changeString = if colsVec[switchInd]
		string("- ", switchInd)
	else
		string("+ ", switchInd)
	end

	# println(string("Converted Cols Vec = ", find(colsVec)))

	newColsVec = [i == switchInd ? !colsVec[i] : colsVec[i] for i in 1:length(colsVec)]
	# candidateCols = find(newColsVec)
	# currentCols = find(colsVec)
	
	errTimeStr = []

	t = time()
	(errs, Rnew, newCols) = if haskey(colsRecord, newColsVec) # in(newColsVec, map(a -> a[1], colsRecord))
		# colsRecord[find(a -> a == newColsVec, map(a -> a[1], colsRecord))][1][2]
		#update newCols with entry from dictionary to ensure column order is preserved
		for i in 1:3 
			# println()
			push!(errTimeStr, "\n")
		end
		push!(errTimeStr, "Got errors from dictionary\n")
		colsRecord[newColsVec]
	else
		#otherwise form it from the previous value and the switching index
		newCols = if !colsVec[switchInd]
			[currentCols; switchInd]
		else
			setdiff(currentCols, switchInd)
		end
		# println(string("Current Cols = ", currentCols))
		# println(string("New Cols = ", newCols))
		# println(string("Switch Index = ", switchInd, " and has a value of ", colsVec[switchInd]))
		t2 = time()
		Rnew = if !colsVec[switchInd]
			t3 = time()
			# Xold = ones(eltype(xCols[1]), length(Y), length(currentCols)+1)
			for j in eachindex(currentCols)	
				X[:, j+1] = xCols[currentCols[j]]	
			end
			# Xold = view(X, :, [1; currentCols+1])
			Xold = view(X, :, 1:length(currentCols)+1)
			tfill = time() - t3
			# println(string("Old X fill time = ", tfill))
			push!(errTimeStr, string("Old X fill time = ", tfill, "\n"))
			qraddcol(Xold, lastR, xCols[switchInd])
		else
			# println()
			push!(errTimeStr, "\n")
			k = findfirst(a -> a == switchInd, currentCols)
			qrdelcol(lastR, k+1) #index of column being removed
		end
		tR = time() - t2
		# println(string("R update time = ", tR))
		push!(errTimeStr, string("R update time = ", tR, "\n"))

		t2 = time()
		# Xtmp = ones(eltype(xCols[1]), length(Y), length(newCols)+1)
		# Xtmp2 = ones(eltype(xCols[1]), length(Ytest), length(newCols)+1)
		#fill matrices with new cols, repeatedly using X and Xtest to avoid allocating new memory
		for j in eachindex(newCols)
			if colsVec[switchInd]
				X[:, j+1] = xCols[newCols[j]]
			elseif j == length(newCols) #if a column was added then only change X in the final added column since it was already formed above
				X[:, j+1] = xCols[newCols[j]]
			end

			Xtest[:, j+1] = xTestCols[newCols[j]]
		end
		# Xtmp = view(X, :, [1; newCols+1])
		# Xtmp2 = view(Xtest, :, [1; newCols+1])
		Xtmp = view(X, :, 1:length(newCols)+1)
		Xtmp2 = view(Xtest, :, 1:length(newCols)+1)
		tfill = time()-t2
		# println(string("New X fill time = ", tfill))
		push!(errTimeStr, string("New X fill time = ", tfill, "\n"))
		t2 = time()
		errs = calcLinRegErr(Rnew, Xtmp, Y, Xtmp2, Ytest)
		tErr = time()-t2
		# println(string("Err update time = ", tErr))
		push!(errTimeStr, string("Err update time = ", tErr, "\n"))
		(errs, Rnew, newCols)
	end
	tErr = time() - t
	# println(string("Total New Err time = ", tErr))
	push!(errTimeStr, string("Total New Err time = ", tErr, "\n"))
	alpha = if errs[2] < lastErrs[2]
		1
	else
		#base acceptance probability on the distance to the naive logLikelihoodError so if the
		#new columns would lower the log likelihood to the naive case the acceptance probability is 0
		# exp((errs[3] - lastErrs[3])/T) # 1 - 2*(log(lastErrs[3]) - log(errs[3]))/(T*nTest) 
		exp(-(errs[2] - lastErrs[2])/T)
	end
	acc = (alpha > rand())

	if printIter #only print if true
		#delete lines for printing new step update only after calculations are complete
		lines = ceil(Int64, size(X, 2)/20) + 13
		for i in 1:lines-1
			print("\r\u1b[K\u1b[A")
		end
		
		# elapsedT = time() - startTime
		# avgIterTime = elapsedT / iNum
		avgIterTime = rpad(round(startTime, digits = 4), 6, '0')
		ETA = round(startTime*(length(Tsteps) - iNum), digits =4)
		ETAmins = floor(Int64, ETA/60)
		ETAsecs = round(ETA - ETAmins*60, digits = 2)
		ETAstr = string(ETAmins, ":", lpad(ETAsecs, 5, '0'))
		println(string("Completed step ", iNum, " of ", length(Tsteps), ", Repeat Step Rate - ", rpad(round(dictHitRate, digits = 4), 6, '0'), ", Avg Step Time - ", avgIterTime, ", ETA - ", ETAstr))

		for s in errTimeStr
			print(s)
		end

		println("Done evaluating candidate columns:")
		printColsVec(firstColsVec, colsVec, switchInd, acc)
		println()
		println(string("Got training and test set errors of ", (errs[1], errs[2])))
		# println(string("New test log likelihood is ", errs[3], " compared to ", lastErrs[3], " for the previous set"))
		
		alphaStr = rpad(round(alpha, digits = 10), 12, '0')
		if acc
			println(string("Accepting new candidate columns with an alpha of ", alphaStr, ", recent acceptance rate = ", round(acceptRate, digits = 5)))
			# println()
		else
			println(string("Rejecting new candidate columns with an alpha of ", alphaStr, ", recent acceptance rate = ", round(acceptRate, digits = 5)))
			# println()
		end
		println(string("Current state training and test errors: ", (lastErrs[1], lastErrs[2])))
		println()
	end
	# println(string("Converted New Cols Vec = ", find(newColsVec)))
	# sleep(0.5)
	(errs, newColsVec, changeString, acc, Rnew, newCols)

end

function runGibbsStep(xCols, X, Y, Rnow, currentCols, BICsRecord, colsRecord, Tsteps, i = 1; startTime = time(), lastPrintTime = time(), acceptRate = 0, iterTime = 0, dictHitRate = 0, firstColsVec = fill(false, length(xCols)), printStep = true, updateInterval = 1.0)
	# println(string("Starting iteration ", i, " of ", length(Tsteps)))
	if i == length(Tsteps)
		return (BICsRecord, colsRecord)
	else
		#print first iteration and subsequent once every second by default
		printIter = if printStep
			if i == 1
				true
			elseif (time() - lastPrintTime) > updateInterval
				true
			else
				false
			end
		else
			false
		end

		stepStartTime = time()		
		out = gibbsStep(xCols, X, Y, Rnow, currentCols, BICsRecord[end][3], colsRecord, BICsRecord[end][2], Tsteps[i], i, Tsteps, iterTime, acceptRate, dictHitRate, firstColsVec, printIter)
		dictEntry = (out[1], out[5], out[6])
		(Rnew, newCols, newBICsRecord, newAcceptRate) = if out[4]
			r = 0.95*acceptRate + 0.05
			(out[5], out[6], [BICsRecord; (out[3], out[2], out[1])], r)
		else
			r = 0.95*acceptRate
			(Rnow, currentCols, BICsRecord, r)
		end

		newDictHitRate = if haskey(colsRecord, out[2])
			dictHitRate*0.95 + 0.05
		else
			dictHitRate*0.95
		end		
		newIterTime = 0.95*iterTime + 0.05*(time() - stepStartTime)
		newPrintTime = printIter ? stepStartTime : lastPrintTime
		runGibbsStep(xCols, X, Y, Rnew, newCols, newBICsRecord, push!(colsRecord, out[2] => dictEntry), Tsteps, i+1, startTime = startTime, lastPrintTime = newPrintTime, acceptRate = newAcceptRate, iterTime = newIterTime, dictHitRate = newDictHitRate, firstColsVec = firstColsVec, printStep = printStep)
	end
end

function runGibbsStep(xCols, X, Y, xTestCols, Xtest, Ytest, Rnow, currentCols, errsRecord, colsRecord, Tsteps, i = 1; startTime = time(), lastPrintTime = time(), acceptRate = 0, iterTime = 0, dictHitRate = 0, firstColsVec = fill(false, length(xCols)), printStep = true, updateInterval = 1.0)

	if i == length(Tsteps)
		return (errsRecord, colsRecord)
	else
		
		#print first iteration and subsequent once every 1 second by default
		printIter = if printStep
			if i == 1
				true
			elseif (time() - lastPrintTime) > updateInterval
				true
			else
				false
			end
		else
			false
		end

		stepStartTime = time()

		out = gibbsStep(xCols, X, Y, xTestCols, Xtest, Ytest, Rnow, currentCols, errsRecord[end][3], colsRecord, errsRecord[end][2], Tsteps[i], i, Tsteps, iterTime, acceptRate, dictHitRate, firstColsVec, printIter)
		dictEntry = (out[1], out[5], out[6])
		(Rnew, newCols, newErrsRecord, newAcceptRate) = if out[4]
			r = 0.95*acceptRate + 0.05
			(out[5], out[6], [errsRecord; (out[3], out[2], out[1])], r)
		else
			r = 0.95*acceptRate
			(Rnow, currentCols, errsRecord, r)
		end

		newDictHitRate = if haskey(colsRecord, out[2])
			dictHitRate*0.95 + 0.05
		else
			dictHitRate*0.95
		end
		
		newIterTime = 0.95*iterTime + 0.05*(time() - stepStartTime)

		newPrintTime = printIter ? stepStartTime : lastPrintTime
		runGibbsStep(xCols, X, Y, xTestCols, Xtest, Ytest, Rnew, newCols, newErrsRecord, push!(colsRecord, out[2] => dictEntry), Tsteps, i+1, startTime = startTime, lastPrintTime = newPrintTime, acceptRate = newAcceptRate, iterTime = newIterTime, dictHitRate = newDictHitRate, firstColsVec = firstColsVec, printStep = printStep)
	end
end

function runStepwiseAnneal(X::Array{T, 2}, Y::Array{T, 1}; initialColsVec = fill(false, size(X, 2)), initialColsRecord = Dict{Array{Bool, 1}, Tuple{Tuple{T, T}, Matrix{T}, Vector{Int64}}}(), iter = 0, printIter = true, updateInterval = 1.0) where T <: Real
	# T = eltype(X)
	newX = [ones(T, length(Y)) X]
	println("Beginning gibbs annealing stepwise regression")
	(err, BIC, R, initialCols) = if sum(initialColsVec) == 0
		# initialColsVec = fill(false, size(X, 2))
		println("Calculating bias term error as a baseline")
		meanY = mean(Y)
		err = sum(abs2, Y .- meanY)/length(Y)
		BIC = length(Y)*log(err)+log(length(Y))
		initialCols = findall(initialColsVec)
		_, R = qr([ones(T, length(Y)) view(X, :, initialCols)])
		(err, BIC, R, initialCols)
	else
		if haskey(initialColsRecord, initialColsVec)
			(errs, R, cols) = initialColsRecord[initialColsVec]
			println("Using dictionary to get errors for initial columns ", cols)
			(errs[1], errs[2], R, cols)
		else
			initialCols = findall(initialColsVec)
			println("Finding errors for initial columns ", initialCols)
			_, R = qr([ones(T, length(Y)) view(X, :, initialCols)])
			errs = calcLinRegErr(R, view(newX, :, [1; initialCols .+ 1]), Y)
			(errs[1], errs[2], R, initialCols)
		end
	end
		
	println(string("Got err and BIC of : ", (err, BIC), " using ", sum(initialColsVec), "/", length(initialColsVec), " columns"))
	println("--------------------------------------------------------------------")
	println("Setting up $(size(X, 2)) columns for step updates")
	println()

	Tmax = BIC/3000 #corresponds to a ~.1% increase in error having a ~1% chance of acceptance
	numSteps = 6*length(initialColsVec) #max(100, round(Int64, length(initialColsVec)*sqrt(length(initialColsVec))/2))
	# Tsteps = initialT*exp.(-7*(0:numSteps-1)/numSteps)
	Tsteps = if iter == 1
		zeros(T, 5*numSteps)
	else
		[Tmax*exp.(-4*(0:3*numSteps-1) ./(3*(numSteps-1))); zeros(T, 2*numSteps)]
	end
	xCols = [X[:, i] for i in 1:size(X, 2)]

	#add line padding for print update if print is turned on
	if printIter
		lines = ceil(Int64, size(X, 2)/20) + 13
		for i in 1:lines-1 
			println()
		end
		println("Waiting for first step to complete")
	else
		println("Starting $(length(Tsteps)) steps of annealing process without printing updates")
	end

	(BICsRecord, colsRecord) = runGibbsStep(xCols, newX, Y, R, initialCols, [("", initialColsVec, (err, BIC))], push!(initialColsRecord, initialColsVec => ((err, BIC), R, initialCols)), Tsteps, firstColsVec = initialColsVec, printStep=printIter, updateInterval = updateInterval)
	BICsOut = [begin
	    (a[1], findall(a[2]), a[3][1], a[3][2])
	end
	for a in BICsRecord]
	(BICsOut, colsRecord)
end


function runStepwiseAnneal(X::Array{T, 2}, Y::Array{T, 1}, Xtest::Array{T, 2}, Ytest::Array{T, 1}; initialColsVec = fill(false, size(X, 2)), initialColsRecord = Dict{Array{Bool, 1}, Tuple{Tuple{T, T}, Matrix{T}, Vector{Int64}}}(), iter = 0, printIter = true, updateInterval = 1.0) where T <: Real
	# T = eltype(X)
	newX = [ones(T, length(Y)) X]
	newXtest = [ones(T, length(Ytest)) Xtest]
	println("Beginning gibbs annealing stepwise regression")
	(trainErr, testErr, R, initialCols) = if sum(initialColsVec) == 0
		# initialColsVec = fill(false, size(X, 2))
		println("Calculating bias term error as a baseline")
		meanY = mean(Y)
		trainErr = sum(abs2, Y .- meanY)/length(Y)
		testErr = sum(abs2, Ytest .- meanY)/length(Ytest)
		initialCols = findall(initialColsVec)
		_, R = qr([ones(T, length(Y)) view(X, :, initialCols)])
		(trainErr, testErr, R, initialCols)
	else
		if haskey(initialColsRecord, initialColsVec)
			(errs, R, cols) = initialColsRecord[initialColsVec]
			println("Using dictionary to get errors for initial columns ", cols)
			(errs[1], errs[2], R, cols)
		else
			initialCols = findall(initialColsVec)
			println("Finding errors for initial columns ", initialCols)
			_, R = qr([ones(T, length(Y)) view(X, :, initialCols)])
			errs = calcLinRegErr(R, view(newX, :, [1; initialCols .+ 1]), Y, view(newXtest, :, [1; initialCols .+ 1]), Ytest)
			(errs[1], errs[2], R, initialCols)
		end
	end
		
	println(string("Got training and test set errors of : ", (trainErr, testErr), " using ", sum(initialColsVec), "/", length(initialColsVec), " columns"))
	println("--------------------------------------------------------------------")
	println("Setting up $(size(X, 2)) columns for step updates")
	Tmax = testErr/3000 #corresponds to a ~.1% increase in error having a ~1% chance of acceptance
	numSteps = 6*length(initialColsVec) #max(100, round(Int64, length(initialColsVec)*sqrt(length(initialColsVec))/2))
	# Tsteps = initialT*exp.(-7*(0:numSteps-1)/numSteps)
	Tsteps = if iter == 1
		zeros(T, 5*numSteps)
	else
		[Tmax*exp.(-4*(0:3*numSteps-1)/(3*(numSteps-1))); zeros(T, 2*numSteps)]
	end
	xCols = [X[:, i] for i in 1:size(X, 2)]
	xTestCols = [Xtest[:, i] for i in 1:size(Xtest, 2)]

	#add line padding for print update if print is turned on
	if printIter
		lines = ceil(Int64, size(X, 2)/20) + 13
		for i in 1:lines-1 
			println()
		end
		println("Waiting for first step to complete")
	else
		println("Starting $(length(Tsteps)) steps of annealing process without printing updates")
	end

	(errsRecord, colsRecord) = runGibbsStep(xCols, newX, Y, xTestCols, newXtest, Ytest, R, initialCols, [("", initialColsVec, (trainErr, testErr))], push!(initialColsRecord, initialColsVec => ((trainErr, testErr), R, initialCols)), Tsteps, firstColsVec = initialColsVec, printStep=printIter, updateInterval = updateInterval)
	errsOut = [begin
	    (a[1], findall(a[2]), a[3][1], a[3][2])
	end
	for a in errsRecord]
	(errsOut, colsRecord)
end


function runStepwiseAnnealProcess(name::String, X::Matrix{Float64}, Y::Vector{Float64}; seed = 1, colNames = map(a -> string("Col ", a), 1:size(X, 2)), initialColsRecord = Dict{Array{Bool, 1}, Tuple{Tuple{T, T}, Matrix{T}, Vector{Int64}}}(), printIter=true, updateInterval = 1.0)
	n = size(X, 2)
	numExamples = size(X, 1)

	Random.seed!(seed)
	println(string("Starting with seed ", seed))
	initialColsVec = if seed == 1
		fill(false, size(X, 2))
	elseif seed == 2
		fill(true, size(X, 2))
	else
		rand(Bool, size(X, 2))
	end
	startTime = time()
	(BICsRecord1, colsRecord1) = runStepwiseAnneal(X, Y, initialColsRecord = initialColsRecord, iter = 1, initialColsVec = initialColsVec, printIter = printIter, updateInterval = updateInterval) #, initialColsVec=initialColsVec)
	iterSeconds = time() - startTime
	iterMins = floor(Int64, iterSeconds/60)
	iterSecs = iterSeconds - 60*iterMins
	iterTimeStr = string(iterMins, ":", round(iterSecs, digits = 2))
	sortInd = sortperm(map(a -> a[4], BICsRecord1))
	writedlm(string("gibbsStepwiseBICAnnealQRupdate_", T, "_", name, "_", seed, "Seed_record1.txt"), [BICsRecord1[sortInd[1]]; BICsRecord1])
	bestCols1 = BICsRecord1[sortInd[1]][2]
	bestErr1 = BICsRecord1[sortInd[1]][3]
	bestBIC1 = BICsRecord1[sortInd[1]][4]
	println(string("Iteration 1 complete after ", iterTimeStr, " with best columns: ", bestCols1, "\nwith error and BIC of ", (bestErr1, bestBIC1)))
	println("--------------------------------------------------------------------")
	println()

	println("Starting second iteration from this starting point")
	colsVec2 = [in(i, bestCols1) ? true : false for i in 1:size(X, 2)]
	startTime = time()
	(BICsRecord2, colsRecord2) = runStepwiseAnneal(X, Y, initialColsVec = colsVec2, initialColsRecord = colsRecord1, printIter = printIter, updateInterval = updateInterval)
	iterSeconds = time() - startTime
	iterMins = floor(Int64, iterSeconds/60)
	iterSecs = iterSeconds - 60*iterMins
	iterTimeStr = string(iterMins, ":", round(iterSecs, digits = 2))

	sortInd = sortperm(map(a -> a[4], BICsRecord2))
	writedlm(string("gibbsStepwiseBICAnnealQRupdate_", T, "_", name, "_", seed, "Seed_record2.txt"), [BICsRecord2[sortInd[1]]; BICsRecord2])
	bestCols2 = BICsRecord2[sortInd[1]][2]
	bestErr2 = BICsRecord2[sortInd[1]][3]
	bestBIC2 = BICsRecord2[sortInd[1]][4]
	println(string("Iteration 2 complete after ", iterTimeStr, " with best columns: ", bestCols2, "\nwith error and BIC of ", (bestErr2, bestBIC2)))
	println("--------------------------------------------------------------------")
	println()

	if bestCols2 == bestCols1
		println("Skipping further iterations because best columns are still from iteration 1")
		println()
		println()
		usedCols = bestCols2
		usedColsCheck = [in(i, usedCols) ? "x" : "" for i in 1:length(colNames)]
		writedlm(string("gibbsStepwiseBICAnnealQRupdateLinReg_", seed, "Seed_UsedCols_", name, ".txt"), [usedColsCheck colNames])
		(usedCols, usedColsCheck, bestErr2, bestBIC2, colsRecord2)
	else
		iter = 3
		newBestCols = bestCols2
		oldBestCols = bestCols1
		newColsRecord = colsRecord2
		newbestErr = Inf
		newBestBIC = Inf
		while newBestCols != oldBestCols
			oldBestCols = newBestCols 
			println(string("Starting iteration ", iter, " from previous best columns"))
			newColsVec = [in(i, oldBestCols) ? true : false for i in 1:size(X, 2)]
			startTime = time()
			(newBICsRecord, newColsRecord) = runStepwiseAnneal(X, Y, initialColsVec = newColsVec, initialColsRecord = newColsRecord, printIter=printIter, updateInterval = updateInterval)
			iterSeconds = time() - startTime
			iterMins = floor(Int64, iterSeconds/60)
			iterSecs = iterSeconds - 60*iterMins
			iterTimeStr = string(iterMins, ":", round(iterSecs, digits = 2))

			sortInd = sortperm(map(a -> a[4], newBICsRecord))
			writedlm(string("gibbsStepwiseBICAnnealQRupdate_", T, "_", name, "_", seed, "Seed_record", iter, ".txt"), [newBICsRecord[sortInd[1]]; newBICsRecord])
			newBestCols = newBICsRecord[sortInd[1]][2]
			newbestErr = newBICsRecord[sortInd[1]][3]
			newBestBIC = newBICsRecord[sortInd[1]][4]
			println(string("Iteration ", iter, " complete after ", iterTimeStr, " with best columns: ", newBestCols, "\nwith error and BIC of ", (newbestErr, newBestBIC)))
			println("--------------------------------------------------------------------")
			println()
			iter += 1
		end
		println(string("Skipping further iterations because best columns are still from iteration ", iter - 2))
		println()
		println()
		usedCols = newBestCols
		usedColsCheck = [in(i, usedCols) ? "x" : "" for i in 1:length(colNames)]
		writedlm(string("gibbsStepwiseBICAnnealQRupdateLinReg_", seed, "Seed_UsedCols_", name, ".txt"), [usedColsCheck colNames])
		(usedCols, usedColsCheck, newbestErr, newBestBIC, newColsRecord)
	end
end



function runStepwiseAnnealProcess(name::String, X::Matrix{Float64}, Y::Vector{Float64}, Xtest::Matrix{Float64}, Ytest::Vector{Float64}; seed = 1, colNames = map(a -> string("Col ", a), 1:size(X, 2)), initialColsRecord = Dict{Array{Bool, 1}, Tuple{Tuple{T, T}, Matrix{T}, Vector{Int64}}}(), printIter=true, updateInterval = 1.0)
	n = size(X, 2)
	numExamples = size(X, 1)

	Random.seed!(seed)
	println(string("Starting with seed ", seed))
	initialColsVec = if seed == 1
		fill(false, size(X, 2))
	elseif seed == 2
		fill(true, size(X, 2))
	else
		rand(Bool, size(X, 2))
	end

	startTime = time()
	(errsRecord1, colsRecord1) = runStepwiseAnneal(X, Y, Xtest, Ytest, iter = 1, initialColsRecord = initialColsRecord, initialColsVec = initialColsVec, printIter=printIter, updateInterval = updateInterval) #, initialColsVec=initialColsVec)
	iterSeconds = time() - startTime
	iterMins = floor(Int64, iterSeconds/60)
	iterSecs = iterSeconds - 60*iterMins
	iterTimeStr = string(iterMins, ":", round(iterSecs, digits = 2))
	sortInd = sortperm(map(a -> a[4], errsRecord1))
	writedlm(string("gibbsStepwiseAnnealQRupdate_", T, "_", name, "_", seed, "Seed_record1.txt"), [errsRecord1[sortInd[1]]; errsRecord1])
	bestCols1 = errsRecord1[sortInd[1]][2]
	bestTrainErr1 = errsRecord1[sortInd[1]][3]
	bestTestErr1 = errsRecord1[sortInd[1]][4]
	println(string("Iteration 1 complete after ", iterTimeStr, " with best columns: ", bestCols1, "\nwith training and test errors of ", (bestTrainErr1, bestTestErr1)))
	println("--------------------------------------------------------------------")
	println()

	println("Starting second iteration from this starting point")
	colsVec2 = [in(i, bestCols1) ? true : false for i in 1:size(X, 2)]
	startTime = time()
	(errsRecord2, colsRecord2) = runStepwiseAnneal(X, Y, Xtest, Ytest, initialColsVec = colsVec2, initialColsRecord = colsRecord1, printIter=printIter, updateInterval = updateInterval)
	iterSeconds = time() - startTime
	iterMins = floor(Int64, iterSeconds/60)
	iterSecs = iterSeconds - 60*iterMins
	iterTimeStr = string(iterMins, ":", round(iterSecs, digits = 2))
	sortInd = sortperm(map(a -> a[4], errsRecord2))
	writedlm(string("gibbsStepwiseAnnealQRupdate_", T, "_", name, "_", seed, "Seed_record2.txt"), [errsRecord2[sortInd[1]]; errsRecord2])
	bestCols2 = errsRecord2[sortInd[1]][2]
	bestTrainErr2 = errsRecord2[sortInd[1]][3]
	bestTestErr2 = errsRecord2[sortInd[1]][4]
	println(string("Iteration 2 complete after ", iterTimeStr, " with best columns: ", bestCols2, "\nwith training and test errors of ", (bestTrainErr2, bestTestErr2)))
	println("--------------------------------------------------------------------")
	println()

	if bestCols2 == bestCols1
		println("Skipping further iterations because best columns are still from iteration 1")
		println()
		println()
		usedCols = bestCols2
		usedColsCheck = [in(i, usedCols) ? "x" : "" for i in 1:length(colNames)]
		writedlm(string("gibbsStepwiseAnnealQRupdateLinReg_", seed, "Seed_UsedCols_", name, ".txt"), [usedColsCheck colNames])
		(usedCols, usedColsCheck, bestTrainErr2, bestTestErr2, colsRecord2)
	else
		iter = 3
		newBestCols = bestCols2
		oldBestCols = bestCols1
		newColsRecord = colsRecord2
		newBestTrainErr = Inf
		newBestTestErr = Inf
		while newBestCols != oldBestCols
			oldBestCols = newBestCols 
			println(string("Starting iteration ", iter, " from previous best columns"))
			newColsVec = [in(i, oldBestCols) ? true : false for i in 1:size(X, 2)]
			startTime = time()
			(newErrsRecord, newColsRecord) = runStepwiseAnneal(X, Y, Xtest, Ytest, initialColsVec = newColsVec, initialColsRecord = newColsRecord, printIter=printIter, updateInterval = updateInterval)
			iterSeconds = time() - startTime
			iterMins = floor(Int64, iterSeconds/60)
			iterSecs = iterSeconds - 60*iterMins
			iterTimeStr = string(iterMins, ":", round(iterSecs, digits = 2))			
			sortInd = sortperm(map(a -> a[4], newErrsRecord))
			writedlm(string("gibbsStepwiseAnnealQRupdate_", T, "_", name, "_", seed, "Seed_record", iter, ".txt"), [newErrsRecord[sortInd[1]]; newErrsRecord])
			newBestCols = newErrsRecord[sortInd[1]][2]
			newBestTrainErr = newErrsRecord[sortInd[1]][3]
			newBestTestErr = newErrsRecord[sortInd[1]][4]
			println(string("Iteration ", iter, " complete after ", iterTimeStr, " with best columns: ", newBestCols, "\nwith training and test errors of ", (newBestTrainErr, newBestTestErr)))
			println("--------------------------------------------------------------------")
			println()
			iter += 1
		end
		println(string("Skipping further iterations because best columns are still from iteration ", iter - 2))
		println()
		println()
		usedCols = newBestCols
		usedColsCheck = [in(i, usedCols) ? "x" : "" for i in 1:length(colNames)]
		writedlm(string("gibbsStepwiseAnnealQRupdateLinReg_", seed, "Seed_UsedCols_", name, ".txt"), [usedColsCheck colNames])
		(usedCols, usedColsCheck, newBestTrainErr, newBestTestErr, newColsRecord)
	end
end


