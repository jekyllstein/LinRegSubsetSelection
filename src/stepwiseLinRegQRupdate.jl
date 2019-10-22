using QRupdate
using LinearAlgebra
using Statistics
using DelimitedFiles

function removeCol(Rold, Xnew, Y, k)
#Rold is the R matrix of the previous fit, Xnew is the updated 
#input matrix, Y is the desired output, k is the column being removed
#from the original data set 
	Rnew = qrdelcol(Rold, k)
	betas, r = csne(Rnew, Xnew, Y)
	linRegTrainErr = sum(abs2, r)/length(Y)
	linRegTestErr = sum(abs2, (Xtest*betas .- Ytest))/length(Ytest)
	(linRegTrainErr, linRegTestErr, Rnew)
end

function addCol(Rold, Xold, Xnew, Y, newCol)
#Rold is the R matrix of the previous fit, Xnew is the updated 
#input matrix, Y is the desired output, Xold is the previous input matrix,
#newCol is the column being added
	Rnew = qraddcol(Xold, Rold, newCol)
	betas, r = csne(Rnew, Xnew, Y)
	linRegTrainErr = sum(abs2, r)/length(Y)
	linRegTestErr = sum(abs2, (Xtest*betas .- Ytest))/length(Ytest)
	(linRegTrainErr, linRegTestErr, Rnew)
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
	linRegTrainErr = sum(abs2, (X*betas .- Y)) #FCANN.calcError(X*betas, Y, costFunc = "sqErr")
	linRegTestErr = sum(abs2, (Xtest*betas .- Ytest))/length(Ytest) #FCANN.calcError(Xtest*betas, Ytest, costFunc = "sqErr")
	(linRegTrainErr, linRegTestErr)
end

function getBestAddCols(xCols, Y, cols, R)
	candidateCols = setdiff(1:length(xCols), cols)
	println(string("Preparing to evaluate ", candidateCols, " remaining columns for addition"))
	println()
	println()
	println()
	counter = 0
	Xtmp = ones(eltype(xCols[1]), length(Y), length(cols)+2)
	Xold = ones(eltype(xCols[1]), length(Y), length(cols)+1)
	for j in 1:length(cols)
		Xtmp[:, j+1] = xCols[cols[j]]
		Xold[:, j+1] = xCols[cols[j]]
	end
	
	newRegErrs = [begin
		Xtmp[:, end] = xCols[c]
		t = time()
		# Rnew = qraddcol(view(X, :, [1; cols+1]), R, xCols[c])
		Rnew = qraddcol(Xold, R, xCols[c])
		tR = time() - t
		t = time()
		# errs = calcLinRegErr(Rnew, view(X, :, [1; [cols;c]+1]), Y, view(Xtest, :, [1; [cols;c]+1]), Ytest) 
		errs = calcLinRegErr(Rnew, Xtmp, Y) 
		tErr = time() - t
		counter += 1
		print("\r\u1b[K\u1b[A")
		print("\r\u1b[K\u1b[A")
		print("\r\u1b[K\u1b[A")
		println(string("R update time = ", tR, ", Fit time = ", tErr))
		println(string("Done with column ", c, ": number ", counter, " out of ", length(candidateCols), " total candidates"))
		println(string("Got error and BIC of : ", errs))
		(errs, Rnew)
	end
	for c in candidateCols]
	print("\r\u1b[K\u1b[A")
	print("\r\u1b[K\u1b[A")
	# (bestTrainErr, bestTestErr, bestR, bestCols)
	(newRegErrs, candidateCols)
end

function getBestAddCols(xCols, Y, cols, R, xTestCols, Ytest)
	candidateCols = setdiff(1:length(xCols), cols)
	println(string("Preparing to evaluate ", candidateCols, " remaining columns for addition"))
	println()
	println()
	println()
	counter = 0
	Xtmp = ones(eltype(xCols[1]), length(Y), length(cols)+2)
	Xold = ones(eltype(xCols[1]), length(Y), length(cols)+1)
	Xtmp2 = ones(eltype(xCols[1]), length(Ytest), length(cols)+2)
	for j in 1:length(cols)
		Xtmp[:, j+1] = xCols[cols[j]]
		Xold[:, j+1] = xCols[cols[j]]
		Xtmp2[:, j+1] = xTestCols[cols[j]]
	end
	
	newRegErrs = [begin
		Xtmp[:, end] = xCols[c]
		Xtmp2[:, end] = xTestCols[c]
		t = time()
		# Rnew = qraddcol(view(X, :, [1; cols+1]), R, xCols[c])
		Rnew = qraddcol(Xold, R, xCols[c])
		tR = time() - t
		t = time()
		# errs = calcLinRegErr(Rnew, view(X, :, [1; [cols;c]+1]), Y, view(Xtest, :, [1; [cols;c]+1]), Ytest) 
		errs = calcLinRegErr(Rnew, Xtmp, Y, Xtmp2, Ytest) 
		tErr = time() - t
		counter += 1
		print("\r\u1b[K\u1b[A")
		print("\r\u1b[K\u1b[A")
		print("\r\u1b[K\u1b[A")
		println(string("R update time = ", tR, ", Fit time = ", tErr))
		println(string("Done with column ", c, ": number ", counter, " out of ", length(candidateCols), " total candidates"))
		println(string("Got training and test set errors of : ", errs))
		(errs, Rnew)
	end
	for c in candidateCols]
	print("\r\u1b[K\u1b[A")
	print("\r\u1b[K\u1b[A")
	# (bestTrainErr, bestTestErr, bestR, bestCols)
	(newRegErrs, candidateCols)
end

function getBestRemovalCols(xCols, Y, cols, R)
	println()
	println()
	println()
	counter = 0
	Xtmp = ones(eltype(xCols[1]), length(Y), length(cols))
	#fill matrices leaving out first column
	for j in 2:length(cols)
		Xtmp[:, j] = xCols[cols[j]]
	end

	newRegErrs = [begin
			newcols = setdiff(cols, c)
			# for j in 1:length(cols)-1
			# 	Xtmp[:, j+1] = xCols[newcols[j]]
			# 	Xtmp2[:, j+1] = xTestCols[newcols[j]]
			# end
			
			#at each step swap out counter column to create substitution matrix.  For example at first we 
			#have selected from cols the indices 2, 3, 4, 5..... and for step two we desire 1, 3, 4, 5....
			#to reach this we replace column 1 with 1 and leave the rest unchanged.  Then for step three we
			#desire 1, 2, 4, 5.... to reach this we replace column 2 with 2 and leave the rest unchanged
			if counter > 0
				Xtmp[:, counter+1] = xCols[newcols[counter]]
			end

			k = findfirst(a -> a == c, cols)
			t = time()
			Rnew = qrdelcol(R, k+1) #index of column being removed
			tR = time() - t
			t = time()
			# errs = calcLinRegErr(Rnew, view(X, :, [1; newcols+1]), Y, view(Xtest, :, [1; newcols+1]), Ytest) 
			errs = calcLinRegErr(Rnew, Xtmp, Y) 
			tErr = time() - t
			counter += 1
			print("\r\u1b[K\u1b[A")
			print("\r\u1b[K\u1b[A")
			print("\r\u1b[K\u1b[A")
			println(string("R update time = ", tR, ", Fit time = ", tErr))
			println(string("Done with column ", c, ": number ", counter, " out of ", length(cols), " total candidates"))
			println(string("Got err and BIC of : ", errs))
			(errs, Rnew)
		end
	for c in cols]
	
	print("\r\u1b[K\u1b[A")
	print("\r\u1b[K\u1b[A")
	newRegErrs
end

function getBestRemovalCols(xCols, Y, cols, R, xTestCols, Ytest)
	println()
	println()
	println()
	counter = 0
	Xtmp = ones(eltype(xCols[1]), length(Y), length(cols))
	Xtmp2 = ones(eltype(xCols[1]), length(Ytest), length(cols))
	#fill matrices leaving out first column
	for j in 2:length(cols)
		Xtmp[:, j] = xCols[cols[j]]
		Xtmp2[:, j] = xTestCols[cols[j]]
	end

	newRegErrs = [begin
			newcols = setdiff(cols, c)
			# for j in 1:length(cols)-1
			# 	Xtmp[:, j+1] = xCols[newcols[j]]
			# 	Xtmp2[:, j+1] = xTestCols[newcols[j]]
			# end
			
			#at each step swap out counter column to create substitution matrix.  For example at first we 
			#have selected from cols the indices 2, 3, 4, 5..... and for step two we desire 1, 3, 4, 5....
			#to reach this we replace column 1 with 1 and leave the rest unchanged.  Then for step three we
			#desire 1, 2, 4, 5.... to reach this we replace column 2 with 2 and leave the rest unchanged
			if counter > 0
				Xtmp[:, counter+1] = xCols[newcols[counter]]
				Xtmp2[:, counter+1] = xTestCols[newcols[counter]]
			end

			k = findfirst(a -> a == c, cols)
			t = time()
			Rnew = qrdelcol(R, k+1) #index of column being removed
			tR = time() - t
			t = time()
			# errs = calcLinRegErr(Rnew, view(X, :, [1; newcols+1]), Y, view(Xtest, :, [1; newcols+1]), Ytest) 
			errs = calcLinRegErr(Rnew, Xtmp, Y, Xtmp2, Ytest) 
			tErr = time() - t
			counter += 1
			print("\r\u1b[K\u1b[A")
			print("\r\u1b[K\u1b[A")
			print("\r\u1b[K\u1b[A")
			println(string("R update time = ", tR, ", Fit time = ", tErr))
			println(string("Done with column ", c, ": number ", counter, " out of ", length(cols), " total candidates"))
			println(string("Got training and test set errors of : ", errs))
			(errs, Rnew)
		end
	for c in cols]
	
	print("\r\u1b[K\u1b[A")
	print("\r\u1b[K\u1b[A")
	newRegErrs
end

function stepwiseRegForward(xCols, Y, BICs, cols, R, record, colsRecord, numSteps = 0)
#function modifies X input
	if isempty(cols) & isempty(BICs)
		println("Calculating bias term error as a baseline")
		err = sum(abs2, (Y .- mean(Y)))/length(Y)
		BIC = length(Y)*log(err)+log(length(Y))
		println(string("Got err and BIC of : ", (err, BIC)))
		println("--------------------------------------------------------------------")
		println()
		stepwiseRegForward(xCols, Y, [BIC], cols, R, ["" err BIC; "" "" ""], [("", cols, err, BIC)], numSteps)
	elseif isempty(BICs)
		Xtmp = ones(eltype(xCols[1]), length(Y), length(cols)+1)
		for j in 1:length(cols)
			Xtmp[:, j+1] = xCols[j]
		end
		(err, BIC) = calcLinRegErr(R, Xtmp, Y)
	 	println(string("Starting with columns ", cols, " with an err and BIC of ", (err, BIC)))
	 	println("--------------------------------------------------------------------")
	 	println()
	 	stepwiseRegForward(xCols, Y, [BIC], cols, R, [record; ["" err BIC; "" "" ""]], [("", cols, err, BIC)], numSteps)
	else
		println(string("Using best columns ", cols, " with err and BIC of : ", (colsRecord[end][3], colsRecord[end][4])))
		if length(cols) == length(xCols)-1
			string("Terminating after adding all available columns")
			(BICs, cols, R, record, colsRecord, numSteps)
		else
			(newRegErrs, candidateCols) = getBestAddCols(xCols, Y, cols, R)

			(BICMin, ind) = findmin(map(a -> a[1][2], newRegErrs))
			# Rnew = newRegErrs[ind][2]
			newRecord = [candidateCols mapreduce(a -> [a[1][1] a[1][2]], vcat, newRegErrs); ["" "" ""]; [candidateCols[ind] newRegErrs[ind][1][1] newRegErrs[ind][1][2]]; ["" "" ""]]
			# newRecord = [["" "" ""]; [bestCols besterr bestBIC]; ["" "" ""]]
			print("\u1b[K")
			println(string("Best column to add is ", candidateCols[ind], " with a BIC of ", BICMin))
			# println(string("Best column to add is ", bestCols, " with a test set error of ", bestBIC))
			println("--------------------------------------------------------------------")
			println()
			if BICMin >= BICs[end]
				println("Terminating forward regression after finding no improvement to BIC")
				(BICs, cols, R, record, colsRecord, numSteps)
			else
				stepwiseRegForward(xCols, Y, [BICs; BICMin], [cols; candidateCols[ind]], newRegErrs[ind][2], [record; newRecord], [colsRecord; (string("+ ", candidateCols[ind]), [cols; candidateCols[ind]], newRegErrs[ind][1][1], newRegErrs[ind][1][2])], numSteps+1)
			end
		end
	end
end

function stepwiseRegForward(xCols, Y, xTestCols, Ytest, testErrs, cols, R, record, colsRecord, numSteps = 0)
#function modifies X input
	if isempty(cols) & isempty(testErrs)
		println("Calculating bias term error as a baseline")
		trainErr = sum(abs2, (Y .- mean(Y)))/length(Y)
		testErr = sum(abs2, (Ytest .- mean(Y)))/length(Ytest)
		println(string("Got training and test set errors of : ", (trainErr, testErr)))
		println("--------------------------------------------------------------------")
		println()
		stepwiseRegForward(xCols, Y, xTestCols, Ytest, [testErr], cols, R, ["" trainErr testErr; "" "" ""], [("", cols, trainErr, testErr)], numSteps)
	elseif isempty(testErrs)
		Xtmp = ones(eltype(xCols[1]), length(Y), length(cols)+1)
		Xtmp2 = ones(eltype(xCols[1]), length(Ytest), length(cols)+1)
		for j in 1:length(cols)
			Xtmp[:, j+1] = xCols[j]
			Xtmp2[:, j+1] = xTestCols[j]
		end
		(trainErr, testErr) = calcLinRegErr(R, Xtmp, Y, Xtmp2, Ytest)
	 	println(string("Starting with columns ", cols, " with a training and test error of ", (trainErr, testErr)))
	 	println("--------------------------------------------------------------------")
	 	println()
	 	stepwiseRegForward(xCols, Y, xTestCols, Ytest, [testErr], cols, R, [record; ["" trainErr testErr; "" "" ""]], [("", cols, trainErr, testErr)], numSteps)
	else
		println(string("Using best columns ", cols, " with training and test errors of : ", (colsRecord[end][3], colsRecord[end][4])))
		if length(cols) == length(xCols)-1
			string("Terminating after adding all available columns")
			(testErrs, cols, R, record, colsRecord, numSteps)
		else
			(newRegErrs, candidateCols) = getBestAddCols(xCols, Y, cols, R, xTestCols, Ytest)
			# for c in candidateCols]
			# Xtmp = nothing
			# Xold = nothing
			# gc()
			
			# testErrMin = bestTestErr
			(testErrMin, ind) = findmin(map(a -> a[1][2], newRegErrs))
			# Rnew = newRegErrs[ind][2]
			newRecord = [candidateCols mapreduce(a -> [a[1][1] a[1][2]], vcat, newRegErrs); ["" "" ""]; [candidateCols[ind] newRegErrs[ind][1][1] newRegErrs[ind][1][2]]; ["" "" ""]]
			# newRecord = [["" "" ""]; [bestCols bestTrainErr bestTestErr]; ["" "" ""]]
			print("\u1b[K")
			println(string("Best column to add is ", candidateCols[ind], " with a test set error of ", testErrMin))
			# println(string("Best column to add is ", bestCols, " with a test set error of ", bestTestErr))
			println("--------------------------------------------------------------------")
			println()
			if testErrMin >= testErrs[end]
				println("Terminating forward regression after finding no improvement to test set error")
				(testErrs, cols, R, record, colsRecord, numSteps)
			else
				stepwiseRegForward(xCols, Y, xTestCols, Ytest, [testErrs; testErrMin], [cols; candidateCols[ind]], newRegErrs[ind][2], [record; newRecord], [colsRecord; (string("+ ", candidateCols[ind]), [cols; candidateCols[ind]], newRegErrs[ind][1][1], newRegErrs[ind][1][2])], numSteps+1)
			end
		end
	end
end

function stepwiseRegBackward(xCols, Y, BICs, cols, R, record, colsRecord, numSteps = 0)
#modifies X input
	if isempty(cols) 
		println("Not enough data columns to remove")
		(BICs, cols, R, record, numSteps)
	elseif length(cols) == 1
		println("Removing only remaining column and calculating bias term error as a baseline")
		err = sum(abs2, (Y .- mean(Y)))
		BIC = log(err)*length(Y) + log(length(Y))
		println(string("Got err and BIC of : ", (err, BIC)))
		println("--------------------------------------------------------------------")
		println()
		([BICs; BIC], Array{Int64, 1}(0), R, [record; ["" err BIC]; ["" "" ""]], [colsRecord; (string("- ", cols[1]), [], err, BIC)], numSteps)
	else
		errs0 = if isempty(colsRecord)
			Xtmp = ones(eltype(xCols[1]), length(Y), length(cols)+1)
			for j in 1:length(cols)
				Xtmp[:, j+1] = xCols[j]
			end
			calcLinRegErr(R, Xtmp, Y)
			# calcLinRegErr(R, view(X, :, [1; cols+1]), Y, view(Xtest, :, [1; cols+1]), Ytest)
		else
			(colsRecord[end][3], colsRecord[end][4])
		end 
		println(string("Using best columns ", cols, " with error and BIC of : ", (errs0[1], errs0[2])))
		println(string("Preparing to evaluate removal of ", cols, " remaining columns"))
		
		newRegErrs = getBestRemovalCols(xCols, Y, cols, R)

		(BICMin, ind) = findmin(map(a -> a[1][2], newRegErrs))
		Rnew = newRegErrs[ind][2]
		newRecord = [cols mapreduce(a -> [a[1][1] a[1][2]], vcat, newRegErrs); ["" "" ""]; [cols[ind] newRegErrs[ind][1][1] newRegErrs[ind][1][2]]; ["" "" ""]]
		print("\u1b[K")
		println(string("Best column to remove is ", cols[ind], " with a BIC of ", BICMin))
		println("--------------------------------------------------------------------")
		println()


		if BICMin >= errs0[2]
			newBICs = if isempty(BICs)
				[errs0[2]]
			else
				BICs
			end
			newColsRecord = if isempty(colsRecord)
				[("", cols, errs0[1], errs0[2])]
			else
				colsRecord
			end
			println("Terminating backward regression after finding no improvement to BIC")
			(newBICs, cols, R, record, newColsRecord, numSteps)
		else
			newBICs = if isempty(BICs)
				[errs0[2]; BICMin]
			else
				[BICs; BICMin]
			end
			newColsRecord = if isempty(colsRecord)
				[("", cols, errs0[1], errs0[2]); (string("- ", cols[ind]), setdiff(cols, cols[ind]), newRegErrs[ind][1][1], newRegErrs[ind][1][2])]
			else
				[colsRecord; (string("- ", cols[ind]), setdiff(cols, cols[ind]), newRegErrs[ind][1][1], newRegErrs[ind][1][2])]
			end
			stepwiseRegBackward(xCols, Y, newBICs, setdiff(cols, cols[ind]), Rnew, [record; newRecord], newColsRecord, numSteps+1)
		end	
	end
end

function stepwiseRegBackward(xCols, Y, xTestCols, Ytest, testErrs, cols, R, record, colsRecord, numSteps = 0)
#modifies X input
	if isempty(cols) 
		println("Not enough data columns to remove")
		(testErrs, cols, R, record, numSteps)
	elseif length(cols) == 1
		println("Removing only remaining column and calculating bias term error as a baseline")
		trainErr = sum(abs2, (Y .- mean(Y)))
		testErr = sum(abs2, (Ytest .- mean(Y)))
		println(string("Got training and test set errors of : ", (trainErr, testErr)))
		println("--------------------------------------------------------------------")
		println()
		([testErrs; testErr], Array{Int64, 1}(undef, 0), R, [record; ["" trainErr testErr]; ["" "" ""]], [colsRecord; (string("- ", cols[1]), [], trainErr, testErr)], numSteps)
	else
		errs0 = if isempty(colsRecord)
			Xtmp = ones(eltype(xCols[1]), length(Y), length(cols)+1)
			Xtmp2 = ones(eltype(xCols[1]), length(Ytest), length(cols)+1)
			for j in 1:length(cols)
				Xtmp[:, j+1] = xCols[j]
				Xtmp2[:, j+1] = xTestCols[j]
			end
			calcLinRegErr(R, Xtmp, Y, Xtmp2, Ytest)
			# calcLinRegErr(R, view(X, :, [1; cols+1]), Y, view(Xtest, :, [1; cols+1]), Ytest)
		else
			(colsRecord[end][3], colsRecord[end][4])
		end 
		println(string("Using best columns ", cols, " with training and test errors of : ", (errs0[1], errs0[2])))
		println(string("Preparing to evaluate removal of ", cols, " remaining columns"))
		
		newRegErrs = getBestRemovalCols(xCols, Y, cols, R, xTestCols, Ytest)

		(testErrMin, ind) = findmin(map(a -> a[1][2], newRegErrs))
		Rnew = newRegErrs[ind][2]
		newRecord = [cols mapreduce(a -> [a[1][1] a[1][2]], vcat, newRegErrs); ["" "" ""]; [cols[ind] newRegErrs[ind][1][1] newRegErrs[ind][1][2]]; ["" "" ""]]
		print("\u1b[K")
		println(string("Best column to remove is ", cols[ind], " with a test set error of ", testErrMin))
		println("--------------------------------------------------------------------")
		println()


		if testErrMin >= errs0[2]
			newTestErrs = if isempty(testErrs)
				[errs0[2]]
			else
				testErrs
			end
			newColsRecord = if isempty(colsRecord)
				[("", cols, errs0[1], errs0[2])]
			else
				colsRecord
			end
			println("Terminating backward regression after finding no improvement to test set error")
			(newTestErrs, cols, R, record, newColsRecord, numSteps)
		else
			newTestErrs = if isempty(testErrs)
				[errs0[2]; testErrMin]
			else
				[testErrs; testErrMin]
			end
			newColsRecord = if isempty(colsRecord)
				[("", cols, errs0[1], errs0[2]); (string("- ", cols[ind]), setdiff(cols, cols[ind]), newRegErrs[ind][1][1], newRegErrs[ind][1][2])]
			else
				[colsRecord; (string("- ", cols[ind]), setdiff(cols, cols[ind]), newRegErrs[ind][1][1], newRegErrs[ind][1][2])]
			end
			stepwiseRegBackward(xCols, Y, xTestCols, Ytest, newTestErrs, setdiff(cols, cols[ind]), Rnew, [record; newRecord], newColsRecord, numSteps+1)
		end	
	end
end

function runStepwiseReg(name, X, Y; colNames = map(a -> string("Col ", a), 1:size(X, 2)))
	T = eltype(X)
	xCols = [X[:, i] for i in 1:size(X, 2)]
	println("Beginning forward stepwise regression")
	_, R = qr(ones(T, size(X, 1), 1))
	(BICs, cols, R, record, colsRecord, numSteps) = stepwiseRegForward(xCols, Y, Array{T, 1}(undef, 0), Array{Int64, 1}(undef, 0), R, ["Columns Added/Removed" "Error" "BIC"], [])
	while numSteps > 0
		println()
		println("--------------------------------------------------------------------")
		println(string("After ", numSteps, " forward steps, starting backwards stepwise regression"))
		(BICs, cols, R, record, colsRecord, numSteps) = stepwiseRegBackward(xCols, Y, BICs, cols, R, record, colsRecord)
		(BICs, cols, R, record, colsRecord, numSteps) = if numSteps > 0
			println(string("After ", numSteps, " backward steps, starting forwards stepwise regression"))
			stepwiseRegForward(xCols, Y, BICs, cols, R, record, colsRecord)
		else
			(BICs, cols, R, record, colsRecord, numSteps)
		end
	end
	writedlm(string("stepwiseBICLinRegQRupdate_", T, "_", name, "_cols+record.csv"), [[cols fill("", length(cols), 2)]; ["" "" ""]; [BICs fill("", length(BICs), 2)]; ["" "" ""]; record], ',')
	writedlm(string("stepwiseBICLinRegQRupdate_", T, "_", name, "_colsRecord.txt"), colsRecord)
	usedCols = sort(cols)
	unusedCols = setdiff(1:size(X, 2), cols)
	usedColsCheck = [in(i, usedCols) ? "x" : "" for i in 1:length(colNames)]
	writedlm(string("stepwiseBICLinRegQRupdateUsedCols_", name, ".txt"), [usedColsCheck colNames])
	(cols, usedColsCheck, colsRecord)
end

function runStepwiseReg(name, X, Y, Xtest, Ytest; colNames = map(a -> string("Col ", a), 1:size(X, 2)))
	T = eltype(X)
	xCols = [X[:, i] for i in 1:size(X, 2)]
	xTestCols = [Xtest[:, i] for i in 1:size(Xtest, 2)]
	println("Beginning forward stepwise regression")
	_, R = qr(ones(T, size(X, 1), 1))
	(testErrs, cols, R, record, colsRecord, numSteps) = stepwiseRegForward(xCols, Y, xTestCols, Ytest, Array{T, 1}(undef, 0), Array{Int64, 1}(undef, 0), R, ["Columns Added/Removed" "Train Error" "Test Error"], [])
	while numSteps > 0
		println()
		println("--------------------------------------------------------------------")
		println(string("After ", numSteps, " forward steps, starting backwards stepwise regression"))
		(testErrs, cols, R, record, colsRecord, numSteps) = stepwiseRegBackward(xCols, Y, xTestCols, Ytest, testErrs, cols, R, record, colsRecord)
		(testErrs, cols, R, record, colsRecord, numSteps) = if numSteps > 0
			println(string("After ", numSteps, " backward steps, starting forwards stepwise regression"))
			stepwiseRegForward(xCols, Y, xTestCols, Ytest, testErrs, cols, R, record, colsRecord)
		else
			(testErrs, cols, R, record, colsRecord, numSteps)
		end
	end
	writedlm(string("stepwiseLinRegQRupdate_", T, "_", name, "_cols+record.csv"), [[cols fill("", length(cols), 2)]; ["" "" ""]; [testErrs fill("", length(testErrs), 2)]; ["" "" ""]; record], ',')
	writedlm(string("stepwiseLinRegQRupdate_", T, "_", name, "_colsRecord.txt"), colsRecord)
	usedCols = sort(cols)
	unusedCols = setdiff(1:size(X, 2), cols)
	usedColsCheck = [in(i, usedCols) ? "x" : "" for i in 1:length(colNames)]
	writedlm(string("stepwiseLinRegQRupdateUsedCols_", name, ".txt"), [usedColsCheck colNames])
	(cols, usedColsCheck, colsRecord)
end