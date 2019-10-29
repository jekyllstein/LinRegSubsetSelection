function getnewtemps(temprecord, cutoff = 0)
	l = length(temprecord)
	high = 0.5
	Ts = [a[1] for a in temprecord]
	Ys = [a[2] for a in temprecord]
	Ybar = [mean(Ys[i:i+10]) for i in 1+floor(Int64, l*cutoff):max(1, length(Ys)-10)]
	Tbar = [mean(Ts[i:i+10]) for i in 1+floor(Int64, l*cutoff):max(1, length(Ys)-10)]
	(Ymin, Ymax) = extrema(Ybar)
	convY(y) = (y - Ymin)/(Ymax-Ymin)
	inds = [findfirst(a -> a <= b, convY.(Ybar)) for b in LinRange(high, high/5, 5)]
	# println("Ts = $(Tbar[inds])")
	Ts = [Tbar[inds]; 0.0]
end

function gibbs_step(data::NTuple{N, InOutPairCols{T}}, currentcols::AbstractVector{W}, Rlast::Matrix{T}, tmpX::NTuple{N, Matrix{T}}, lasterr::T, lastbic::T, colsrecord::RecordType{T}, colsvec::ColVec, temp, inum, tsteps, t_start, accept_rate, dicthitrate, firstcolsvec, printiter = true; xfillcol = 0) where N where T <: AbstractFloat where W <: Integer

	testerr = length(data) > 1
	traincols = data[1][1]
	y = data[1][2]
	Xtmp = tmpX[1]
	if testerr
		Xtmp2 = tmpX[2]
		testcols = data[2][1]
		ytest = data[2][2]
	end

	str = testerr ? "train and test error" : "error and BIC"


	#randomly select an index to switch
	switchind = ceil(Int64, rand()*length(colsvec))
	addcol = !colsvec[switchind] #true if adding a column
	changestring = "$switchind$(addcol ? "+" : "-")"

	newcolsvec = [(i == switchind ? !c : c) for (i, c) in enumerate(colsvec)]
	errtimestr = []

	t = time()
	(newerr, newbic, Rnew, newcols) = if haskey(colsrecord, newcolsvec)
		for i in 1:3 push!(errtimestr, "\n") end
		push!(errtimestr, "Got errors from dictionary\n")
		colsrecord[newcolsvec] #return record for newcolsvec
	else
		#otherwise form it from the previous value and the switching index
		newcols = if !colsvec[switchind]
			[currentcols; switchind]
		else
			setdiff(currentcols, switchind)
		end

		#determine where to start filling in Xtmp based on how updated it is
		fillstartcol = if addcol
			xfillcol+1
		else
			k = findfirst(a -> a == switchind, currentcols)
			min(xfillcol+1, k)
		end

		#####################Update Xtmp with newcols########################
		t2 = time()
		for (j, c) in enumerate(newcols)
			if j >= fillstartcol
				view(Xtmp, :, j+1) .= traincols[c]
				if testerr
					view(Xtmp2, :, j+1) .= testcols[c]
				end
			end
		end
		tfill = time() - t2
		push!(errtimestr, "Xtmp fill time = $tfill\n")
	
		#####################Generate new R update###########################
		t2 = time()
		Rnew = if addcol			
			qraddcol(view(Xtmp, :, 1:length(currentcols)+1), Rlast, traincols[switchind])
		else
			k = findfirst(a -> a == switchind, currentcols)
			qrdelcol(Rlast, k+1) #index of column being removed
		end
		tR = time() - t2
		push!(errtimestr, "R update time = $tR\n")

		#####################Calculate New Errors#############################
		t2 = time()
		newerr, newbeta = calc_linreg_error(Rnew, view(Xtmp, :, 1:length(newcols) + 1), y)
		newbic = if testerr
			calc_linreg_error(newbeta, view(Xtmp2, :, 1:length(newcols) + 1), ytest)
		else
			calc_linreg_bic(newerr, view(Xtmp, :, 1:length(newcols)+1), y)
		end

		terr = time()-t2
		push!(errtimestr, "Err update time = $terr\n")
		(newerr, newbic, Rnew, newcols)
	end
	terr = time() - t
	push!(errtimestr, "Total New Err time = $terr\n")
	
	##########################Determine Step Acceptance######################
	delt = newbic - lastbic
	alpha = (delt <= 0) ? 1.0 : exp(-delt/temp)
	acc = (alpha > rand())
	
	##########################Print Update Selectively########################
	if printiter #only print if true
		#delete lines for printing new step update only after calculations are complete
		lines = ceil(Int64, length(traincols)/20) + 14
		for i in 1:lines-1
			print("\r\u1b[K\u1b[A")
		end
	
		avg_iter_time = rpad(round(t_start, digits = 4), 6, '0')
		ETA = round(t_start*(length(tsteps) - inum), digits =4)
		ETAmins = floor(Int64, ETA/60)
		ETAsecs = round(ETA - ETAmins*60, digits = 2)
		ETAstr = string(ETAmins, ":", lpad(ETAsecs, 5, '0'))
		println(string("Completed step ", inum, " of ", length(tsteps), ", Repeat Step Rate - ", rpad(round(dicthitrate, digits = 4), 6, '0'), ", Avg Step Time - ", avg_iter_time, ", ETA - ", ETAstr))

		for s in errtimestr
			print(s)
		end

		println("Done evaluating candidate columns:")
		printcolsvec(firstcolsvec, colsvec, switchind, acc)
		println()
		println(string("Got $str of ", (newerr, newbic)))
		
		alphastr = rpad(round(alpha, digits = 10), 12, '0')
		accstr = acc ? "Accepting" : "Rejecting"
		println("$accstr new candidate columns with an alpha of $alphastr, recent acceptance rate = $(round(accept_rate, digits = 5))")
		println(string("Current state $str: ", (lasterr, lastbic)))
		println()
	end

	#update the accurate fill column for Xtmp
	xfillcol = if haskey(colsrecord, newcolsvec)
		#if used dictionary reference then assume no column is correct
		acc ? 0 : xfillcol 
	else
		#if change is accepted then Xtmp is accurate through all newcols
		if acc
			length(newcols)
		#if the change is rejected then Xtmp is accurate based on the add/remove col update procedure
		elseif addcol
			length(currentcols)
		else
			k - 1
		end
	end

	##############################Return Values###############################
	(newerr, newbic, newcolsvec, changestring, acc, Rnew, newcols, xfillcol)
end

function run_gibbs_step(regdata, Rnow, tmpX, currentcols, errsrecord, colsrecord::RecordType, tsteps; printstep = true, updateinterval = 2.0, accept_rate = 0.0, dicthitrate = 0.0, itertime = 0.0, calibrate = true)
	
	##############################Initialize Values###############################################
	f = 0.99 #controls how long running average is for iter time and accept and repeat rate
	t_stepstart = time()
	deltCT = 0.0
	deltST = 0.0
	deltCasum = 0.0
	temprecord = Vector{Tuple{Float64, Float64}}()
	T0 = calibrate ? Inf : tsteps[1]

	#############################Get dimensions of reg data#####################
	l = length(regdata[1][1][1])
	n = length(regdata[1][1])
	m = length(tsteps)

	#N is the number of potential columns
	firstcolsvec = subset_to_bin(currentcols, n)

	membuffer = if Sys.isapple()
		n*n*min(m, 1000)*8
	else
		1e9 + (n*l*4 + n*n*1000)*8 #number of bytes to make sure are available for dictionary addition
	end

	memcheck = (Sys.free_memory() > membuffer) #make sure enough free memory to add to dictionary

	###############################Get initial values from first iteration#######
	colsvec = errsrecord[end][2]
	(err1, err2) = errsrecord[end][3:4]
	(newerr1, newerr2, newcolsvec, changestring, acc, Rnew, newcols, xfillcol) = gibbs_step(regdata, currentcols, Rnow, tmpX, err1, err2, colsrecord, colsvec, T0, 1, tsteps, itertime, accept_rate, dicthitrate, firstcolsvec, false)

	
	###############################Purge record if insufficient memory###########
	keycheck = haskey(colsrecord, newcolsvec)
	dicthitrate = Float64(keycheck)
	if keycheck
		delete!(colsrecord, newcolsvec)
	elseif !memcheck #if not enough memory purge 10% of colsrecord weighed more towards earlier entries
		while !memCheck
			purge_record!(colsrecord)
			memcheck = (Sys.free_memory() > membuffer) #make sure enough free memory to add to dictionary
		end
	end

	push!(colsrecord, newcolsvec => (newerr1, newerr2, Rnew, newcols))
	accept_rate = f*accept_rate + (1-f)*acc
	deltCk = newerr2 - err2
	deltCasum += abs(deltCk)
	
	if acc
		push!(errsrecord, (changestring, newcolsvec, newerr1, newerr2))
		Rnow = Rnew
		currentcols = newcols
		colsvec = newcolsvec
		(err1, err2) = (newerr1, newerr2)
		deltCT += deltCk
	end

	t_iter = time() - t_stepstart
	itertime = if itertime == 0
		t_iter
	else
		f*itertime + (1-f)*t_iter
	end
	t_lastprint = time()

	currenttemp = T0 #for initial calibration accept all changes 
	calsteps = calibrate ? 5*length(colsvec) : 0 #number of calibration steps to determine initial temperature

	for i in 2:length(tsteps)
		if i%1000 == 0 #update memCheck every 1000 steps
			memcheck = (Sys.free_memory() > membuffer) #make sure enough free memory to add to dictionary
		end

		#print second iteration and subsequent once every 1 second by default
		printiter = if printstep
			if i == 2
				true
			elseif (time() - t_lastprint) > updateinterval
				true
			else
				false
			end
		else
			false
		end
		
		################################Perform gibbs step iteration#############
		t_stepstart = time()
		(newerr1, newerr2, newcolsvec, changestring, acc, Rnew, newcols, xfillcol) = gibbs_step(regdata, currentcols, Rnow, tmpX, err1, err2, colsrecord, colsvec, currenttemp, i, tsteps, itertime, accept_rate, dicthitrate, firstcolsvec, printiter, xfillcol = xfillcol)

		
		###############################Purge record if insufficient memory#######
		keycheck = haskey(colsrecord, newcolsvec)
		dicthitrate = dicthitrate*f + (1-f)*keycheck
		if keycheck
			delete!(colsrecord, newcolsvec)
		elseif !memcheck #if not enough memory purge 10% of colsrecord weighed more towards earlier entries
			while !memCheck
				purge_record!(colsrecord)
				memcheck = (Sys.free_memory() > membuffer) #make sure enough free memory to add to dictionary
			end
		end

		push!(colsrecord, newcolsvec => (newerr1, newerr2, Rnew, newcols))
		accept_rate = f*accept_rate + (1-f)*acc
		deltCk = newerr2 - err2
		deltCasum += abs(deltCk)

		#after calibration steps are done calculate T0 based on average cost variation of steps
		if i == calsteps
			T0 = -deltCasum/(log(0.8)*calsteps)
			deltCT = 0.0 #reset deltCT
			currenttemp = T0
			# println("T0 = $T0")
		end
		
		if acc
			push!(errsrecord, (changestring, newcolsvec, newerr1, newerr2))
			Rnow = Rnew
			currentcols = newcols
			colsvec = newcolsvec
			(err1, err2) = (newerr1, newerr2)
		end

		if i > calsteps
			if printiter
				println("Current temperature = $currenttemp, T0 = $T0 after $calsteps steps of calibration")
				println()
			end

			#update entropy variation
			if deltCk > 0
				deltST -= deltCk/currenttemp
			end

			if acc 
				push!(temprecord, (currenttemp, newerr2))
			else
				push!(temprecord, (currenttemp, err2))
			end

			if calibrate
				currenttemp = currenttemp * 0.95^(1000/(length(tsteps)-calsteps))
			else
				currenttemp = tsteps[i]
			end
		elseif printiter
			println("Current temperature = $currenttemp, waiting $calsteps steps for T0 calibration")
			println()
		end

		itertime = f*itertime + (1-f)*(time() - t_stepstart)

		t_lastprint = printiter ? time() : t_lastprint
	end

	return (errsrecord, colsrecord, temprecord)
end

function run_stepwise_anneal(data::InOutPair{T}...; initialcolsvec = BitVector(undef, size(data[1][1], 2)), initialcolsrecord = RecordType{T}(), iter = 0, printiter = true, updateinterval = 1.0, errdelt = 0.0, tpoints = Vector{T}()) where T <: AbstractFloat

	testerr = (length(data) > 1)
	initialcols = findall(initialcolsvec)
	println("Initializing gibbs annealing stepwise selection from $(sum(initialcolsvec)) columns")

	#convert X's in data to vectors of columns
	datainput = map(a -> (collect.(eachcol(a[1])), a[2]), data)

	y = datainput[1][2]
	traincols = datainput[1][1]

	Xtmp = ones(T, length(y), length(traincols)+1)
	tmpX = if testerr
		Xtmp2 = ones(T, length(datainput[2][2]), length(datainput[1][1])+1)
		ytest = datainput[2][2]
		(Xtmp, Xtmp2)
	else
		(Xtmp,)
	end

	#####################Get initial errors and R matrix#########################
	if sum(initialcolsvec) == 0
		println("Calculating bias term error as a baseline")
		err1 = sum(abs2, (y .- mean(y)))/length(y)
		err2 = testerr ? sum(abs2, (data[2][2] .- mean(y)))/length(data[2][2]) : length(y)*log(err1)+log(length(y))
		#get initial R vector for bias terms
		_, R = qr(ones(T, length(y), 1))
	elseif haskey(initialcolsrecord, initialcolsvec)
		println("Using dictionary to get errors for initial columns")
		(err1, err2, R, cols) = initialcolsrecord[initialcolsvec]
	else
		println("Calculating errors for initial columns")
		
		#fill in tmpX with for initialcols
		for i in eachindex(tmpX)
			for (j, c) in enumerate(initialcols) 
				view(tmpX[i], :, j+1) .= traincols[c]
			end
		end
		_, R = qr(view(tmpX[1], :, 1:length(initialcols)+1))
		err1, beta = calc_linreg_error(R, view(tmpX[1], :, 1:length(initialcols)+1), y)
		err2 = if testerr
			calc_linreg_error(beta, view(tmpX[2], :, 1:length(initialcols)+1), ytest)
		else
			calc_linreg_bic(err1, view(tmpX[1], :, 1:length(initialcols)+1), y)
		end
	end
		
	str = testerr ? "train and test error" : "error and BIC"
	println(string("Got $str of : ", (err1, err2), " using ", sum(initialcolsvec), "/", length(initialcolsvec), " columns"))
	println("--------------------------------------------------------------------")
	println("Setting up $(length(traincols)) columns for step updates")
	println()

	# Tmax = abs(BIC)/2000 #corresponds to a ~.03% increase in error having a ~50% chance of acceptance
	tmax = max(err2/3000, -errdelt/log(0.5)) #corresponds to a typical delta seen at the end of the last iteration having a ~50% chance of acceptance
	numsteps = 6*length(initialcolsvec) #max(100, round(Int64, length(initialColsVec)*sqrt(length(initialColsVec))/2))
	# Tsteps = initialT*exp.(-7*(0:numSteps-1)/numSteps)
	tsteps = if iter == 1
		zeros(T, 5*numsteps)
	else
		#with this schedule Tmax is reduced to 0.0067*Tmax
		# [Tmax*exp.(-5*(0:3*numSteps-1)/(3*(numSteps-1))); zeros(T, 2*numSteps)]
		[tmax*LinRange(1, 0, 3*numsteps); zeros(T, 5*numsteps)]
	end

	tsteps = if isempty(tpoints)
		zeros(T, 5*numsteps)
	else
		reduce(vcat, [LinRange(tpoints[i], tpoints[i+1], numsteps) for i in 1:length(tpoints)-1])
	end

	#add line padding for print update if print is turned on
	if printiter
		lines = ceil(Int64, length(traincols)/20) + 13
		for i in 1:lines-1 
			println()
		end
		println("Waiting for first step to complete")
	else
		println("Starting $(length(tsteps)) steps of annealing process without printing updates")
	end

	(err_record, colsrecord, temprecord) = run_gibbs_step(datainput, R, tmpX, initialcols, [("", initialcolsvec, err1, err2)], push!(initialcolsrecord, initialcolsvec => (err1, err2, R, initialcols)), tsteps, printstep=printiter, updateinterval = updateinterval, calibrate = (iter == 1))
	errs_out = [begin
	    (a[1], findall(a[2]), a[3], a[4])
	end
	for a in err_record]
	(errs_out, colsrecord, temprecord)
end


function extract_record(errs_record)
	errs = [a[4] for a in errs_record]
	sortind = sortperm(errs)
	errdelt = quantile(abs.(errs[2:end] .- errs[1:end-1]), 0.25)
	bestcols = errs_record[sortind[1]][2]
	besterr1 = errs_record[sortind[1]][3]
	besterr2 = errs_record[sortind[1]][4]
	(besterr1, besterr2, errdelt, bestcols) 
end

function run_stepwise_anneal_process(data::InOutPair{T}...; seed = 1, colnames = map(a -> string("Col ", a), 1:size(data[1][1], 2)), initialcolsrecord = RecordType{T}(), printiter=true, updateinterval = 2.0) where T <: AbstractFloat
	Random.seed!(seed)
	println(string("Starting with seed ", seed))
	
	n = size(data[1][1], 2)
	initialcolsvec = BitVector(undef, size(data[1][1], 2))
	if seed == 2
		fill!(initialcolsvec, true)
	else
		initialcolsvec .= rand(Bool, length(initialcolsvec))
	end

	testerr = length(data) > 1
	str = testerr ? "training and test errors" : "error and BIC"

	tstart = time()
	(errs_record1, colsrecord1, temprecord1) = run_stepwise_anneal(data..., iter = 1, initialcolsrecord = initialcolsrecord, initialcolsvec = initialcolsvec, printiter = printiter, updateinterval = updateinterval)
	newtemps = getnewtemps(temprecord1)
	iterseconds = time() - tstart
	itertimestr = maketimestr(iterseconds)

	(besterr1, besterr2, errdelt, bestcols1) = extract_record(errs_record1)
	
	println("Iteration 1 complete after $itertimestr with $(length(bestcols1)) columns\nwith $str of $((besterr1, besterr2))")
	println("--------------------------------------------------------------------")
	println()


	println("Starting second iteration from this starting point")
	colsvec2 = subset_to_bin(bestcols1, n)
	tstart = time()
	(errs_record2, colsrecord2, temprecord2) = run_stepwise_anneal(data..., iter = 2, initialcolsrecord=colsrecord1, initialcolsvec=colsvec2, printiter=printiter, updateinterval=updateinterval, errdelt=errdelt, tpoints = newtemps)
	newtemps = getnewtemps(temprecord2, 0.5) #only use second half of record to determine new temp range 
	iterseconds = time() - tstart
	itertimestr = maketimestr(iterseconds)

	(besterr1, besterr2, errdelt, bestcols2) = extract_record(errs_record2)
	println("Iteration 2 complete after $itertimestr with $(length(bestcols2)) columns\nwith $str of $((besterr1, besterr2))")
	println("--------------------------------------------------------------------")
	println()

	if bestcols2 == bestcols1
		println("Skipping further iterations because best columns are still from iteration 1")
		println()
		println()
		usedcols = bestcols2
		usedcolscheck = [in(i, usedcols) ? "x" : "" for i in 1:length(colnames)]
		(usedcols, usedcolscheck, besterr1, besterr2, colsrecord2, temprecord2)
	else
		iter = 3
		newbestcols = bestcols2
		oldbestcols = bestcols1
		newcolsrecord = colsrecord2
		newtemprecord = temprecord2
		besterr1 = Inf
		besterr2 = Inf
		while newbestcols != oldbestcols
			oldbestcols = newbestcols 
			println("Starting iteration $iter from previous best columns")
			newcolsvec = subset_to_bin(oldbestcols, n)
			tstart = time()
			(newerrs_record, newcolsrecord, newtemprecord) = run_stepwise_anneal(data..., iter = iter, initialcolsrecord=newcolsrecord, initialcolsvec=newcolsvec, printiter=printiter, updateinterval=updateinterval, errdelt = errdelt, tpoints=newtemps) #only use second half to record to determine new temp range
			newtemps = getnewtemps(newtemprecord, 0.5)
			iterseconds = time() - tstart	
			itertimestr = maketimestr(iterseconds)
			(besterr1, besterr2, errdelt, newbestcols) = extract_record(newerrs_record)
			println("Iteration $iter complete after $itertimestr with best columns: $newbestcols\nwith $str of $((besterr1, besterr2))")
			println("--------------------------------------------------------------------")
			println()
			iter += 1
		end
		println("Skipping further iterations because best columns are still from iteration $(iter - 2)")
		println()
		println()
		usedcols = newbestcols 
		usedcolscheck = [in(i, usedcols) ? "x" : "" for i in 1:length(colnames)]
		(usedcols, usedcolscheck, besterr1, besterr2, newcolsrecord, newtemprecord)
	end
end

# function run_quasistatic_anneal_process(regdata::InOurPair{T}...; seed = 1, colnames = map(a -> string("Col ", a), 1:size(regdata[1], 2)), initialcolsrecord::RecordType{T} = RecordType{T}(), printiter=true, updateinterval = 2.0, chi = 0.9, delt = 0.001, M = round(Int64, size(regdata[1], 2)*log(size(regdata[1], 2))))

# 	X = regdata[1][1]
# 	y = regdata[1][2]
# 	n = size(X, 2)
# 	m = size(X, 1)

# 	@assert m == length(y) "X and y must have the same number of rows"

# 	testerr = (length(regdata) == 2)

# 	if testerr 
# 		Xtest = regdata[2][1]
# 		ytest = regdata[2][1]
# 		@assert size(Xtest, 1) == length(ytest) "Xtest and ytest must have the same number of rows"
# 	end

# 	badCols = findall([sum(X[:, c]) for c in 1:n] == 0)
# 	@assert isempty(badCols) "Cannot continue with columns $badCols being zero valued"

# 	namePrefix = "QuasistaticAnnealReheatLinReg"
	
# 	(errsrecord1, colsrecord, tsteps, accs, testerrs) = runTempCalibrate(regdata..., seed = seed, initialcolsrecord = initialcolsrecord, printiter = printiter, updateinterval = updateinterval, chi = chi)
# 	l = length(accs)
# 	accrate = mean(accs[round(Int64, l/2):end])
# 	newtemp = tsteps[end]
# 	println("Achieved an acceptance rate of $accrate compared to a target of $chi with a final temperature of $newtemp")


# 	# (newBestTestErr, bestInd) = findmin([a[3][2] for a in errsRecord1])
# 	newBestInd = length(errsRecord1)
# 	newBestRecord = errsRecord1[end]
# 	newBestTestErr = newBestRecord[3][2]
# 	newBestCols = newBestRecord[2]
# 	startingColsVec = newBestCols

# 	C0 = mean(testErrs[round(Int64, l/2):end])
# 	errsRecord = deepcopy(errsRecord1)
# 	bestTestErr = Inf
# 	fullTempRecord = []
# 	origSeed = seed
# 	while newBestTestErr < bestTestErr
# 		bestTestErr = newBestTestErr
# 		startTime = time()
# 		(newErrsRecord, colsRecord, tempRecord) = runStationarySteps(regData..., startingColsVec, colsRecord, newTemp, C0, seed = seed, delt = delt, M = M)

# 		seed = rand(UInt32)

# 		startingColsVec = newErrsRecord[end][2]
# 		temps = [a[1] for a in tempRecord]
# 		ARs = [a[3] for a in tempRecord]
# 		avgErrs = [a[2] for a in tempRecord]
# 		ind = findfirst(a -> a < 0.25, ARs)

# 		if ind > 1
# 			newTemp = (temps[ind] + temps[ind-1])/2
# 			C0 = (avgErrs[ind] + avgErrs[ind-1])/2
# 		else
# 			newTemp = temps[ind]
# 			C0 = avgErrs[ind]
# 		end

# 		iterSeconds = time() - startTime
# 		iterMins = floor(Int64, iterSeconds/60)
# 		iterSecs = iterSeconds - 60*iterMins
# 		iterTimeStr = string(iterMins, ":", round(iterSecs, digits = 2))

# 		errsRecord = [errsRecord; newErrsRecord[2:end]]
# 		fullTempRecord = [fullTempRecord; tempRecord]
# 		testErrs = [a[3][2] for a in errsRecord]
# 		(newBestTestErr, bestInd) = findmin(testErrs)
# 		newBestRecord = errsRecord[bestInd]
# 		newBestCols = newBestRecord[2]

# 		println(string("Quasitatic annealing complete after ", iterTimeStr, " with best columns: ", findall(newBestCols), "\nwith training and test errors of ", newBestRecord[3]))
# 		if newBestTestErr < bestTestErr
# 			println(string("Resetting temperature to ", newTemp, " to try to find a better configuration"))
# 		else
# 			println("No improvement found after reheating so terminating process")
# 		end
# 	end

# 	recordFileOut = [("Col Changes", "Columns", "Train Err", "Test Err"); reduce(vcat, [(newBestRecord[1], findall(newBestCols), newBestRecord[3][1], newBestRecord[3][2]); [(a[1], findall(a[2]), a[3][1], a[3][2]) for a in errsRecord]])]

# 	tempFileOut = [["Temperature" "Cost Avg" "Accept Rate"]; reduce(vcat, [[a[1] a[2] a[3]] for a in fullTempRecord])]


# 	usedColsCheck = [a ? "x" : "" for a in newBestCols]
# 	writedlm("$(namePrefix)_$(origSeed)Seed$(delt)Delt$(M)Steps$(chi)CHI_BestUsedCols_$name.txt", [usedColsCheck colNames])
	
# 	writedlm("$(namePrefix)_$(origSeed)Seed$(delt)Delt$(M)Steps$(chi)CHI_Record_$name.txt", recordFileOut, '\t')
	
# 	writedlm("$(namePrefix)_$(origSeed)Seed$(delt)Delt$(M)Steps$(chi)CHI_TemperatureSteps_$name.csv", tempFileOut, ',')

# 	(errsRecord, colsRecord, fullTempRecord, (newBestTestErr, newBestInd))
# end

