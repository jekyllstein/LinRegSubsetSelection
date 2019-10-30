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

function getinitialerrs(regdata::NTuple{N, InOutPairCols{T}}, colsvec::ColVec, colsrecord::RecordType{T}) where T <: AbstractFloat where N
	testerr = (length(regdata) > 1)
	traincols = regdata[1][1]
	n = length(traincols)
	y = regdata[1][2]

	Xtmp = ones(T, length(y), n+1)
	tmpX = if testerr
		testcols = regdata[2][1]
		Xtmp2 = ones(T, length(regdata[2][2]), length(regdata[1][1])+1)
		ytest = regdata[2][2]
		(Xtmp, Xtmp2)
	else
		(Xtmp,)
	end	

	if sum(colsvec) == 0
		initialcols = Vector{Integer}()
		println("Calculating bias term error as a baseline")
		meany = mean(y)
		err1 = sum(abs2, y .- meany)/length(y)
		err2 = if testerr
			sum(abs2, ytest .- meany)/length(ytest)
		else
			length(y)*log(err1) + log(length(y))
		end
		_, R = qr(view(Xtmp, :, 1:1))
	else
		if haskey(colsrecord, colsvec)
			println("Using dictionary to get errors for $(sum(colsvec)) initial columns ")
			(err1, err2, R, initialcols) = colsrecord[colsvec]
		else
			initialcols = findall(colsvec)
			println("Finding errors for $(length(initialcols)) initial columns ")
			for (i, c) in enumerate(initialcols)
				view(Xtmp, :, i+1) .= traincols[c]
				if testerr 
					view(Xtmp2, :, i+1) .= testcols[c]
				end
			end
			_, R = qr(view(Xtmp, :, 1:length(initialcols)+1))
			err1, beta = calc_linreg_error(R, view(tmpX[1], :, 1:length(initialcols)+1), y)
			err2 = if testerr
				calc_linreg_error(beta, view(tmpX[2], :, 1:length(initialcols)+1), ytest)
			else
				calc_linreg_bic(err1, view(tmpX[1], :, 1:length(initialcols)+1), y)
			end
		end
	end
	(testerr, tmpX, initialcols, err1, err2, R)
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
			# println("currentcols = $currentcols")
			# println("switchind = $switchind")
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

function run_gibbs_step(regdata, Rnow, tmpX, currentcols, errsrecord, colsrecord::RecordType, tsteps; printstep = true, updateinterval = 2.0, accept_rate = 0.0, dicthitrate = 0.0, itertime = 0.0, calibrate = true, xfillcol=0)
	
	##############################Initialize Values###############################################
	f = 0.99 #controls how long running average is for iter time and accept and repeat rate
	t_stepstart = time()
	deltCT = 0.0
	deltST = 0.0
	deltCasum = 0.0
	temprecord = Vector{Tuple{Float64, Float64}}()
	costsequence = Vector{Float64}()
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
	colsvec = subset_to_bin(currentcols, n)
	(err1, err2) = colsrecord[colsvec][1:2]
	push!(costsequence, err2)
	(newerr1, newerr2, newcolsvec, changestring, acc, Rnew, newcols, xfillcol) = gibbs_step(regdata, currentcols, Rnow, tmpX, err1, err2, colsrecord, colsvec, T0, 1, tsteps, itertime, accept_rate, dicthitrate, firstcolsvec, false, xfillcol = xfillcol)

	
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
	push!(costsequence, err2)

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

		push!(costsequence, err2)

		itertime = f*itertime + (1-f)*(time() - t_stepstart)

		t_lastprint = printiter ? time() : t_lastprint
	end

	return (errsrecord, colsrecord, temprecord, costsequence, currentcols, xfillcol, Rnow)
end

function calibrate_gibbs_temp(regdata::NTuple{N, InOutPairCols{T}}, tmpX::NTuple{N, Matrix{T}}, Rnow::Matrix{T}, currentcols, errsrecord, colsrecord::RecordType{T}; printstep = true, updateinterval = 2.0, chi = 0.9) where T <: AbstractFloat where N
	
	##########Set up initial values and print first iteration##############################
	t_stepstart = time()
	t_iter = 0.0
	accept_rate = 0.0
	dicthitrate = 0.0
	testerr = (length(regdata) > 1)
	traincols = regdata[1][1]
	y = regdata[1][1]
	Xtmp = tmpX[1]
	if testerr
		testcols = regdata[2][1]
		ytest = regdata[2][2]
		Xtmp2 = tmpX[2]
		tmpX = (Xtmp, Xtmp2)
	else
		tmpX = (Xtmp,)
	end

	n = length(traincols)
	firstcolsvec = subset_to_bin(currentcols, n)
	
	l = length(y)
	m = 2*round(Int64, n*log(n))
	tsteps = ones(m+1)
	accs = zeros(Int64, m)
	err2record = Vector{Float64}(undef, m)
	deltCbar = 0.0
	m1 = 0
	m2 = 0
	f = 0.99 #controls how long running average is for iter time and accept and repeat rate

	membuffer = if Sys.isapple()
		n*n*min(m, 1000)*8
	else
		1e9 + (n*l*4 + n*n*1000)*8 #number of bytes to make sure are available for dictionary addition
	end

	memcheck = (Sys.free_memory() > membuffer) #make sure enough free memory to add to dictionary
	colsvec = errsrecord[end][2]
	err1 = errsrecord[end][3]
	err2 = errsrecord[end][4]
	T0 = tsteps[1]

	#########################################Perform first step iteration######################################
	(newerr1, newerr2, newcolsvec, changestring, acc, Rnew, newcols, xfillcol) = gibbs_step(regdata, currentcols, Rnow, tmpX, err1, err2, colsrecord, colsvec, T0, 1, tsteps, t_iter, accept_rate, dicthitrate, firstcolsvec, false)

	###############################Purge record if insufficient memory#######
	keycheck = haskey(colsrecord, newcolsvec)
	push!(colsrecord, newcolsvec => (newerr1, newerr2, Rnew, newcols))
	accept_rate = Float64(acc)
	dicthitrate = Float64(keycheck)
	deltC = newerr2 - err2
	if deltC > 0
		deltCbar += deltC
		m2 += 1
	else
		m1 += 1
	end

	tsteps[2] = deltCbar / log(m2 / (m2*chi - (1-chi)*m1)) / m2
	
	if acc
		accs[1] = 1
		push!(errsrecord, (changestring, newcolsvec, newerr1, newerr2))
		Rnow = Rnew
		currentcols = newcols
		colsvec = newcolsvec
		(err1, err2) = (newerr1, newerr2)
	end

	err2record[1] = err2

	t_iter = (time() - t_stepstart)
	t_lastprint = time()

	for i in 2:length(tsteps)-1
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

		t_stepstart = time()
		(newerr1, newerr2, newcolsvec, changestring, acc, Rnew, newcols, xfillcol) = gibbs_step(regdata, currentcols, Rnow, tmpX, err1, err2, colsrecord, colsvec, tsteps[i], i, tsteps, t_iter, accept_rate, dicthitrate, firstcolsvec, printiter, xfillcol = xfillcol)

		keycheck = haskey(colsrecord, newcolsvec)
		if keycheck
			delete!(colsrecord, newcolsvec)
		elseif !memcheck #if not enough memory purge 10% of colsRecord weighted towards earlier elements
			while !memcheck
				purgerecord!(colsrecord)
				memcheck = (Sys.free_memory() > membuffer) #make sure enough free memory to add to dictionary
			end
		end

		dicthitrate = dicthitrate*f + (1-f)*keycheck
		
		push!(colsrecord, newcolsvec => (newerr1, newerr2, Rnew, newcols))
		accept_rate = f*accept_rate + (1-f)*acc

		deltC = newerr2 - err2
		if deltC > 0
			deltCbar += deltC
			m2 += 1
		else
			m1 += 1
		end

		c = m2 / (m2*chi - (1-chi)*m1)
		tsteps[i+1] = if c <= 0
			0.0
		elseif c <= 1
			Inf 
		else
			deltCbar / log(c) / m2
		end

		if printiter
			println("Current temperature = $(tsteps[i+1]) after $i steps of calibration")
			println()
		end

		#in second half try resetting values to get more accurate calibration
		if i == round(Int64, length(tsteps)/2) #((i < round(Int64, length(Tsteps)*0.75)) && ((i % N) == 0))
			m1 = 0
			m2 = 0
			deltCbar = 0.0
		end
		
		if acc
			accs[i] = 1
			push!(errsrecord, (changestring, newcolsvec, newerr1, newerr2))
			Rnow = Rnew
			currentcols = newcols
			colsvec = newcolsvec
			(err1, err2) = (newerr1, newerr2)
		end
		err2record[i] = err2

		t_iter = f*t_iter + (1-f)*(time() - t_stepstart)

		t_lastprint = printiter ? time() : t_lastprint
	end
	(errsrecord, colsrecord, tsteps, accs, err2record)
end


function run_stepwise_anneal(data::InOutPair{T}...; initialcolsvec = BitVector(undef, size(data[1][1], 2)), initialcolsrecord = RecordType{T}(), iter = 0, printiter = true, updateinterval = 1.0, errdelt = 0.0, tpoints = Vector{T}()) where T <: AbstractFloat

	testerr = (length(data) > 1)
	initialcols = findall(initialcolsvec)
	println("Initializing gibbs annealing stepwise selection from $(sum(initialcolsvec)) columns")

	#convert X's in data to vectors of columns
	datainput = map(a -> (collect.(eachcol(a[1])), a[2]), data)

	y = datainput[1][2]
	traincols = datainput[1][1]

	(testerr, tmpX, initialcols, err1, err2, R) = getinitialerrs(datainput, initialcolsvec, initialcolsrecord)
		
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

function run_temp_calibrate(regdata::InOutPair{T}...; seed = 1, colsrecord::RecordType{T} = RecordType{T}(), printiter = true, updateinterval = 2.0, chi = 0.9) where T <: AbstractFloat where N

	Random.seed!(seed)
	
	datainput = map(a -> (collect.(eachcol(a[1])), a[2]), regdata)

	y = datainput[1][2]
	traincols = datainput[1][1]
	n = length(traincols)

	initialcolsvec = rand(Bool, length(traincols))
	(testerr, tmpX, initialcols, err1, err2, R) = getinitialerrs(datainput, initialcolsvec, colsrecord)
	str = testerr ? "train and test error" : "error and BIC"

	println("Beginning temperature calibration with a target select rate of $chi")
		
	println("Got $str of: $((err1, err2)) using $(sum(initialcolsvec))/$n columns")
	println("--------------------------------------------------------------------")
	println("Setting up $(length(traincols)) columns for step updates")

	#add line padding for print update if print is turned on
	if printiter
		lines = ceil(Int64, n/20) + 12
		for i in 1:lines-1 
			println()
		end
		println("Waiting for first step to complete")
	else
		numsteps = 2*round(Int64, n*log(n))+1
		println("Starting $numsteps steps of sampling process without printing updates")
	end

	push!(colsrecord, initialcolsvec => (err1, err2, R, initialcols))
	calibrate_gibbs_temp(datainput, tmpX, R, initialcols, [("", initialcolsvec, err1, err2)], colsrecord; printstep = printiter, updateinterval = updateinterval, chi = chi)
end

function extract_record(errs_record)
	errs = [a[4] for a in errs_record]
	sortind = sortperm(errs)
	errdelt = try 
		quantile(abs.(errs[2:end] .- errs[1:end-1]), 0.25) 
	catch 
		0.0 
	end
	bestcols = errs_record[sortind[1]][2]
	besterr1 = errs_record[sortind[1]][3]
	besterr2 = errs_record[sortind[1]][4]
	(besterr1, besterr2, errdelt, bestcols) 
end

function calc_accept_rate(C::Vector{T}) where T <: AbstractFloat
	l = length(C)
	numacc = 0
	for i in 1:(l-1)
		if C[i] != C[i+1]
			numacc += 1
		end
	end
	numacc/(l-1)
end

function run_stationary_steps(regdata::NTuple{N, InOutPairCols{T}}, initialcolsvec::ColVec, initialcolsrecord::RecordType{T}, initialtemp::Float64, C0::Float64; seed = 1, printiter = true, updateinterval = 2.0, delt = 0.001, n = length(regdata[1][1]), m = round(Int64, n*log(n))) where T <: AbstractFloat where N
	
	Random.seed!(seed)
	
	println()
	println("-------------------------------------------------------------------------------------------------------------------")
	println("Beginning quasistatic steps at temperature of $initialtemp with a delta of $delt and equilibrium plateau of $m steps")
	println("-------------------------------------------------------------------------------------------------------------------")
	println()

	(testerr, tmpX, initialcols, err1, err2, R) = getinitialerrs(regdata, initialcolsvec, initialcolsrecord)
	
	traincols = regdata[1][1]
	y = regdata[1][2]

	str = testerr ? "train and test error" : "error and BIC"
		
	println("Got $str of: $((err1, err2)) using $(sum(initialcolsvec))/$n columns")
	println("--------------------------------------------------------------------")
	println("Setting up $(length(traincols)) columns for step updates")

	tsteps = fill(initialtemp, m)
	
	#add line padding for print update if print is turned on
	lines = ceil(Int64, n/20) + 14
	if printiter
		println("On initial step using starting temperature of $initialtemp")
		for j in 1:lines-1
			println()
		end
	else
		println("Starting $m steps of sampling process without printing updates")
	end

	(err_record, colsrecord, temprecord, costsequence, currentcols, xfillcol, R) = run_gibbs_step(regdata, R, tmpX, initialcols, [("", initialcolsvec, err1, err2)], push!(initialcolsrecord, initialcolsvec => (err1, err2, R, initialcols)), tsteps, printstep=printiter, updateinterval = updateinterval, calibrate = false)
	Cs = mean(costsequence)
	ar = calc_accept_rate(costsequence)

	fulltemprecord = [(initialtemp, Cs, ar)]

	sig = std(costsequence)
	newtemp = initialtemp/(1+(log(1+delt)*initialtemp/(3*sig)))
	dT = newtemp - initialtemp
	dC = Cs-C0
	thresh = abs(dC/dT * newtemp/C0)
	i = 1

	t_report = time()
	while thresh > eps(Cs)
		if time() - t_report > updateinterval
			printcheck = true
			t_report = time()
		else
			printcheck = false
		end
		if printiter && printcheck
			print("\u001b[$(lines)F") #move cursor to beginning of lines lines+1 lines up
			print("\u001b[2K") #clear entire line
			println("Reducing temperature from #$(i-1):$(round(initialtemp, sigdigits = 3)) to #$i:$(round(newtemp, sigdigits = 3)) with thresh:$(round(thresh, digits = 3))")
			print("\u001b[$(lines)E") #move cursor to beginning of lines lines+1 lines down
		end
		tsteps = fill(newtemp, m)
		initialtemp = newtemp
		(err_record, colsrecord, temprecord, costsequence, currentcols, xfillcol, R) = run_gibbs_step(regdata, R, tmpX, currentcols, err_record, colsrecord, tsteps, printstep=printiter&&printcheck, updateinterval = updateinterval, calibrate = false, xfillcol = xfillcol)
		ar = calc_accept_rate(costsequence)
		push!(fulltemprecord, (newtemp, mean(costsequence), ar))
		sig = std(costsequence)
		newtemp = initialtemp/(1+(log(1+delt)*initialtemp/(3*sig)))
		dC = mean(costsequence) - Cs
		Cs = mean(costsequence)
		dT = newtemp - initialtemp
		thresh = abs(dC/dT * newtemp/C0)
		i += 1
	end

	colscheck = true
	while colscheck
		if time() - t_report > updateinterval
			printcheck = true
			t_report = time()
		else
			printcheck = false
		end
		if printiter && printcheck
			print("\u001b[$(lines+1)F") #move cursor to beginning of lines lines+1 lines up
			print("\u001b[2K") #clear entire line
			println("Confirming local minimum at a temperature of 0.0")
			print("\u001b[$(lines+1)E") #move cursor to beginning of lines lines+1 lines down
		end
		tsteps = fill(0.0, round(Int64, n*log(n)))
		(err_record, colsrecord, temprecord, costsequence, currentcols, xfillcol, R) = run_gibbs_step(regdata, R, tmpX, currentcols, err_record, colsrecord, tsteps, printstep=printiter&&printcheck, updateinterval = updateinterval, calibrate = false, xfillcol = xfillcol)
		ar = calc_accept_rate(costsequence)
		push!(fulltemprecord, (0.0, mean(costsequence), ar))
		colscheck = length(unique(costsequence)) > 1 #verify that sequence is still fluctuating
	end
	(err_record, colsrecord, fulltemprecord)
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
		usedcolscheck = [in(i, usedcols) ? "x\t$n" : " \t$n" for (i, n) in enumerate(colnames)]
		(usedcols, usedcolscheck, besterr1, besterr2, newcolsrecord, newtemprecord)
	end
end

function run_quasistatic_anneal_process(regdata::InOutPair{T}...; seed = 1, colnames = map(a -> string("Col ", a), 1:size(regdata[1][1], 2)), initialcolsrecord::RecordType{T} = RecordType{T}(), printiter=true, updateinterval = 2.0, chi = 0.9, delt = 0.001, M = round(Int64, size(regdata[1][1], 2)*log(size(regdata[1][1], 2)))) where T <: AbstractFloat

	X = regdata[1][1]
	y = regdata[1][2]
	n = size(X, 2)
	m = size(X, 1)

	#convert X's in data to vectors of columns
	datainput = map(a -> (collect.(eachcol(a[1])), a[2]), regdata)

	@assert m == length(y) "X and y must have the same number of rows"

	testerr = (length(regdata) == 2)
	str = testerr ? "training and test errors" : "error and BIC"

	if testerr 
		Xtest = regdata[2][1]
		ytest = regdata[2][2]
		@assert size(Xtest, 1) == length(ytest) "Xtest and ytest must have the same number of rows"
	end

	badCols = findall([sum(X[:, c]) for c in 1:n] == 0)
	@assert isempty(badCols) "Cannot continue with columns $badCols being zero valued"

	namePrefix = "QuasistaticAnnealReheatLinReg"
	
	(errsrecord, colsrecord, tsteps, accs, errs) = run_temp_calibrate(regdata..., seed = seed, colsrecord = initialcolsrecord, printiter = printiter, updateinterval = updateinterval, chi = chi)
	l = length(accs)
	accrate = mean(accs[round(Int64, l/2):end])
	newtemp = tsteps[end]
	println("Achieved an acceptance rate of $accrate compared to a target of $chi with a final temperature of $newtemp")


	# (newBestTestErr, bestInd) = findmin([a[3][2] for a in errsRecord1])
	
	bestrecord = errsrecord[end]
	newbesterr1 = bestrecord[3]
	newbesterr2 = bestrecord[4]
	bestcols = bestrecord[2]
	startingcolsvec = bestcols

	C0 = mean(errs[round(Int64, l/2):end])
	besterr2 = Inf
	fulltemprecord = []
	origseed = seed
	while newbesterr2 < besterr2
		besterr2 = newbesterr2
		t_start = time()
		(newerrs_record, colsrecord, temprecord) = run_stationary_steps(datainput, startingcolsvec, colsrecord, newtemp, C0, seed = seed, delt = delt)

		seed = rand(UInt32)

		startingcolsvec = newerrs_record[end][2]
		temps = [a[1] for a in temprecord]
		ARs = [a[3] for a in temprecord]
		avgerrs = [a[2] for a in temprecord]
		ind = findfirst(a -> a < 0.25, ARs)

		if ind > 1
			newtemp = (temps[ind] + temps[ind-1])/2
			C0 = (avgerrs[ind] + avgerrs[ind-1])/2
		else
			newtemp = temps[ind]
			C0 = avgerrs[ind]
		end

		iterseconds = time() - t_start
		itertimestr = maketimestr(iterseconds)

		for r in newerrs_record[2:end]
			push!(errsrecord, r)
		end

		for r in temprecord
			push!(fulltemprecord, r)
		end

		
		(newbesterr1, newbesterr2, errdelt, bestcols) = extract_record(newerrs_record) 

		println("Quasitatic annealing complete after $itertimestr with $(sum(bestcols)) columns\nwith $str of $((newbesterr1, newbesterr2))")
		if newbesterr2 < besterr2
			println(string("Resetting temperature to ", newtemp, " to try to find a better configuration"))
		else
			println("No improvement found after reheating so terminating process")
		end
	end
	usedcols = findall(bestcols)
	usedcolscheck = [in(i, usedcols) ? "x\t$n" : " \t$n" for (i, n) in enumerate(colnames)]
	(usedcols, usedcolscheck, newbesterr1, newbesterr2, colsrecord, fulltemprecord, errsrecord)
end

