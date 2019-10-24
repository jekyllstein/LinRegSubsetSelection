function gibbs_step(data::NTuple{N, InOutPairCols{T}}, currentcols::AbstractVector{W}, Rlast::Matrix{T}, Xtmp::Matrix{T}, lasterr::T, lastbic::T, colsrecord::RecordType{T}, colsvec::BinVec, temp, inum, tsteps, t_start, accept_rate, dicthitrate, firstcolsvec, printiter = true; xfillcol = 0) where N where T <: AbstractFloat where W <: Integer

	traincols = data[1][1]
	y = data[1][2]

	testerr = length(data) > 1

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
		newbic = calc_linreg_bic(newerr, view(Xtmp, :, 1:length(newcols)+1), y)
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
		lines = ceil(Int64, length(traincols)/20) + 13
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
		println(string("Got error and BIC of ", (newerr, newbic)))
		
		alphastr = rpad(round(alpha, digits = 10), 12, '0')
		accstr = acc ? "Accepting" : "Rejecting"
		println("$accstr new candidate columns with an alpha of $alphastr, recent acceptance rate = $(round(accept_rate, digits = 5))")
		println(string("Current state error and BIC: ", (lasterr, lastbic)))
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

function run_gibbs_step(regdata, Rnow, tmpX, currentcols, errsrecord, colsrecord::RecordType, tsteps; printstep = true, updateinterval = 2.0, accept_rate = 0.0, dicthitrate = 0.0, itertime = 0.0)
	
	f = 0.99 #controls how long running average is for iter time and accept and repeat rate
	
	t_stepstart = time()

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
	costsequence = Vector{Float64}()

	###############################Get initial values from first iteration#######
	colsvec = errsrecord[end][2]
	(err1, err2) = errsrecord[end][3:4]
	push!(costsequence, err2)
	(newerr1, newerr2, newcolsvec, changestring, acc, Rnew, newcols, xfillcol) = gibbs_step(regdata, currentcols, Rnow, tmpX..., err1, err2, colsrecord, colsvec, tsteps[1], 1, tsteps, itertime, accept_rate, dicthitrate, firstcolsvec, false)

	
	###############################Purge record if insufficient memory###########
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
	
	if acc
		push!(errsrecord, (changestring, newcolsvec, newerr1, newerr2))
		Rnow = Rnew
		currentcols = newcols
		colsvec = newcolsvec
		(err1, err2) = (newerr1, newerr2)
	end
	push!(costsequence, err2)

	t_iter = time() - t_stepstart
	itertime = if itertime == 0
		t_iter
	else
		f*itertime + (1-f)*t_iter
	end
	t_lastprint = time()

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
		(newerr1, newerr2, newcolsvec, changestring, acc, Rnew, newcols, xfillcol) = gibbs_step(regdata, currentcols, Rnow, tmpX..., err1, err2, colsrecord, colsvec, tsteps[i], i, tsteps, itertime, accept_rate, dicthitrate, firstcolsvec, printiter, xfillcol = xfillcol)

		
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
		
		if acc
			push!(errsrecord, (changestring, newcolsvec, newerr1, newerr2))
			Rnow = Rnew
			currentcols = newcols
			colsvec = newcolsvec
			(err1, err2) = (newerr1, newerr2)
		end
		push!(costsequence, err2)

		itertime = f*itertime + (1-f)*(time() - t_stepstart)

		t_lastprint = printiter ? time() : t_lastprint
	end

	return (errsrecord, colsrecord, costsequence, accept_rate, dicthitrate, currentcols, Rnow, itertime)
end

function run_stepwise_anneal(data::InOutPair{T}...; initialcolsvec = BitVector(undef, size(data[1][1], 2)), initialcolsrecord = RecordType{T}(), iter = 0, printiter = true, updateinterval = 1.0, errdelt = 0.0) where T <: AbstractFloat

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
		
	str = testerr ? "train and test error" : "error and bic"
	println(string("Got $str of : ", (err1, err2), " using ", sum(initialcolsvec), "/", length(initialcolsvec), " columns"))
	println("--------------------------------------------------------------------")
	println("Setting up $(length(traincols)) columns for step updates")
	println()

	# Tmax = abs(BIC)/2000 #corresponds to a ~.03% increase in error having a ~50% chance of acceptance
	tmax = -errdelt/log(0.5) #corresponds to a typical delta seen at the end of the last iteration having a ~50% chance of acceptance
	numsteps = 6*length(initialcolsvec) #max(100, round(Int64, length(initialColsVec)*sqrt(length(initialColsVec))/2))
	# Tsteps = initialT*exp.(-7*(0:numSteps-1)/numSteps)
	tsteps = if iter == 1
		zeros(T, 5*numsteps)
	else
		#with this schedule Tmax is reduced to 0.0067*Tmax
		# [Tmax*exp.(-5*(0:3*numSteps-1)/(3*(numSteps-1))); zeros(T, 2*numSteps)]
		[tmax*LinRange(1, 0, 3*numsteps); zeros(T, 5*numsteps)]
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

	(err_record, colsrecord) = run_gibbs_step(datainput, R, tmpX, initialcols, [("", initialcolsvec, err1, err2)], push!(initialcolsrecord, initialcolsvec => (err1, err2, R, initialcols)), tsteps, printstep=printiter, updateinterval = updateinterval)
	errs_out = [begin
	    (a[1], findall(a[2]), a[3], a[4])
	end
	for a in err_record]
	(errs_out, colsrecord)
end

function run_stepwise_anneal_process(data::InOutPair{T}...; seed = 1, colnames = map(a -> string("Col ", a), 1:size(data[1][1], 2)), initialcolsrecord = RecordType{T}(), printiter=true, updateinterval = 2.0) where T <: AbstractFloat
	Random.seed!(seed)
	println(string("Starting with seed ", seed))
	
	initialcolsvec = BitVector(undef, size(data[1][1], 2))
	if seed == 2
		fill!(initialcolsvec, true)
	else
		initialcolsvec .= rand(Bool, length(initialcolsvec))
	end

	testerr = length(data) > 1
	str = testerr ? "training and test errors" : "error and BIC"

	tstart = time()
	(errs_record1, colsrecord1) = run_stepwise_anneal(data..., iter = 1, initialcolsrecord = initialcolsrecord, initialcolsvec = initialcolsvec, printiter = printiter, updateinterval = updateinterval)
	iterseconds = time() - tstart
	itermins = floor(Int64, iterseconds/60)
	itersecs = iterseconds - 60*itermins
	itertimestr = "$itermins:$(round(itersecs, digits=2))"
	errs = [a[4] for a in errs_record1]
	sortind = sortperm(errs)
	errdelt = quantile(abs.(errs[2:end] .- errs[1:end-1]), 0.25)
	bestcols1 = errs_record1[sortind[1]][2]
	besterr1 = errs_record1[sortind[1]][3]
	besterr2 = errs_record1[sortind[1]][4]
	println("Iteration 1 complete after $itertimestr with $(length(bestcols1)) columns\nwith $str of $((besterr1, besterr2))")
	(bestcols1, besterr1, besterr2, colsrecord1)
end
