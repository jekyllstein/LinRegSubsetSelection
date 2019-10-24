recordType{T} = OrderedDict{Array{Bool, 1}, Tuple{Tuple{T, T}, Matrix{T}, Vector{Int64}}}

function gibbsStep(traincols, y, currentcols, Rlast, Xmp, lasterr, lastbic, colsrecord, colsvec, temp, inum, tsteps, start_time, accept_rate, dicthitrate, firstcolsvec, printiter = true)

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

		#####################Update Xtmp with newcols########################
		t2 = time()
		for (j, c) in eachindex(newcols)
			view(Xtmp, :, j+1) .= traincols[c]
		end
		tfill = time() - t2
		push!(errtimestr, "Xtmp fill time = $tfill\n")
	
		#####################Generate new R update###########################
		t2 = time()
		Rnew = if addcol			
			qraddcol(view(Xtmp, :, length(currentcols+1)), Rlast, traincols[switchInd])
		else
			k = findfirst(a -> a == switchInd, currentCols)
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
		lines = ceil(Int64, size(X, 2)/20) + 13
		for i in 1:lines-1
			print("\r\u1b[K\u1b[A")
		end
	
		avgIterTime = rpad(round(startTime, digits = 4), 6, '0')
		ETA = round(startTime*(length(Tsteps) - iNum), digits =4)
		ETAmins = floor(Int64, ETA/60)
		ETAsecs = round(ETA - ETAmins*60, digits = 2)
		ETAstr = string(ETAmins, ":", lpad(ETAsecs, 5, '0'))
		println(string("Completed step ", iNum, " of ", length(Tsteps), ", Repeat Step Rate - ", rpad(round(dictHitRate, digits = 4), 6, '0'), ", Avg Step Time - ", avgIterTime, ", ETA - ", ETAstr))

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

	##############################Return Values###############################
	(newerr, newbic, newcolsvec, changestring, acc, Rnew, newcols)
end