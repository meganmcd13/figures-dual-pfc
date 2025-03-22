function numWorkers = startParPool(numParRuns)

	currParPool = gcp('NoCreate');

	numWorkers = min(feature('NumCores'), numParRuns);

	if isempty(currParPool)
        parpool(numWorkers);
	elseif currParPool.NumWorkers < numWorkers
		delete(currParPool)
		parpool(numWorkers);
    end
    
    fprintf('%d parallel workers started\n',numWorkers);
    
end
