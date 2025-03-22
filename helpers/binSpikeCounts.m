function [countmat,trainingdat] = binSpikeCounts(trainingdat, starttimecode, endtimecode, bin)

    Fs = 30000; % Hz, sampling rate of recording system
    countmat = [];
    histchannels = [1:96 257:352]; % Bank A and C for arrays

    for n = 1:length(trainingdat)
        % calculate global spike times (in seconds) from the spike time diffs saved in the dat
        newspiketimes = cumsum([double(trainingdat(n).firstspike); double(trainingdat(n).spiketimesdiff)]) / Fs;
        spikes = [double(trainingdat(n).spikeinfo) newspiketimes];

        % filter out spike sorted units from noise (0 and 255 codes are noise waveforms, should not be included)
        if length(unique(trainingdat(n).spikeinfo(:,2)))>1
            spikes = spikes(~ismember(spikes(:,2),[0 255]),:);
        end

        % find global time of start and end of trial
        trialcodes = trainingdat(n).trialcodes;
        starttime = trialcodes(trialcodes(:,2)==starttimecode,3); % trial start
        spiketimes = (spikes(:,3));
        spikechannels = spikes(:,1);
        endtime = trialcodes(trialcodes(:,2)==endtimecode,3);

        % bin spike times into "bin" second bins
        tempcounts = [];
        if ~isempty(starttime) && ~isempty(endtime)
            times = starttime(1):bin:endtime(1);
            if length(times)>1
            tempcounts = histcn([spikechannels spiketimes],histchannels,times);
            if ~isempty(find(spiketimes==times(end),1))
                if size(tempcounts,2)>1
                    tempcounts = tempcounts(:,1:(end-1));
                else
                    tempcounts = [];
                end
            end
            if size(tempcounts,1)+1 == length(histchannels)
                tempcounts(end+1,:) = zeros(1,size(tempcounts,2)); 
            end
            trainingdat(n).counttimes = times;
            trainingdat(n).nBins = length(times)-1;
            end
        end

        trainingdat(n).counts = tempcounts;
        countmat = [countmat tempcounts];

    end
end