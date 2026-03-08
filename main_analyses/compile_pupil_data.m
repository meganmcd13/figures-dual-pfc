%% COMPILE PUPIL DATA - 
% take preprocessed all_data_delay_<sub> files and add pupil data to them

clear; close all;
addpath('../helpers/')

subjects = {'satchel', 'wakko', 'pepe'};

data_dir = '../raw_data_forms/';
save_dir = '../preprocessed_data/';

% useful trial codes - Smith lab
fixon_code = 2; targon_code = 70; targoff_code = 100; fixoff_code = 3;
PRE_TO = 200; % ms of pupil data to take before target onset - for event-related baselining
ar_order = 25; % length of trial window to subtract slow component

% manual selection of peak of evoked response
peaks = {[150, 127, 161, 181, 165, 149, 154, 129, 161, 170, 177, 175, 200, 166, 178, 150], ... % satchel
         [330, 356, 347, 351, 306, 297, 319, 276, 285, 281], ...                               % wakko
         [297, 280, 297, 224, 221, 269, 294, 290, 293, 287, 275, 264, 252, 288, 276, 273]};    % pepe

for i_sub=1:length(subjects)
    sub_str = subjects{i_sub};

    % get all data 
    load(sprintf('%s/all_data_delay_%s.mat',save_dir,sub_str),'all_data');
    load(sprintf('%s/pupil_%s.mat',data_dir,sub_str)); % downsampled raw pupil traces

    n_pupil = length(behav);

    for ii = 1:n_pupil
        fprintf('Getting pupil responses for %s...\n',behav{ii}.sess_name(1:8));

        curr_pupil = behav{ii}.pupil;

        % compute pupil information
        N = length(curr_pupil);
        trial = cell(1,N);
        avg = nan(1,N);
        baseline = nan(1,N);
        evoked = nan(1,N);
        n_samp = nan(1,N);
        for jj = 1:N
            codes = curr_pupil(jj).codesamples;
            start_idx = codes(find(codes(:,1)==targon_code,1),2) - PRE_TO;
            delay_start_idx = codes(find(codes(:,1)==targoff_code,1),2);
            end_idx = codes(find(codes(:,1)==fixoff_code,1),2);
            smoothed_trace = unpadded_medfilt1(double(curr_pupil(jj).trial((start_idx+1):end_idx)),50); % remove high-frequency noise from signal
            delay_smoothed_trace = unpadded_medfilt1(double(curr_pupil(jj).trial((delay_start_idx+1):end_idx)),50); % remove high-frequency noise from signal

            trial{jj} = smoothed_trace;
            avg(jj) = mean(delay_smoothed_trace); % compute average over the delay, including outlier points
            baseline(jj) = mean(smoothed_trace(PRE_TO-200+1:PRE_TO-100)); % -200 to -100 pre target onset
            n_samp(jj) = length(smoothed_trace); % number of pupil samples in the delay period

            % find evoked response magnitude
            evoked_start = PRE_TO + peaks{i_sub}(ii);
            evoked(jj) = mean(smoothed_trace(evoked_start-49:evoked_start+50)); % -50 to +50 centered on manual peak of evoked response
        end
        pupil.trial = trial;
        pupil.avg = avg;
        pupil.baseline = baseline;
        pupil.evoked = evoked;
        pupil.n_samp = int16(n_samp);

        % estimate slow and fast component of pupil
        slow_pupil_component = movmean(avg, ar_order, "Endpoints", "shrink");
        fast_pupil_component = avg - slow_pupil_component;

        % find the right session in all data
        all_data_fnames = fieldnames(all_data); 
        for i_sess = 1:length(all_data_fnames)
            if strcmpi(behav{ii}.sess_name(1:8), all_data_fnames{i_sess})
                % add avg pupil response to all_data
                all_data.(all_data_fnames{i_sess}).pupil = pupil;
                all_data.(all_data_fnames{i_sess}).slow_component_pupil = slow_pupil_component;
                all_data.(all_data_fnames{i_sess}).fast_component_pupil = fast_pupil_component;
            end
        end
    end

    % save updated all data
    save(sprintf('%s/all_data_delay_%s.mat',save_dir,sub_str),'all_data');
end