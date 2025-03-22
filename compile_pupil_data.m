%% COMPILE PUPIL DATA - 
% take preprocessed all_data_delay_<sub> files and add pupil data to them

clear; close all;
addpath('helpers/')

subjects = {'satchel', 'wakko', 'pepe'};

data_dir = 'raw_data_forms/';
save_dir = 'preprocessed_data/';

% useful trial codes - Smith lab
fixon_code = 2; targon_code = 70; targoff_code = 100; fixoff_code = 3;
ar_order = 25; % length of trial window to subtract slow component
for sub=subjects
    sub_str = sub{1};

    % get all data 
    load(sprintf('%s/all_data_delay_%s.mat',save_dir,sub_str),'all_data');
    load(sprintf('%s/behav_%s.mat',data_dir,sub_str)); % downsampled raw pupil traces

    n_pupil = length(behav);

    for ii = 1:n_pupil
        fprintf('Getting pupil responses for %s...\n',behav{ii}.sess_name(1:8));
        pupil.scale = behav{ii}.pupil(1).scale;

        curr_pupil = behav{ii}.pupil;

        % compute pupil information
        trial = cell(1,n_pupil);
        avg_outlier = nan(1,n_pupil);
        avg = nan(1,n_pupil);
        baseline = nan(1,n_pupil);
        evoked = nan(1,n_pupil);
        n_samp = nan(1,n_pupil);
        for jj = 1:length(curr_pupil)
            codes = curr_pupil(jj).codesamples;
            start_idx = codes(find(codes(:,1)==targon_code,1),2) - 200;
            end_idx = codes(find(codes(:,1)==fixoff_code,1),2);
            smoothed_trace = unpadded_medfilt1(double(curr_pupil(jj).trial((start_idx+1):end_idx)),50); % remove high-frequency noise from signal

            trial{jj} = smoothed_trace;
            avg_outlier(jj) = trimmean(smoothed_trace,20); % remove outlier values and compute average over the delay
            avg(jj) = mean(smoothed_trace); % compute average over the delay, including outlier points
            baseline(jj) = mean(smoothed_trace(51:150)); % -150 to -50 pre target onset
            evoked(jj) = mean(smoothed_trace(251:350)); % +50 to +150 post target onset
            n_samp(jj) = length(smoothed_trace); % number of pupil samples in the delay period
        end
        pupil.trial = trial;
        pupil.avg_outlier = avg_outlier;
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