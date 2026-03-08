%% MAIN PREPROCESSING SCRIPT - 
% loads cmp file containing array layout and information for each subject
% loads per session dat files
% computes trial metrics and spike count matrices
% saves out all_data_delay_<sub>.mat

clear; close all;
addpath('../helpers');

subjects = {'satchel', 'wakko', 'pepe'};

data_dir = '../raw_data_forms/';
save_dir = '../preprocessed_data/';

ar_order = 25; % length of trial window to subtract slow component
for sub=subjects
    sub_str = sub{1};
    
    % get per session dat files
    sub_prefix = [upper(sub_str(1)), lower(sub_str(2))];
    fnames = dir(sprintf('%s/%s*_sort0.2.mat',data_dir, sub_prefix));
    n_sess = length(fnames);

    results = cell(n_sess,1);
    sess_names = cell(n_sess,1);

    startParPool(min([n_sess,10]));
    parfor ii = 1:n_sess
        sess_names{ii} = fnames(ii).name(1:8);
        results{ii} = compile_session_data(fnames(ii).name,data_dir,ar_order);
    end; clear ii

    % compile session info into struct
    all_data = cell2struct(results, sess_names);
    all_data.ar_order = ar_order;

    % add array layout per subject
    load(sprintf('%s/cmp_%s.mat',data_dir,sub_str));
    all_data.arr_spatial.LH_layout = LH;
    all_data.arr_spatial.RH_layout = RH;
    all_data.arr_spatial.LH_dists = LH_dists;
    all_data.arr_spatial.RH_dists = RH_dists;
    clear LH RH LH_dists RH_dists
    
    % reorder fields to have general info at beginning
    all_data = orderfields(all_data,[n_sess+1,n_sess+2,1:n_sess]);

    % save per subject info
    save(sprintf('%s/all_data_delay_%s.mat',save_dir,sub_str),'all_data');

end



%% HELPER FUNCTION - per session

function results = compile_session_data(fname,data_dir,ar_order)
    rng('default');

    % optional parameters
    bin_width = 10; % ms
    skip_delay_bins = 20; % number of bins to skip after target onset when computing delay counts
    
    % some useful constants - Smith lab trial codes
    targ_on_code = 70; targ_off_code = 100; fix_off_code = 3; saccade_code = 141; 
    
    %% load data
    sub_date = fname(1:8);
    
    if strcmpi(sub_date(1:2),'pe')
        load(sprintf('%s/%s',data_dir,fname),'dat','testDat');
        dat = [dat testDat];
        clear testDat;
    else
        load(sprintf('%s/%s',data_dir,fname),'dat');
    end
    
    %% get some info about this session (# trials, # targets, stim length, delay period length, etc)
    n_trials = length(dat);

    % target angles and reaction times
    targ_angs = nan(1,n_trials);
    rts = nan(1,n_trials);
    fix_durs = nan(1,n_trials);
    for i_trial = 1:n_trials
        curr_codes = dat(i_trial).trialcodes;
        fixoff = curr_codes(curr_codes(:,2)==fix_off_code,3);
        sacc = curr_codes(curr_codes(:,2)==saccade_code,3);
        rts(i_trial) = sacc-fixoff;
        targ_angs(i_trial) = dat(i_trial).params.trial.angle;
        fix_durs(i_trial) = dat(i_trial).params.trial.fixDuration;
    end
    n_targs = length(unique(targ_angs));
    
    results.n_targs = n_targs;
    results.n_trials = n_trials;
    results.targ_angs = targ_angs;
    results.rt = rts;

    % trial structure
    cue_on_delay = dat(1).params.block.targetOnsetDelay;
    cue_dur = dat(1).params.block.targetDuration;
    delay_durs = fix_durs - (cue_dur + cue_on_delay);
    
    results.cue_dur = cue_dur;
    results.delay_dur = delay_durs;
        
    %% remove neurons with low frs or cross-talk
    [~,tmp] = binSpikeCounts(dat,targ_on_code,fix_off_code,0.001);
    good_chans = get_good_channels(tmp);
    array1_idx = good_chans<=96;
    array2_idx = good_chans>96;
    n_chans = length(good_chans);

    % channel and array indices
    if strcmpi(sub_date(1:2),'wa')
        % Wakko arrays: left hemisphere array = C, right hemisphere array = A
        results.arr.LH_idx = array2_idx;
        results.arr.RH_idx = array1_idx;
        results.arr.LH_chans = good_chans(array2_idx);
        results.arr.RH_chans = good_chans(array1_idx);
    else
        % Satchel & Pepe arrays: left hemisphere array = A, right hemisphere array = C
        results.arr.LH_idx = array1_idx;
        results.arr.RH_idx = array2_idx;
        results.arr.LH_chans = good_chans(array1_idx);
        results.arr.RH_chans = good_chans(array2_idx);
    end
    results.n_arr1 = sum(results.arr.LH_idx);
    results.n_arr2 = sum(results.arr.RH_idx);
    
    %% extract spikes and spike counts during delay period
    [~,delay_dat] = binSpikeCounts(dat,targ_off_code,fix_off_code,bin_width/1000);
    min_delay_bins = min([delay_dat.nBins]);
    n_delay_bins = min_delay_bins - skip_delay_bins;
    delay_counts = nan(n_chans,n_delay_bins,n_trials);

    for i_trial = 1:n_trials
        delay_counts(:,:,i_trial) = delay_dat(i_trial).counts(good_chans,skip_delay_bins+1:min_delay_bins);
    end
    clear delay_dat;

    % get last 1 sec of shared delay time
    results.ex_spike_times = delay_counts(:,:,1); % keep one example trial of spike times
    counts_mat = squeeze(sum(delay_counts(:,end-99:end,:),2))'; % n_trials x n_neurons
    results.raw_counts = counts_mat;
    results.binsize = 1; % sec
    
    %% preprocess counts for analyses
    % 1) remove condition means
    [N,D] = size(counts_mat);

    counts_mat_nomean = nan(N,D);
    for cond = unique(targ_angs)
        cond_mask = targ_angs==cond;
        cond_counts = counts_mat(cond_mask,:);
        cond_mean = mean(cond_counts,1); % mean across trials, per neuron
        counts_mat_nomean(cond_mask,:) = cond_counts - cond_mean;
    end
    assert(~any(isnan(counts_mat_nomean),'all'), 'Some counts missing from condition mean subtraction')

    % 2) remove moving average
    slow_component_est = movmean(counts_mat_nomean, ar_order, 1, "Endpoints", "shrink");
    counts_mat_nomean_resid = counts_mat_nomean - slow_component_est;

    % 3) separate into left and right hemisphere
    left_idx = results.arr.LH_idx > 0;
    right_idx = results.arr.RH_idx > 0;
    
    results.slow_component_left = slow_component_est(:,left_idx);
    results.slow_component_right = slow_component_est(:,right_idx);
    results.fast_component_left = counts_mat_nomean_resid(:,left_idx);
    results.fast_component_right = counts_mat_nomean_resid(:,right_idx);

end