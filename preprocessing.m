monkey_names = ["W", "V"];
aligned_events = ["RewFeedback", "StimOnset"];

addpath '/Users/f005d7d/Documents/Attn_MdPRL/what-where-task/RasterVec_binSize_10ms'
addpath '/Users/f005d7d/Documents/Attn_MdPRL/what-where-task/Behavior/'

bhv_path = '/Users/f005d7d/Documents/Attn_MdPRL/what-where-task/Behavior/';
processed_path = '/Users/f005d7d/Documents/Attn_MdPRL/what-where-task/processed/';
filelists = dir('/Users/f005d7d/Documents/Attn_MdPRL/what-where-task/RasterVec_binSize_10ms');

filelists = {filelists.name};
filelists = filelists(4:end);

binsize = 0.01;
gauss_window = 0.5./binsize;
gauss_SD = 0.05./binsize; 
gk = gausskernel(gauss_window,gauss_SD); 
gk = gk./binsize;


% 12 regressors, separate by blocks
regressor_names = {'S_curr', 'C_what_curr', 'C_where_curr', 'R_curr', ...
                   'C_what_prev', 'C_where_prev', 'R_prev', ...
                   'C_where_prevXR_prev', 'C_what_prevXR_prev', ...
                   'SXC_what_prev', 'SXR_prev', ...
                   'SXC_what_prevXR_prev'};


%%


all_sess_regression_info = ...
    struct('monkey', {{}}, 'sess', {{}}, ...
           'betas', {{}}, 'exp_var', {{}});

for f=filelists
    % load neural data
    curr_sess_info = load(f{1});

    % load monkey and session names
    mname_sess = regexp(f{1}, 'RastVect-(\w*)-binsize10ms-align2StimOnset', 'tokens');
    mname_sess = mname_sess{1}{1};
    monkey_name = mname_sess(1);
    sess_name = mname_sess(2:end);

    all_sess_regression_info.monkey{end+1} = monkey_name;
    all_sess_regression_info.sess{end+1} = sess_name;

    % smooth neural data
    neural_data = convn(curr_sess_info.aligned2event, gk, 'same');
    % neural_data = neural_data(1:stride:end,:,:);
    [num_timesteps, num_units, num_trials] = size(neural_data);

    % load behavioral data
    bhv_filename = strcat(bhv_path, 'SPKcounts_', mname_sess, 'cue_MW_250X250ms.mat');
    bhv_file = load(bhv_filename);
    task_info = bhv_file.Y;
    % only keep chosen image, chosen loc, reward, and block type
    task_info = task_info(:, [1 2 3 10]); 

    disp("===============================================")
    disp(strcat('loaded file ', f{1}));

    % create array to save regression coeffs and explained variance
    all_blocks_beta = ones(2, num_timesteps, num_units, length(regressor_names)+1)*nan;
    all_blocks_exp_var = ones(2, num_timesteps, num_units, length(regressor_names))*nan;

    for block_type=[1 2]
        disp('-------------------------------------------------')
        disp(strcat('block type ', block_type));

        block_type_mask = task_info(:,4)==block_type;
        
        block_trial_info = task_info(block_type_mask,:);
        block_trial_info(:,1:3) = block_trial_info(:,1:3)*2-1;
        
        X = [block_trial_info(2:end,1).*block_trial_info(2:end,2)...
             block_trial_info(2:end,1)...
             block_trial_info(2:end,2)...
             block_trial_info(2:end,3)...
             block_trial_info(1:end-1,1)...
             block_trial_info(1:end-1,2)...
             block_trial_info(1:end-1,3)...
             block_trial_info(1:end-1,2).*block_trial_info(1:end-1,3)...
             block_trial_info(1:end-1,1).*block_trial_info(1:end-1,3)...
             block_trial_info(2:end,1).*block_trial_info(2:end,2).*block_trial_info(1:end-1,1)...
             block_trial_info(2:end,1).*block_trial_info(2:end,2).*block_trial_info(1:end-1,3)...
             block_trial_info(2:end,1).*block_trial_info(2:end,2).*...
             block_trial_info(1:end-1,1).*block_trial_info(1:end-1,3)];

        X = [block_trial_info(1,1).*block_trial_info(1,2)...
             block_trial_info(1,1)...
             block_trial_info(1,2)...
             block_trial_info(1,3)...
             zeros(1,8); X];

        all_units_beta = ones(num_timesteps, num_units, length(regressor_names)+1)*nan;
        all_units_exp_var = ones(num_timesteps, num_units, length(regressor_names))*nan;
        for unit=1:num_units
            disp(strcat(num2str(unit), '/', num2str(num_units)))
            curr_unit_betas = [];
            curr_unit_exp_var = [];
            for timestep=1:num_timesteps
                tbl_to_fit = array2table([squeeze(neural_data(timestep, unit, block_type_mask)), X], ...
                    "VariableNames", ['fr', regressor_names]);
                mdl = fitlm(tbl_to_fit, strcat('fr~', strjoin(regressor_names, '+')));
                curr_unit_betas(timestep,:) = mdl.Coefficients.Estimate;
                anova_mdl = anova(mdl);
                curr_unit_exp_var(timestep,:) = anova_mdl.SumSq(1:end-1)/sum(anova_mdl.SumSq);
            end
            all_units_beta(:,unit,:) = curr_unit_betas;
            all_units_exp_var(:,unit,:) = curr_unit_exp_var;
        end

        all_blocks_beta(block_type,:,:,:) = all_units_beta;
        all_blocks_exp_var(block_type,:,:,:) = all_units_exp_var;
    end

    all_sess_regression_info.betas{end+1} = all_blocks_beta;
    all_sess_regression_info.exp_var{end+1} = all_blocks_exp_var;

    save(strcat(processed_path, 'all_sess_regression_info'), all_sess_regression_info)
end


