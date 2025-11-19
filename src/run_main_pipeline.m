function run_main_pipeline()
% RUN_MAIN_PIPELINE - Enhanced Accelerometer-Based Authentication System
% End-to-end: ingest CSV → filter → window → features → (PCA) →
% train NN → Accuracy + CM + FAR/FRR/EER + ROC → save outputs
% Enhanced with comprehensive evaluation metrics and visualizations

clc; close all; rng(123);
warning('off','stats:pca:ColRankDefX');   % harmless PCA rank warning

% Add current directory to MATLAB path to ensure all functions are available
addpath(pwd);

fprintf('=== Enhanced Accelerometer-Based Authentication Pipeline ===\n');

%% ── 1) Auto-detect data folder ─────────────────────────────────────────────
% Recursively search for CSV files starting from current location
function dataPath = findDataFolder(searchDir, maxDepth)
    dataPath = "";
    if maxDepth <= 0, return; end
    
    % Check current directory for CSV files
    csvFiles = dir(fullfile(searchDir, '*.csv'));
    if ~isempty(csvFiles)
        % Verify it contains our data files (U*NW pattern)
        for f = csvFiles.'
            if contains(f.name, 'NW_') && (contains(f.name, 'FD') || contains(f.name, 'MD'))
                dataPath = searchDir;
                return;
            end
        end
    end
    
    % Search subdirectories
    try
        subDirs = dir(searchDir);
        subDirs = subDirs([subDirs.isdir] & ~startsWith({subDirs.name}, '.'));
        for d = subDirs.'
            result = findDataFolder(fullfile(searchDir, d.name), maxDepth - 1);
            if result ~= ""
                dataPath = result;
                return;
            end
        end
    catch
        % Skip inaccessible directories
    end
end

% Start searching from current directory
dataDir = findDataFolder(pwd, 4);  % Search up to 4 levels deep

% If not found locally, try parent directories
if dataDir == ""
    parentDir = fileparts(pwd);
    dataDir = findDataFolder(parentDir, 3);
end

if dataDir==""
    error("Data files not found! Please ensure CSV files (U*NW_FD.csv, U*NW_MD.csv) are in the project.");
end
fprintf("Auto-detected data folder: %s\n", dataDir);

%% ── 2) Configuration ────────────────────────────────────────────────────────
CONFIG.dataDir     = dataDir;
CONFIG.fs          = 32;       % sampling rate (Hz)
CONFIG.lp_acc_hz   = 5;        % accel low-pass cutoff (Hz)
CONFIG.lp_gyro_hz  = 10;       % gyro low-pass cutoff (Hz)
CONFIG.win_sec     = 3.0;      % window length (seconds)
CONFIG.step_ratio  = 0.5;      % 50% overlap
CONFIG.split_mode  = "session";% 'session' (FD→train, MD→test) or 'random'
CONFIG.train_ratio = 0.7;      % used if split_mode='random'

% Neural Network configuration
CONFIG.hiddenSizes = [32 16];
CONFIG.trainFcn    = 'trainscg';
CONFIG.maxEpochs   = 200;

% PCA controls
CONFIG.usePCA      = true;
CONFIG.pcaVarKeep  = 0.95;

% Output directories (timestamped like mathlab_project)
RUN.runStamp = make_run_stamp();

% Find correct results directory (look in parent directory if needed)
if isfolder(fullfile(pwd, "results"))
    resultsBase = fullfile(pwd, "results");
elseif isfolder(fullfile(pwd, "..", "results"))
    resultsBase = fullfile(pwd, "..", "results");
else
    resultsBase = fullfile(fileparts(pwd), "results");
    if ~isfolder(resultsBase), mkdir(resultsBase); end
end

RUN.outDir   = fullfile(resultsBase, RUN.runStamp);
RUN.figDir   = fullfile(RUN.outDir,"figures");
RUN.tabDir   = fullfile(RUN.outDir,"tables");
RUN.logDir   = fullfile(RUN.outDir,"logs");
for d = [RUN.outDir, RUN.figDir, RUN.tabDir, RUN.logDir]
    if ~isfolder(d), mkdir(d); end
end

% Log pipeline start
fid = fopen(fullfile(RUN.outDir,'_pipeline_started.txt'),'w');
ts = datetime('now','Format','yyyy-MM-dd HH:mm:ss');
fprintf(fid,'Pipeline started: %s\n', char(ts));
fclose(fid);

%% ── 3) Ingest all files ─────────────────────────────────────────────────────
allFiles = dir(fullfile(CONFIG.dataDir,'*'));
files = allFiles(~[allFiles.isdir] & endsWith({allFiles.name},{'.csv','.CSV','.xlsx','.XLSX'}));

ALL = table();
for k = 1:numel(files)
    f = fullfile(files(k).folder, files(k).name);
    ALL = [ALL; readOneFile(f)]; %#ok<AGROW>
end
fprintf('Loaded %d rows total from %d files.\n', height(ALL), numel(files));

%% ── 4) Filter signals ───────────────────────────────────────────────────────
ALL_raw = ALL;   % keep raw copy for visualization
[ALL.acc_x,ALL.acc_y,ALL.acc_z]    = apply_sensor_filter(ALL.acc_x,ALL.acc_y,ALL.acc_z,CONFIG.fs,CONFIG.lp_acc_hz);
[ALL.gyro_x,ALL.gyro_y,ALL.gyro_z] = apply_sensor_filter(ALL.gyro_x,ALL.gyro_y,ALL.gyro_z,CONFIG.fs,CONFIG.lp_gyro_hz);

%% ── 5) Window + feature extraction ──────────────────────────────────────────
winN  = round(CONFIG.win_sec * CONFIG.fs);
stepN = max(1, round((1 - CONFIG.step_ratio) * winN));
[XF, yUser, sessTag] = extract_sensor_features(ALL, winN, stepN);
XF = zscore(XF);

% Feature analysis before PCA / training
try
    analyze_sensor_features(XF, yUser, RUN, CONFIG);
catch ME
    try txt = getReport(ME,'basic','hyperlinks','off'); catch, txt = char(ME.message); end
    warning('PIPE:FeatAnalysis','%s %s','Feature analysis skipped:', txt);
end

%% ── 6) Train/Test split ─────────────────────────────────────────────────────
switch lower(CONFIG.split_mode)
  case 'session'
    isTrain = strcmp(sessTag,'day1');    % FD → train
    isTest  = strcmp(sessTag,'day2');    % MD → test
  case 'random'
    cv = cvpartition(yUser,'HoldOut',1 - CONFIG.train_ratio);
    isTrain = training(cv); isTest = test(cv);
  otherwise
    error('Unknown split_mode: %s', CONFIG.split_mode);
end
Xtr0 = XF(isTrain,:); ytr = yUser(isTrain);
Xte0 = XF(isTest ,:); yte = yUser(isTest);

%% ── 7) PCA (fit on TRAIN only) with variance-prune + rank-cap ───────────────
if CONFIG.usePCA
    std_tr   = std(Xtr0, 0, 1);
    keep_var = std_tr > 1e-10;
    Xtr0_red = Xtr0(:, keep_var);
    Xte0_red = Xte0(:, keep_var);

    [coeff_all, score_tr_all, ~, ~, explained, mu] = pca(Xtr0_red, ...
        'Centered', true, 'Algorithm', 'svd');

    rank_tr        = rank(Xtr0_red);
    target_by_var  = find(cumsum(explained)/100 >= CONFIG.pcaVarKeep, 1, 'first');
    if isempty(target_by_var), target_by_var = size(score_tr_all,2); end
    k              = min([rank_tr, target_by_var, size(score_tr_all,2)]);

    coeff = coeff_all(:,1:k);
    Xtr   = score_tr_all(:,1:k);
    Xte   = (Xte0_red - mu) * coeff;

    fprintf('PCA: kept %d/%d feats (var-prune), rank=%d, k=%d (keep=%.0f%% var)\n', ...
        sum(keep_var), numel(keep_var), rank_tr, k, 100*CONFIG.pcaVarKeep);

    % PCA plots (scree + 2D scatter on TRAIN scores)
    try
        save_pca_analysis(explained, Xtr, ytr, RUN, CONFIG, "main");
    catch ME
        try txt = getReport(ME,'basic','hyperlinks','off'); catch, txt = char(ME.message); end
        warning('PIPE:PCAplots','%s %s','PCA plots skipped:', txt);
    end
else
    Xtr = Xtr0; Xte = Xte0; k = size(Xtr0,2);
end

%% ── 8) Train + evaluate + save outputs ─────────────────────────────────────
res = train_evaluate_model(Xtr, ytr, Xte, yte, CONFIG, RUN, "main");

% Enhanced precision/recall/F1 CSVs (per-class + summary)
try
    [~, ~, yte_idx] = labelToIndex(ytr, yte);
    YP_te = res.net(Xte')';
    [~, yhat_te] = max(YP_te, [], 2);
    save_pr_metrics(yte_idx, yhat_te, YP_te, RUN, CONFIG, "main");
catch ME
    try txt = getReport(ME,'basic','hyperlinks','off'); catch, txt = char(ME.message); end
    warning('PIPE:PRcsv','%s %s','PR metrics CSVs skipped:', txt);
end

% Additional visualizations: per-class F1 and raw-vs-filtered overlay
try
    save_user_performance(yte_idx, yhat_te, RUN, CONFIG, "main");
catch ME
    try txt = getReport(ME,'basic','hyperlinks','off'); catch, txt = char(ME.message); end
    warning('PIPE:F1plot','%s %s','Per-class F1 plot skipped:', txt);
end

try
    save_signal_analysis(ALL_raw, ALL, CONFIG, RUN, "main");
catch ME
    try txt = getReport(ME,'basic','hyperlinks','off'); catch, txt = char(ME.message); end
    warning('PIPE:RawFilt','%s %s','Raw-vs-filtered example skipped:', txt);
end

%% ── 9) Generate comprehensive summary ───────────────────────────────────────
fid = fopen(fullfile(RUN.outDir,"run_summary.txt"),'w');
fprintf(fid,"=== ENHANCED ACCELEROMETER-BASED AUTHENTICATION RESULTS ===\n");
fprintf(fid,"DataDir      : %s\n", CONFIG.dataDir);
fprintf(fid,"Split        : %s  | TrainN=%d  TestN=%d\n", CONFIG.split_mode, sum(isTrain), sum(isTest));
fprintf(fid,"Model        : hidden=%s  trainFcn=%s  epochs=%d\n", mat2str(CONFIG.hiddenSizes), CONFIG.trainFcn, CONFIG.maxEpochs);
fprintf(fid,"PCA          : use=%d  keep=%.2f  k=%d\n", CONFIG.usePCA, CONFIG.pcaVarKeep, k);
fprintf(fid,"Results      : TrainAcc=%.4f  TestAcc=%.4f  EER=%.4f\n", res.acc_tr, res.acc_te, res.EER);
fprintf(fid,"Timestamp    : %s\n", datestr(now));
fclose(fid);

% Copy to latest results folder for easy access
try
    lat = fullfile(resultsBase,'latest');
    if isfolder(lat), rmdir(lat,'s'); end
    copyfile(RUN.outDir, lat);
catch
    fprintf('Note: Could not create latest results link\n');
end

fprintf('\n=== PIPELINE COMPLETED SUCCESSFULLY ===\n');
fprintf(' Test Accuracy: %.2f%% | EER: %.4f\n', res.acc_te*100, res.EER);
fprintf(' Results saved to: %s\n', RUN.outDir);
fprintf(' Also available in: results/latest/\n');

end