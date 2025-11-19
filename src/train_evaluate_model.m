function result = train_evaluate_model(Xtr,ytr,Xte,yte,CONFIG,RUN,tag)
% TRAIN_EVALUATE_MODEL - Train neural network and perform comprehensive evaluation
% Trains once, saves confusion matrix, ROC curves + summary CSV

touchLog = fullfile(RUN.outDir, "_touch_log.txt");   % running log

% Convert labels to indices and create targets
[ytr_idx, labels, yte_idx] = labelToIndex(ytr, yte);
num_classes = length(labels);
num_samples = length(ytr_idx);

% Create one-hot encoded targets (classes x samples)
T = zeros(num_classes, num_samples);
for i = 1:num_samples
    T(ytr_idx(i), i) = 1;
end

fprintf('Data shapes: Xtr=%dx%d, ytr_idx=%dx%d, T=%dx%d\n', ...
    size(Xtr), size(ytr_idx), size(T));

% Disable GUI for training
if exist('nntraintool','file'), nntraintool('close'); nntraintool('disable'); end

% Configure neural network
net = patternnet(CONFIG.hiddenSizes, CONFIG.trainFcn);
net.trainParam.epochs = CONFIG.maxEpochs;
net.performParam.regularization = 0.1;
net.divideFcn = 'dividetrain';
net.trainParam.showWindow = false; 
net.trainParam.showCommandLine = false; 
net.trainParam.show = NaN;

% Debug: Check dimensions before training
fprintf('Training dimensions: Xtr'' = %dx%d, T = %dx%d\n', size(Xtr'), size(T));

% Train the network (inputs: features x samples, targets: classes x samples)
[net, tr] = train(net, Xtr', T);

% Save training plots
try
    save_training_curves(tr, RUN, CONFIG, tag);
    fid=fopen(touchLog,'a'); fprintf(fid,"Saved training plots\n"); fclose(fid);
catch ME
    fid=fopen(touchLog,'a'); fprintf(fid,"Training plot failed: %s\n", ME.message); fclose(fid);
end

% Generate predictions
YP_tr = net(Xtr'); [~, yhat_tr] = max(YP_tr, [], 1); yhat_tr = yhat_tr(:);
YP_te = net(Xte'); [~, yhat_te] = max(YP_te, [], 1); yhat_te = yhat_te(:);
acc_tr = mean(yhat_tr == ytr_idx); 
acc_te = mean(yhat_te == yte_idx);

% Save confusion matrix
try
    figCM = figure('Visible','off','Color','w','Name',"CM_"+tag);
    cm = confusionchart(yte_idx, yhat_te);
    cm.Title = sprintf('Confusion Matrix (%s)', tag);
    cm.RowSummary = 'row-normalized'; 
    cm.ColumnSummary = 'column-normalized';
    cm.FontColor = 'k';
    cm.DiagonalColor = 'k';
    cm.OffDiagonalColor = 'k';

    save_fig(figCM, fullfile(RUN.figDir, fname_base(CONFIG, tag) + "_CM.png"));
    fid=fopen(touchLog,'a'); fprintf(fid,"Saved confusion matrix\n"); fclose(fid);
catch ME
    fid=fopen(touchLog,'a'); fprintf(fid,"CM save failed: %s\n", ME.message); fclose(fid);
end

% Compute ROC and EER metrics
scores_te = YP_te';
[EER, FRR, FAR, thr, fpr, tpr] = compute_authentication_metrics(scores_te, yte_idx);

% Save ROC curve
try
    figROC = figure('Visible','off','Color','w','Name',"ROC_"+tag);
    plot(fpr, tpr, 'LineWidth', 1.5); grid on;
    xlabel('False Positive Rate'); ylabel('True Positive Rate');
    title(sprintf('ROC Curve (%s) | EER=%.3f', tag, EER));
    plot([0,1], [0,1], 'k--', 'Alpha', 0.5); % diagonal line
    xlim([0,1]); ylim([0,1]);
    save_fig(figROC, fullfile(RUN.figDir, fname_base(CONFIG, tag) + "_ROC.png"));
    fid=fopen(touchLog,'a'); fprintf(fid,"Saved ROC curve\n"); fclose(fid);
catch ME
    fid=fopen(touchLog,'a'); fprintf(fid,"ROC save failed: %s\n", ME.message); fclose(fid);
end

% Save summary CSV
try
    hs_str = string(join(string(CONFIG.hiddenSizes), ","));
    row = struct('tag',string(tag),'acc_tr',acc_tr,'acc_te',acc_te,'EER',EER, ...
                 'FRR_atEER',FRR,'FAR_atEER',FAR,'thr_atEER',thr, ...
                 'hidden',hs_str,'trainFcn',string(CONFIG.trainFcn),'epochs',double(CONFIG.maxEpochs));
    outcsv = fullfile(RUN.tabDir, fname_base(CONFIG, tag) + "_summary.csv");
    writetable(struct2table(row), outcsv);
    fid=fopen(touchLog,'a'); fprintf(fid,"Saved summary CSV -> %s\n", outcsv); fclose(fid);
catch ME
    fid=fopen(touchLog,'a'); fprintf(fid,"Summary CSV failed: %s\n", ME.message); fclose(fid);
end

% Return comprehensive results
result = struct('acc_tr',acc_tr,'acc_te',acc_te,'EER',EER,'FRR',FRR,'FAR',FAR,'thr',thr,'net',net,'tr',tr);
end