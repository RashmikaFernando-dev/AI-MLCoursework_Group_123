function save_pr_metrics(yte_idx, yhat_te, YP_te, RUN, CONFIG, tag)
% SAVE_PR_METRICS - Save precision/recall/F1 metrics per class and summary

try
    unique_classes = unique([yte_idx; yhat_te]);
    n_classes = length(unique_classes);
    
    % Per-class metrics
    precision = zeros(n_classes, 1);
    recall = zeros(n_classes, 1);
    f1_score = zeros(n_classes, 1);
    
    for i = 1:n_classes
        class_id = unique_classes(i);
        tp = sum((yte_idx == class_id) & (yhat_te == class_id));
        fp = sum((yte_idx ~= class_id) & (yhat_te == class_id));
        fn = sum((yte_idx == class_id) & (yhat_te ~= class_id));
        
        precision(i) = tp / max(tp + fp, eps);
        recall(i) = tp / max(tp + fn, eps);
        f1_score(i) = 2 * precision(i) * recall(i) / max(precision(i) + recall(i), eps);
    end
    
    % Save per-class results
    perclass_table = table(unique_classes, precision, recall, f1_score, ...
        'VariableNames', {'class_id', 'precision', 'recall', 'f1_score'});
    writetable(perclass_table, fullfile(RUN.tabDir, fname_base(CONFIG, tag) + "_perclass.csv"));
    
    % Save summary metrics
    summary_table = table(mean(precision), mean(recall), mean(f1_score), ...
        'VariableNames', {'mean_precision', 'mean_recall', 'mean_f1'});
    writetable(summary_table, fullfile(RUN.tabDir, fname_base(CONFIG, tag) + "_summary_metrics.csv"));
    
catch ME
    warning('save_pr_metrics:SaveFailed', 'PR metrics save failed: %s', ME.message);
end
end