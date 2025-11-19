function save_user_performance(yte_idx, yhat_te, RUN, CONFIG, tag)
% SAVE_PERCLASS_F1_PLOT - Save bar plot of F1 scores per class

try
    unique_classes = unique([yte_idx; yhat_te]);
    n_classes = length(unique_classes);
    f1_scores = zeros(n_classes, 1);
    
    % Calculate F1 score for each class
    for i = 1:n_classes
        class_id = unique_classes(i);
        tp = sum((yte_idx == class_id) & (yhat_te == class_id));
        fp = sum((yte_idx ~= class_id) & (yhat_te == class_id));
        fn = sum((yte_idx == class_id) & (yhat_te ~= class_id));
        
        precision = tp / max(tp + fp, eps);
        recall = tp / max(tp + fn, eps);
        f1_scores(i) = 2 * precision * recall / max(precision + recall, eps);
    end
    
    % Create bar plot
    fig = figure('Visible', 'off', 'Color', 'w');
    bar(unique_classes, f1_scores, 'FaceColor', [0.2, 0.6, 0.8]);
    xlabel('User ID');
    ylabel('F1 Score');
    title('Per-Class F1 Scores');
    grid on;
    ylim([0, 1]);
    
    % Add value labels on bars
    for i = 1:length(f1_scores)
        text(unique_classes(i), f1_scores(i) + 0.02, sprintf('%.3f', f1_scores(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 8);
    end
    
    save_fig(fig, fullfile(RUN.figDir, fname_base(CONFIG, tag) + "_f1_perclass.png"));
    
catch ME
    warning('save_perclass_f1_plot:PlotFailed', 'Per-class F1 plot failed: %s', ME.message);
end
end