function analyze_sensor_features(XF, yUser, RUN, CONFIG)
% ANALYZE_FEATURES - Enhanced feature analysis and visualizations
% Generates comprehensive feature analysis plots and statistics

try
    fprintf('  - Feature matrix: %d samples x %d features\n', size(XF,1), size(XF,2));
    fprintf('  - Users: %d unique\n', length(unique(yUser)));
    
    %% 1) Feature summary statistics plot
    figure('Position',[100,100,1200,800], 'Color', 'w');
    subplot(2,2,1);
    mu = mean(XF,1); sigma = std(XF,1);
    plot(1:length(mu), mu, 'b-', 'LineWidth',1.5); hold on;
    plot(1:length(sigma), sigma, 'r-', 'LineWidth',1.5);
    xlabel('Feature Index', 'Color', 'k'); ylabel('Value', 'Color', 'k');
    title('Feature Statistics Overview', 'Color', 'k');
    legend('Mean','Std Dev','Location','best', 'TextColor', 'k');
    grid on;
    set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
    
    %% 2) Feature correlation heatmap
    subplot(2,2,2);
    if size(XF,2) <= 50  % Only for reasonable number of features
        C = corrcoef(XF);
        imagesc(C); colorbar; colormap('jet');
        title('Feature Correlation Matrix');
        xlabel('Feature Index'); ylabel('Feature Index');
    else
        % For many features, show sample correlation
        idx_sample = 1:min(30,size(XF,2));
        C = corrcoef(XF(:,idx_sample));
        imagesc(C); colorbar; colormap('jet');
        title(sprintf('Sample Correlation (first %d features)',length(idx_sample)));
        xlabel('Feature Index'); ylabel('Feature Index');
    end
    
    %% 3) Class distribution visualization
    subplot(2,2,3);
    [users,~,user_idx] = unique(yUser);
    hist_data = histcounts(user_idx, 1:length(users)+1);
    bar(hist_data);
    xlabel('User ID'); ylabel('Sample Count');
    title('Samples per User');
    grid on;
    
    %% 4) Feature variance ranking
    subplot(2,2,4);
    feature_var = var(XF,1);
    [sorted_var, sort_idx] = sort(feature_var, 'descend');
    semilogy(1:length(sorted_var), sorted_var, 'g-', 'LineWidth',1.5);
    xlabel('Feature Rank'); ylabel('Variance (log scale)');
    title('Feature Importance by Variance');
    grid on;
    
    save_fig(gcf, fullfile(RUN.figDir, 'feature_summary_stats.png'));
    close(gcf);
    
    %% 5) Feature correlation heatmap (separate detailed plot)
    if size(XF,2) <= 100
        figure('Position',[200,200,800,600]);
        C = corrcoef(XF);
        imagesc(C); colorbar; colormap('jet');
        title('Feature Correlation Heatmap');
        xlabel('Feature Index'); ylabel('Feature Index');
        save_fig(gcf, fullfile(RUN.figDir, 'feature_correlation_heatmap.png'));
        close(gcf);
    end
    
    %% 6) Class distribution detailed plot
    figure('Position',[300,300,1000,600]);
    subplot(1,2,1);
    bar(hist_data);
    xlabel('User ID'); ylabel('Sample Count');
    title('Samples per User');
    grid on;
    
    subplot(1,2,2);
    % Box plot of first few features by class
    if size(XF,2) >= 3
        feat_idx = 1:min(3,size(XF,2));
        for i = 1:length(feat_idx)
            subplot(1,length(feat_idx)+1,i+1);
            boxplot(XF(:,feat_idx(i)), user_idx);
            xlabel('User ID'); ylabel(sprintf('Feature %d',feat_idx(i)));
            title(sprintf('Feature %d Distribution',feat_idx(i)));
        end
    end
    save_fig(gcf, fullfile(RUN.figDir, 'feature_class_distribution.png'));
    close(gcf);
    
    %% 7) Feature F-statistics (ANOVA-like)
    figure('Position',[400,400,1000,500]);
    f_stats = zeros(1,size(XF,2));
    for j = 1:size(XF,2)
        try
            [~,~,stats] = anova1(XF(:,j), user_idx, 'off');
            f_stats(j) = stats.F;
        catch
            f_stats(j) = 0;
        end
    end
    
    subplot(1,2,1);
    plot(1:length(f_stats), f_stats, 'b-', 'LineWidth',1.5);
    xlabel('Feature Index'); ylabel('F-statistic');
    title('Feature Discriminative Power (F-stats)');
    grid on;
    
    subplot(1,2,2);
    [sorted_f, sort_idx_f] = sort(f_stats, 'descend');
    semilogy(1:length(sorted_f), sorted_f, 'r-', 'LineWidth',1.5);
    xlabel('Feature Rank'); ylabel('F-statistic (log scale)');
    title('Ranked Feature F-statistics');
    grid on;
    
    save_fig(gcf, fullfile(RUN.figDir, 'feature_fstats.png'));
    close(gcf);
    
    %% 8) PCA preview (before main PCA)
    if size(XF,1) > size(XF,2) && size(XF,2) > 2
        figure('Position',[500,500,800,600]);
        [coeff_preview,score_preview,~,~,explained_preview] = pca(XF);
        
        subplot(2,2,1);
        plot(1:min(20,length(explained_preview)), explained_preview(1:min(20,end)), 'bo-');
        xlabel('PC Number', 'Color', 'black'); ylabel('Variance Explained (%)', 'Color', 'black');
        title('PCA Scree Plot Preview', 'Color', 'black');
        grid on;
        set(gca, 'XColor', 'black', 'YColor', 'black');
        
        subplot(2,2,2);
        plot(cumsum(explained_preview(1:min(20,end))), 'ro-');
        xlabel('PC Number'); ylabel('Cumulative Variance (%)');
        title('Cumulative Variance Explained');
        grid on;
        
        if size(score_preview,2) >= 2
            subplot(2,1,2);
            gscatter(score_preview(:,1), score_preview(:,2), user_idx);
            xlabel('PC1'); ylabel('PC2');
            title('PCA Preview: PC1 vs PC2 by User');
            legend('Location','best');
        end
        
        save_fig(gcf, fullfile(RUN.figDir, 'feature_pca_preview.png'));
        close(gcf);
    end
    
    %% 9) Save detailed statistics table
    feature_stats = table();
    feature_stats.feature_idx = (1:size(XF,2))';
    feature_stats.mean_val = mean(XF, 1)';
    feature_stats.std_val = std(XF, 1)';
    feature_stats.var_val = var(XF, 1)';
    feature_stats.min_val = min(XF, [], 1)';
    feature_stats.max_val = max(XF, [], 1)';
    feature_stats.f_stat = f_stats';
    
    writetable(feature_stats, fullfile(RUN.tabDir, 'feature_analysis_detailed.csv'));
    fprintf('  - Generated %d feature analysis plots\n', 5);
    fprintf('  - Saved detailed feature statistics\n');
    
catch ME
    warning('analyze_features:AnalysisFailed', 'Feature analysis failed: %s', ME.message);
end
end