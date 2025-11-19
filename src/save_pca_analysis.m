function save_pca_analysis(explained, Xtr, ytr, RUN, CONFIG, tag)
% SAVE_PCA_PLOTS - Save PCA analysis plots (scree plot and 2D scatter)

try
    % Scree plot
    fig1 = figure('Visible', 'off', 'Color', 'w');
    plot(cumsum(explained), 'o-', 'LineWidth', 1.5);
    xlabel('Principal Component');
    ylabel('Cumulative Variance Explained (%)');
    title('PCA Scree Plot');
    grid on;
    yline(CONFIG.pcaVarKeep * 100, 'r--', sprintf('%.0f%% threshold', CONFIG.pcaVarKeep * 100));
    save_fig(fig1, fullfile(RUN.figDir, fname_base(CONFIG, tag) + "_pca_scree.png"));
    
    % 2D scatter plot of first two PCs
    if size(Xtr, 2) >= 2
        fig2 = figure('Visible', 'off', 'Color', 'w');
        unique_users = unique(ytr);
        colors = lines(length(unique_users));
        
        for i = 1:length(unique_users)
            user_idx = (ytr == unique_users(i));
            scatter(Xtr(user_idx, 1), Xtr(user_idx, 2), 50, colors(i,:), 'filled', ...
                'DisplayName', sprintf('User %d', unique_users(i)));
            hold on;
        end
        
        xlabel('PC1'); ylabel('PC2');
        title('PCA 2D Projection');
        legend('show'); grid on;
        save_fig(fig2, fullfile(RUN.figDir, fname_base(CONFIG, tag) + "_pca_2d.png"));
    end
    
catch ME
    warning('save_pca_plots:PlotFailed', 'PCA plots failed: %s', ME.message);
end
end