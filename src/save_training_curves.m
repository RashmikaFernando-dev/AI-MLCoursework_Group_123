function save_training_curves(tr, RUN, CONFIG, tag)
% SAVE_TRAINING_PLOTS - Save neural network training performance plots
%
% Inputs:
%   tr     - training record from neural network
%   RUN    - run configuration with output directories
%   CONFIG - pipeline configuration
%   tag    - identifier tag for filename

try
    % Create training performance plot
    fig = figure('Visible', 'off', 'Color', 'w');
    
    % Plot training performance
    if isfield(tr, 'perf') && ~isempty(tr.perf)
        semilogy(tr.perf, 'b-', 'LineWidth', 1.5);
        hold on;
        
        if isfield(tr, 'vperf') && ~isempty(tr.vperf)
            semilogy(tr.vperf, 'r-', 'LineWidth', 1.5);
            legend('Training', 'Validation', 'Location', 'best');
        else
            legend('Training', 'Location', 'best');
        end
        
        xlabel('Epoch');
        ylabel('Performance (MSE)');
        title(sprintf('Training Performance (%s)', tag));
        grid on;
    else
        % Fallback if training record is incomplete
        text(0.5, 0.5, 'Training data not available', ...
            'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', 'k');
        xlim([0 1]); ylim([0 1]);
        title(sprintf('Training Performance (%s)', tag), 'Color', 'k');
        set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
    end
    
    % Save figure
    filename = fullfile(RUN.figDir, fname_base(CONFIG, tag) + "_training.png");
    save_fig(fig, filename);
    
catch ME
    warning('save_training_plots:PlotFailed', 'Failed to save training plots: %s', ME.message);
end
end