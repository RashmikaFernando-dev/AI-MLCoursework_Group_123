function save_signal_analysis(ALL_raw, ALL_filtered, CONFIG, RUN, tag)
% SAVE_SIGNAL_EXAMPLE_PLOTS - Save raw vs filtered signal comparison

try
    % Get first user's data for example
    users = unique(ALL_raw.User);
    if isempty(users), return; end
    
    user_data_raw = ALL_raw(ALL_raw.User == users(1), :);
    user_data_filt = ALL_filtered(ALL_filtered.User == users(1), :);
    
    % Take first 1000 samples for visualization
    n_samples = min(1000, height(user_data_raw));
    if n_samples < 10, return; end
    
    time_vec = (0:n_samples-1) / CONFIG.fs;
    
    % Create comparison plot
    fig = figure('Visible', 'off', 'Color', 'w');
    
    % Accelerometer comparison
    subplot(2,1,1);
    plot(time_vec, user_data_raw.acc_x(1:n_samples), 'b-', 'LineWidth', 0.5, 'DisplayName', 'Raw');
    hold on;
    plot(time_vec, user_data_filt.acc_x(1:n_samples), 'r-', 'LineWidth', 1, 'DisplayName', 'Filtered');
    xlabel('Time (s)'); ylabel('Acceleration X');
    title(sprintf('Signal Filtering Example - Accelerometer (User %d)', users(1)));
    legend('show'); grid on;
    
    % Gyroscope comparison  
    subplot(2,1,2);
    plot(time_vec, user_data_raw.gyro_x(1:n_samples), 'b-', 'LineWidth', 0.5, 'DisplayName', 'Raw');
    hold on;
    plot(time_vec, user_data_filt.gyro_x(1:n_samples), 'r-', 'LineWidth', 1, 'DisplayName', 'Filtered');
    xlabel('Time (s)'); ylabel('Gyroscope X');
    title('Signal Filtering Example - Gyroscope');
    legend('show'); grid on;
    
    save_fig(fig, fullfile(RUN.figDir, fname_base(CONFIG, tag) + "_signal_example.png"));
    
catch ME
    warning('save_signal_example_plots:PlotFailed', 'Signal example plots failed: %s', ME.message);
end
end