function save_fig(fig_handle, filepath)
% SAVE_FIG - Save figure to file with proper formatting
% 
% Inputs:
%   fig_handle - figure handle
%   filepath   - full path for output file

try
    % Ensure directory exists
    [dir_path, ~, ~] = fileparts(filepath);
    if ~exist(dir_path, 'dir')
        mkdir(dir_path);
    end
    
    % Set figure properties for better output
    set(fig_handle, 'Color', 'white');
    set(fig_handle, 'PaperPositionMode', 'auto');
    
    % Save figure
    saveas(fig_handle, filepath);
    
    % Close figure to save memory
    close(fig_handle);
    
catch ME
    warning('save_fig:SaveFailed', 'Failed to save figure: %s', ME.message);
    if ishandle(fig_handle)
        close(fig_handle);
    end
end
end