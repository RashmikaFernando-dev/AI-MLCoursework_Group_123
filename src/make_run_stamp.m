function stamp = make_run_stamp()
% MAKE_RUN_STAMP - Generate timestamp for output directories
% Returns: string like "2025-11-14_15-30-45"

stamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
end