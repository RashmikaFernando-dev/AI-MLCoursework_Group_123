function T = readOneFile(fpath)
% READONEFILE - Robust reader for headerless CSV motion logs
% Resolves to 6 signal columns: [acc_x acc_y acc_z gyro_x gyro_y gyro_z]
% Supports filename formats: U{id}NW_FD.csv (day1) or U{id}NW_MD.csv (day2)

[~, fname, ext] = fileparts(fpath);

% Parse user + session from filename like U10NW_FD.csv / U10NW_MD.csv
tokU = regexp(fname, '^U(\d+)', 'tokens', 'once');
assert(~isempty(tokU), 'Cannot parse user from: %s', fname);
userID = str2double(tokU{1});

tokS = regexp(fname, '_(F|M)D', 'tokens', 'once');
assert(~isempty(tokS), 'Cannot parse session (FD/MD) from: %s', fname);
if strcmpi(tokS{1}, 'F')
    sess = "day1";  % FD = First Day
elseif strcmpi(tokS{1}, 'M')
    sess = "day2";  % MD = Middle Day
else
    error('Unknown session in %s', fname);
end

% Read numeric matrix
switch lower(ext)
    case {'.csv','.txt'},  M = readmatrix(fpath);
    case {'.xlsx','.xls'}, M = readmatrix(fpath,'OutputType','double');
    otherwise, error('Unsupported file: %s', fpath);
end
assert(~isempty(M) && isnumeric(M), 'File not numeric: %s', fpath);

% Clean columns: drop NaN-only & constant columns
nanCols = all(isnan(M),1);
M(:, nanCols) = [];

colStd = std(M, 0, 1, 'omitnan');
constCols = colStd < 1e-12;
M(:, constCols) = [];

% Detect monotonic columns (likely time columns)
isMonotonic = false(1,size(M,2));
for c = 1:size(M,2)
    v = M(:,c);
    v = v(~isnan(v));
    if numel(v) > 1 && (all(diff(v) >= 0) || all(diff(v) <= 0))
        isMonotonic(c) = true;
    end
end

% Remove likely time columns if too many columns
if size(M,2) > 6
    cand = find(isMonotonic);
    if ~isempty(cand)
        ranges = range(M(:,cand), 1);
        [~,ix] = max(ranges);
        M(:, cand(ix)) = [];  % remove time column with largest range
    end
end

% Keep last 6 columns if still too many
if size(M,2) > 6
    M = M(:, end-5:end);
end

% Handle 7-column case (likely has time as first column)
if size(M,2) == 7
    first = M(:,1);
    if all(abs(first) < 1e-8) || isMonotonic(1)
        M = M(:,2:end);  % remove first column
    end
end

assert(size(M,2)==6, 'Could not resolve to 6 signal columns in %s (found %d).', fpath, size(M,2));

% Map columns to accelerometer and gyroscope data
acc_x  = M(:,1); acc_y  = M(:,2); acc_z  = M(:,3);
gyro_x = M(:,4); gyro_y = M(:,5); gyro_z = M(:,6);

% Create labeled table
N = size(M,1);
T = table((0:N-1)', acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, ...
          repmat(userID,N,1), repmat(sess,N,1), ...
    'VariableNames', {'time','acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z','User','Session'});
end