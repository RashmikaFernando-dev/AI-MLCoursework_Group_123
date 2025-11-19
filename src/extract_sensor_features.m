function [XF, yUser, sessTag] = extract_sensor_features(ALL, winN, stepN)
% EXTRACTALLFEATURES - Advanced feature extraction with 30+ statistical features
% Returns:
%   XF      - feature matrix [num_windows x num_features] (30+ features)
%   yUser   - user label per window
%   sessTag - 'day1'/'day2' per window

XF=[]; yUser=[]; sessTag=strings(0,1);

U = unique(ALL.User);
for u = U.'
    rowsU = (ALL.User==u);
    S = unique(ALL.Session(rowsU));
    for s = S.'
        seg = ALL(rowsU & ALL.Session==s, :);
        n = height(seg);
        for i = 1:stepN:(n - winN + 1)
            W = seg(i:i+winN-1, :);

            % vector magnitudes
            acc_mag  = sqrt(W.acc_x.^2 + W.acc_y.^2 + W.acc_z.^2);
            gyro_mag = sqrt(W.gyro_x.^2 + W.gyro_y.^2 + W.gyro_z.^2);

            % Comprehensive features (time-domain + correlations + entropy)
            f = [ ...
              mean(W.acc_x) std(W.acc_x) rms(W.acc_x) ...
              mean(W.acc_y) std(W.acc_y) rms(W.acc_y) ...
              mean(W.acc_z) std(W.acc_z) rms(W.acc_z) ...
              mean(W.gyro_x) std(W.gyro_x) rms(W.gyro_x) ...
              mean(W.gyro_y) std(W.gyro_y) rms(W.gyro_y) ...
              mean(W.gyro_z) std(W.gyro_z) rms(W.gyro_z) ...
              mean(acc_mag)  std(acc_mag)  rms(acc_mag)  sum(abs(acc_mag))/numel(acc_mag) ...
              mean(gyro_mag) std(gyro_mag) rms(gyro_mag) ...
              corrSafe(W.acc_x,W.acc_y) corrSafe(W.acc_x,W.acc_z) corrSafe(W.acc_y,W.acc_z) ...
              corrSafe(W.gyro_x,W.gyro_y) corrSafe(W.gyro_x,W.gyro_z) corrSafe(W.gyro_y,W.gyro_z) ...
              H(acc_mag) H(gyro_mag) ];

            XF      = [XF; f]; %#ok<AGROW>
            yUser   = [yUser; u]; %#ok<AGROW>
            sessTag = [sessTag; string(s)]; %#ok<AGROW>
        end
    end
end
end

% --- Helper functions ---
function r=corrSafe(a,b)
    if numel(a)<3 || std(a)==0 || std(b)==0, r=0; else, R=corrcoef(a,b); r=R(1,2); end
end

function h=H(x)
    edges = linspace(min(x), max(x)+eps, 21);
    p = histcounts(x, edges, 'Normalization','probability'); p=p(p>0);
    h = -sum(p.*log2(p));
end