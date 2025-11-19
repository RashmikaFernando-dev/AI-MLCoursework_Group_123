function [x_f, y_f, z_f] = apply_sensor_filter(x, y, z, fs, cutoffHz)
% LPFILTERTRIAD - Low-pass filter for 3-axis sensor data (toolbox-free)
% Zero-phase low-pass using forward-backward moving-average (boxcar) filter
% Approximates a cutoff near 'cutoffHz' without Signal Processing Toolbox
%
% Inputs:
%   x, y, z     - 3-axis sensor data vectors
%   fs          - sampling frequency (Hz)
%   cutoffHz    - desired cutoff frequency (Hz)
%
% Outputs:
%   x_f, y_f, z_f - filtered 3-axis data

% Choose window length L so that the -3 dB frequency of a boxcar
% (~0.443*fs/L) is close to cutoffHz  =>  L ≈ 0.443*fs/cutoffHz
L = max(3, round(0.443 * fs / max(cutoffHz, eps)));

% Ensure odd L for symmetric impulse response (better zero-phase)
if mod(L,2)==0, L = L + 1; end

b = ones(L,1) / L;   % FIR coefficients (boxcar filter)
a = 1;

% Apply zero-phase filtering to each axis
x_f = zeroPhaseMA(x, b, a);
y_f = zeroPhaseMA(y, b, a);
z_f = zeroPhaseMA(z, b, a);
end

function y = zeroPhaseMA(x, b, a)
% Zero-phase forward-backward filtering (manual implementation)
% Equivalent to filtfilt but without toolbox dependency
y = filter(b, a, x);    % Forward filtering
y = flipud(y);          % Reverse
y = filter(b, a, y);    % Backward filtering
y = flipud(y);          % Reverse back to original order
end