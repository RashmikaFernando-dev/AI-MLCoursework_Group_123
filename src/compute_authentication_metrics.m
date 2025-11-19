function [EER, FRR_atEER, FAR_atEER, thr_atEER, fpr_micro, tpr_micro] = compute_authentication_metrics(scores, true_idx)
% COMPUTEFAR_FRR_EER - Compute comprehensive authentication metrics
% 
% Inputs:
%   scores (N x K)   - network probabilities per class
%   true_idx (N x 1) - true class index (1..K)
%
% Outputs:
%   EER              - Equal Error Rate
%   FRR_atEER        - False Rejection Rate at EER
%   FAR_atEER        - False Acceptance Rate at EER  
%   thr_atEER        - Threshold at EER
%   fpr_micro, tpr_micro - ROC curve points

[N,K] = size(scores);

% Extract genuine and impostor scores
lin = sub2ind([N K], (1:N)', true_idx(:));
genuine = scores(lin);                    % genuine scores (correct class)
mask = true(size(scores)); mask(lin) = false;
impostor = scores(mask);                  % impostor scores (all others)

% Compute FAR and FRR across threshold range
ths = linspace(0, 1, 400); 
FAR = zeros(size(ths)); 
FRR = FAR;

for i = 1:numel(ths)
    t = ths(i);
    FRR(i) = mean(genuine < t);           % genuine rejected
    FAR(i) = mean(impostor >= t);         % impostor accepted
end

% Find EER (where FAR ≈ FRR)
[~, ix] = min(abs(FAR - FRR));
EER = 0.5 * (FAR(ix) + FRR(ix));
thr_atEER = ths(ix); 
FAR_atEER = FAR(ix); 
FRR_atEER = FRR(ix);

% Generate ROC curve data
y_scores = [genuine; impostor];
y_labels = [ones(numel(genuine),1); zeros(numel(impostor),1)];

% Use perfcurve if available, otherwise simple approximation
if exist('perfcurve', 'file')
    [fpr_micro, tpr_micro] = perfcurve(y_labels, y_scores, 1);
else
    % Simple ROC approximation
    sorted_scores = sort(y_scores, 'descend');
    thresholds = [inf; sorted_scores; -inf];
    fpr_micro = zeros(length(thresholds), 1);
    tpr_micro = zeros(length(thresholds), 1);
    
    for i = 1:length(thresholds)
        predictions = y_scores >= thresholds(i);
        tp = sum(predictions & y_labels);
        fp = sum(predictions & ~y_labels);
        fn = sum(~predictions & y_labels);
        tn = sum(~predictions & ~y_labels);
        
        tpr_micro(i) = tp / (tp + fn);
        fpr_micro(i) = fp / (fp + tn);
    end
end
end