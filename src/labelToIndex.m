function [ytr_idx, labels, yte_idx] = labelToIndex(ytr, yte)
% LABELTOINDEX - Convert categorical labels to numeric indices
% 
% Inputs:
%   ytr, yte - training and test labels (can be strings or numbers)
%
% Outputs:
%   ytr_idx, yte_idx - numeric indices starting from 1
%   labels           - unique label values

labels = unique([ytr; yte]);
ytr_idx = arrayfun(@(x) find(labels == x), ytr);
yte_idx = arrayfun(@(x) find(labels == x), yte);
end