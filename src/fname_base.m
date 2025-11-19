function base = fname_base(CONFIG, tag)
% FNAME_BASE - Generate standardized filename base for outputs
% Returns: string like "main_HS3216_FCNtrainscg_E200"

hs_str = sprintf("%d", CONFIG.hiddenSizes(1));
if numel(CONFIG.hiddenSizes) > 1
    for i = 2:numel(CONFIG.hiddenSizes)
        hs_str = hs_str + sprintf("%d", CONFIG.hiddenSizes(i));
    end
end

base = sprintf("%s_HS%s_FCN%s_E%d", tag, hs_str, CONFIG.trainFcn, CONFIG.maxEpochs);
end