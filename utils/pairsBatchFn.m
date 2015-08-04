function [batchX, batchY, idx] = pairsBatchFn(X, Y, batchSz, idx)
    nSamples = size(X.pairs, 1);
    
    if isempty(idx) % first mini-batch
        idx = randperm(nSamples);
    end
    batchIdx = idx(1:min(batchSz, end));
    batchX   = {X.data(:, :, X.pairs(batchIdx, 1)), ...
                X.data(:, :, X.pairs(batchIdx, 2))};
    batchY   = Y(idx(1:min(batchSz, end)));
    
    if numel(idx) > batchSz
        idx = idx(batchSz + 1:end);
    else
        idx = [];
    end
end