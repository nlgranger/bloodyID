function dataset = make_dataset(dbPath, sz, partition, pairing, varargin)
% MAKE_DATASET loads Hong Kong Polytechnic University Finger Image Database
%   dataset = MAKE_DATASET(path, [h w], [train val], [np nn]) performs the 
%   following operations:
%   - load images under FingerVein/ directory found in provided path
%   - cut a ROI centered between the intermediate and distal phalanges, 
%     images are then resized to h x w
%   - remove background color variations and slightly improve contrast
%   - assign individuals into categories following the proportions train and
%     val, remaining individuals go to testing
%   - build positive and negative pairs of images inside each partition, 
%     np specifies the number of images of session 2 to associate to each image
%     from session 1 when generating positive pairs and nn the number of 
%     negative ones.
%
%   dataset is a structure with the following fields:
%   TODO
% 
%   dataset = MAKE_DATASET(path, [h w], [train val], [np nn], 'nFolds', n) 
%   TODO


% Notes for this file ------------------------------------------------------- %
% X is a h*w*N array with the N loaded images. 
% I is an N long vector of individual ids mapped to the dataset ones for
% finger 1 and offset by 156 for finger 2.
% S contain the session number, 1 or 2 for the original images or 3 and 4
% for extra images generated from session 1 and 2 respectively.


% Parameters ---------------------------------------------------------------- %

h            = sz(1);
w            = sz(2);
ratio        = w/h;
trainRatio   = partition(1);
valRatio     = partition(2);
nMatching    = pairing(1);
nNonMatching = pairing(2);

assert(mod(numel(varargin), 2) == 0, 'options should be ''name'', value pairs');
for i = 1:2:numel(varargin)
    if strcmp(varargin{i}, 'nExtra')
        nExtra = varargin{i+1};
        angleStd     = 1.5; % angle std deviation in degrees
        shiftStd     = 2;   % shift std deviation in pixels
    elseif strcmp(varargin{i}, 'nFolds')
        nFolds = varargin{i+1};
    elseif strcmp(varargin{i}, 'preprocessed')
        preprocessed = varargin{i+1};
    else
        error('unknown option ''%s''', varargin{i});
    end
end


% Image Loading ------------------------------------------------------------- %

fprintf(1, 'Loading image files ...\n');
if ~(exist('preprocessed', 'var') && exist(preprocessed, 'file'))
    if exist(fullfile(dbPath, 'raw.mat'), 'file')
        load(fullfile(dbPath, 'raw.mat'));
    else
        [X, I, S] = load_hk_original(dbPath);
        save(fullfile(dbPath, 'raw.mat'), 'X', 'I', 'S');
    end
end


% Image Preprocessing ------------------------------------------------------- %

if ~(exist('preprocessed', 'var') && exist(preprocessed, 'file'))
    fprintf(1, 'Extracting fingers ...\n');
    raw = X;
    X = zeros(h, w, size(raw, 3));
    M = true(h, w, size(raw, 3));
    keep = true(size(raw, 3), 1);
    for i = 1:size(raw, 3)
        [O, Ma] = fingerExtraction(raw(:,:,i), 150, ratio);
        if numel(O) == 0
            warning('rejected finger %d', i);
            keep(i) = false;
        else
            X(:,:,i) = imresize(O, [h w]);
            M(:,:,i) = imresize(Ma, [h w]);
        end
    end
    rejected = X(:,:, ~keep);%#ok
    X = X(:,:,keep);
    M = M(:,:,keep);
    S = S(keep);
    I = I(keep);
    fprintf(1, 'Saving ...\n');
    save(preprocessed, 'M', 'X', 'I', 'S', 'rejected');
else
    load(preprocessed);
end


% Partitionning ------------------------------------------------------------- %

fprintf(1, 'Assigning caterogries ...\n');

% move pretraining samples apart
preOnly = false(numel(I), 1);
for id = 1:312
    fromS1 = find(I == id & S == 1, 6);
    fromS2 = find(I == id & S == 2, 6);
    
    if isempty(fromS1) || numel(fromS2) < nMatching || id == 35
        preOnly([fromS1; fromS2]) = 4;
        nPretrain = nPretrain + numel(fromS1) + numel(fromS2);
    end
end

dataset.pretrain_x = X(:,:, preOnly);

% Distribute ids into categories
uniqId = unique(I(~preonly));
uniqId = uniqId(randperm(numel(uniqId)));
if isset('nFolds', 'var') % cross validation
    nTrain   = round(numel(uniqId) * trainRatio);
    
    % move test samples aside
    testIds  = uniqId(nTrain + 1:end);
    uniqId   = uniqId(1:nTrain);
    
    trainIds = cell(nFolds, 1);
    valIds   = cell(nFolds, 1);
    batchSz  = round(nTrain / nFolds);
    for i = 1:nFolds
        val = false(numel(uniqIds));
        val((i-1)*batchSz+1:min(i*batchSz, end)) = true;
        trainIds{i} = uniqIds(val);
        valIds{i}   = uniqIds(~val);
    end
else % simple validation
    nTrain   = round(numel(uniqId) * trainRatio);
    trainIds = uniqId(1:nTrain);
    nVal     = round(numel(uniqId) * valRatio);
    valIds   = uniqId(nTrain+1:nTrain+nVal);
    testIds  = uniqId(nTrain+nVal+1:end);
end


% Extra artificial samples generation --------------------------------------- %

fprintf(1, 'Generating artificial images ...\n');

idx   = find(ismember(I, uniqId));
Xtra  = zeros(h ,w, nExtra*numel(idx));
Ixtra = zeros(nExtra*numel(idx), 1, 'uint32');
Sxtra = zeros(nExtra*numel(idx), 1, 'uint32');
for k = 1:numel(idx)
    for l = 1:nExtra
        n           = (k-1)*nExtra+l;
        shift       = round(randn(2, 1) * angleStd);
        a           = randn() * shiftStd;
        tmp         = imrotate(imshift(X(:,:,idx(k)), shift, 1), a, 'bilinear', 'crop');
        tmpM        = imrotate(imshift(M(:,:,idx(k)), shift, 1), a, 'bilinear', 'crop');
        tmp(~tmpM)  = 1;
        Xtra(:,:,n) = tmp;
        Ixtra(n)    = I(idx(k));
        Sxtra(n)    = S(idx(k))+2;
    end
end

X = cat(3, X, Xtra);
I = [I; Ixtra];
S = [S; Sxtra];


% Pairing ------------------------------------------------------------------- %

fprintf(1, 'Building pairs ...\n');
[dataset.test_x, dataset.test_y] = make_pairs(I, S, testIds);

if isset('nFolds', 'var') % cross validation
    S(S==3) = 1; S(S==4)=2; % merge normal and extra samples for training
    dataset.train_x = cell(numel(trainIds), 1);
    for i = 1:numel(trainIds)
        [dataset.train_x{i}, dataset.train_y{i}] = ...
            make_pairs(I, S, trainIds{i}, nMatching, nNonMatching);
        [dataset.val_x{i}, dataset.val_y{i}] = ...
            make_pairs(I, S, valIds{i}, nMatching, nNonMatching);
    end
else % simple validation
    [dataset.val_x, dataset.val_y] = ...
            make_pairs(I, S, valIds, nMatching, nNonMatching);
    S(S==3) = 1; S(S==4)=2; % merge normal and extra samples for training
    [dataset.train_x, dataset.train_y] = ...
            make_pairs(I, S, trainIds, nMatching, nNonMatching);
end

% Packing up ---------------------------------------------------------------- %

fprintf(1, 'Packing up dataset ...\n');
dataset.X = X;
dataset.h = h;
dataset.w = w;

fprintf(1, 'Done.\n');
end