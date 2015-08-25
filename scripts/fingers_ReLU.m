%% Parameters

h            = 35;
ratio        = 2.3;
w            = round(ratio*h);
nMatching    = 4; % number of matching pairs for each image
nNonMatching = 7; % number of non matching pairs for each image
patchSz      = [19 19];
overlap      = [15 15];
nFolds       = 10;

%% load initial data

if exist('data/hk_original/dataset_small_pos.mat', 'file')
    load('data/hk_original/dataset_small_pos.mat');
else
    dataset = make_dataset('data/hk_original', ...
        [h w], [0.7 0], [nMatching nNonMatching], ...
        'preprocessed', 'data/hk_original/preprocessed_small.mat', ...
        'nFolds', nFolds);
    dataset.X = 1 - 0.5 * (dataset.X + 1);
    save('data/hk_original/dataset_small_pos.mat', 'dataset');
end

%% Extraction layers

% Patch filter RBM
RBM1pretrainingOpts = struct( ...
    'lRate', 1e-5, ...
    'momentum', 0.5, ...
    'nEpochs', 100, ...
    'batchSz', 400, ...
    'dropVis', 0.3, ...
    'selectivity', 10, ...
    'selectAfter', 25, ...
    'displayEvery', 10);
RBM1trainOpts = struct( ...
    'lRate', 1e-3);

% fusion RBM
RBM2pretrainingOpts = struct( ...
    'lRate', 1e-6, ...
    'momentum', 0.5, ...
    'nEpochs', 60, ...
    'batchSz', 400, ...
    'dropVis', 0.3, ...
    'displayEvery', 10);
RBM2trainOpts = struct( ...
    'dropout', 0.1, ...
    'lRate', 1e-3);

RBM3pretrainingOpts = struct();
RBM3trainOpts = struct( ...
    'dropout', 0.2, ...
    'lRate', 1e-3);

RBM4pretrainingOpts = struct();
RBM4trainOpts = struct( ...
    'dropout', 0.3, ...
    'lRate', 1e-3);

%% Training

trainOpts = struct(...
    'nIter', 50, ...
    'batchSz', 400, ...
    'batchFn', @pairsBatchFn, ...
    'displayEvery', 3);

for i = 1:1%nFolds % Cross-validation loop
%     extractionNet = MultiLayerNet();
%     
%     % Filter layers
%     patchMaker    = PatchNet([h, w], patchSz, overlap);
%     extractionNet.add(patchMaker);
%     extractionNet.freezeBelow(1);% don't train patch extraction
%     
%     rbm = RELURBM(prod(patchSz), 50, ...
%         RBM1pretrainingOpts, RBM1trainOpts, false);
%     imRedux = SiameseNet(rbm, numel(patchMaker.outsize()));
%     extractionNet.add(imRedux);
%     
%     % Pretraining for bottom layers
%     pretrainIdx = unique([dataset.pretrain_x; ...
%                 dataset.train_x{i}(:,1); ...
%                 dataset.train_x{i}(:,2)]);
%     extractionNet.pretrain(dataset.X(:,:, pretrainIdx));
%     
%     patchMerge = ReshapeNet(extractionNet, ...
%         sum(cellfun(@prod, extractionNet.outsize())));
%     extractionNet.add(patchMerge);
% 
%     % Dimension reduction layers
%     rbm = RELURBM(extractionNet.outsize(), 500, ...
%         RBM2pretrainingOpts, RBM2trainOpts, true);
%     extractionNet.add(rbm);
%     
%     rbm = RELURBM(extractionNet.outsize(), 200, ...
%         RBM2pretrainingOpts, RBM3trainOpts, true);
%     extractionNet.add(rbm);
%     rbm = RELURBM(extractionNet.outsize(), 200, ...
%         RBM3pretrainingOpts, RBM4trainOpts, true);
%     extractionNet.add(rbm);
    
    extractionNet.nets{2}.net.trainOpts.lRate   = RBM1trainOpts.lRate;
    extractionNet.nets{4}.trainOpts.lRate       = RBM2trainOpts.lRate;
    extractionNet.nets{5}.trainOpts.lRate       = RBM3trainOpts.lRate;
    extractionNet.nets{5}.trainOpts.lRate       = RBM4trainOpts.lRate;
    %extractionNet.nets{2}.net.trainOpts.dropout = RBM1trainOpts.dropout;
    extractionNet.nets{4}.trainOpts.dropout     = RBM2trainOpts.dropout;
    extractionNet.nets{5}.trainOpts.dropout     = RBM3trainOpts.dropout;
    extractionNet.nets{5}.trainOpts.lRate       = RBM4trainOpts.lRate;

    % Combined network
    compareNet = SiameseNet(extractionNet, 2);
    wholeNet   = MultiLayerNet();
    metric     = JaccardDistance(extractionNet.outsize(), 0.1);
    wholeNet.add(compareNet);
    wholeNet.add(metric);
    
    % Training
    X = struct('data', dataset.X, 'pairs', dataset.train_x{i});
    X.data = X.data - 0.5;
    %Y = 0.8 * single(~dataset.train_y{i})' + 0.1;
    %Y = single(~dataset.train_y{i})';
    Y = ~dataset.train_y{i}';
    train(wholeNet, @contrastiveLoss, X, Y, trainOpts);
    r = zeros(1, 4);
    
    % Training performances
    [allX, allY] = trainOpts.batchFn(X, Y, inf, []);
    o = wholeNet.compute(allX);
    eer = fminsearch(@(t) abs(mean(o(allY < 0.5)<t) - mean(o(allY > 0.5)>=t)), double(mean(o)));
    r(1) = mean(o(allY < 0.5) > eer);
    r(2) = mean(o(allY > 0.5) < eer);

    % Validation performances
    Xv = struct('data', dataset.X, 'pairs', dataset.val_x{i});
    Xv.data = Xv.data - 0.5;
    %Yv = 0.8 * single(~dataset.val_y{i})' + 0.1;
    %Yv = single(~dataset.val_y{i})';
    Yv = ~dataset.val_y{i}';
    [allX, allY] = trainOpts.batchFn(Xv, Yv, inf, []);
    o = wholeNet.compute(allX);
    m = (o > eer) ~= (allY > 0);
    r(3) = mean(o(allY < 0.5) > eer);
    r(4) = mean(o(allY > 0.5) < eer);
    disp(i);
    disp(r);
end

%% Testing

