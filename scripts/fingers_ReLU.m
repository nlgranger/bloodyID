%% Parameters

h            = 35;
ratio        = 2.3;
w            = round(ratio*h);
nMatching    = 4; % number of matching pairs for each image
nNonMatching = 6; % number of non matching pairs for each image
patchSz      = [19 19];
overlap      = [11 11];
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
    'nEpochs', 150, ...
    'batchSz', 400, ...
    'selectivity', 10, ...
    'selectAfter', 25, ...
    'displayEvery', 10);
RBM1trainOpts = struct( ...
    'lRate', 3e-4, ...
    'dropout', 0.5);

% fusion RBM
RBM2pretrainingOpts = struct( ...
    'lRate', 1e-5, ...
    'momentum', 0.5, ...
    'nEpochs', 30, ...
    'batchSz', 400, ...
    'displayEvery', 5);
RBM2trainOpts = struct( ...
    'dropout', 0.4, ...
    'lRate', 3e-4);

RBM3pretrainingOpts = RBM2pretrainingOpts;
RBM3trainOpts = struct( ...
    'lRate', 3e-4);

%% Training

trainOpts = struct(...
    'nIter', 80, ...
    'batchSz', 400, ...
    'batchFn', @pairsBatchFn, ...
    'displayEvery', 5);

for i = 1:1%nFolds % Cross-validation loop
    extractionNet = MultiLayerNet();
    
    % Filter layers
    patchMaker    = PatchNet([h, w], patchSz, overlap);
    extractionNet.add(patchMaker);
    extractionNet.freezeBelow(1);% don't train patch extraction
    
    rbm = RELURBM(prod(patchSz), 150, ...
        RBM1pretrainingOpts, RBM1trainOpts, false);
    imRedux = SiameseNet(rbm, numel(patchMaker.outsize()));
    extractionNet.add(imRedux);
    
    % Pretraining
    pretrainIdx = unique([dataset.pretrain_x; ...
                dataset.train_x{i}(:,1); ...
                dataset.train_x{i}(:,2)]);
    extractionNet.pretrain(dataset.X(:,:, pretrainIdx));
    
    patchMerge = ReshapeNet(extractionNet, ...
        sum(cellfun(@prod, extractionNet.outsize())));
    extractionNet.add(patchMerge);

    % Dimension reduction layers
    rbm = RELURBM(extractionNet.outsize(), 120, ...
        RBM2pretrainingOpts, RBM2trainOpts, true);
    extractionNet.add(rbm);
    rbm = RELURBM(extractionNet.outsize(), 80, ...
        RBM3pretrainingOpts, RBM3trainOpts, true);
    extractionNet.add(rbm);
    
    % Combined network
    compareNet = SiameseNet(extractionNet, 2);
    wholeNet   = MultiLayerNet();
    metric     = L2Compare(extractionNet.outsize());
    wholeNet.add(compareNet);
    wholeNet.add(metric);
    
    % Training
    X = struct('data', dataset.X, 'pairs', dataset.train_x{i});
    Y = 2*(~dataset.train_y{i})';
    train(wholeNet, @L2Cost, X, Y, trainOpts);
    r = zeros(1, 4);
    
    % Training performances
    [allX, allY] = trainOpts.batchFn(X, Y, inf, []);
    o = wholeNet.compute(allX);
    eer = fminsearch(@(t) abs(mean(o(allY == 0)<t) - mean(o(allY > 0)>=t)), 1);
    m = (o > eer) ~= (allY > 0);
    r(1) = mean(m(allY > 0));
    r(2) = mean(m(allY == 0));

    % Validation performances
    X = struct('data', dataset.X, 'pairs', dataset.val_x{i});
    Y = single((~dataset.val_y{i}))';
    [allX, allY] = trainOpts.batchFn(X, Y, inf, []);
    o = wholeNet.compute(allX);
    m = (o > eer) ~= (allY > 0);
    r(3) = mean(m(allY > 0));
    r(4) = mean(m(allY == 0));
    disp(i);
    disp(r);
end

%% Testing

