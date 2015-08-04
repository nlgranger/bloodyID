%% Parameters

h            = 35;
ratio        = 2.3;
w            = round(ratio*h);
nMatching    = 4; % number of matching pairs for each image
nNonMatching = 6; % number of non matching pairs for each image
patchSz      = [19 19];
overlap      = [11 11];
nFolds       = 7;

%% load initial data

if exist('data/hk_original/dataset_small.mat', 'file')
    load('data/hk_original/dataset_small.mat');
else
    dataset = make_dataset('data/hk_original', ...
        [h w], [0.5 0], [nMatching nNonMatching], ...
        'preprocessed', 'data/hk_original/preprocessed_small.mat', ...
        'nFolds', nFolds);
    dataset.X = 1 - 0.5 * (dataset.X + 1);
    save('data/hk_original/dataset_small.mat', 'dataset');
end

%% Extraction layers

extractionNet = MultiLayerNet();

patchMaker = PatchNet([h, w], patchSz, overlap);
extractionNet.add(patchMaker);
extractionNet.freezeBelow(1);

RBMtrainOpts = struct( ...
    'lRate', 2e-4, ...
    'dropout', 0.2);
RBMpretrainingOpts = struct( ...
    'lRate', 2e-5, ...
    'momentum', 0.5, ...
    'nEpochs', 120, ...
    'batchSz', 400, ...
    'displayEvery', 10);
rbm = RELURBM(prod(patchSz), 150, RBMpretrainingOpts, RBMtrainOpts, false);

imRedux = SiameseNet(rbm, numel(patchMaker.outsize()));
extractionNet.add(imRedux);

patchMerge = ReshapeNet(imRedux, sum(cellfun(@prod, imRedux.outsize())));
extractionNet.add(patchMerge);

RBMtrainOpts = struct( ...
    'lRate', 2e-4);
RBMpretrainingOpts = struct( ...
    'lRate', 2e-5, ...
    'momentum', 0.5, ...
    'nEpochs', 30, ...
    'batchSz', 400, ...
    'displayEvery', 5);
rbm = RELURBM(150 * numel(patchMaker.outsize()), 150, RBMpretrainingOpts, RBMtrainOpts, true);
extractionNet.add(rbm);

%% Training

trainOpts = struct('nIter', 100, ...
                   'batchSz', 400, ...
                   'batchFn', @pairsBatchFn, ...
                   'displayEvery', 3);

for i = 1:1%nFolds
    extraction = extractionNet.copy();
%     extraction.pretrain(dataset.X(:,:, ...
%         unique([dataset.pretrain_x; ...
%                 dataset.train_x{i}(:,1); ...
%                 dataset.train_x{i}(:,2)])));
    rbm = RELURBM(150, 100, RBMpretrainingOpts, RBMtrainOpts, true);
    extraction.add(rbm);

    compareNet = SiameseNet(extraction, 2, 'skipPretrain');
    wholeNet   = MultiLayerNet();
    metric     = CosineCompare(100);
    wholeNet.add(compareNet);
    wholeNet.add(metric);
    
    X = struct('data', dataset.X, 'pairs', dataset.train_x{i});
    Y = 2*(~dataset.train_y{i})';
    net = wholeNet.copy(); % start from random weights
    train(net, @L2Cost, X, Y, trainOpts);
    r = zeros(1, 4);
    
    % Compute mis-classification on training data
    [allX, allY] = trainOpts.batchFn(X, Y, inf, []);
    o = net.compute(allX);
    eer = fminsearch(@(t) abs(mean(o(allY == 0)<t) - mean(o(allY > 0)>=t)), 1);
    m = (o > eer) ~= (allY > 0);
    r(1) = mean(m(allY > 0));
    r(2) = mean(m(allY == 0));

    % Compute mis-classification on testing data
    X = struct('data', dataset.X, 'pairs', dataset.val_x{i});
    Y = single((~dataset.val_y{i}))';
    [allX, allY] = trainOpts.batchFn(X, Y, inf, []);
    o = net.compute(allX);
    m = (o > eer) ~= (allY > 0);
    r(3) = mean(m(allY > 0));
    r(4) = mean(m(allY == 0));
    disp(i);
    disp(r);
end

%% Testing

