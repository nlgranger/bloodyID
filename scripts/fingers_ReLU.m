

h     = 35;
ratio = 2.3;
w     = round(ratio*h);
nMatching    = 4; % number of matching pairs for each image
nNonMatching = 6; % number of non matching pairs for each image
patchSz      = [10 23];
overlap      = [5 11];

%% load initial data

% load('data/hk_original/dataset.mat');
% dataset.pretrain_x = 0.5 * (dataset.pretrain_x + 1);
% dataset.train_x{1} = 0.5 * (dataset.train_x{1} + 1);
% dataset.train_x{2} = 0.5 * (dataset.train_x{2} + 1);
% dataset.test_x{1}  = 0.5 * (dataset.test_x{1} + 1);
% dataset.test_x{2}  = 0.5 * (dataset.test_x{2} + 1);

%% Extraction layers

extractionNet = MultiLayerNet(struct('skipBelow', 1));

patchMaker = PatchNet([h, w], patchSz, overlap);
extractionNet.add(patchMaker);

patchRedux = MultiLayerNet(struct());
RBMtrainOpts = struct('lRate', 0.05);
RBMpretrainingOpts = struct( ...
    'lRate', 0.00001, ...
    'momentum', 0.5, ...
    'nEpochs', 90, ...
    'batchSz', 600, ...
    'displayEvery', 5);
rbm = RELURBM(prod(patchSz), 100, RBMpretrainingOpts, RBMtrainOpts);
patchRedux.add(rbm);

% RBMtrainOpts = struct('lRate', 1);
% rbm = RELURBM(200, 70, RBMpretrainingOpts, RBMtrainOpts);
% patchRedux.add(rbm);

imRedux = SiameseNet(patchRedux, numel(patchMaker.outsize()));
extractionNet.add(imRedux);

patchMerge = ReshapeNet(imRedux, sum(cellfun(@prod, imRedux.outsize())));
extractionNet.add(patchMerge);

RBMtrainOpts = struct('lRate', 0.05);
RBMpretrainingOpts = struct( ...
    'lRate', 0.00001, ...
    'momentum', 0.5, ...
    'nEpochs', 70, ...
    'batchSz', 300, ...
    'displayEvery', 5);
rbm = RELURBM(100 * numel(patchMaker.outsize()), 200, RBMpretrainingOpts, RBMtrainOpts);
extractionNet.add(rbm);
% RBMtrainOpts = struct('lRate', 1);
% rbm = RELURBM(300, 150, RBMpretrainingOpts, RBMtrainOpts);
% extractionNet.add(rbm);

extractionNet.pretrain(dataset.pretrain_x);

save('data/workspaces/pretrained.mat', 'extractionNet');

%% Comparison layers

trainOpts = struct('nIter', 50, ...
                   'batchSz', 300, ...
                   'displayEvery', 1);

wholeNet  = MultiLayerNet(trainOpts);

% Duplicate extraction network
compareNet = SiameseNet(extractionNet, 2, 'skipPretrain');
wholeNet.add(compareNet);

cosine = CosineCompare(100);
wholeNet.add(cosine);

%% Training

% o = wholeNet.compute(dataset.test_x);
% m = o > 0.47 ~= dataset.test_y';
% mean(m(dataset.test_y))
% mean(m(~dataset.test_y))
% o = wholeNet.compute(dataset.train_x);
% m = o > .47 ~= dataset.train_y';
% mean(m(dataset.train_y))
% mean(m(~dataset.train_y))

wholeNet.train(dataset.train_x, dataset.train_y');

o = wholeNet.compute(dataset.test_x);
m = o > 0.48 ~= dataset.test_y';
mean(m(dataset.test_y))
mean(m(~dataset.test_y))
o = wholeNet.compute(dataset.train_x);
m = o > .48 ~= dataset.train_y';
mean(m(dataset.train_y))
mean(m(~dataset.train_y))

%% Testing

% Xa = wholeNet.compute(dataset.train_x);
% Xt = wholeNet.compute(dataset.test_x);
% 
% SVMModel = fitcsvm(Xa',dataset.train_y,'KernelFunction','rbf', 'OutlierFraction',0.05);
% y = predict(SVMModel,Xa');
% m = y ~= dataset.train_y;
% fprintf(1, 'Training fpr : %f\n', mean(m(~dataset.train_y)));
% fprintf(1, '         frr : %f\n', mean(m(dataset.train_y)));
% y = predict(SVMModel,Xt');
% m = y ~= dataset.test_y;
% fprintf(1, 'Testing  fpr : %f\n', mean(m(~dataset.test_y)));
% fprintf(1, '         frr : %f\n', mean(m(dataset.test_y)));