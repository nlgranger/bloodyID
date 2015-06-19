h = 32;
w = 75;
nMatching     = 3; % number of matching pairs for each image
nNonMatching  = 6; % number of non matching pairs for each image
patchSz       = [15 20];
overlap       = [11 16];

%% load initial data

% fill a dataset object with fields 
% train_x : training samples pairs, nbsamples x (2*h*w) matrix
% train_y : associated label
% test_x  : same as train_x for testing
% test_y  : same as train_y for testing
% h       : height of the images
% w       : width of the images

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
    'lRate', 0.05, ...
    'momentum', 0.9, ...
    'nEpochs', 30, ...
    'batchSz', 300, ...
    'dropoutHid', 0.4, ...
    'displayEvery', 5);
rbm = RBM(prod(patchSz), 60, RBMpretrainingOpts, RBMtrainOpts);
patchRedux.add(rbm);

rbm = RBM(60, 25, RBMpretrainingOpts, RBMtrainOpts);
patchRedux.add(rbm);

imRedux = SiameseNet(patchRedux, numel(patchMaker.outsize()));
extractionNet.add(imRedux);

patchMerge = ReshapeNet(imRedux, sum(cellfun(@prod, imRedux.outsize())));
extractionNet.add(patchMerge);

RBMtrainOpts = struct('lRate', 0.05);
RBMpretrainingOpts = struct( ...
    'lRate', 0.07, ...
    'momentum', 0.9, ...
    'nEpochs', 30, ...
    'batchSz', 300, ...
    'dropoutHid', 0.4, ...
    'displayEvery', 5);
rbm = RBM(25 * numel(patchMaker.outsize()), 200, RBMpretrainingOpts, RBMtrainOpts);
extractionNet.add(rbm);
rbm = RBM(200, 100, RBMpretrainingOpts, RBMtrainOpts);
extractionNet.add(rbm);

extractionNet.pretrain(dataset.pretrain_x);

save('data/workspaces/pretrained.mat', 'extractionNet');

%% Comparison layers

trainOpts = struct('nIter', 150, ...
                   'batchSz', 300, ...
                   'displayEvery', 1);

wholeNet  = MultiLayerNet(trainOpts);

% Duplicate extraction network
compareNet = SiameseNet(extractionNet, 2, 'skipPretrain');
wholeNet.add(compareNet);

cosine = L2Compare(100);
wholeNet.add(cosine);

%% Training

o = wholeNet.compute(dataset.test_x);
m = o < 0.53 ~= dataset.test_y';
mean(m(dataset.test_y))
mean(m(~dataset.test_y))
o = wholeNet.compute(dataset.train_x);
m = o < .53 ~= dataset.train_y';
mean(m(dataset.train_y))
mean(m(~dataset.train_y))

wholeNet.train(dataset.train_x, ~dataset.train_y');

o = wholeNet.compute(dataset.test_x);
m = o < 0.5 ~= dataset.test_y';
mean(m(dataset.test_y))
mean(m(~dataset.test_y))
o = wholeNet.compute(dataset.train_x);
m = o < .5 ~= dataset.train_y';
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