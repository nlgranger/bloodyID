%% Parameters

h     = 35;
ratio = 2.3;
w     = round(ratio*h);
nMatching    = 4; % number of matching pairs for each image
nNonMatching = 6; % number of non matching pairs for each image
patchSz      = [19 19];
overlap      = [11 11];

%% load initial data

% load('data/hk_original/dataset.mat');
% dataset.pretrain_x = single(dataset.pretrain_x);
% dataset.train_x{1} = single(dataset.train_x{1});
% dataset.train_x{2} = single(dataset.train_x{2});
% dataset.test_x{1}  = single(dataset.test_x{1});
% dataset.test_x{2}  = single(dataset.test_x{2});

%% Extraction layers

extractionNet = MultiLayerNet(struct());

CNNtrainOpts = struct('lRate', 1e-5);
cnn = CNN([h w], [8 10], 24, CNNtrainOpts, 'pool', [4 4]);
extractionNet.add(cnn);

outSz = extractionNet.outsize();
cnn   = CNN(outSz, [2 3], 5, CNNtrainOpts, 'pool', [2 4]);
extractionNet.add(cnn);

concat = ReshapeNet(extractionNet, prod(extractionNet.outsize()));
extractionNet.add(concat);

%% Comparison layers

trainOpts  = struct('nIter', 50, ...
                    'batchSz', 300, ...
                    'displayEvery', 1);
wholeNet   = MultiLayerNet(trainOpts);
compareNet = SiameseNet(extractionNet, 2, 'skipPretrain');
wholeNet.add(compareNet);

l2 = L2Compare(60);
wholeNet.add(l2);

%% Training

wholeNet.train(dataset.train_x, 5*(~dataset.train_y)');

o = wholeNet.compute(dataset.test_x);
m = o < 2 ~= dataset.test_y';
mean(m(dataset.test_y))
mean(m(~dataset.test_y))
o = wholeNet.compute(dataset.train_x);
m = o < 2 ~= dataset.train_y';
mean(m(dataset.train_y))
mean(m(~dataset.train_y))