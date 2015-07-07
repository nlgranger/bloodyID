%% Parameters

h     = 35;
ratio = 2.3;
w     = round(ratio*h);
nMatching    = 4; % number of matching pairs for each image
nNonMatching = 6; % number of non matching pairs for each image

%% load initial data
% CNN requires single precision input!
if exist('data/hk_original/dataset.mat', 'file')
    load('data/hk_original/dataset.mat');
else
    dataset = make_dataset('data/hk_original', [35 81], [0.7 0], [4 6], ...
        'preprocessed', 'data/hk_original/preprocessed.mat');
    dataset.X = -single(dataset.X);
    save('data/hk_original/dataset.mat', 'dataset');
end

%% Extraction layers

extractionNet = MultiLayerNet(struct());

trainOpts = struct('lRate', 1e-5, 'dropout', .2);
cnn = CNN([h w], [8 10], 20, trainOpts, 'pool', [4 4]);
extractionNet.add(cnn);

trainOpts = struct('lRate', 5e-6, 'dropout', .1);
outSz = extractionNet.outsize();
cnn   = CNN(outSz, [2 3], 5, trainOpts, 'pool', [2 2]);
extractionNet.add(cnn);

concat = ReshapeNet(extractionNet, prod(extractionNet.outsize()));
extractionNet.add(concat);

rbm = RELURBM(extractionNet.outsize(), 120, struct(), trainOpts);
extractionNet.add(rbm);

%% Comparison layers

trainOpts  = struct('nIter', 80, ...
                    'batchSz', 300, ...
                    'displayEvery', 5);
wholeNet   = MultiLayerNet(trainOpts);
compareNet = SiameseNet(extractionNet, 2, 'skipPretrain');
wholeNet.add(compareNet);

l2 = L2Compare(120);
wholeNet.add(l2);

%% Training
X = {dataset.X(:,:,dataset.train_x(:,1)), dataset.X(:,:,dataset.train_x(:,2))};
Y = 3*(~dataset.train_y)';
Xt = {dataset.X(:,:,dataset.test_x(:,1)), dataset.X(:,:,dataset.test_x(:,2))};
Yt = 3*(~dataset.test_y)';

wholeNet.train(X, Y);


o = wholeNet.compute(X)';
eer = fminsearch(@(t) abs(mean(o(dataset.train_y)<t) - mean(o(~dataset.train_y)>t)), 1.5);
m = (o > eer) == dataset.train_y;
mean(m(dataset.train_y))
mean(m(~dataset.train_y))

o = wholeNet.compute(Xt)';
m = (o > eer) == dataset.test_y;
mean(m(dataset.test_y))
mean(m(~dataset.test_y))

colormap gray
W = wholeNet.nets{1}.net.nets{1}.filters;
for i = 1:size(W, 4)
    imagesc(W(:,:,1,i));
    axis image;
    pause
end