%% Parameters

h            = 35;
ratio        = 2.3;
w            = round(ratio*h);
nMatching    = 4; % number of matching pairs for each image
nNonMatching = 6; % number of non matching pairs for each image
nFolds       = 7;

%% load initial data
% CNN requires single precision input!
if exist('data/hk_original/dataset_small.mat', 'file')
    load('data/hk_original/dataset_small.mat');
else
    dataset = make_dataset('data/hk_original', [35 81], [1 0], [4 6], ...
        'preprocessed', 'data/hk_original/preprocessed_small.mat', ...
        'nFolds', nFolds);
    dataset.X = -single(dataset.X);
    save('data/hk_original/dataset_small.mat', 'dataset');
end

%% Extraction layers

extractionNet = MultiLayerNet();

trainOpts = struct('lRate', 3e-6);
cnn = CNN([h w], [12 14], 16, trainOpts, 'pool', [4 4], 'dropout', 0.4);

L = [15, 18];
for j = 1:numel(L)
    l = L(j);
    for k = 1:4
        cnn.filters(:,:,1,(j-1)*4+k) = ...
            gaborfilter([12 14], [5 5], l, k * pi/4) / (8*10*16);
    end
end
extractionNet.add(cnn);

trainOpts = struct('lRate', 3e-6);
outSz     = extractionNet.outsize();
cnn       = CNN(outSz, [3 6], 8, trainOpts, 'pool', [2 2], 'dropout', 0.2);
extractionNet.add(cnn);

concat = ReshapeNet(extractionNet, prod(extractionNet.outsize()));
extractionNet.add(concat);

trainOpts = struct('lRate', 3e-7);
rbm = RELURBM(extractionNet.outsize(), 50, struct(), trainOpts);
extractionNet.add(rbm);

%% Comparison layers

wholeNet   = MultiLayerNet();
compareNet = SiameseNet(extractionNet, 2);
wholeNet.add(compareNet);

cosine = CosineCompare(100);
wholeNet.add(cosine);

%% Training

trainOpts = struct('nIter', 25, ...
                   'batchSz', 500, ...
                   'displayEvery', 5);

res = zeros(nFolds, 4, 8);
for i = 1:nFolds
    X = {dataset.X(:,:,dataset.train_x{i}(:,1)), dataset.X(:,:,dataset.train_x{i}(:,2))};
    Y = single((~dataset.train_y{i})');
    
    for j = 1:8
        
        net = train(wholeNet, @L2Cost, X, Y, trainOpts);

        r = zeros(1, 4);
        o = net.compute(X)';
        eer = fminsearch(@(t) abs(mean(o(dataset.train_y{i})<t) - mean(o(~dataset.train_y{i})>t)), 0.5);
        m = (o > eer) == dataset.train_y{i};
        r(1) = mean(m(dataset.train_y{i}));
        r(2) = mean(m(~dataset.train_y{i}));

        Xv = {dataset.X(:,:,dataset.val_x{i}(:,1)), dataset.X(:,:,dataset.val_x{i}(:,2))};
        o = net.compute(Xv)';
        clear Xv;
        
        m = (o > eer) == dataset.val_y{i};
        r(3) = mean(m(dataset.val_y{i}));
        r(4) = mean(m(~dataset.val_y{i}));
        res(i,:,j) = r;
        disp(i);
        disp(r);
    end
end

disp(res);
