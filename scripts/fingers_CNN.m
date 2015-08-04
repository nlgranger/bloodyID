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
    dataset = make_dataset('data/hk_original', [h w], [0.5 0], [4 6], ...
        'preprocessed', 'data/hk_original/preprocessed_small.mat', ...
        'nFolds', nFolds);
    dataset.X = -single(padarray(dataset.X, [6 7 0], 1));
    save('data/hk_original/dataset_small.mat', 'dataset');
end

%% Extraction layers

% All 1e-5, 
% [47 95], [12 16], 16 'pool', [4 4], 'dropout', 0.4
% [2 7], 10, trainOpts, 'pool', [2 2]
% trains, but doesn't learn Gabor filter

extractionNet = MultiLayerNet();

trainOpts = struct('lRate', 1e-5, 'dropout', 0.1);
cnn = CNN([47 95], [12 16], 10, trainOpts, 'pool', [4 4]);

% L = [3.5, 4.2];
% for j = 1:numel(L)
%     l = L(j);
%     for k = 1:4
%         cnn.filters(:,:,1,(j-1)*4+k) = ...
%             gaborfilter([12 16], [1.16 1.16], l, k * pi/4) / sqrt(8*12*16);
%     end
% end
extractionNet.add(cnn);

trainOpts = struct('lRate', 1e-5);
cnn       = CNN(extractionNet.outsize(), [2 7], 10, trainOpts, 'pool', [2 2]);
extractionNet.add(cnn);

concat = ReshapeNet(extractionNet, prod(extractionNet.outsize()));
extractionNet.add(concat);

trainOpts = struct('lRate', 1e-6);
rbm = RELURBM(extractionNet.outsize(), 70, struct(), trainOpts);
extractionNet.add(rbm);

%% Comparison layers

wholeNet   = MultiLayerNet();
compareNet = SiameseNet(extractionNet, 2);
wholeNet.add(compareNet);

metric = L2Compare(70);
wholeNet.add(metric);

%% Training

trainOpts = struct('nIter', 10, ...
                   'batchSz', 500, ...
                   'displayEvery', 1);

res = zeros(nFolds, 4, 8);
for i = 1:nFolds
    X = {dataset.X(:,:,dataset.train_x{i}(:,1)), dataset.X(:,:,dataset.train_x{i}(:,2))};
    Y = single((~dataset.train_y{i}))';
    
    net = wholeNet.copy(); % start from random weights
    
    for j = 1:8
        % Train for a few iterations
        net = train(net, @testCost, X, Y, trainOpts);

        % Compute mis-classification on training data
        r = zeros(1, 4);
        o = net.compute(X)';
        eer = fminsearch(@(t) abs(mean(o(dataset.train_y{i})<t) - mean(o(~dataset.train_y{i})>t)), 0.25);
        m = (o > eer) == dataset.train_y{i};
        r(1) = mean(m(dataset.train_y{i}));
        r(2) = mean(m(~dataset.train_y{i}));

        % Compute mis-classification on testing data
        Xv = {dataset.X(:,:,dataset.val_x{i}(:,1)), dataset.X(:,:,dataset.val_x{i}(:,2))};
        o = net.compute(Xv)';
        clear Xv;
        m = (o > eer) == dataset.val_y{i};
        r(3) = mean(m(dataset.val_y{i}));
        r(4) = mean(m(~dataset.val_y{i}));
        res(i,:,j) = r;
        disp(i);
        disp(r);
        
        % Save network state
        save('data/workspaces/trained.mat', 'net');
    end
end

disp(res);
