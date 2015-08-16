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
    dataset = make_dataset('data/hk_original', [h w], [0.7 0], [4 5], ...
        'preprocessed', 'data/hk_original/preprocessed_small.mat', ...
        'nFolds', nFolds);
    dataset.X = single(padarray(-dataset.X, [6 7 0], -1));
    save('data/hk_original/dataset_small.mat', 'dataset');
end

%% Extraction layers

% All 1e-5, 
% [47 95], [12 16], 16 'pool', [4 4], 'dropout', 0.4
% [2 7], 10, trainOpts, 'pool', [2 2]
% trains, but doesn't learn Gabor filter

extractionNet = MultiLayerNet();

trainOpts = struct('lRate', 4e-5);
cnn = CNN([47 95], [12 16], 14, trainOpts, 'pool', [4 4]);
% L = [3.5, 4.2];
% for j = 1:numel(L)
%     l = L(j);
%     for k = 1:4
%         cnn.filters(:,:,1,(j-1)*4+k) = ...
%             gaborfilter([12 16], [1.16 1.16], l, k * pi/4) / sqrt(16*12*16);
%     end
% end
extractionNet.add(cnn);

% trainOpts = struct('lRate', 1e-5, 'dropout', 0.3);
% cnn       = CNN(extractionNet.outsize(), [3 4], 20, trainOpts);
% extractionNet.add(cnn);

concat = ReshapeNet(extractionNet, prod(extractionNet.outsize()));
extractionNet.add(concat);

trainOpts = struct('lRate', 4e-5, 'dropout', 0.5);
rbm = RELURBM(extractionNet.outsize(), 1000, struct(), trainOpts);
extractionNet.add(rbm);

trainOpts = struct('lRate', 4e-5, 'dropout', 0.3);
rbm = RELURBM(extractionNet.outsize(), 500, struct(), trainOpts);
extractionNet.add(rbm);

%% Comparison layers

wholeNet   = MultiLayerNet();
compareNet = SiameseNet(extractionNet, 2);
wholeNet.add(compareNet);

metric = L2Compare(extractionNet.outsize());
wholeNet.add(metric);

%% Training

trainOpts = struct('batchFn', @pairsBatchFn, ...
                   'nIter', 20, ...
                   'batchSz', 500, ...
                   'displayEvery', 10);

res = zeros(nFolds, 8, 4);
for i = 2:nFolds
    net = wholeNet.copy(); % start from random weights
    X  = struct('data', dataset.X, 'pairs', dataset.train_x{i});
    Y  = ~dataset.train_y{i}';
    Xv = struct('data', dataset.X, 'pairs', dataset.val_x{i});
    Yv = ~dataset.val_y{i}';
    
    for j = 1:8
        % Train for a few iterations
        train(net, @expCost, X, Y, trainOpts);

        r = zeros(1, 4);
        % Training performances
        [allX, allY] = trainOpts.batchFn(X, Y, inf, []);
        o = net.compute(allX);
        eer = fminsearch(@(t) abs(mean(o(allY == 0) < t) ...
            - mean(o(allY > 0) >= t)), double(mean(o)));
        m = (o > eer) ~= (allY > 0);
        r(1) = mean(m(allY > 0));
        r(2) = mean(m(allY == 0));

        % Validation performances
        [allX, allY] = trainOpts.batchFn(Xv, Yv, inf, []);
        o = net.compute(allX);
        m = (o > eer) ~= (allY > 0);
        r(3) = mean(m(allY > 0));
        r(4) = mean(m(allY == 0));
        
        res(i, j, :) = r;
        disp(r);        
    end
    
    % Save network state
    save('data/workspaces/trained.mat', 'net');
end

disp(res);
