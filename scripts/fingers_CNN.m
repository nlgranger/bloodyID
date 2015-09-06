%% Parameters

h            = 35;
ratio        = 2.3;
w            = round(ratio*h);
nMatching    = 4; % number of matching pairs for each image
nNonMatching = 8; % number of non matching pairs for each image
nFolds       = 7;

%% load initial data
% CNN requires single precision input!
if exist('data/hk_original/dataset_small.mat', 'file')
    load('data/hk_original/dataset_small.mat');
    dataset.X = single(padarray(dataset.X, [6 7 0], -1));
else
    dataset = make_dataset('data/hk_original', [h w], [0.7 0], [5 5], ...
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

trainOpts = struct('lRate', 2e-4, 'dropout', 0.1);
cnn = CNN([47 95], [12 12], 30, trainOpts, 'pool', [4 4]);
% L = [3.5, 4.2];
% for j = 1:numel(L)
%     l = L(j);
%     for k = 1:4
%         cnn.filters(:,:,1,(j-1)*4+k) = ...
%             gaborfilter([12 16], [1.16 1.16], l, k * pi/4) / sqrt(16*12*16);
%     end
% end
load('data/workspaces/filters.mat');
cnn.filters = single(filters);
clear filters;
extractionNet.add(cnn);

trainOpts = struct('lRate', 2e-4, 'dropout', 0.2);
cnn       = CNN(extractionNet.outsize(), [3 3], 30, trainOpts);
extractionNet.add(cnn);

concat = ReshapeNet(extractionNet, prod(extractionNet.outsize()));
extractionNet.add(concat);

trainOpts = struct('lRate', 1e-6, 'dropout', 0.4);
rbm = RELURBM(extractionNet.outsize(), 400, struct(), trainOpts);
extractionNet.add(rbm);

trainOpts = struct('lRate', 1e-6);
rbm = RELURBM(extractionNet.outsize(), 100, struct(), trainOpts);
extractionNet.add(rbm);

%% Comparison layers

wholeNet   = MultiLayerNet();
compareNet = SiameseNet(extractionNet, 2);
wholeNet.add(compareNet);

metric = L1Distance(extractionNet.outsize());%, 0.0005); % <<<<<<------ METRIC
wholeNet.add(metric);

%% Training

trainOpts = struct('batchFn', @pairsBatchFn, ...
                   'nIter', 20, ...
                   'batchSz', 500, ...
                   'displayEvery', 6);

res = zeros(nFolds, 8, 4);
for i = 1:nFolds
    net = wholeNet.copy(); % start from random weights
    X  = struct('data', dataset.X, 'pairs', dataset.train_x{i});
    %Y  = single(~dataset.train_y{i}') * 0.8 + 0.1;
    Y  = ~dataset.train_y{i}';
    %Y  = single(~dataset.train_y{i}');
    Xv = struct('data', dataset.X, 'pairs', dataset.val_x{i});
    %Yv = single(~dataset.val_y{i}');
    Yv = ~dataset.val_y{i}';
    %Yv = single(~dataset.val_y{i}');
    
    for j = 1:3
        % Train for a few iterations
        train(net, @expCost, X, Y, trainOpts); %<<<<<<<<<<<<<<---------- TRAIN
        
        r = zeros(1, 4);
        % Training performances
        [batchX, allY, idx] = trainOpts.batchFn(X, Y, 2000, []);
        o    = [];
        while ~isempty(idx)
            o = [o net.compute(batchX)];
            [batchX, batchY, idx] = trainOpts.batchFn(X, Y, 2000, idx);
            allY = [allY batchY];
        end
        o = [o net.compute(batchX)];
        eer  = fminsearch(@(t) abs(mean(o(allY < 0.5) < t) ...
             - mean(o(allY > 0.5) >= t)), double(mean(o)));
        r(1) = mean(o(allY > 0.5) < eer);
        r(2) = mean(o(allY < 0.5) > eer);
        subplot(3,2,2*j-1)
        hold off
        histogram(o(allY > 0.5), 'binWidth', 0.05);
        hold on
        histogram(o(allY < 0.5), 'binWidth', 0.05);
        plot(eer, 0, 'r*')
        hold off

        % Validation performances
        [allX, allY] = trainOpts.batchFn(Xv, Yv, inf, []);
        o = net.compute(allX);
        r(3) = mean(o(allY > 0.5) < eer);
        r(4) = mean(o(allY < 0.5) > eer);        
        subplot(3, 2, 2*j)
        hold off
        histogram(o(allY > 0.5), 'binWidth', 0.05);
        hold on
        histogram(o(allY < 0.5), 'binWidth', 0.05);
        plot(eer, 0, 'r*')
        hold off
        drawnow
        clear allX allY o
        
        res(i, j, :) = r;
        disp(r);
    end
    
    % Save network state
    save('data/workspaces/trained.mat', 'net');
end

disp(res);
