h             = 39;
w             = 96;
nMatching     = 1; % number of matching pairs for each image
nNonMatching  = 3; % number of non matching pairs for each image

%% load initial data

% fill a database object with fields 
% train_x : training samples pairs, nbsamples x (2*h*w) matrix
% train_y : associated label
% test_x  : same as train_x for testing
% test_y  : same as train_y for testing
% h       : height of the images
% w       : width of the images

if exist('data/hk_Qin_preprocessing.mat', 'file')
    load('data/hk_Qin_preprocessing.mat');
else % load images and build data pairs
    database = load_hk_Qin_preprocessing('./data/hk_Qin_preprocessing', h, w, 0.6);
    Na = numel(database.train_y);
    Nt = numel(database.test_y);
    save('data/hk_Qin_preprocessing.mat', 'database', 'Na', 'Nt', 'nPairs');
end

%% Extraction layers

RBMpretrainingOpts = struct( ...
    'lRate', 0.005, ...
    'momentum', 0.6, ...
    'nEpochs', 100, ...
    'batchSz', 60, ...
    'nGS', 1, ...
    'dropoutVis', 0.5, ...
    'wPenalty', 0.001, ...
    'wDecayDelay', 0, ...
    'sparsity', 0.10, ...
    'sparseGain', 2, ...
    'displayEvery', 5);

RBMtrainOpts = struct( ...
    'lRate', 1, ...
    'displayEvery', 5);
trainOpts = struct('nIter', 100, ...
                   'batchSz', 60, ...
                   'skipBelow', 2, ...
                   'displayEvery', 5);

extractionNet = MultiLayerNet(struct());
               
% layer 1
rbm = RBM(h*w, 200, RBMpretrainingOpts, RBMtrainOpts);
extractionNet.add(rbm);

% layer 2
rbm = RBM(200, 100, RBMpretrainingOpts, RBMtrainOpts);
extractionNet.add(rbm);

extractionNet.pretrain(database.train_x);

%% Comparison layers

wholeNet   = MultiLayerNet(trainOpts);
reshapeNet = ReshapeNet(h*w*2, {h*w, h*w});
wholeNet.add(reshapeNet);

% Duplicate extraction network
comparePreTrainOpts = struct('skip', true);
compareNet = CompareNet(extractionNet, 2, comparePreTrainOpts);
wholeNet.add(compareNet);

reshapeNet = ReshapeNet(compareNet, 200);
wholeNet.add(reshapeNet);

RBMtrainOpts = struct( ...
    'lRate', 1, ...
    'decayNorm', 2, ...
    'decayRate', 0.05, ...
    'displayEvery', 5);

% layer 3
rbm = RBM(200, 50, RBMpretrainingOpts, RBMtrainOpts);
wholeNet.add(rbm);

% layer 4
rbm = RBM(50, 1, RBMpretrainingOpts, RBMtrainOpts);
wholeNet.add(rbm);

database = makepairs(database, nMatching, nNonMatching);

%% Training
% wholeNet.train(database.train_x, database.train_y);

%% Testing

% Xa = wholeNet.compute(database.train_x);
% Xt = wholeNet.compute(database.test_x);
% 
% SVMModel = fitcsvm(Xa',database.train_y,'KernelFunction','rbf', 'OutlierFraction',0.05);
% y = predict(SVMModel,Xa');
% m = y ~= database.train_y;
% fprintf(1, 'Training fpr : %f\n', mean(m(~database.train_y)));
% fprintf(1, '         frr : %f\n', mean(m(database.train_y)));
% y = predict(SVMModel,Xt');
% m = y ~= database.test_y;
% fprintf(1, 'Testing  fpr : %f\n', mean(m(~database.test_y)));
% fprintf(1, '         frr : %f\n', mean(m(database.test_y)));