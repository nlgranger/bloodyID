h             = 39;
w             = 96;
nMatching     = 4; % number of matching pairs for each image
nNonMatching  = 4; % number of non matching pairs for each image

%% load initial data

% fill a database object with fields 
% train_x : training samples pairs, nbsamples x (2*h*w) matrix
% train_y : associated label
% test_x  : same as train_x for testing
% test_y  : same as train_y for testing
% h       : height of the images
% w       : width of the images

load('data/hk_Qin_preprocessing/dataset.mat');

%% Create Network

trainOpts = struct('nIter', 100, ...
                   'batchSz', 60, ...
                   'skipBelow', 2, ...
                   'displayEvery', 5);

wholeNet   = MultiLayerNet(trainOpts);

%% Extraction layers

RBMpretrainingOpts = struct( ...
    'lRate', 0.007, ...
    'momentum', 0.6, ...
    'nEpochs', 400, ...
    'batchSz', 100, ...
    'nGS', 1, ...
    'dropoutVis', 0.4, ...
    'wPenalty', 0.03, ...
    'wDecayDelay', 10, ...
    'sparsity', 0.04, ...
    'sparseGain', 7, ...
    'displayEvery', 5);

RBMtrainOpts = struct( ...
    'lRate', 1, ...
    'displayEvery', 5);

extractionNet = MultiLayerNet(struct());
               
% layer 1
rbm = RBM(h*w, 300, RBMpretrainingOpts, RBMtrainOpts);
extractionNet.add(rbm);

% layer 2
rbm = RBM(300, 100, RBMpretrainingOpts, RBMtrainOpts);
extractionNet.add(rbm);

extractionNet.pretrain(database.pretrain_x);

%% Comparison layers

% Duplicate extraction network
compareNet = CompareNet(extractionNet, 2, struct());
wholeNet.add(compareNet);

reshapeNet = ReshapeNet(compareNet, 200);
wholeNet.add(reshapeNet);

RBMtrainOpts = struct( ...
    'lRate', 0.1, ...
    'decayNorm', 2, ...
    'decayRate', 0.05, ...
    'displayEvery', 5);

% % layer 3
% rbm = RBM(200, 50, RBMpretrainingOpts, RBMtrainOpts);
% wholeNet.add(rbm);
% 
% % layer 4
% rbm = RBM(50, 1, RBMpretrainingOpts, RBMtrainOpts);
% wholeNet.add(rbm);

database2 = {};
[database2.train_x, database2.train_y] = makepairs(database.train_x, database.train_y, nMatching, nNonMatching);
[database2.val_x, database2.val_y] = makepairs(database.val_x, database.val_y, nMatching, nNonMatching);
[database2.test_x, database2.test_y] = makepairs(database.test_x, database.test_y, nMatching, nNonMatching);

save('data/workspaces/veinDBN.mat', 'wholeNet', 'database2');

%% Training

Xa = wholeNet.compute(database2.train_x);
Xv = wholeNet.compute(database2.val_x);
Xt = wholeNet.compute(database2.test_x);
Ya = database2.train_y;
Yv = database2.val_y;
Yt = database2.test_y;

save('/tmp/features.mat', 'Xa', 'Xv', 'Xt', 'Ya', 'Yv', 'Yt');
clear all
load '/tmp/features.mat'
svmcrossval(Xa, Ya, Xv, Yv);

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