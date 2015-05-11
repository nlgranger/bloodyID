% image sizes, images will be *cropped*
h      = 39;
w      = 96;
nPairs = 4; % # of random pairs for each initial sample

%% load initial data

% fill a database object with fields 
% train_x : training samples, nbsamples x (h*w) matrix
% train_y : associated label
% test_x  : same as train_x for testing
% test_y  : same as train_y for testing
% h       : height of the images
% w       : width of the images
if exist('data/hk_Qin_preprocessing.mat', 'file')
    load('data/hk_Qin_preprocessing.mat');
else % load images and build data pairs
    database = load_hk_Qin_preprocessing('./data/hk_Qin_preprocessing', h, w, 0.6);
    save('data/hk_Qin_preprocessing.mat', 'database', 'Na', 'Nt', 'nPairs');
end

Na = numel(database.train_y);
Nt = numel(database.test_y);

shuffle           = [1:Na randi(Na, 1, (nPairs-1)*Na)];
database.train_x  = [repmat(database.train_x, 1, nPairs); ...
                     database.train_x(:,shuffle)];
database.train_i1 = database.train_y(shuffle);
database.train_i2 = repmat(database.train_y, nPairs, 1);
database.train_y  = abs(database.train_i1) == abs(database.train_i2);

shuffle           = [1:Nt randi(Nt, 1, (nPairs-1)*Nt)];
database.test_x   = [repmat(database.test_x, 1, nPairs); ...
                   database.test_x(:,shuffle)];
database.test_i1 = database.test_y(shuffle);
database.test_i2 = repmat(database.test_y, nPairs, 1);
database.test_y  = abs(database.test_i1) == abs(database.test_i2);

%% Extraction layers

extractionNet = MultiLayerNet(struct());
RBMpretrainingOpts = {'verbose', 1, ...
    'lRate', 0.02, ...
    'momentum', 0.6, ...
    'nEpoch', 100, ...
    'wPenalty', 0.005, ...
    'sparsity', 0.005, ...
    'batchSz', 50, ...
    'nGibbs', 2, ...
    'dropout', 0.5, ...
    'displayEvery', 10};
RBMtrainOpts = struct( ...
    'nIter', 0, ...
    'batchSz', 50, ...
    'lRate', 0.04, ...
    'decayNorm', 2, ...
    'decayRate', 0.001);
trainOpts = struct('nIter', 100, ...
                   'batchSz', 50);

% layer 1
arch = struct('size', [h*w 200], ...
              'classifier',false, ...
              'inputType','binary');
rbm = RBM(arch, RBMpretrainingOpts, RBMtrainOpts);
extractionNet.add(rbm);

% layer 2
arch = struct('size', [200 100], ...
              'classifier',false, ...
              'inputType','binary');
rbm = RBM(arch, RBMpretrainingOpts, RBMtrainOpts);
extractionNet.add(rbm);

% layer 3
% arch = struct('size', [500 100], ...
%               'classifier',false, ...
%               'inputType','binary');
% rbm = RBM(arch, RBMpretrainingOpts, RBMtrainOpts);
% extractionNet.add(rbm);

%% Comparison layers

wholeNet   = MultiLayerNet(trainOpts);
% Duplicate extraction network
comparePreTrainOpts = struct('limit', Na);
compareNet = CompareNet(extractionNet, 2, comparePreTrainOpts);
reshapeNet = ReshapeNet(h*w*2, {h*w, h*w});
wholeNet.add(reshapeNet);
wholeNet.add(compareNet);

% layer 3
% arch = struct('size', [200 50], ...
%               'classifier',false, ...
%               'inputType','binary');
% rbm = RBM(arch, RBMpretrainingOpts, RBMtrainOpts);
reshapeNet = ReshapeNet(compareNet, 200);
wholeNet.add(reshapeNet);
% wholeNet.add(rbm);

% layer 4
% RBMpretrainingOpts = {'verbose', 1, ...
%     'lRate', 0.01, ...
%     'momentum', 0.6, ...
%     'nEpoch', 75, ...
%     'wPenalty', 0.0001, ...
%     'batchSz', 50, ...
%     'nGibbs', 2, ...
%     'displayEvery', 10};
% arch = struct('size', [50 30], ...
%               'classifier',false, ...
%               'inputType','binary');
% rbm = RBM(arch, RBMpretrainingOpts, RBMtrainOpts);
% wholeNet.add(rbm);

% layer 5
% arch = struct('size', [30 1], ...
%               'classifier',false, ...
%               'inputType','binary');
% rbm = RBM(arch, RBMpretrainingOpts, RBMtrainOpts);
% wholeNet.add(rbm);

% Pretrain
wholeNet.pretrain(database.train_x);

X = wholeNet.compute(database.train_x);
Xt = wholeNet.compute(database.test_x);
SVMModel = fitcsvm(X',database.train_y,'KernelFunction','rbf');
label = predict(SVMModel,X');
m = label == database.train_y;

label = predict(SVMModel,Xt');
m = label == database.test_y;
mean(m(database.test_i1 == - database.test_i2))

% train
% wholeNet.train(database.train_x, database.train_y);