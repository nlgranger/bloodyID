% image sizes, images will be *cropped*
h      = 39;
w      = 96;
nPairs = 6; % # of random pairs for each initial sample

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
    Na = numel(database.train_y);
    Nt = numel(database.test_y);
    
    shuffle          = [1:Na randi(Na, 1, (nPairs-1)*Na)];
    database.train_x = [repmat(database.train_x, 1, nPairs); ...
                        database.train_x(:,shuffle)];
    database.train_y = repmat(1:Na, 1, nPairs) == shuffle;
    
    shuffle         = [1:Nt randi(Nt, 1, (nPairs-1)*Nt)];
    database.test_x = [repmat(database.test_x, 1, nPairs); ...
                       database.test_x(:,shuffle)];
    database.test_y = repmat(1:Nt, 1, nPairs) == shuffle;
    save('data/hk_Qin_preprocessing.mat', 'database', 'Na', 'Nt');
end

%% Extraction layers

extractionNet = MultiLayerNet(struct());
RBMpretrainingOpts = {'verbose', 1, ...
    'lRate', 0.05, ...
    'momentum', 0.8, ...
    'nEpoch', 200, ...
    'wPenalty', 0.001, ...
    'batchSz', 40, ...
    'nGibbs', 2, ...
    'displayEvery', 1};
RBMtrainOpts = struct(...
    'nIter', 0, ...
    'batchSz', 40, ...
    'lRate', 0.01, ...
    'decayNorm', 2, ...
    'decayRate', 0.001);
trainOpts = struct('nIter', 100, ...
                   'batchSz', 50);

% layer 1

arch = struct('size', [h*w 1000], ...
              'classifier',false, ...
              'inputType','binary');
rbm = RBM(arch, RBMpretrainingOpts, RBMtrainOpts);
extractionNet.add(rbm);

% layer 2
arch = struct('size', [1000 500], ...
              'classifier',false, ...
              'inputType','binary');
rbm = RBM(arch, RBMpretrainingOpts, RBMtrainOpts);
extractionNet.add(rbm);

% layer 3
arch = struct('size', [500 100], ...
              'classifier',false, ...
              'inputType','binary');
rbm = RBM(arch, RBMpretrainingOpts, RBMtrainOpts);
extractionNet.add(rbm);

%% Comparison layers

wholeNet   = MultiLayerNet(trainOpts);
% Duplicate layers 1-3
comparePreTrainOpts = struct('limit', Na);
compareNet = CompareNet(extractionNet, 2, comparePreTrainOpts);
reshape = ReshapeNet(h*w*2, {h*w, h*w});
wholeNet.add(reshape);
wholeNet.add(compareNet);

% layer 4
arch = struct('size', [200 50], ...
              'classifier',false, ...
              'inputType','binary');
rbm = RBM(arch, RBMpretrainingOpts, RBMtrainOpts);
reshape = ReshapeNet(compareNet, rbm);
wholeNet.add(reshape);
wholeNet.add(rbm);

% layer 5
arch = struct('size', [50 1], ...
              'classifier',false, ...
              'inputType','binary');
rbm = RBM(arch, RBMpretrainingOpts, RBMtrainOpts);
wholeNet.add(rbm);

% Pretrain
wholeNet.pretrain(database.train_x);

% train
wholeNet.train(database.train_x, database.train_y);