% image sizes, images will be *cropped*
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

extractionNet = MultiLayerNet(struct());
RBMpretrainingOpts = struct(...
    'lRate', 0.01, ...
    'momentum', 0.6, ...
    'nEpochs', 100, ...
    'batchSz', 60, ...
    'nGS', 1, ...
    'dropoutVis', 0.5, ...
    'wPenalty', 0.0001, ...
    'wDecayDelay', 10, ...
    'displayEvery', 5);

    %'sparsity', 0.01, ...
    %'sparseGain', 2, ...
RBMtrainOpts = struct( ...
    'nIter', 0, ...
    'batchSz', 60, ...
    'lRate', 0.04, ...
    'decayNorm', 2, ...
    'decayRate', 0.001);
trainOpts = struct('nIter', 100, ...
                   'batchSz', 50);

% layer 1
rbm = RBM(h*w, 200, RBMpretrainingOpts, RBMtrainOpts);
extractionNet.add(rbm);
extractionNet.pretrain(database.train_x);

% layer 2
% arch = struct('size', [200 100], ...
%               'classifier',false, ...
%               'inputType','binary');
% rbm = RBM(arch, RBMpretrainingOpts, RBMtrainOpts);
% extractionNet.add(rbm);

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
reshapeNet = ReshapeNet(compareNet, 400);
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

database = makepairs(database, nMatching, nNonMatching);
Xa = wholeNet.compute(database.train_x);
Xt = wholeNet.compute(database.test_x);



SVMModel = fitcsvm(Xa',database.train_y,'KernelFunction','rbf', 'OutlierFraction',0.05);
y = predict(SVMModel,Xa');
m = y ~= database.train_y;
fprintf(1, 'Training fpr : %f\n', mean(m(~database.train_y)));
fprintf(1, '         frr : %f\n', mean(m(database.train_y)));
y = predict(SVMModel,Xt');
m = y ~= database.test_y;
fprintf(1, 'Testing  fpr : %f\n', mean(m(~database.test_y)));
fprintf(1, '         frr : %f\n', mean(m(database.test_y)));

% train
% wholeNet.train(database.train_x, database.train_y);