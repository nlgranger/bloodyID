

h     = 35;
ratio = 2.3;
w     = round(ratio*h);
nMatching    = 4; % number of matching pairs for each image
nNonMatching = 6; % number of non matching pairs for each image
patchSz      = [18 18];
overlap      = [8 8];

%% load initial data

% load('data/hk_original/dataset.mat');
% dataset.pretrain_x = 0.5 * (dataset.pretrain_x + 1);
% dataset.train_x{1} = 0.5 * (dataset.train_x{1} + 1);
% dataset.train_x{2} = 0.5 * (dataset.train_x{2} + 1);
% dataset.test_x{1}  = 0.5 * (dataset.test_x{1} + 1);
% dataset.test_x{2}  = 0.5 * (dataset.test_x{2} + 1);

%% Extraction layers

extractionNet = MultiLayerNet(struct('skipBelow', 1));

patchMaker = PatchNet([h, w], patchSz, overlap);
extractionNet.add(patchMaker);

RBMtrainOpts = struct('lRate', 3e-3);
RBMpretrainingOpts = struct( ...
    'lRate', 4e-3, ...
    'momentum', 0.5, ...
    'nEpochs', 100, ...
    'batchSz', 400, ...
    'dropVis', 0.3, ...
    'dropHid', 0.1, ...
    'wPenalty', 0.001, ...
    'displayEvery', 5);
rbm = RELURBM(prod(patchSz), 70, RBMpretrainingOpts, RBMtrainOpts, false);

imRedux = SiameseNet(rbm, numel(patchMaker.outsize()));
extractionNet.add(imRedux);

patchMerge = ReshapeNet(imRedux, sum(cellfun(@prod, imRedux.outsize())));
extractionNet.add(patchMerge);

RBMtrainOpts = struct('lRate', 9e-3);
RBMpretrainingOpts = struct( ...
    'lRate', 5e-3, ...
    'momentum', 0.5, ...
    'nEpochs', 100, ...
    'batchSz', 400, ...
    'displayEvery', 5);
rbm = RELURBM(70 * numel(patchMaker.outsize()), 300, RBMpretrainingOpts, RBMtrainOpts, false);
extractionNet.add(rbm);

rbm = RELURBM(300, 200, RBMpretrainingOpts, RBMtrainOpts);
extractionNet.add(rbm);

extractionNet.pretrain(dataset.pretrain_x);

rbm = RELURBM(200, 150, RBMpretrainingOpts, RBMtrainOpts);
extractionNet.add(rbm);



save('data/workspaces/pretrained.mat', 'extractionNet');
% o = extractionNet.nets{1}.compute(dataset.pretrain_x);
% p = extractionNet.nets{2}.compute(o);
% q = extractionNet.nets{3}.compute(p);
% W = extractionNet.nets{2}.net.W;
% colormap gray
% for i = 1:70
%     subplot(2,1,1)
%     imagesc(reshape(W(:,i), patchSz))
%     colorbar
%     axis image
%     subplot(2,1,2)
%     hist(reshape(q(i:70:end,:), 1, []));
%     pause
% end
% 
% r = extractionNet.compute(dataset.pretrain_x);
% for i = 1:70
%     subplot(1,1,1)
%     hist(reshape(r(i:70:end,:), 1, []));
%     pause
% end


%% Comparison layers

trainOpts = struct('nIter', 50, ...
                   'batchSz', 300, ...
                   'displayEvery', 5);

wholeNet  = MultiLayerNet(trainOpts);

% Duplicate extraction network
compareNet = SiameseNet(extractionNet, 2, 'skipPretrain');
wholeNet.add(compareNet);

cosine = L2Compare(150);
wholeNet.add(cosine);

%% Training

% o = wholeNet.compute(dataset.test_x);
% m = o > 0.47 ~= dataset.test_y';
% mean(m(dataset.test_y))
% mean(m(~dataset.test_y))
% o = wholeNet.compute(dataset.train_x);
% m = o > .47 ~= dataset.train_y';
% mean(m(dataset.train_y))
% mean(m(~dataset.train_y))

wholeNet.train(dataset.train_x, 10*(~dataset.train_y)');

o = wholeNet.compute(dataset.test_x);
m = o > .5 ~= dataset.test_y';
mean(m(dataset.test_y))
mean(m(~dataset.test_y))
o = wholeNet.compute(dataset.train_x);
m = o > .20 ~= dataset.train_y';
mean(m(dataset.train_y))
mean(m(~dataset.train_y))

%% Testing

% Xa = wholeNet.compute(dataset.train_x);
% Xt = wholeNet.compute(dataset.test_x);
% 
% SVMModel = fitcsvm(Xa',dataset.train_y,'KernelFunction','rbf', 'OutlierFraction',0.05);
% y = predict(SVMModel,Xa');
% m = y ~= dataset.train_y;
% fprintf(1, 'Training fpr : %f\n', mean(m(~dataset.train_y)));
% fprintf(1, '         frr : %f\n', mean(m(dataset.train_y)));
% y = predict(SVMModel,Xt');
% m = y ~= dataset.test_y;
% fprintf(1, 'Testing  fpr : %f\n', mean(m(~dataset.test_y)));
% fprintf(1, '         frr : %f\n', mean(m(dataset.test_y)));