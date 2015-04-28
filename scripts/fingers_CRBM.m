% image sizes, images will be *cropped*
h  = 99;
w  = 231;
nF = 7;
filterSize = [10 20];
stride     = [3 4];
rbm2OutSz  = 700;
rbm3OutSz  = 700;

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
else
    database = load_hk_Qin_preprocessing('./data/hk_Qin_preprocessing', h, w, 0.6);
    save('data/hk_Qin_preprocessing.mat', 'database');
end

Na = size(database.train_x, 1);
Nt = size(database.test_x, 1);

% format data for toolbox
database.train_x = double(reshape(database.train_x', database.h, database.w, Na));
database.test_x  = double(reshape(database.test_x', database.h, database.w, Nt));

%% Unsupervised Training of convolutional RBM as first layer

wholeNet = MultiLayerNet();
CRBMPretrainingOpts = {'nEpoch', 6, ...
    'lRate', .0003, ...
    'displayEvery',20, ...
    'wPenalty', .05, ...
    'sparsity', .01, ...
    'sparseGain', 5};
RBMpretrainingOpts = {'verbose', 1, ...
    'lRate', 0.01, ...
    'momentum', 0.5, ...
    'nEpoch', 200, ...
    'wPenalty', 0.01, ...
    'sparsity', 0.01, ...
    'batchSz', 20, ...
    'nGibbs', 2, ...
    'displayEvery', 20};

% layer 1
arch = struct('dataSize', [h w], ...
        'nFM', nF, ...
        'filterSize', filterSize, ...
        'stride', stride, ...
        'inputType', 'binary');
l1crbm = crbm(arch);
wholeNet.add(l1crbm, CRBMPretrainingOpts, {});

% layer 2
l1OutDim = ([h w] - filterSize + 1) ./stride;
rbms = MetaNet();
arch = struct('size', [prod(l1OutDim) rbm2OutSz], ...
              'classifier',false, ...
              'inputType','binary');
for i = 1:nF
    filterAnalyzer = MultiLayerNet();
    reshapeNet = ReshapeNet(l1OutDim, prod(l1OutDim));
    filterAnalyzer.add(reshapeNet, struct(), struct());
    l2rbm = rbm(arch);
    filterAnalyzer.add(l2rbm, RBMpretrainingOpts, struct());
    rbms.add(filterAnalyzer, (1:nF) == i, struct(), struct());
end
wholeNet.add(rbms, struct(), struct());

% layer 3
reshapeNet = ReshapeNet(wholeNet.outSize(), nF * rbm2OutSz);
wholeNet.add(reshapeNet, {}, {});
arch = struct('size', [nF*rbm2OutSz rbm3OutSz], ...
              'classifier',false, ...
              'inputType','binary');
rbm3 = rbm(arch);
wholeNet.add(rbm3, RBMpretrainingOpts, {});

% pretraining

wholeNet.pretrain(database.train_x);

%% Observe separation between individuals

[~, s] = sort(database.train_y);
out    = wholeNet.compute(database.train_x);
D      = dist(out) + 1000 * eye(numel(s));
[~, m] = min(D);
m2     = m(1:end-1)-1 == m(2:end);
fprintf(1, 'correct testing veins proximity : %f\n', mean(m2));
blah   = [];
bluh   = zeros(Nt/2, 1);
for i = 1:2:Nt
    bluh(ceil(i/2)) = D(i, i+1);
end
for i = 1:2:Nt-2
    blah = [blah, D(i,i+2:end)];
end
for i = 2:2:Nt-2
    blah = [blah, D(i,i+1:end)];
end
[h1, x1] = hist(blah,20);
h1 = h1 / sum(h1);
[h2, x2] = hist(bluh,20);
h2 = h2 / sum(h2);
plot(x1,h1,'g');
hold on
plot(x2,h2,'r');
