% image sizes, images will be *cropped*
h = 99;
w = 231;

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

%% Train unsupervised convolutional RBM as first layer

arch = struct('dataSize', [h w], ...
		'nFM', 7, ...
        'filterSize', [10 20], ...
        'stride', [3 4], ...
        'inputType', 'binary');

arch.opts = {'nEpoch', 6, ...
			 'lRate', .0003, ...
			 'displayEvery',20, ...
             'wPenalty', .05, ...
			 'sparsity', .01, ...
			 'sparseGain', 5};

crbm = crbm(arch);
crbm.train(database.train_x);
save('data/workspaces/veinsCRBM.mat', 'crbm');

%% Generate output data

outputSize        = prod(crbm.hidSize ./ crbm.stride) * 1;% crbm.nFM;
database2.train_x = zeros(Na, outputSize);
database2.train_y = database.train_y;
database2.test_x  = zeros(Nt, outputSize);
database2.test_y  = database.test_y;

% pool doesn't change output size so it repeats pooling value over the pool
% below we keep a reduced matrix
poolYIdx          = mod(0:crbm.hidSize(1)-1,crbm.stride(1)) == 0;
poolXIdx          = mod(0:crbm.hidSize(2)-1,crbm.stride(2)) == 0;

for s = 1:Na
    output      = crbm.poolGivVis(double(database.train_x(:,:,s)));
    subsampling = output(poolYIdx, poolXIdx, 1);
    database2.train_x(s, :) = ...
        reshape(subsampling, 1, outputSize);
end
for s = 1:Nt
    output = crbm.poolGivVis(double(database.test_x(:,:,s)));
    subsampling = output(poolYIdx, poolXIdx, 1);
    database2.test_x(s, :) = ...
        reshape(subsampling, 1, outputSize);
end

%% Observe separation between individuals

[~, s] = sort(database2.test_y);
D      = dist(database2.test_x(s,:)');
[~, m] = min(D+eye(length(s))*inf);
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

%% don't pollute workspace and free some memory
clear crbm poolSize database poolYIdx poolXIdx arch output subsampling s dataSize D m2 m s

%% Train DBN on top

arch = struct('size', [outputSize 300], ...
              'classifier',false, ...
              'inputType','binary');

arch.opts = {'verbose', 1, ...
             'lRate', 0.0001, ...
             'momentum', 0.5, ...
             'nEpoch', 40, ...
             'wPenalty', 0.5, ...
             'sparsity', 0.01, ...
             'batchSz', 20, ...
             'nGibbs', 2, ...
             'varyEta',7, ...
             'displayEvery', 20};

rbm1 = rbm(arch);

rbm1 = rbm1.train(database2.train_x);

save('data/workspaces/veinsRBM1.mat', 'rbm1');

%% Generate output data

