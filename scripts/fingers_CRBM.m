% image sizes, images will be *cropped*
h  = 99;
w  = 231;
nF = 7;
filterSize = [10 20];
stride     = [3 4];
rbm1OutSz  = 1000;
rbm2OutSz  = 700;

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
		'nFM', nF, ...
        'filterSize', filterSize, ...
        'stride', stride, ...
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

outputSize        = crbm.hidSize ./ crbm.stride;
database2.train_x = zeros(Na, outputSize(1) * outputSize(2), nF);
database2.train_y = database.train_y;
database2.test_x  = zeros(Nt, outputSize(1) * outputSize(2), nF);
database2.test_y  = database.test_y;

% pool doesn't change output size and repeats pooling value over the pool
% below we keep a reduced matrix
poolYIdx          = mod(0:crbm.hidSize(1)-1,crbm.stride(1)) == 0;
poolXIdx          = mod(0:crbm.hidSize(2)-1,crbm.stride(2)) == 0;

for s = 1:Na
    output = crbm.poolGivVis(double(database.train_x(:,:,s)));
    database2.train_x(s, :, :) = ...
        reshape(output(poolYIdx, poolXIdx, :), prod(outputSize), nF);
end
for s = 1:Nt
    output = crbm.poolGivVis(double(database.test_x(:,:,s)));
    database2.test_x(s, :, :) = ...
        reshape(output(poolYIdx, poolXIdx, :), prod(outputSize), nF);
end

%% Observe separation between individuals

% [~, s] = sort(database2.test_y);
% D      = dist(database2.test_x(s,:)');
% [~, m] = min(D+eye(length(s))*inf);
% m2     = m(1:end-1)-1 == m(2:end);
% fprintf(1, 'correct testing veins proximity : %f\n', mean(m2));
% blah   = [];
% bluh   = zeros(Nt/2, 1);
% for i = 1:2:Nt
%     bluh(ceil(i/2)) = D(i, i+1);
% end
% for i = 1:2:Nt-2
%     blah = [blah, D(i,i+2:end)];
% end
% for i = 2:2:Nt-2
%     blah = [blah, D(i,i+1:end)];
% end
% [h1, x1] = hist(blah,20);
% h1 = h1 / sum(h1);
% [h2, x2] = hist(bluh,20);
% h2 = h2 / sum(h2);
% plot(x1,h1,'g');
% hold on
% plot(x2,h2,'r');

%% don't pollute workspace and free some memory
clear crbm poolSize database poolYIdx poolXIdx arch output subsampling s dataSize D m2 m

%% Train DBN on top

arch = struct('size', [prod(outputSize) rbm1OutSz], ...
              'classifier',false, ...
              'inputType','binary');

arch.opts = {'verbose', 1, ...
             'lRate', 0.01, ...
             'momentum', 0.5, ...
             'nEpoch', 200, ...
             'wPenalty', 0.01, ...
             'sparsity', 0.01, ...
             'batchSz', 20, ...
             'nGibbs', 2, ...
             'varyEta',7, ...
             'displayEvery', 20};

rbm1 = cell(nF,1);
parfor i = 1:nF
    rbm1{i} = rbm(arch);
    rbm1{i} = rbm1{i}.train(database2.train_x(:,:,i));
end

save('data/workspaces/veinsRBM1.mat', 'rbm1');

% rbm1 = rbm1.hidGivVis(database2.test_x(s,:), [], false);
% D      = dist(rbm1.pHid');
% m2     = m(1:end-1)-1 == m(2:end);
% fprintf(1, 'correct testing veins proximity : %f\n', mean(m2));
% blah   = [];
% bluh   = zeros(Nt/2, 1);
% for i = 1:2:Nt
%     bluh(ceil(i/2)) = D(i, i+1);
% end
% for i = 1:2:Nt-2
%     blah = [blah, D(i,i+2:end)];
% end
% for i = 2:2:Nt-2
%     blah = [blah, D(i,i+1:end)];
% end
% [h1, x1] = hist(blah,20);
% h1 = h1 / sum(h1);
% [h2, x2] = hist(bluh,20);
% h2 = h2 / sum(h2);
% plot(x1,h1,'g');
% hold on
% plot(x2,h2,'r');

%% Generate output data

tmptrain_x = zeros(Na, rbm1OutSz, nF);
database3.train_y = database2.train_y;
tmptest_x  = zeros(Nt, rbm1OutSz, nF);
database3.test_y  = database2.test_y;

for i=1:nF
    rbm1{i} = rbm1{i}.hidGivVis(database2.train_x(:,:,i), [], false);
    tmptrain_x(:,:,i) = rbm1{i}.pHid;
    rbm1{i} = rbm1{i}.hidGivVis(database2.test_x(:,:,i), [], false);
    tmptest_x(:,:,i) = rbm1{i}.pHid;
end

database3.train_x = reshape(permute(tmptrain_x, [2 3 1]), nF*rbm1OutSz, Na)';
database3.test_x  = reshape(permute(tmptest_x,  [2 3 1]), nF*rbm1OutSz, Nt)';

clear tmptrain_x tmptest_x

%% Second layer RBM

arch = struct('size', [rbm1OutSz*nF rbm2OutSz], ...
              'classifier',false, ...
              'inputType','binary');

arch.opts = {'verbose', 1, ...
             'lRate', 0.005, ...
             'momentum', 0.5, ...
             'nEpoch', 200, ...
             'wPenalty', 0.005, ...
             'sparsity', 0.09, ...
             'batchSz', 20, ...
             'nGibbs', 2, ...
             'varyEta',7, ...
             'displayEvery', 20};
rbm2 = rbm(arch);
rbm2 = rbm2.train(database3.train_x);

%% plot separation performance

[~, s] = sort(database2.test_y);
rbm2 = rbm2.hidGivVis(database3.test_x(s,:), [], false);
D      = dist(rbm2.pHid');
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
