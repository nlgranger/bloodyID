%% Parameters

h            = 150;
w            = 347;
nMatching    = 4; % number of matching pairs for each image
nNonMatching = 6; % number of non matching pairs for each image
coeff        = 0.4;% this proportion of the pixels should match

%% load initial data
if exist('data/hk_original/dataset_big.mat', 'file')
    load('data/hk_original/dataset_big.mat');
else
    dataset = make_dataset('data/hk_original', [h w], [0.7 0], [4 6], ...
        'preprocessed', 'data/hk_original/preprocessed_big.mat');
    X = false([h w size(dataset.X, 3)]);
    for i = 1:size(dataset.X, 3)
        X(:,:,i) = vein_extraction(dataset.X(:,:,i)) ...
            .* dataset.M(:,:,i);
    end
    dataset.X = X;
    save('data/hk_original/dataset_big.mat', 'dataset');
end

%% Testing
optimizer = registration.optimizer.OnePlusOneEvolutionary;
metric    = registration.metric.MeanSquares;
optimizer.InitialRadius = 0.001;
optimizer.Epsilon = 5e-3;
optimizer.GrowthFactor = 1.5;
optimizer.MaximumIterations = 70;

res = zeros(numel(dataset.train_y), 3);
parfor i = 1:numel(dataset.train_y)
    I1 = dataset.X(:,:,dataset.train_x(i,1));
    I2 = dataset.X(:,:,dataset.train_x(i,2));
    Ir = imregister(uint8(I1), uint8(I2), 'rigid', optimizer, metric);
    res(i, :) = [sum(sum(I2)), sum(sum(Ir)), sum(sum(I2 & Ir))];
end

r   = res(:,3)./mean([res(:,1) res(:,2)], 2);
thr = fminsearch(@(t) abs(mean(r(dataset.train_y)<t) - mean(r(~dataset.train_y)>t)), .5);

res = zeros(numel(dataset.test_y), 3);
parfor i = 1:numel(dataset.test_y)
    I1 = dataset.X(:,:,dataset.test_x(i,1));
    I2 = dataset.X(:,:,dataset.test_x(i,2));
    Ir = imregister(uint8(I1), uint8(I2), 'rigid', optimizer, metric);
    res(i, :) = [sum(sum(I2)), sum(sum(I1)), sum(sum(I2 & Ir))];
end

r = res(:,3)./mean(res(:,1),res(:,2));
e = (r > thr) ~= dataset.test_y;

previewpairs(dataset.X(:,:,dataset.test_x(e,1)), ...
dataset.X(:,:,dataset.test_x(e,2)), dataset.test_y(e));
