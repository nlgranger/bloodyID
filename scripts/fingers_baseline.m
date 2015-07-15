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
    dataset = make_dataset('data/hk_original', [h w], [1 0], [4 6], ...
        'preprocessed', 'data/hk_original/preprocessed_big.mat');
    X = dataset.X;
    dataset.X = false([h w size(X, 3)]);
    for i = 1:size(dataset.X, 3)
        dataset.X(:,:,i) = vein_extraction(X(:,:,i)) ...
            .* dataset.M(:,:,i);
    end
    save('data/hk_original/dataset_big.mat', 'dataset');
end

%% Testing
optimizer                   = registration.optimizer.OnePlusOneEvolutionary;
metric                      = registration.metric.MattesMutualInformation;
optimizer.InitialRadius     = 0.001;
optimizer.Epsilon           = 1.5e-4;
optimizer.GrowthFactor      = 1.005;
optimizer.MaximumIterations = 70;

res = zeros(numel(dataset.train_y), 3);

pool = parpool();
parfor i = 1:numel(dataset.train_y)
    I1 = dataset.X(:,:,dataset.train_x(i,1));
    I2 = dataset.X(:,:,dataset.train_x(i,2));    
    Ir = imregister(uint8(I1), uint8(I2), 'rigid', optimizer, metric);
    res(i, :) = [sum(sum(I2)), sum(sum(Ir)), sum(sum(I2 & Ir))];
        
%     subplot(3,1,1)
%     imagesc(I1); axis image; axis off;
%     subplot(3,1,2)
%     imagesc(I2); axis image; axis off;
%     subplot(3,1,3)
%     imshowpair(Ir, uint8(I2));
%     pause
end
delete(pool);
