h               = 39; % resised image height
w               = 96; % resized image width
trainingRatio   = 0.35; % proportion of training images
validationRatio = 0.35; % proportion of validation images
dbpath = fileparts(mfilename('fullpath'));

copies  = 6; % number of images by finger
[rh, rw] = size(imread(fullfile(dbpath, 'finger_veins', '(1)/(1).bmp')));

dataset = {};

veinsPath = fullfile(dbpath, 'finger_veins');
masksPath = fullfile(dbpath, 'finger_masks');
veinDirs = dir(veinsPath);

x   = false(h * w, 0);
m   = false(h * w, 0);
ids = zeros(0, 1, 'int32');

for d = 1:length(veinDirs)
    subDir = veinDirs(d);
    if ~subDir.isdir ...
        || strcmp(subDir.name,'.') ...
        || strcmp(subDir.name, '..')
        continue
    else
        isS2 = strcmp(subDir.name(end-1:end), 's2');
        imv = imread(fullfile(veinsPath, subDir.name, sprintf('(%d).bmp', 1 + 6 * isS2)));
        if size(imv, 1) ~= rh || size(imv, 2) ~= rw % skip corrupted files
            continue;
        end
        if isS2
            id = - str2double(subDir.name(2:end-4));
        else
            id = str2double(subDir.name(2:end-1));
        end
        for i = 1:copies
            imFileName = sprintf('(%d).bmp', i + 6 * isS2);
            imv = imread(fullfile(veinsPath, subDir.name, imFileName));
            imv = imresize(imv, [h w], 'bilinear') > 128;
            x   = [x reshape(imv, h*w, 1)];
            imm = imread(fullfile(masksPath, subDir.name, imFileName));
            imm = imresize(imv, [h w], 'bilinear') > 128;
            m   = [m reshape(imm, h*w, 1)];
            ids = [ids; id];
        end
    end
end

s1only   = setdiff(ids, -ids);
both     = intersect(ids, -ids);
nBoth    = length(both);    
shuffle  = randperm(nBoth);

nTrain   = round(nBoth * trainingRatio);
trainIds = both(shuffle(1:nTrain));
trainIdx = sum(bsxfun(@eq, repmat(ids, 1, length(trainIds)), trainIds'), 2) > 0;
dataset.train_x = x(:, trainIdx);
dataset.train_y = ids(trainIdx);

nVal   = round(nBoth * validationRatio);
valIds = both(shuffle(nTrain+1:nTrain+nVal));
valIdx = sum(bsxfun(@eq, repmat(ids, 1, length(valIds)), valIds'), 2) > 0;
dataset.val_x = x(:, valIdx);
dataset.val_y = ids(valIdx);

testIds = both(shuffle(nTrain+nVal+1:end));
testIdx = sum(bsxfun(@eq, repmat(ids, 1, length(testIds)), testIds'), 2) > 0;
dataset.test_x = x(:, testIdx);
dataset.test_y = ids(testIdx);

pretrainIdx = sum(bsxfun(@eq, repmat(ids, 1, length(s1only)), s1only'), 2) > 0;
pretrainIdx = pretrainIdx | trainIdx | valIdx;
dataset.pretrain_x = x(:, pretrainIdx);

save('dataset.mat', 'dataset', 'h', 'w');