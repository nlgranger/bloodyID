function dataset = load_hk_Qin_preprocessing(h, w, copies, ...
    nMatching, nNonMatching, trainRatio, valRatio)

% Load images

dbpath = fileparts(mfilename('fullpath'));

[rh, rw] = size(imread(fullfile(dbpath, 'finger_veins', '(1)/(1).bmp')));

veinsPath = fullfile(dbpath, 'finger_veins');
% masksPath = fullfile(dbpath, 'finger_masks');
veinDirs = dir(veinsPath);

x   = false(h, w, 0);
% m   = false(h * w, 0);
ids = zeros(0, 1, 'int32');
ses = zeros(0, 1, 'int32');

for d = 1:length(veinDirs)
    subDir = veinDirs(d);
    if ~subDir.isdir || strcmp(subDir.name,'.') || strcmp(subDir.name, '..')
        continue
    else
        isS2 = strcmp(subDir.name(end-1:end), 's2');
        fName = sprintf('(%d).bmp', 1 + 6 * isS2);
        imv = imread(fullfile(veinsPath, subDir.name, fName));
        if size(imv, 1) ~= rh || size(imv, 2) ~= rw % skip corrupted files
            continue;
        end
        if isS2
            id = str2double(subDir.name(2:end-4));
        else
            id = str2double(subDir.name(2:end-1));
        end
        for i = 1:copies
            imFileName = sprintf('(%d).bmp', i + 6 * isS2);
            imv = imread(fullfile(veinsPath, subDir.name, imFileName));
            imv = imresize(imv, [h w], 'bilinear') > 128;
            x   = cat(3, x, imv);
            % imm = imread(fullfile(masksPath, subDir.name, imFileName));
            % imm = imresize(imv, [h w], 'bilinear') > 128;
            % m   = [m reshape(imm, h*w, 1)];
            ids = [ids; id];
            ses = [ses; 1 + isS2];
        end
    end
end

% Assign individuals to training, validation and testing
category  = zeros(numel(ids), 1, 'int32');

for id = 1:312
    fromS1 = find(ids == id & ses == 1, copies);
    fromS2 = find(ids == id & ses == 2, copies);
    
    if isempty(fromS2)
        category(fromS1) = 4;
    else
        c = rand();
        if c < trainRatio
            category(fromS1) = 1;
            category(fromS2) = 1;
        elseif c < trainRatio + valRatio
            category(fromS1) = 2;
            category(fromS2) = 2;
        else
            category(fromS1) = 3;
            category(fromS2) = 3;
        end
    end
end

% Make pairs
pairs     = cell(2,1);
paircat   = zeros(0, 1, 'int32');
pairs{1}  = false(h, w, 0);
pairs{2}  = false(h, w, 0);
y         = false(1,0);

for id = 1:210
    fromS1 = find(ids == id & ses == 1, copies);
    fromS2 = find(ids == id & ses == 2, copies);
    c = category(fromS1(1));
    paircat = [paircat; repmat(c, numel(fromS1) * (nMatching+nNonMatching), 1)];
    
    nonpeers = find(ids ~= id & category == c);
    for i = 1: numel(fromS1)
        s = fromS1(i);
        pairs{1} = cat(3, pairs{1}, ...
                          repmat(x(:,:,s), 1, 1, nMatching + nNonMatching));
        y        = [y; true(nMatching, 1); false(nNonMatching, 1)];
        shuffle  = randperm(copies, nMatching);
        pairs{2} = cat(3, pairs{2}, x(:,:,fromS2(shuffle)));
        shuffle  = randperm(numel(nonpeers), nNonMatching);
        pairs{2} = cat(3, pairs{2}, x(:,:, nonpeers(shuffle)));
    end
end

% Build dataset

dataset.x          = x;
dataset.category   = category;
dataset.id         = ids;
dataset.ses        = ses;
dataset.pretrain_x = x(:,:, (category == 1) | (category == 4));
dataset.train_x    = {pairs{1}(:,:, paircat == 1); pairs{2}(:,:, paircat == 1)};
dataset.train_y    = y(paircat == 1);
dataset.val_x      = {pairs{1}(:,:, paircat == 2); pairs{2}(:,:, paircat == 2)};
dataset.val_y      = y(paircat == 2);
dataset.test_x     = {pairs{1}(:,:, paircat == 3); pairs{2}(:,:, paircat == 3)};
dataset.test_y     = y(paircat == 3);
end