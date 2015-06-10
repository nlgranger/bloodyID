trainRatio   = 0.7;
valRatio     = 0;
nMatching    = 4;
nNonMatching = 6;
h = 100;
ratio = 2.3;
w = round(ratio * 200);

curDir = fileparts(mfilename('fullpath'));

% Image Loading
fprintf(1, 'Loading image files...\n');
if exist(fullfile(curDir, 'raw.mat'), 'file')
    load(fullfile(curDir, 'raw.mat'));
else
    [X, I, S] = load_hk_original();
    save(fullfile(curDir, 'raw.mat'), 'X', 'I', 'S');
end

% Image Preprocessing
fprintf(1, 'Extracting fingers...\n');
X2 = zeros(h, w, size(X, 3));
M = true(h, w, size(X, 3));
keep = true(size(X, 3), 1);
for i = 1:size(X, 3)
    [O, Ma] = fingerExtraction(X(:,:,i));
    if numel(O) == 0
        warning('rejected finger %d', i);
        keep(i) = false;
    else
        X2(:,:,i) = O;
        M(:,:,i)  = Ma;
    end
end
X = X2;
clear X2;

fprintf(1, 'Saving...\n');
save(fullfile(curDir, 'preprocessed.mat'), 'M', 'X', 'I', 'S');

% % Preview
% for i = 1:6:size(X, 3)
%     if mod(i-1, 6) == 0
%         disp((i-1)/6);
%     end
%     for j = 0:5
%         subplot(3,2,j+1)
%         imagesc(X(:,:,i));
%         axis image
%     end
%     pause
% end

% Assign individuals to training, validation and testing
fprintf(1, 'Assigning caterogries...\n');
category  = zeros(numel(I), 1, 'int32');
for id = 1:312
    fromS1 = find(I == id & S == 1, 6);
    fromS2 = find(I == id & S == 2, 6);
    
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
fprintf(1, 'Building pairs...\n');
pairs     = cell(2,1);
paircat   = zeros(0, 1, 'int32');
pairs{1}  = zeros(h, w, 'uint8');
pairs{2}  = zeros(h, w, 'uint8');
y         = false(1,0);

for id = [1:105 157:261]
    fromS1 = find(I == id & S == 1, 6);
    fromS2 = find(I == id & S == 2, 6);
    c = category(fromS1(1));
    paircat = [paircat; repmat(c, numel(fromS1) * (nMatching+nNonMatching), 1)];
    
    nonpeers = find(I ~= id & category == c);
    for i = 1: numel(fromS1)
        s = fromS1(i);
        pairs{1} = cat(3, pairs{1}, ...
                          repmat(X(:,:,s), 1, 1, nMatching + nNonMatching));
        y        = [y; true(nMatching, 1); false(nNonMatching, 1)];
        shuffle  = randperm(6, nMatching);
        pairs{2} = cat(3, pairs{2}, X(:,:,fromS2(shuffle)));
        shuffle  = randperm(numel(nonpeers), nNonMatching);
        pairs{2} = cat(3, pairs{2}, X(:,:, nonpeers(shuffle)));
    end
end

% Build dataset
fprintf(1, 'Packing up dataset...\n');
dataset.category   = category;
dataset.pretrain_x = X(:,:, (category == 1) | (category == 4));
dataset.train_x    = {pairs{1}(:,:, paircat == 1); pairs{2}(:,:, paircat == 1)};
dataset.train_y    = y(paircat == 1);
dataset.val_x      = {pairs{1}(:,:, paircat == 2); pairs{2}(:,:, paircat == 2)};
dataset.val_y      = y(paircat == 2);
dataset.test_x     = {pairs{1}(:,:, paircat == 3); pairs{2}(:,:, paircat == 3)};
dataset.test_y     = y(paircat == 3);

fprintf(1, 'Saving dataset...\n');
save(fullfile(curDir, 'dataset.mat'), 'dataset');

fprintf(1, 'Done.\n');