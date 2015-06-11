trainRatio   = 0.7;
valRatio     = 0;
nMatching    = 4;
nNonMatching = 6;
h = 100;
ratio = 2.3;
w = round(ratio * h);

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
raw = X;
X = zeros(h, w, size(raw, 3));
M = true(h, w, size(raw, 3));
keep = true(size(raw, 3), 1);
for i = 1:size(raw, 3)
    [O, Ma] = fingerExtraction(raw(:,:,i), h, ratio);
    if numel(O) == 0
        warning('rejected finger %d', i);
        keep(i) = false;
    else
        X(:,:,i) = O;
        M(:,:,i) = Ma;
    end
end
X = X(:,:,keep);
M = M(:,:,keep);
S = S(keep);
I = I(keep);

fprintf(1, 'Saving...\n');
save(fullfile(curDir, 'preprocessed.mat'), 'M', 'X', 'I', 'S');

% Preview
% colormap gray
% for i = 1:312
%     s1 = find(I == i & S == 1);
%     s2 = find(I == i & S == 2);
%     for j = 1:numel(s1)
%         subplot(6,2,2*j-1)
%         imagesc(X(:, :, s1(j)));
%         axis image
%         axis off
%     end
%     for j = numel(s1)+1:6
%         cla(subplot(6,2,2*j-1))
%     end
%     for j = 1:numel(s2)
%         subplot(6,2,2*j)
%         imagesc(X(:, :, s2(j)));
%         axis image
%         axis off
%     end
%     for j = numel(s2)+1:6
%         cla(subplot(6,2,2*j-1))
%     end
%     pause
% end

% Assign individuals to training, validation and testing
fprintf(1, 'Assigning caterogries...\n');
category  = zeros(numel(I), 1, 'int32');
for id = 1:312
    fromS1 = find(I == id & S == 1, 6);
    fromS2 = find(I == id & S == 2, 6);
    
    if numel(fromS1) == 0 || numel(fromS2) < nMatching || id == 35
            category([fromS2; fromS2]) = 4;
    else
        c = rand();
        if c < trainRatio
            category([fromS1; fromS2]) = 1;
        elseif c < trainRatio + valRatio
            category([fromS1; fromS2]) = 2;
        else
            category([fromS1; fromS2]) = 3;
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
        shuffle  = randperm(numel(fromS2), nMatching);
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