% Configuration ------------------------------------------------------------- %

% ratio of training/validation individuals among the participants of both 
% sessions, the remaining ones go to testing
trainRatio   = 0.7;
valRatio     = 0;

% # of matching and non matching pairs from session 2 generated for each
% sample of session 1
nMatching    = 4;
nNonMatching = 6;

% Height and ratio (w/h) of the images
h            = 35;
ratio        = 2.3;
w            = round(ratio * h);

% Artificial images generation
nExtra       = 0; % # of artificial *training* images for each original one
angleStd     = 1.5; % angle std deviation in degrees
shiftStd     = 2; % shift std deviation in pixels

% path to FingerVein/ directory (not included)
curDir = fileparts(mfilename('fullpath'));

% Main Script --------------------------------------------------------------- %

% Purge existing dataset
clear dataset

% % Image Loading
% fprintf(1, 'Loading image files ...\n');
% if exist(fullfile(curDir, 'raw.mat'), 'file') % use backup file if available
%     load(fullfile(curDir, 'raw.mat'));
% else
%     [X, I, S] = load_hk_original();
%     save(fullfile(curDir, 'raw.mat'), 'X', 'I', 'S');
% end
% 
% % Image Preprocessing
% fprintf(1, 'Extracting fingers ...\n');
% raw = X;
% X = zeros(h, w, size(raw, 3));
% M = true(h, w, size(raw, 3));
% keep = true(size(raw, 3), 1);
% for i = 1:size(raw, 3)
%     [O, Ma] = fingerExtraction(raw(:,:,i), 150, ratio);
%     if numel(O) == 0
%         warning('rejected finger %d', i);
%         keep(i) = false;
%     else
%         X(:,:,i) = imresize(O, [h w]);
%         M(:,:,i) = imresize(Ma, [h w]);
%     end
% end
% X = X(:,:,keep);
% M = M(:,:,keep);
% S = S(keep);
% I = I(keep);
% 
% fprintf(1, 'Saving ...\n');
% save(fullfile(curDir, 'preprocessed.mat'), 'M', 'X', 'I', 'S');

load(fullfile(curDir, 'preprocessed.mat'));

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

% load( fullfile(curDir, 'preprocessed.mat'))

% Assign individuals to training, validation and testing
fprintf(1, 'Assigning caterogries ...\n');
C = zeros(numel(I), 1, 'int32');
nPretrain = 0;
nTrain    = 0;
nVal      = 0;
nTest     = 0;
for id = 1:312
    fromS1 = find(I == id & S == 1, 6);
    fromS2 = find(I == id & S == 2, 6);
    
    if isempty(fromS1) || numel(fromS2)*(1+nExtra) < nMatching || id == 35
            C([fromS1; fromS2]) = 0; % Pretraining
            nPretrain = nPretrain + numel(fromS1) + numel(fromS2);
    else
        c = rand();
        if c < trainRatio
            C([fromS1; fromS2]) = 1; % Training
            nTrain = nTrain + numel(fromS1) + numel(fromS2);
        elseif c < trainRatio + valRatio
            C([fromS1; fromS2]) = 2; % Validation
        else
            C([fromS1; fromS2]) = 3; % Testing
        end
    end
end

% Artificial samples
fprintf(1, 'Generating artificial training images ...\n');

T     = find(C == 1 | C == 0);
Xtra  = zeros(h ,w, nExtra*numel(T));
Ixtra = zeros(nExtra*numel(T), 1, 'uint32');
Sxtra = zeros(nExtra*numel(T), 1, 'uint32');
Cxtra = zeros(nExtra*numel(T), 1, 'uint32');
for k = 1:numel(T)
    for l = 1:nExtra
        n = (k-1)*nExtra+l;
        
        shift       = round(randn(2, 1) * angleStd);
        a           = randn() * shiftStd;
        tmp         = imrotate(imshift(X(:,:,T(k)), shift, 1), a, 'bilinear', 'crop');
        tmpM        = imrotate(imshift(M(:,:,T(k)), shift, 1), a, 'bilinear', 'crop');
        tmp(~tmpM)  = 1;
        Xtra(:,:,n) = tmp;
        Ixtra(n)    = I(T(k));
        Sxtra(n)    = S(T(k));
        Cxtra(n)    = C(T(k));
    end
end

X = cat(3, X, Xtra);
I = [I; Ixtra];
S = [S; Sxtra];
C = [C; Cxtra];
nPretrain = nPretrain * (nExtra+1);
nTrain    = nTrain * (nExtra+1);

% Sort samples by category
[C, s] = sort(C);
X = X(:,:,s);
I = I(s);
S = S(s);

dataset.pretrain_x = X(:,:,1:nPretrain + nTrain);
C = C(nPretrain+1:end);
X = X(:, :, nPretrain+1:end);
I = I(nPretrain+1:end);
S = S(nPretrain+1:end);

% Make pairs
fprintf(1, 'Building pairs ...\n');

shuffle = zeros(sum(S==1)*(nMatching + nNonMatching), 2 ,'uint32');
Y       = false(sum(S==1)*(nMatching + nNonMatching), 1);
Cnew    = zeros(sum(S==1)*(nMatching + nNonMatching), 1, 'uint32');

i = 1;
for id = unique(I)'
    fromS1 = find(I == id & S == 1, 6 * (1+nExtra));
    fromS2 = find(I == id & S == 2, 6 * (1+nExtra));
    c = C(fromS1(1));
    nonpeers = find(I ~= id & C == c);
    Cnew(i:i+nMatching+nNonMatching) = c;
    
    shuffle(i:i + numel(fromS1) * nMatching - 1, 1) = ...
        reshape(repmat(fromS1', nMatching, 1), 1, []);
    shuffle(i:i + numel(fromS1) * nMatching - 1, 2) = ...
        fromS2(randi(numel(fromS2), 1, numel(fromS1)*nMatching));
    Y(i:i + numel(fromS1) * nMatching) = true;
    
    i = i + numel(fromS1) * nMatching;
    
    shuffle(i:i + numel(fromS1) * nNonMatching - 1, 1) = ...
        reshape(repmat(fromS1, nNonMatching, 1), 1, []);
    shuffle(i:i + numel(fromS1) * nNonMatching - 1, 2) = ...
        nonpeers(randi(numel(nonpeers), 1, numel(fromS1)*nNonMatching));
    i = i + numel(fromS1) * nNonMatching;
end

% Build dataset
fprintf(1, 'Packing up dataset ...\n');
dataset.train_x    = { X(:,:,shuffle(C == 1, 1)), ...
                       X(:,:,shuffle(C == 1, 2)) };
dataset.train_y    = Y(C == 1);
dataset.val_x    = { X(:,:,shuffle(C == 2, 1)), ...
                     X(:,:,shuffle(C == 2, 2)) };
dataset.val_y    = Y(C == 2);
dataset.test_x    = { X(:,:,shuffle(C == 3, 1)), ...
                      X(:,:,shuffle(C == 3, 2)) };
dataset.test_y    = Y(C == 3);

fprintf(1, 'Saving dataset...\n');
save(fullfile(curDir, 'dataset.mat'), 'dataset');

fprintf(1, 'Done.\n');