function [X, I, S] = load_hk_original(dbPath)
dbPath = fullfile(dbPath, 'FingerVein');

repeat = 6;
h      = 256;
w      = 513;

X = zeros(h,w,0, 'uint8');
I = zeros(1,0, 'uint32');
S = zeros(1,0, 'uint32');

for n = 1:156
    % finger 1 session 1
    for r = 1:repeat
        fileName = sprintf('%s_%d_f1_1.bmp', num2str(n), r);
        filePath = fullfile(dbPath, num2str(n), 'f1', '1', fileName);
        im = imread(filePath);
        X  = cat(3, X, im);
        I  = cat(1, I, n);
        S  = cat(1, S, 1);
    end
    
    % finger 2 session 1
    for r = 1:repeat
        fileName = sprintf('%s_%d_f2_1.bmp', num2str(n), r);
        filePath = fullfile(dbPath, num2str(n), 'f2', '1', fileName);
        im = imread(filePath);
        X  = cat(3, X, im);
        I  = cat(1, I, 156 + n);
        S  = cat(1, S, 1);
    end
    
    if exist(fullfile(dbPath, num2str(n), 'f1', '2'), 'dir') == 7
        % finger 1 session 2
        for r = 1:repeat
            fileName = sprintf('%s_%d_f1_2.bmp', num2str(n), r);
            filePath = fullfile(dbPath, num2str(n), 'f1', '2', fileName);
            im = imread(filePath);
            X  = cat(3, X, im);
            I  = cat(1, I, n);
            S  = cat(1, S, 2);
        end
        
        % finger 2 session 2
        for r = 1:repeat
            fileName = sprintf('%s_%d_f2_2.bmp', num2str(n), r);
            filePath = fullfile(dbPath, num2str(n), 'f2', '2', fileName);
            im = imread(filePath);
            X  = cat(3, X, im);
            I  = cat(1, I, 156 + n);
            S  = cat(1, S, 2);
        end
    end
end
end