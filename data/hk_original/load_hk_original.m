function database = load_hk_original(path)
    dbpath = fileparts(mfilename('fullpath'));
    
    folders = dir(dbPath);
    
    N = length(folders);
    repeat = 6;
    h      = 256;
    w      = 513;
    
    database.train_x = zeros(2 * N * repeat, h*w, 'uint8');
    database.train_m = true(2 * N * repeat, h*w);
    database.train_y = zeros(2 * N * repeat, 1, 'uint32');
    database.test_x  = zeros(2 * N * repeat,  h*w, 'uint8');
    database.test_m  = true(2 * N * repeat,  h*w);
    database.test_y  = zeros(2 * N * repeat,  1, 'uint32');
    
    na = 0;
    nt = 0;
    for n = 1:length(folders)
        f = folders(n);
        if ~(f.isdir) || strcmp(f.name, '.') || strcmp(f.name, '..')
            continue
        end
        
        % finger 1 session 1
        for r = 1:repeat
            na = na + 1;
            fileName = sprintf('%s_%d_f1_1.bmp', f.name, r);
            filePath = fullfile(dbPath, f.name, 'f1', '1', fileName);
            im = imread(filePath);
            database.train_x(na,:) = reshape(im, 1, h*w);
            database.train_y(na)   = str2double(f.name) * 10;
        end
        % finger 1 session 2
        if exist(fullfile(dbPath, f.name, 'f1', '2'), 'dir') == 7
            for r = 1:repeat
                nt = nt + 1;
                fileName = sprintf('%s_%d_f1_2.bmp', f.name, r);
                filePath = fullfile(dbPath, f.name, 'f1', '2', fileName);
                im = imread(filePath);
                database.test_x(nt,:) = reshape(im, 1, h*w);
                database.test_y(nt)   = str2double(f.name) * 10;
            end
        end
        % finger 2 session 1
        for r = 1:repeat
            na = na + 1;
            fileName = sprintf('%s_%d_f2_1.bmp', f.name, r);
            filePath = fullfile(dbPath, f.name, 'f2', '1', fileName);
            im = imread(filePath);
            database.train_x(na,:) = reshape(im, 1, h*w);
            database.train_y(na)   = str2double(f.name) * 10 + 1;
        end
        % finger 2 session 2
        if exist(fullfile(dbPath, f.name, 'f2', '2'), 'dir') == 7
            for r = 1:repeat
                nt = nt + 1;
                fileName = sprintf('%s_%d_f2_2.bmp', f.name, r);
                filePath = fullfile(dbPath, f.name, 'f2', '2', fileName);
                im = imread(filePath);
                database.test_x(nt,:) = reshape(im, 1, h*w);
                database.test_y(nt)   = str2double(f.name) * 10 + 1;
            end
        end
    end
    
    database.train_x = database.train_x(1:na, :);
    database.train_y = database.train_y(1:na);
    database.train_m = database.train_m(1:na, :);
    database.test_x = database.test_x(1:nt, :);
    database.test_y = database.test_y(1:nt);
    database.test_m = database.test_m(1:nt, :);
    database.h      = h;
    database.w      = w;
end