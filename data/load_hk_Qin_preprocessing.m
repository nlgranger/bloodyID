function database = load_hk_Qin_preprocessing(dbpath, h, w, trainingRatio)
    assert(exist(dbpath, 'dir') == 7,'Path not found');
    
    copies  = 6; % number of images by finger
    [rh,rw] = size(imread(fullfile(dbpath, 'finger_veins', '(1)/(1).bmp')));

    session1.data = false(156 * copies, h*w);
    session2.data = false(156 * copies, h*w);
    session1.mask = false(156 * copies, h*w);
    session2.mask = false(156 * copies, h*w);
    session1.id   = zeros(156 * copies, 1, 'int32');
    session2.id   = zeros(156 * copies, 1, 'int32');
    
    database = {};
    database.train_x = false(0,h*w);
    database.train_m = false(0,h*w);
    database.train_y = [];
    database.test_x = false(0,h*w);
    database.test_m = false(0,h*w);
    database.test_y = [];

    veinsPath = fullfile(dbpath, 'finger_veins');
    masksPath = fullfile(dbpath, 'finger_masks');
    veinDirs = dir(veinsPath);
    
    n1 = 0;
    n2 = 0;

    for d = 1:length(veinDirs)
        subDir = veinDirs(d);
        if ~subDir.isdir ...
            || strcmp(subDir.name,'.') ...
            || strcmp(subDir.name, '..')
            continue
        elseif strcmp(subDir.name(end-1:end), 's2')
            imv = imread(fullfile(veinsPath, subDir.name, '(7).bmp'));
            if size(imv, 1) ~= rh || size(imv, 2) ~= rw % skip corrupted files
                continue;
            end
            n2 = n2 + 1;
            id = str2double(subDir.name(2:end-4));
            if id == 0
                pause;
            end
            isFirstSession = false;
        else
            n1 = n1 + 1;
            id = str2double(subDir.name(2:end-1));
            if id == 0
                pause;
            end
            isFirstSession = true;
        end
        fprintf(1, '%d\n', id);
        for i = 1:copies
            imFileName = sprintf('(%d).bmp', round(i + 6 * ~isFirstSession));
            imv = imread(fullfile(veinsPath, subDir.name, imFileName));
            imv = imresize(imv, [h w], 'bilinear') > 128;
            imm = imread(fullfile(masksPath, subDir.name, imFileName));
            imm = imresize(imm, [h w], 'bilinear') > 128;
            
            if isFirstSession
                session1.data((n1-1)*copies+i,:) = reshape(imv, 1, h*w);
                session1.mask((n1-1)*copies+i,:) = reshape(imm, 1, h*w);
                session1.id((n1-1)*copies+i)     = id;
            else
                session2.data((n2-1)*copies+i,:) = reshape(imv, 1, h*w);
                session2.mask((n2-1)*copies+i,:) = reshape(imm, 1, h*w);
                session2.id((n2-1)*copies+i)     = id;
            end
        end
    end
    
    session1.id = session1.id(1:n1*copies);
    session2.id = session2.id(1:n2*copies);
    
    % number of individuals used for training
    ntrain       = round((n1 + n2) * trainingRatio);
    % participants of session 1 only serve for training
    s1only      = setdiff(session1.id, session2.id);
    both        = intersect(session1.id, session2.id);
    both        = both(randperm(numel(both)));
    nbmoretrain = max(0, ntrain - length(s1only));
    
    for i = 1:numel(s1only)
        t = s1only(i);
        idx = find(bsxfun(@eq, session1.id, t));
        database.train_x = [database.train_x;
                            session1.data(idx,:)];
        database.train_m = [database.train_m;
                            session1.mask(idx,:)];
        database.train_y = [database.train_y;
                            session1.id(idx)];
    end
    for i = 1:round(nbmoretrain/2)
        t = both(i);
        idx = find(bsxfun(@eq, session1.id, t));
        database.train_x = [database.train_x;
                            session1.data(idx,:)];
        database.train_m = [database.train_m;
                            session1.mask(idx,:)];
        database.train_y = [database.train_y;
                            session1.id(idx)];
        idx = find(bsxfun(@eq, session2.id, t));
        database.train_x = [database.train_x;
                            session2.data(idx,:)];
        database.train_m = [database.train_m;
                            session2.mask(idx,:)];
        database.train_y = [database.train_y;
                            -1 * session2.id(idx)];
    end
    for i = round(nbmoretrain/2)+1:numel(both)
        t = both(i);
        idx = find(bsxfun(@eq, session1.id, t));
        database.test_x = [database.test_x;
                            session1.data(idx,:)];
        database.test_m = [database.test_m;
                            session1.mask(idx,:)];
        database.test_y = [database.test_y;
                            session1.id(idx)];
        idx = find(bsxfun(@eq, session2.id, t));
        database.test_x = [database.test_x;
                            session2.data(idx,:)];
        database.test_m = [database.test_m;
                            session2.mask(idx,:)];
        database.test_y = [database.test_y;
                           -1 * session2.id(idx)];
    end
    
    database.train_x  = database.train_x';
    database.test_x   = database.test_x';
    database.h        = h;
    database.w        = w;
end
