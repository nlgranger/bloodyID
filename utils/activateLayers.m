function [samples, scores] = activateLayers(dnn, layer, clusters)
    % ACTIVATELAYERS sample input from indicator output
    %   TODO: write description
    inDim    = dnn.nets{1}.insize();
    outDim   = dnn.nets{layer}.outsize();
    samples  = zeros(inDim, clusters, outDim);
    scores   = zeros(outDim, clusters);

    parfor k = 10:min(30, outDim)
        % generates samples
        tmp      = zeros(clusters*150, outDim);
        tmp(:,k) = 1;
        for l = layer:-1:1
            tmp = dnn.nets{l}.visGivHid(tmp, true);
        end

        % group similar samples using k-means
        [idx, C]       = kmeans(tmp, clusters);
        samples(:,:,k) = C';
        scores(k,:)    = hist(idx, clusters);
    end
end
