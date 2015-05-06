function [samples, scores] = activateLayers(dnn, l, copies, method, varargin)

    if strcmp(method, 'activationMax')
        % Use gradient descent to have an output that has 1 on the observed 
        % neuron and 0 otherwise.
        
        if nargin < 4
            error('missing input data samples');
        end
        a  = varargin{1};
        maxIter  = varargin{2};
        eps = varargin{3};
        
        
        in = varargin{4};
        
        inDim    = size(dnn.rbm{1}.W, 1);
        outDim   = size(dnn.rbm{l}.W, 2);
        dnn.rbm  = dnn.rbm(1:l);
        scores   = zeros(outDim, copies);
        
        % Normalize
        in       = bsxfun(@minus, in, min(in,[], 2));
        in       = bsxfun(@rdivide, in, max(in,[], 2) + 0.001);
        out      = v2h(dnn, in);
        
        % Iterations start from best samples
        samples  = zeros(inDim, copies, outDim);

        for k = 1:min(10, outDim)
            u    = zeros(1, outDim);
            u(k) = 1;
            tmp = bsxfun(@minus, out, u);
            tmp = sum(tmp .* tmp, 2);
            [~, idx] = sort(tmp, 1, 'ascend');
            idx      = idx(1:min(10*copies, size(in,1)), :);
            
            A = in(idx, :);
            [~, C] = kmeans(A, copies);
            samples(:,:,k) = C';
        end

%         samples = reshape(in, inDim, copies, outDim);

        gbak = zeros(inDim,1);
        for k = 1:min(10, outDim)
            u    = zeros(1, outDim);
            u(k) = 1;
            for j = 1:copies
                for i = 1:maxIter % Gradient descent
                    [G, H, S] =  inputgradient(dnn, samples(:, j, k)');
                    e = S - u;
                    g = (G * e') ./  (0.01 + H * e' + sum(G.^2,2));
                    if norm(g) > 1
                        g = g / norm(g);
                    end
                    
                    % Update input
                    samples(:, j, k) = bsxfun(@minus, samples(:, j, k), g);

                    if mod(i, 20) == 1 % Print progress
                        q = norm(g - gbak);
                        fprintf(1,'%2d,%2d : %.4f , %.4f, %.4f, %.4f , %.4f\n', ...
                            k, j, norm(g), norm(e), q/norm(e), ...
                            min(samples(:, j, k)), max(samples(:, j, k)));
                    end
                    
                    gbak = g;
                end
                scores(k, j) = norm(e);
            end
        end
        
    elseif strcmp(method, 'generate')
        dnn.rbms = dnn.rbms(1:l);
        inDim    = dnn.rbms{1}.rbmParams.numVis;
        outDim   = dnn.rbms{end}.rbmParams.numHid;
        samples  = zeros(inDim, copies, outDim);
        scores   = zeros(outDim, copies);
        
        %tmp = zeros(copies * 100, inDim);
        
        for k = 1:min(10, outDim)
            % generates samples
            out      = zeros(copies*150, outDim);
            out(:,k) = 1;
            tmp      = dnn.generateData(out, 1);
            
            % group similar samples using k-means
            [idx, C]       = kmeans(tmp, copies);
            samples(:,:,k) = C';
            scores(k,:)    = hist(idx, copies);
        end
        
    elseif strcmp(method, 'posterior')
        inDim    = dnn.rbms{1}.rbmParams.numVis;
        outDim   = dnn.rbms{l}.rbmParams.numHid;
        samples  = zeros(inDim, 1, outDim);
        scores   = ones(outDim, 1);
        
        for k = 1:outDim
            % generates samples
            out      = zeros(1, outDim);
            out(:,k) = 1;
            for layer = l:-1:1
                [~, out] =  dnn.rbms{layer}.sampler.down(dnn.rbms{layer}.rbmParams, out);
            end
            samples(:,1,k) = out;
        end
    else
        error('Undefined method');
    end
end
