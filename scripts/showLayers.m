function [] = showLayers(samples, scores, copies, h, w)

    cols = ceil(sqrt(copies));

    % Display
    colormap('gray');
    for k = 1:size(samples, 3)
        for j = 1:copies
            subplot(cols, min(cols, copies), j);
            tmp = samples(:, j, k);
            imagesc(reshape(tmp, h, w));
            title(sprintf('%d: %f', k, scores(k, j)));
            axis equal;
            axis off;
        end
        pause;
    end

end