function [] = showClassSeparation(X1, X2, label)
    D = sum((X1 - X2) .^ 2, 1);
    [h1, x1] = hist(D(label),30);
%     h1 = h1 / sum(h1);
    [h2, x2] = hist(D(~label),30);
%     h2 = h2 / sum(h2);
    bar(x1,h1, 0.3, 'g');
    hold on
    bar(x2,h2, 0.3, 'r');
    hold off
    legend({'inClass', 'outClass'});
end

