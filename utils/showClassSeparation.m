function [] = showClassSeparation(X1, X2, label)
    D = sqrt(sum((X1 - X2) .^ 2, 1));
%     D  = min(1, sum(X1 .* X2) ./ (sum(X1.^2) .* sum(X2.^2)));
    histogram(D(label),'BinWidth',0.1);
	hold on;
    histogram(D(~label),'BinWidth',0.1);
    legend({'inClass', 'outClass'});
    hold off
end

