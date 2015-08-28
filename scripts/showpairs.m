function [] = showpairs(X1, X2)

colormap gray

for i = 1:size(X1, 3)
    X3 = zeros(size(X1(:,:,1)));
    subplot(2,1,1)
    imagesc(X1(:,:,i));
    axis image
    subplot(2,1,2)
    imagesc(cat(3,X1(:,:,i),X2(:,:,i), X3));
    axis image
    pause
end