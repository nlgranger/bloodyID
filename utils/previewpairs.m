function [] = previewpairs(X1, X2, y)

colormap gray

optimizer = registration.optimizer.OnePlusOneEvolutionary;
metric    = registration.metric.MeanSquares;
optimizer.InitialRadius = 0.002;
optimizer.Epsilon = 5e-3;
optimizer.GrowthFactor = 1.5;
optimizer.MaximumIterations = 70;

for i = 30:size(X1, 3)
    subplot(3,1,1)
    imagesc(X1(:,:,i));
    axis image
    subplot(3,1,2)
    imagesc(X2(:,:,i));
    axis image
    
    Ir = imregister(uint8(X1(:,:,i)), uint8(X2(:,:,i)), 'similarity', optimizer, metric);
    subplot(3,1,3)
    imshowpair(Ir, X2(:,:,i));
    
    res = [sum(sum(X2(:,:,i))), sum(sum(X1(:,:,i))), sum(sum(X2(:,:,i) & Ir))];
    
    fprintf('%-.3f . %d\n', res(3)./mean(res(1),res(2)) , y(i));
    
    pause
end