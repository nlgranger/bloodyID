colormap gray; 
for i=1:150;
    subplot(2,1,1)
    imagesc(reshape(extractionNet.nets{2}.net.W(:, i), 19, 19));
    axis image
    colorbar;
    subplot(2,1,2)
    imagesc(reshape(wholeNet.nets{1}.net.nets{2}.net.W(:, i), 19, 19));
    axis image
    colorbar
    pause;
end

subplot(1,1,1)
y = wholeNet.nets{1}.compute(allX);

for i = 1:80
    hist([y{1}(i,:) y{2}(i,:)], 30);
    pause;
end    

subplot(1,1,1)
y = wholeNet.nets{1}.compute(allX);

for i = 1:80
    hist([y{1}(i,:) y{2}(i,:)], 30);
    pause;
end    