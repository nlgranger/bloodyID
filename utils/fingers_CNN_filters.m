colormap gray; 
for i=1:10
    imagesc(net.nets{1}.net.nets{1}.filters(:,:, i));
    axis image
    colorbar;
    pause;
end

y = wholeNet.nets{1}.compute(allX);

for i = 1:70
    hist([y{1}(i,:) y{2}(i,:)], 30);
    pause;
end

subplot(1,1,1)
y = wholeNet.nets{1}.compute(allX);

for i = 1:80
    hist([y{1}(i,:) y{2}(i,:)], 30);
    pause;
end