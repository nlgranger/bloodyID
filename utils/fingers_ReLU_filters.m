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
    disp(mean(abs(extractionNet.nets{2}.net.W(:, i) ...
        - wholeNet.nets{1}.net.nets{2}.net.W(:, i))));
end

subplot(1,1,1)
y = wholeNet.nets{1}.compute(allX);

for i = 1:80
    hist([y{1}(i,:) y{2}(i,:)], 30);
    pause;
end



[allX, allY] = trainOpts.batchFn(X, Y, inf, []);
o = wholeNet.compute(allX);
eer = fminsearch(@(t) abs(mean(o(allY < 0.5)<t) - mean(o(allY > 0.5)>=t)), double(mean(o)));
subplot(2,1,1)
hold off
histogram(o(allY > 0.5), 'binWidth', 0.02);
hold on
histogram(o(allY < 0.5), 'binWidth', 0.02);
plot(eer, 0, 'r*')
hold off

[allX, allY] = trainOpts.batchFn(Xv, Yv, inf, []);
o = wholeNet.compute(allX);
subplot(2,1,2)
hold off
histogram(o(allY > 0), 'binWidth', 0.02);
hold on
histogram(o(allY < 0.5), 'binWidth', 0.02);
plot(eer, 0, 'r*')
hold off

colormap gray;
o = wholeNet.compute(allX);
eer = fminsearch(@(t) abs(mean(o(allY == 0) < t) ...
    - mean(o(allY > 0) >= t)), double(mean(o)));
m = (o > eer) ~= (allY > 0);

for posErr = find(m & (allY > 0.5))
    colormap gray;
   subplot(2, 2, 1);
   imagesc(allX{1}(:,:,posErr));
   colorbar
   axis image;
   subplot(2, 2, 3);
   imagesc(allX{2}(:,:,posErr));
   colorbar
   axis image;
   subplot(2, 2, [2; 4]);
   p = wholeNet.nets{1}.net.compute(allX{1}(:,:,posErr));
   n = wholeNet.nets{1}.net.compute(allX{2}(:,:,posErr));
   newplot
   hold on;
   bh = bar(p, 'r');
   set(bh,'edgecolor','none');
   bh = bar(-n, 'g');
   set(bh,'edgecolor','none');
   bh = bar(p-n, 'b');
   set(bh,'edgecolor','none');
   hold off
   title(sprintf('%e < %e\n', o(posErr), eer));
   pause
end

for negErr = find(m & (allY < 0.5))
    colormap gray;
   subplot(2, 2, 1);
   imagesc(allX{1}(:,:,negErr));
   colorbar
   axis image;
   subplot(2, 2, 3);
   imagesc(allX{2}(:,:,negErr));
   colorbar
   axis image;
   subplot(2, 2, [2; 4]);
   p = wholeNet.nets{1}.net.compute(allX{1}(:,:,negErr));
   n = wholeNet.nets{1}.net.compute(allX{2}(:,:,negErr));
   newplot
   hold on;
   bh = bar(p, 'r');
   set(bh,'edgecolor','none');
   bh = bar(-n, 'g');
   set(bh,'edgecolor','none');
   bh = bar(p-n, 'b');
   set(bh,'edgecolor','none');
   hold off
   title(sprintf('%e > %e\n', o(negErr), eer));
   pause
end