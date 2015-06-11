function [O, M] = fingerExtraction(I, h, ratio)

M = I < 245;
M = imopen(M, strel('disk', 5));

[y, x] = find(M(10:end-10, 100:end-100));
tmp = cov(x, y);
a  = atan(tmp(1,2)/var(x));
R = imrotate(I, a*180/pi, 'bilinear');
M = imrotate(M, a*180/pi, 'bilinear');
left  = find(M(round(size(M, 1)/2), :), 1);
right = find(M(round(size(M, 1)/2), :), 1, 'last');
M(:, 1:left+40) = false;
M(:, right-40:end) = false;

G = imgradient(R);
M(G>60) = false;


[y, ~] = find(M);
m      = mean(y);
[L, n] = bwlabel(M, 4);
for k = 1:n
    idx = (L == k);
    [y, ~] = find(idx);
    ma     = mean(y);
    sa     = std(y);
    if (abs(m-ma) > 110) || (sa < 30)
        M(idx) = false;
    end
end

if ~any(any(M))
    O = [];
    M = [];
    return
end

M = imfill(imclose(M, strel('disk', 5)), 'holes');

[y, x] = find(M);
tmp = cov(x, y);
a  = atan(tmp(1,2)/var(x));
R = imrotate(R, a*180/pi, 'bilinear');
M = imrotate(M, a*180/pi, 'bilinear');

right = find(M(round(size(M, 1)/2), :), 1, 'last');
his   = filter2(ones(1,30), sum(R .* uint8(M))./(sum(M)+1));
his   = filter2(ones(1,15), his)/(30*15);
% subplot(2,1,2);plot(his)
d     = .5 * (his(11:end-12) - his(13:end-10));
dipj  = find(11:numel(his)-12 < right - 50 ... % crop region
    & 11:numel(his)-12 > 280 ... % crop region
    & abs(d) < 0.05 ... % null derivative
    & .5 * (his(1:end-22) + his(23:end)) <= his(12:end-11), ... % curvature
    1, 'first') + 11;
if numel(dipj) == 0 || dipj <= left+200
    O = [];
    M = [];
    return
end

% define ROI
[his, centers] = hist(sum(M(:, dipj-120:dipj+60)));
[~, tmp] = max(his);
height = round(centers(tmp));
width  = round(ratio*height);
[y, ~] = find(M(:, dipj-120:dipj+60));
m = mean(y);
x      = round(dipj - 2/3 * width);
y      = round(m - height/2);

if x < 1 || x+width > size(R, 2) || y < 1 || y+height > size(R, 1)
    O = [];
    M = [];
    return
end

% Remove low freq
O  = double(R);
bg = ones(size(O)) * mean(O(M));
bg(M) = O(M);
bg = filter2(fspecial('gaussian', [20 20], 15), bg);
O  = tanh((O - bg)/20);

% Cut ROI and resize
O = O(y:y+height, x:x+width);
M = M(y:y+height, x:x+width);
M = imresize(M, [h round(ratio*h)]);
O = imresize(O, [h round(ratio*h)]);

% Spread histogram
M2    = bwmorph(M, 'erode', 8);
[~, bins] = hist(O(M2));
mini  = bins(1);
maxi  = bins(end);
O     = max(min(tanh((O-mini)/(maxi-mini))*255, 255), 0);
O(~M) = 0;
end