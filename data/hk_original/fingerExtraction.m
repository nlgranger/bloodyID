function [O, M] = fingerExtraction(I, h, ratio)
%fingerExtraction extract a ROI in the finger
%   [O, M] = fingerExtraction(I, h, ratio) returns the region of interest
%   around the third phalangeal distal joint. The ROI takes the average
%   height of the finger in this region and its width is computed using
%   ratio. Horizontal alignment is 2/3 on the left of the joint and 1/3 on
%   the right. The image is then resized to h x round(ratio*h).
%   This function also does a little bit of contrast enhancement at the end.

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

[y, x] = find(M(50:end-20));
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
% [his, centers] = hist(sum(M(:, dipj-120:dipj+60)));
% [~, tmp] = max(his);
% height = round(centers(tmp));
% width  = round(ratio*height);
w = round(h*ratio);
[y, ~] = find(M(:, dipj-120:dipj+60));
m      = mean(y);
x      = round(dipj - 3/4 * w);
y      = round(m - h/2);

if x < 1 || x+w > size(R, 2) || y < 1 || y+h > size(R, 1)
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
O = O(y:y+h, x:x+w);
M = M(y:y+h, x:x+w);

% Spread histogram
M2    = bwmorph(M, 'erode', 6);
m     = mean(O(M2));
s     = std(O(M2));
O     = tanh((O-m)/(2*s));
O(~M2) = 1;
end