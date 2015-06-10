function [O, M] = fingerExtraction(I)
patchSize = 40;
ratio = 2;
[h, w] = size(I);
g = fspecial('gaussian', [30 20], 5);
e = fspecial('gaussian', 15, 15);
p = fspecial('gaussian',[patchSize patchSize], patchSize);

% Center and align image
B = imfilter(I, g);
m = mean(mean(B(100:end-100, 25:end-25)));
M = B < m;
[y, x] = find(M(100:end-100, 25:end-25));
tmp    = cov(x,y);
angle  = atan(tmp(2,1)/var(x));
O      = imrotate(I, 180*angle/pi, 'bilinear');
M      = imrotate(M, 180*angle/pi, 'bilinear');
A      = imrotate(true(h,w), 180*angle/pi, 'bilinear');
[y, ~] = find(M(100:end-100, 25:end-25));
off    = round(size(O,1)/2-mean(y))+25;
if off > 0
    O = [zeros(off, size(O, 2)); O(1:end-off, :)];
    A = [zeros(off, size(O, 2)); A(1:end-off, :)];
elseif off < 0
    O = [O(-off:end, :); zeros(-off, size(O, 2))];
    A = [A(-off:end, :); zeros(-off, size(O, 2))];
end
offy   = floor((size(O,1)-h)/2)+1;
offx   = floor((size(O,2)-w)/2)+1;
O      = O(offy:offy+h-1, offx:offx+w-1);
A      = A(offy:offy+h-1, offx:offx+w-1);

% Remove overexposed or underexposed areas and border
B = imfilter(O, e);
M = B > min(min(B)) + 10 & B < max(max(B)) - 10 & A ...
    & repmat((1:w < round(11/12*w)) & (1:w > round(w/12)), h,1) ...
    & repmat(((1:h)' < round(11/12*h)) & ((1:h)' > round(h/12)), 1,w);

% Find edges
G = imgradient(B);
G = tanh(std(reshape(G, numel(G), 1)));
bg = imfilter(G, p);
G = G - bg;
m = mean(G(M));
s = std(G(M));
U = (G > m);
M = U | ~M;
M = ~imopen(M, strel('disk', 5'));

% Keep finger areas (along the horizon)
[L,n] = bwlabel(M, 4);
for k = 1:n
    idx = (L == k);
    [y, ~] = find(L.*idx);
    m = mean(y);
    s = std(y);
    if m < h/3 || m > 2*h/3 || s > h/4;
        M(idx) = 0;
    end
end

M = imclose(M, strel('disk', 8));
M = imfill(M, 'holes');

if any(sum(M(:,100:end-100))==0)
    M = false;
    O = [];
    return
end

% Find ROI
width = sum(M);
width(width < 50) = -1;
his   = filter2(ones(1,40), sum(O .* uint8(M)));
dipj   = find((his(2:end-1) > 10000) ...
    & abs(.5 * (his(1:end-2) - his(3:end))) < 300, ...
    1, 'last') + 1;
if numel(dipj) == 0 || dipj <= 250 || dipj >= 473
    M = false;
    O = [];
    return
end

[y, x] = find(M(:, dipj-250:dipj+50));
tmp    = cov(x,y);
angle2 = atan(tmp(2,1)/var(x));
O      = imrotate(O, 180*angle2/pi, 'bilinear');
M      = imrotate(M, 180*angle2/pi, 'bilinear');
h      = round(sum(sum(M(:, dipj-250:dipj+50)))/600);
w      = round(ratio*h*2/5);
m      = round(mean(sum(bsxfun(@times, ...
                               M(:, dipj-250:dipj+50), ...
                               (1:size(M, 1))')) ...
                    ./ (sum(M(:, dipj-250:dipj+50))+1)));
                
% Remove low freq colors
% b    = ones(2*patchSize+1);
% bg   = conv2(double(O), b, 'same');
% d    = conv2(double(M), b, 'same');
% d(d==0) = 1;
% bg      = bg ./ d;
O  = double(O);
bg = ones(size(O)) * mean(O(M));
bg(M) = O(M);
bg = filter2(p, bg);
O       = O - bg;

% Cut ROI and resize
M = M(m-h:m+h, dipj-4*w:dipj+w);
O = O(m-h:m+h, dipj-4*w:dipj+w);
M = imresize(M, [100 200]);
O = imresize(O, [100 200]);

% Spread histogram
mini  = min(O(M));
maxi  = max(O(M));
O     = (O-mini)*255/(maxi-mini);
O(~M) = 0;
end