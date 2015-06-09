function [O, M] = fingerExtraction(I)
[h, w] = size(I);
% Center image
g = fspecial('gaussian', [40 30], 8);
e = fspecial('gaussian', 10, 4);

B = 255-imfilter(I, g);
m = mean(mean(B));
M = B > m;
[y, x] = find(M);
tmp    = cov(x,y);
angle  = atan(tmp(2,1)/var(x));
O      = imrotate(I, 180*angle/pi, 'bilinear');
M      = imrotate(M, 180*angle/pi, 'bilinear');
A      = imrotate(true(h,w), 180*angle/pi, 'bilinear');
[y, ~] = find(M);
off    = round(size(O,1)/2-mean(y));
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
m = mean(G(M));
s = std(G(M));
U = (G > m + s/6);
M = U | ~M;
M = ~imclose(M, strel('disk', 5'));

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

M = imclose(M, strel('disk', 6));
M = imfill(M, 'holes');

% Find ROI
hist = filter2(ones(1,40), sum(O .* uint8(M)));
find

end