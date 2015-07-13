function B = vein_extraction(I)
% VEINEXTRACTION Label pixels as veins using gabor filters detection
%   B = VEINEXTRACTION(I) takes a grayscale image from the database and returns
%   a binary image of the veins using a gabor filter based extraction.
%   [Multi-Channel Gabor Filter Design for Finger-vein Image Enhancement, 
%   Jinfeng Yang, Jinli Yang, ICGIP 2009]

g = fspecial('gaussian');
E = imfilter(I, g);

colormap gray

d = 30;
s = 5;
L = [15, 18];
bank = zeros(d,d,numel(L)*4);

for j = 1:numel(L)
    l = L(j);
    for k = 1:4
        bank(:,:,(j-1)*4+k) = ...
            gaborfilter([d d], [s s], l, k * pi/4);
    end
end

F = zeros([size(E), size(bank,3)]);

for i = 1:size(bank,3)
    G        = bank(:,:,i);
    F(:,:,i) = imfilter(E, G);
end

B = false(size(I));
for i = 1:numel(L)
    R = min(F(:,:,(i-1)*4+1:i*4), [], 3);
    m = mean(reshape(R, 1, []));
    s = std(reshape(R, 1, []));
    B = B | (R < 0.9*m-0.4*s*L(i)/10);
end
B = bwareaopen(B, 50);
end