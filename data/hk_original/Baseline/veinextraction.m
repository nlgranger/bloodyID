% function B = veinextraction(I)
% VEINEXTRACTION Label pixels as veins using gabor filters detection
%   B = VEINEXTRACTION(I) takes a grayscale image from the database and returns
%   a binary image of the veins using a gabor filter based extraction.
%   [Multi-Channel Gabor Filter Design for Finger-vein Image Enhancement, 
%   Jinfeng Yang, Jinli Yang, ICGIP 2009]

I = imread('FingerVein/2/f1/1/2_1_f1_1.bmp');
[E, M] = fingerExtraction(I, 150, 81/35);
g = fspecial('gaussian');
E = imfilter(E, g);

colormap gray

d = 30;
S = 6;
L = [14, 20];
bank = zeros(d,d,numel(S)*numel(L)*4);
for i = 1:numel(S)
    s = S(i);
    for j = 1:numel(L)
        l = L(j);
        for k = 1:4
            bank(:,:,((i-1)*numel(L)+(j-1))*4+k) = ...
                gaborfilter([d d], [s s], l, k * pi/4);
        end
    end
end

F = zeros([size(E), size(bank,3)]);

for i = 1:size(bank,3)
    G        = bank(:,:,i);
    F(:,:,i) = imfilter(E, G);
    
    subplot(numel(L)*numel(S)+1, 4, i)
    imagesc(F(:,:,i)); axis image; axis off
end

H = min(F, [], 3);

subplot(numel(L)*numel(S)+1, 4, (numel(L)*numel(S))*4+1)
imagesc(E); axis image; axis off

subplot(numel(L)*numel(S)+1, 4, (numel(L)*numel(S))*4+2)
imagesc(H); axis image; axis off

m = mean(reshape(H, 1, []));
si = std(reshape(H, 1, []));
subplot(numel(L)*numel(S)+1, 4, (numel(L)*numel(S))*4+2)
imagesc(H<m); axis image; axis off




% subplot(3,1,1);
% imagesc(R.*double(M)); axis off;axis image;
% subplot(3,1,2);
% imagesc(E.*double(M)); axis off;axis image;
% subplot(3,1,3);
% imagesc(G); axis off;axis image;