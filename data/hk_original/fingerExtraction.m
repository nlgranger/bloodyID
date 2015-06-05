function [O] = fingerExtraction(I)
  [h,w] = size(I);

  se = strel('disk', 7);
  C = imopen(I, se) < 245;
  C = C & (imclose(I, se) > 10);

  D = imgradient(I .* uint8(C ~= 0));
  m = mean(reshape(D,w*h,1));
  s = std(reshape(D,w*h,1));
  G = D > 0.5 * s;
  G = imdilate(G,strel('rectangle', [12,22]));
  G = imerode(G,strel('rectangle', [8,15]));
%  m = mean(reshape(G, h*w, 1));
%  G = ~(G > m/2);
  [L,n] = bwlabel(~G, 4);
  tmp = repmat([1:h]',1,w);

  M = L;
  for k = 1:n
    idx = (L == k);
    [y ~] = find(L.*idx);
    m = mean(y);
    s = std(y);
    if m < h/4 || m > 3*h/4 || s > h/4;
      M(idx) = 0;
    end
  end
  mask = zeros(h,w);
  mask(ceil(w/15):end-floor(w/15), ceil(w/15):end-floor(w/15)) = 1;
  se3 = strel('disk', round(w/16));
  N = imclose(M>0 & mask, se3);
  
  O = I .* uint8(C & N);
  
%  colormap('gray');
%  subplot(2,2,1);
%  imagesc(I);
%  axis equal;
%  subplot(2,2,3);
%  imagesc(M);
%  axis equal;
%  subplot(2,2,4);
%  imagesc(N);
%  axis equal;
%  subplot(2,2,2);
%  imagesc(O);
%  axis equal;
%  pause;
end