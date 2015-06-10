function [V] = veinExtraction(K, M)
% for w = 3:5
% for s = 11:0.5:14
%   colormap('gray');
  w = 4;
  s = 13;
  scale = 1/w;
  L = spatialgabor(K, scale, 0, s, s);
  thres = mean(L(M))*0.8;
  V = (1 - mat2gray(-L<thres)) .* M;
  %V = imopen(V, strel('rectangle',[3,3]));
  V = imclose(V, strel('disk',2));
  [V,n] = bwlabel(V, 4);
  for i = 1:n
    tmp = V == i;
    if sum(sum(tmp)) < numel(K)/100
        V(tmp) = 0;
    else
        V(tmp) = 1;
    end
  end
%   subplot(3,1,1);
%   imagesc(K);
%   axis equal;
%   subplot(3,1,2);
%   imagesc(V);
%   axis equal;
% imwrite(1 - mat2gray(-L) .* M, sprintf('test_%d_%d.png',w,s));
%   pause
% end
% end
end