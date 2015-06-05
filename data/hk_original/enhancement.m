function [O, mask] = enhancement(I, patchSize)
  [h,w] = size(I);
  
%  colormap('gray');
%  subplot(3,2,1);
%  imagesc(I);
%  axis equal;
  
  O = double(I);
  mask = I > 0;
  
  % spread histogram
  m = min(O(mask));
  M = max(O(mask));
  O(mask) = (O(mask) - m)*255/(M-m);
  
  % remove low freq color
  b    = ones(2*patchSize+1);
  bg   = conv2(O, b, 'same');
  d    = conv2(double(mask), b, 'same');
  d(d==0) = 1;
  bg   = bg ./ d;
  O = O - bg;
%  subplot(3,2,2);
%  imagesc(bg);
%  axis equal;
%  subplot(3,2,3);
%  imagesc(O);
%  axis equal;
  
  % spread and equalize histogram
  m = mean(O(mask));
  s = std(O(mask));
  O(mask) = 255*tanh((O(mask)-m)/s);
%  subplot(3,2,4);
%  imagesc(O);
%  axis equal;
  
  m = min(O(mask));
  M = max(O(mask));
  O(mask) = (O(mask) - m)*255/(M-m);
  O = O .* mask;

  % O(mask) = histeq(O(mask));
%  subplot(3,2,5);
%  imagesc(O);
%  axis equal;
  
  
  % align and center image
  [y, x] = find(mask>0);
  tmp = cov(x,y);
  angle = atan(tmp(2,1)/var(x));
  O = imrotate(O, 180*angle/pi, 'bilinear');
  mask = imrotate(uint8(mask*255), 180*angle/pi, 'bilinear') > 0;
  [y, ~] = find(mask>0);
  O = circshift(O, [round(size(O,1)/2-mean(y)), 0]);
  mask = circshift(mask, [round(size(O,1)/2-mean(y)), 0]);
  offy = floor((size(O,1)-h)/2)+1;
  offx = floor((size(O,2)-w)/2)+1;
  O = O(offy:offy+h-1, offx:offx+w-1);
  mask = mask(offy:offy+h-1, offx:offx+w-1);
  
%  subplot(3,2,6);
%  imagesc(O);
%  axis equal;
end