function mask = XxCalMask(img, ksize, thresh)

% ------------------------------------------------------------------------
% XxCalMask: calculate the foreground mask of img
%
% usage:  mask = XxCalMask(img, ksize, thresh)
% where,
%    img       -- 2D img to calculate foreground
%    ksize     -- the size of first gaussian kernel, typically set as 5~10
%    thresh    -- lower thresh leads to larger mask, typically set as 
%                 [1e-3, 5e-2]
%
% Author: Chang Qiao
% Email: qc17@mails.tsinghua.edu.cn
% Version: 2020/5/15
% ------------------------------------------------------------------------

if nargin < 3, thresh = 5e-2; end
if nargin < 2, ksize = 10; end
kernel = fspecial('gaussian',[ksize,ksize],ksize);
fd = imfilter(img,kernel,'replicate');
kernel = fspecial('gaussian',[100,100],50);
bg = imfilter(img,kernel,'replicate');
mask = fd - bg;
mask(mask >= thresh) = 1;
mask(mask ~= 1) = 0;
mask = logical(mask);

end