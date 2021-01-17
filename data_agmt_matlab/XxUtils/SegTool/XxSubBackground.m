function data_subBG = XxSubBackground(data, thresh, radius, tmin, tmax, ndirs, nphases, show)

% ------------------------------------------------------------------------
% XxSubBackground: calculate the foreground mask of img
%
% usage:  data_subBG = XxSubBackground(data, thresh, radius, tmin, tmax)
% where,
%    data       -- raw data to be processed
%    thresh     -- lower thresh leads to larger foreground mask, typically 
%                  set as [1e-3, 5e-2]
%    radius     -- average radius
%    tmin, tmax -- lower and upper bound of normalization
%    ndirs      -- direction numbers of structured illumination
%    nphases    -- phase numbers per direction
%    show       -- 1 for showing intermediate results
%
% Author: Chang Qiao
% Email: qc17@mails.tsinghua.edu.cn
% Version: 2020/5/13
% ------------------------------------------------------------------------

% set default parameters
if nargin < 8, show = 0; end
if nargin < 7, nphases = 3; end
if nargin < 6, ndirs = 3; end
if nargin < 5, tmax = 100; end
if nargin < 4, tmin = 0; end


n_perSIM = ndirs * nphases;
[height,width,nframe] = size(data);
num_SIM = size(data, 3) / n_perSIM;
data_subBG = zeros(height,width,nframe);
for j = 1:num_SIM
    d = double(data(:,:,(j-1)*n_perSIM+1:j*n_perSIM));
    % calculate mask
    sd = sum(d,3) / n_perSIM;
    kernel = fspecial('gaussian',[5,5],5);
    fd = imfilter(sd,kernel,'replicate');
    kernel = fspecial('gaussian',[20,20],20);
    bg = imfilter(sd,kernel,'replicate');
    mask = fd - bg;
    mask(mask >= thresh) = 1;
    mask(mask ~= 1) = 0;
    mask = imresize(mask, 1/3, 'bilinear');
    mask(mask > 0) = 1;
    
    if show
        figure(1);
        subplot(1,7,1), imshow(sd,[]);
        subplot(1,7,2), imshow(bg,[]);
        subplot(1,7,3), imshow(mask,[]);
    end
    
    for dr = 1:ndirs
        % cal background
        bg = sum(d(:,:,(dr-1)*nphases+1:dr*nphases),3) / nphases;
        bg = imresize(bg, 1/3, 'bicubic');
        bg_mask = bg .* (1-mask);
        [h,w] = size(bg_mask);
        flag = (bg_mask == 0);
%         radius = 20;
        for y = 1:h
            for x = 1:w
                if flag(y,x)
                    xmin = max(x - radius, 1);
                    ymin = max(y - radius, 1);
                    xmax = min(x + radius, w);
                    ymax = min(y + radius, h);
                    rect = bg_mask(ymin:ymax, xmin:xmax);
                    list = (rect > 0);
                    num = sum(list(:));
                    bg_mask(y,x) = sum(rect(:)) / num;
                end
            end
        end
        bg_final = imresize(bg_mask, [height,width], 'bicubic');
        for ph = 1:nphases
            frame = d(:,:,(dr-1)*nphases+ph);
            % sub background
            frame_subBG = frame - bg_final;
            frame_subBG(frame_subBG < 0) = 0;
            frame_subBG = XxNorm(frame_subBG, tmin, tmax);
            
            if show
                subplot(1,7,4), imshow(frame,[]);
                subplot(1,7,5), imshow(bg,[]);
                subplot(1,7,6), imshow(bg_final,[]);
                subplot(1,7,7), imshow(frame_subBG,[]);
            end
            
            data_subBG(:,:,(j-1)*n_perSIM + (dr-1)*nphases + ph) = frame_subBG;
        end
    end
    
%     kernel = fspecial('gaussian',[100,100],100);
%     bg = imfilter(sd,kernel,'replicate');
%     figure();
%     subplot(1,4,1), imshow(sd,[]);
%     subplot(1,4,2), imshow(bg,[]);
%     for k = 1:nphases
%         frame = double(d(:,:,k));
%         frame_subBG = frame - bg;
%         frame_subBG(frame_subBG < 0) = 0;
%         subplot(1,4,3), imshow(frame,[]);
%         subplot(1,4,4), imshow(frame_subBG,[]);
%         data(:,:,(j-1)*nphases+k) = frame_subBG;
%     end
    
end