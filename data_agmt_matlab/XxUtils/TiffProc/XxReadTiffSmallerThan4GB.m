function data = XxReadTiffSmallerThan4GB(TiffPath)

pic_info = imfinfo(TiffPath);
[frame_CCD,~] = size(pic_info);
row_CCD = pic_info.Height;
colum_CCD = pic_info.Width;

data = zeros(row_CCD,colum_CCD,frame_CCD);
for frame_temp = 1:frame_CCD
    data(:,:,frame_temp) = imread(TiffPath,'Index',frame_temp);
end

end