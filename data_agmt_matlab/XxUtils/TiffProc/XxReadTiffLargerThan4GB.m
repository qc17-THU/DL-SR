function data = XxReadTiffLargerThan4GB(TiffPath, DataType)



pic_info = imfinfo(TiffPath,'tif');
pic_info = pic_info(1); 
img_desc = pic_info.ImageDescription;
img_amount = txt_find_value(img_desc,{'images=','\n'},1);
slcies_amount = txt_find_value(img_desc,{'slices=','\n'},1);
frames_amount = txt_find_value(img_desc,{'frames=','\n'},1);
if isempty(img_amount)+isempty(slcies_amount)+isempty(frames_amount) == 3 
    img_amount_read = 1;
else
    img_amount_read = img_amount;
end
file_handle = fopen(TiffPath,'r');
fseek(file_handle, pic_info.StripOffsets, 'bof');
data = fread(file_handle,pic_info.Width*pic_info.Height*img_amount_read,DataType,'b');
data = reshape(data, [pic_info.Width,pic_info.Height,img_amount_read]);
data = permute(data,[2 1 3]);
fclose(file_handle);

if strcmp(DataType,'uint8')
    data = uint8(data);
elseif strcmp(DataType,'uint16')
    data = uint16(data);
elseif strcmp(DataType,'double')
    data = double(data);
elseif strcmp(DataType,'float')
    data = float(data);
end

end