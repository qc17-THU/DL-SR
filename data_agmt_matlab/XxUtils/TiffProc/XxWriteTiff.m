function XxWriteTiff(data, FileName)

imwrite(data(:,:,1), FileName, 'Compression','none');

for i = 2:size(data,3)
    imwrite(data(:,:,i), FileName, 'WriteMode', 'append', 'Compression','none');
end
