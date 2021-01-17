function header = XxReadMRCHeader(file)

handle = fopen(file,'r');
header = int32(fread(handle,256,'int32'));
if header(4)>7 
    frewind(handle);
    header = int32(fread(handle,256,'int32','b'));
    
end
fclose(handle);
