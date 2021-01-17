function handle = XxWriteMRC_SmallEndian(handle, image, header)

position=ftell(handle);
switch header(4)
    case 6 
        dataclass='uint16';
    case 4 
        dataclass='complex';
    case 2 
        dataclass='single';
end

if position==-1
    fprintf('Invalid MRC file handle\n');
    return;
end
 
if position>1024
   switch dataclass
     case 'single'
         fwrite(handle, image, 'single');
     case 'uint16'
         fwrite(handle, image, 'uint16');
     case 'complex'
         for ii=1:numel(image)
             fwrite(handle, real(image(ii)), 'single');
             fwrite(handle, imag(image(ii)), 'single');
         end
   end
elseif position == 0
     fwrite(handle,header,'int32');
     switch dataclass
       case 'single'
            fwrite(handle, image, 'single');
       case 'uint16'
            fwrite(handle, image, 'uint16');
       case 'complex'
         for ii=1:numel(image)
             fwrite(handle, real(image(ii)), 'single');
             fwrite(handle, imag(image(ii)), 'single');
         end
     end
    end
end