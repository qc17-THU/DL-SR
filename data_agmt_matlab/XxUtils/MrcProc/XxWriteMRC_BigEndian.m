function handle = XxWriteMRC_BigEndian(handle, image, header)

% disp('write position')
position=ftell(handle);
% dataclass=class(image);
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
         fwrite(handle, image, 'single','b');
     case 'uint16'
         fwrite(handle, image, 'uint16','b');
     case 'complex'
         for ii=1:numel(image)
             fwrite(handle, real(image(ii)), 'single','b');
             fwrite(handle, imag(image(ii)), 'single','b');
         end
   end
else if position == 0
     fwrite(handle,header,'int32','b');
     switch dataclass
       case 'single'
            fwrite(handle, image, 'single','b');
       case 'uint16'
            fwrite(handle, image, 'uint16','b');
       case 'complex'
         for ii=1:numel(image)
             fwrite(handle, real(image(ii)), 'single','b');
             fwrite(handle, imag(image(ii)), 'single','b');
         end
     end
    end
end