function [header, data] = XxReadMRC(file)

handle = fopen(file,'r');
header = int32(fread(handle,256,'int32'));

if header(4)>7  % big endian
    frewind(handle);
    header=int32(fread(handle,256,'int32','b'));
    
    if nargin==1
        switch header(4)
            case 6
                rawimage=uint16(fread(handle,header(1)*header(2)*header(3),'uint16','b'));
            case 4
                fseek(handle, header(24), 'cof');
                rawimage=single(fread(handle,header(1)*header(2)*header(3)*2,'single','b'));
            case 2
                fseek(handle, header(24), 'cof');
                rawimage=single(fread(handle,header(1)*header(2)*header(3),'single','b'));
        end
        data=rawimage;
    end
    fclose(handle);
    
else
    if nargin==1  % small endian
        switch header(4)
            case 6
                rawimage=uint16(fread(handle,double(header(1))*double(header(2))*double(header(3)),'uint16'));
            case 4
                fseek(handle, header(24), 'cof');
                rawimage=single(fread(handle,double(header(1))*double(header(2))*double(header(3))*2,'single'));
            case 2
                fseek(handle, header(24), 'cof');
                rawimage=single(fread(handle,double(header(1))*double(header(2))*double(header(3)),'single'));
        end
        data=rawimage;
    end
    fclose(handle);
end

end

% The MRC image header has a fixed size of 1024 bytes. The information within the header includes a description of the extended header and image data. The column, row, and section are equivalent to the x, y, and z axes.
% 
% Byte    
% Numbers    No.     Variable Type Variable Name	  Contents
%  1 - 4	 01      i	     NumCol	            Number of columns. Typically, NumCol represents the number of image elements along the X axis.
%  5 - 8	 02      i	     NumRow	            Number of rows. Typically, NumRow represents the number of image elements along the Y axis.
%  9 - 12	 03      i	     NumSections	    Total number of sections. (NumZSec*NumWave*NumTimes)
% 13 - 16	 04      i    	 PixelType	        The format of each pixel value. See the Pixel Data Types table below.
% 17 - 20	 05      i	     mxst	            Starting point along the X axis of sub-image (pixel number). Default is 0.
% 21 - 24	 06      i	     myst	            Starting point along the Y axis of sub-image (pixel number). Default is 0.
% 25 - 28	 07      i   	 mzst	            Starting point along the Z axis of sub-image (pixel number). Default is 0.
% 29 - 32	 08      i	     mx	                Sampling frequency in x; commonly set equal to one or the number of columns.
% 33 - 36	 09      i	     my	                Sampling frequency in y; commonly set equal to one or the number of rows.
% 37 - 40	 10      i	     mz	                Sampling frequency in z; commonly set equal to one or the number of z sections.
% 41 - 44	 11      f	     dx	                Cell dimension in x; for non-crystallographic data, set to the x sampling frequency times the x pixel spacing.
% 45 - 48	 12      f	     dy	                Cell dimension in y; for non-crystallographic data, set to the y sampling frequency times the y pixel spacing.
% 49 - 52	 13      f    	 dz	                Cell dimension in z; for non-crystallographic data, set to the z sampling frequency times the z pixel spacing.
% 53 - 56	 14      f	     alpha	            Cell angle (alpha) in degrees. Default is 90.
% 57 - 60	 15      f	     beta	            Cell angle (beta) in degrees. Default is 90.
% 61 - 64	 16      f	     gamma	            Cell angle (gamma) in degrees. Default is 90.
% 65 - 68	 17      i	     -	                Column axis. Valid values are 1,2, or 3. Default is 1.
% 69 - 72	 18      i	     -	                Row axis. Valid values are 1,2, or 3. Default is 2.
% 73 - 76	 19      i	     -	                Section axis. Valid values are 1,2, or 3. Default is 3.
% 77 - 80	 20      f	     min	            Minimum intensity of the 1st wavelength image.
% 81 - 84	 21      f	     max	            Maximum intensity of the 1st wavelength image.
% 85 - 88	 22      f	     mean	            Mean intensity of the first wavelength image.
% 89 - 92	 23      i	     nspg	            Space group number. Applies to crystallography data.
% 93 - 96	 24      i	     next	            Extended header size, in bytes.
% 97 - 98	 25      n	     dvid	            ID value. (-16224)
% 99 - 100	 25      n	     nblank	            Unused.
% 101 - 104	 26      i	     ntst	            Starting time index.
% 105 - 128	 27     c24	     blank	            Blank section. 24 bytes.
% 129 - 130  33     n	     NumIntegers	    Number of 4 byte integers stored in the extended header per section.
% 131 - 132	 33     n	     NumFloats	        Number of 4 byte floating-point numbers stored in the extended header per section.
% 133 - 134  34     n	     sub	            Number of sub-resolution data sets stored within the image. Typically, this equals 1.
% 135 - 136	 34     n	     zfac	            Reduction quotient for the z axis of the sub-resolution images.
% 137 - 140	 35     f	     min2	            Minimum intensity of the 2nd wavelength image.
% 141 - 144	 36     f	     max2	            Maximum intensity of the 2nd wavelength image.
% 145 - 148	 37     f	     min3	            Minimum intensity of the 3rd wavelength image.
% 149 - 152	 38     f	     max3	            Maximum intensity of the 3rd wavelength image.
% 153 - 156	 39     f	     min4	            Minimum intensity of the 4th wavelength image.
% 157 - 160  40     f	     max4	            Maximum intensity of the 4th wavelength image.
% 161 - 162	 41     n	     type	            Image type. See the Image Type table below.
% 163 - 164	 41     n	     LensNum	        Lens identification number.
% 165 - 166	 42     n	     n1	                Depends on the image type.
% 167 - 168	 42     n	     n2	                Depends on the image type.
% 169 - 170	 43     n	     v1	                Depends on the image type.
% 171 - 172	 43     n	     v2	                Depends on the image type.
% 173 - 176	 44     f	     min5	            Minimum intensity of the 5th wavelength image.
% 177 - 180	 45     f	     max5	            Maximum intensity of the 5th wavelength image.
% 181 - 182	 46     n	     NumTimes	        Number of time points.
% 183 - 184	 46     n	     ImgSequence	    Image sequence. 0=ZTW, 1=WZT, 2=ZWT.
% 185 - 188	 47     f	     -	                X axis tilt angle (degrees).
% 189 - 192	 48     f	     -	                Y axis tilt angle (degrees).
% 193 - 196	 49     f	     -	                Z axis tilt angle (degrees).
% 197 - 198	 50     n	     NumWaves	        Number of wavelengths.
% 199 - 200	 50     n	     wave1	            Wavelength 1, in nm.
% 201 - 202	 51     n	     wave2	            Wavelength 2, in nm.
% 203 - 204	 51     n	     wave3	            Wavelength 3, in nm.
% 205 - 206	 52     n	     wave4	            Wavelength 4, in nm.
% 207 - 208	 52     n	     wave5	            Wavelength 5, in nm.
% 209 - 212	 53     f	     z0	                Z origin, in um.
% 213 - 216	 54     f	     x0	                X origin, in um.
% 217 - 220	 55     f	     y0	                Y origin, in um.
% 221 - 224	 56     i	     NumTitles	        Number of titles. Valid numbers are between 0 and 10.
% 225 - 304	      c80	     -	                Title 1. 80 characters long.
% 305 - 384	      c80	     -	                Title 2. 80 characters long.
% 385 - 464	      c80	     -	                Title 3. 80 characters long.
% 465 - 544	      c80	     -	                Title 4. 80 characters long.
% 545 - 624	      c80	     -	                Title 5. 80 characters long.
% 625 - 704	      c80	     -	                Title 6. 80 characters long.
% 705 - 784	      c80	     -	                Title 7. 80 characters long.
% 785 - 864	      c80	     -	                Title 8. 80 characters long.
% 865 - 944	      c80	     -         	        Title 9. 80 characters long.
% 945 -1024	      c80	     -	                Title 10. 80 characters long.



% Pixel Data Types
% The data type of an image, stored in header bytes 13-16, is designated by one of the code numbers in the following table. 
%
% Code	C/C++ Macro	        Description
%  0	IW_BYTE	            1-byte unsigned integer
%  1	IW_SHORT	        2-byte signed integer
%  2	IW_FLOAT	        4-byte floating-point (IEEE)
%  3	IW_COMPLEX_SHORT	4-byte complex value as 2 2-byte signed integers
%  4	IW_COMPLEX	        8-byte complex value as 2 4-byte floating-point values
%  5	IW_EMTOM	        2-byte signed integer
%  6	IW_USHORT	        2-byte unsigned integer
%  7	IW_LONG	            4-byte signed integer

