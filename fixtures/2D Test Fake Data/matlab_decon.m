% Script to compare Richardson-Lucy Algorithms

% change current directory to this file's location
[PATHSTR,NAME,EXT] = fileparts(mfilename('fullpath'));
cd(PATHSTR)

% read in data
psf = imread('psf.tif');
image = imread('image.tif');
% deconvolve data
tic
decon = deconvlucy(image, psf, 10);
toc
% save decon
tifffile = Tiff('decon matlab.tif', 'w');
tifffile.setTag('Photometric',Tiff.Photometric.MinIsBlack);
tifffile.setTag('PlanarConfiguration',Tiff.PlanarConfiguration.Chunky)
tifffile.setTag('BitsPerSample',32);
tifffile.setTag('ImageLength',512);
tifffile.setTag('ImageWidth',512);
tifffile.setTag('SampleFormat',Tiff.SampleFormat.IEEEFP)
tifffile.write(single(decon))
tifffile.close