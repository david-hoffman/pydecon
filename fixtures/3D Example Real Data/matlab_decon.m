% Script to compare Richardson-Lucy Algorithms

% change current directory to this file's location
[PATHSTR,NAME,EXT] = fileparts(mfilename('fullpath'));
cd(PATHSTR)

% read in data
% https://www.mathworks.com/matlabcentral/fileexchange/35684-save-and-load-a-multiframe-tiff-image
psf = double(loadtiff('Real 3D PSF.tif'));
psf = max(psf, 0);
psf = psf./sum(psf(:));
image = double(loadtiff('Real 3D Data.tif'));
% deconvolve data
tic
decon = deconvlucy2(image, psf, 10);
toc
% save decon
options = struct('overwrite', true);
saveastiff(single(decon), 'decon matlab.tif', options)