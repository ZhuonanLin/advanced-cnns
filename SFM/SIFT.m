% SIFT.m, carrying on SIFT detector and descriptor on images
% output 000abc_sift.mat files for SFM

% setup VLFEAT package
run('../../../../vlfeat-0.9.21/toolbox/vl_setup.m');

% clear up
delete *.mat;
clear all; clc;

disp('Carrying on SIFT on sample images...');
% reading all image names as a struct 
imgs = dir('*.png');
N = size(imgs, 1);
fprintf('Total number of images: %d.\n', N);
disp('SIFT begins...');
for i = 1:N
    % output message for every 20 images
    if rem(i, 20) == 0
        fprintf('%d images have been done...\n', i);
    end
    % readingin image name and output file name
    img_name = imgs(i).name;
    file_name = [img_name(1:end-4), '_sift'];
    
    % read in image and convert to single data type for VLFEAT package
    I = single(imread(img_name));
    % carry our SIFT, f is feature dectected and d is the descriptor output
    [f, d] = vl_sift(I);
    % get output data, use f(1:2,:) for position only
    % note that to convert to single data type for output
    feature = [single(f(1:2, :)); single(d)];
    % save output for one single image
    save(file_name, 'feature');
end
disp('SIFT has finished on all images successfully!\n');