function resize_data = prepare_image_Alexnet_single(im)
% caffe/matlab/+caffe/imagenet/ilsvrc_2012_mean.mat contains mean_data that
% is already in W x H x C with BGR channels
d = load('../need_for_caffe/ilsvrc_2012_mean.mat');
mean_data = d.mean_data;
IMAGE_DIM = 256;
CROPPED_DIM = 227;

% Convert an image returned by Matlab's imread to im_data in caffe's data
% format: W x H x C with BGR channels
im_data = im(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
im_data = permute(im_data, [2, 1, 3]);  % flip width and height
im_data = single(im_data);  % convert from uint8 to single
im_data = ...
    imresize(im_data, [IMAGE_DIM IMAGE_DIM], 'bilinear');  % resize im_data
im_data = ...
    im_data - mean_data;  % subtract mean_data (already in W x H x C, BGR)

resize_data = zeros(CROPPED_DIM, CROPPED_DIM, 3, 1, 'single');
resize_data = imresize(im_data, [CROPPED_DIM CROPPED_DIM]);