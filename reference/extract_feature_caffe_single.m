function features = extract_feature_caffe_single(im, net, layer_name)
% prepare oversampled input
% input_data is Height x Width x Channel x Num
%% check if the image is grayscale, change to color image
if ismatrix(im)
    im = cat(3, im, im, im);
end

% tic;
% input_data = prepare_image_Alexnet(im);
input_data = prepare_image_Alexnet_single(im);
% toc;

%% change to extract layers
% tic;
net.blobs('data').set_data(input_data);
net.forward_prefilled();
features = net.blobs(layer_name).get_data();
% toc;