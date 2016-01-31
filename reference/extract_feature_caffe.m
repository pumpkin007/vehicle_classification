function feature = extract_feature_caffe(dirs, net, params)
%% get parameters
% lists and directories
dir_image = dirs.dir_image;
list_image = dirs.list_image;
% feature properties
layer_extract = params.layer_extract;
dim_feature = params.dim_feature;

%% extract feature
feature = zeros(dim_feature, length(list_image));
%% set a waitbar
% hwaitbar = waitbar(0, 'progress>>>');
for i = 1:length(list_image)
    image = imread([dir_image, list_image(i).name]);
    % If you have multiple images, cat them with cat(4, ...)
    feature_single = extract_feature_caffe_single(image, net, layer_extract);
    
    % get average for feature
    feature(:, i) = feature_single;
    %% get progress
    percentage = fix(i / length(list_image) * 100);
%     waitbar(i/length(list_image), hwaitbar, [num2str(percentage), '%']);
    disp([num2str(i), '/', num2str(length(list_image))]);
end

%% close waitbar
% close(hwaitbar);