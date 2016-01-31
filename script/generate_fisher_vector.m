function [] = generate_fisher_vector(dirs, dim_layer)
%% generate fisher vector for vehicle image


%% set parameter
dim_feature = dim_layer;
width_side = 256;

%% set directory and get image list
%% for vehicle dataset
dir_passenger = dirs.passenger;
dir_other = dirs.other;
dir_save = dirs.feature;
if ~isdir(dir_save)
    mkdir(dir_save);
end
list_passenger = dir([dir_passenger, '*.jpg']);
list_other = dir([dir_other, '*.jpg']);
feature_save = [dir_save, 'feature_fv.mat'];

%% get SIFT from image
if exist([dirs.feature, 'feature_SIFT.mat']) == 2 %#ok<EXIST>
    load([dirs.feature, 'feature_SIFT.mat'])
else
    %% passenger
    feature_passenger = cell(1, length(list_passenger));
    for i = 1:length(list_passenger)
        disp(['Getting SIFT on image: ', num2str(i)]);
        img = imread([dir_passenger, list_passenger(i).name]);
        % resize the image
        img = imresize(img, [width_side, width_side]);
        [~, descriptor] = vl_sift(single(rgb2gray(img)));
        % save the descriptor as double
        feature_passenger{i} = double(descriptor);
    end
    %% other
    feature_other = cell(1, length(list_other));
    for i = 1:length(list_other)
        disp(['Getting SIFT on image: ', num2str(i)]);
        img = imread([dir_other, list_other(i).name]);
        % resize the image
        img = imresize(img, [width_side, width_side]);
        [~, descriptor] = vl_sift(single(rgb2gray(img)));
        % save the descriptor as double
        feature_other{i} = double(descriptor);
    end
    feature = [feature_passenger, feature_other];
    label = [ones(1, length(list_passenger)), 2*ones(1, length(list_other))];
    save([dirs.feature, 'feature_SIFT.mat'], 'feature', 'label')
end


%% get PCA from SIFT
if exist([dirs.feature, 'feature_PCA_SIFT.mat']) == 2 %#ok<EXIST>
    load([dirs.feature, 'feature_PCA_SIFT.mat'])
else
    % get all SIFT descriptor
    feature_total = [];
    for i = 1:length(label)
        disp(['Add feature: ', num2str(i)]);
        feature_total = [feature_total, feature{i}];
    end
    
    % transpose the feature
    feature_total = feature_total';
    
    %% PCA
    % make feature standardized
    feature_total = zscore(feature_total);
    [coeff,score,latent] = pca(feature_total);
    
    % get result
    for i = 1:length(label)
        disp(['Get PCA: ', num2str(i)]);
        feature_PCA{i} = score(1:size(feature{i}, 2), :);
        score(1:size(feature{i}, 2), :) = [];
    end
    feature = feature_PCA;
    save([dirs.feature, 'feature_PCA_SIFT.mat'], 'feature', 'label');
end

%% generate fisher vector
% set parameter for fisher vector
percentC = 0.1;
% number of class for GMM
numClusters = 32;
dim_keep = 64;

% random select codebook
feature_rand = [];
% feature_total = [];
for i = 1:length(label)
    disp(['Adding to code book: ', num2str(i)]);
    seq_rand = randperm(size(feature{i}, 1));
    seq_rand = seq_rand(1:ceil(percentC*size(feature{i}, 1)));
    feature_rand = [feature_rand; feature{i}(seq_rand, :)];
    %     feature_total = [feature_total; feature{i}];
end

% transpose the feature
feature_rand = feature_rand';

% get first half PCA
feature_rand = feature_rand(1:dim_keep, :);

% get GMM
[means, covariances, priors] = vl_gmm(feature_rand, numClusters);
disp('Finish GMM');

% doing fisher encoding
fisher_encoding = [];
for i = 1:length(label)
    disp(['Encoding: ', num2str(i)]);
    % create data to be encoded
    dataToBeEncoded = feature{i};
    % transpose the data to be encoded
    dataToBeEncoded = dataToBeEncoded';
    % get first half PCA
    dataToBeEncoded = dataToBeEncoded(1:dim_keep, :);
    % encode using fisher encoding
    encoding = vl_fisher(dataToBeEncoded, means, covariances, priors);
    % transpose the encoding data
    encoding = encoding';
    fisher_encoding = [fisher_encoding; encoding];
end

feature = fisher_encoding;
save([dirs.feature, 'feature_fv.mat'], 'feature', 'label');