function feature = generate_PCA_feature(feature)
%% generate PCA from feature
[m, n] = size(feature);

%% if not enough sample
if m < n
    num_rep = ceil(n / m);
    feature = repmat(feature, [num_rep 1]);
end

%% PCA
tic;
% make feature standardized
feature = zscore(feature);
[~, score, ~] = pca(feature);
toc;
% try result
% feature_PCA = feature * coeff;
% feature = feature_PCA;
feature = score(1:m, :);