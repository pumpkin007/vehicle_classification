%% run experiment for vehicle dataset
clear;
clc;
%% initialize
path_initialize;
%% setup vlfeat
vl_setup;
%% set directory
set_dir;

%% set param here
% Alexnet;
name_layer = 'fc7';
dim_layer = 4096;
% for experiment
% number of class
num_class = 2;
% cross validation
num_cv = 10;
% set iteration number to run classification
num_iteration = 5;
% only keep first 50 dimension for PCA
num_keep = 50;

%% generate feature
% Alexnet
% need to set up caffe directory, see the details in 
% generate_Alexnet_feature
if exist([dirs.feature, 'feature_', name_layer, '.mat']) ~= 2
    generate_Alexnet_feature(dirs, name_layer, dim_layer);
end
% fisher vector
if exist([dirs.feature, 'feature_fv.mat']) ~= 2
    generate_fisher_vector(dirs, dim_layer);
end

%% generate PCA feature
% Alexnet
if exist([dirs.feature, 'feature_PCA_', name_layer, '.mat']) ~= 2
    load([dirs.feature, 'feature_', name_layer, '.mat']);
    feature = generate_PCA_feature(feature);
    save([dirs.feature, 'feature_PCA_', name_layer, '.mat'], ...
        'feature', 'label');
end
% fisher vector
if exist([dirs.feature, 'feature_PCA_fv.mat']) ~= 2
    load([dirs.feature, 'feature_fv.mat']);
    feature = generate_PCA_feature(feature);
    save([dirs.feature, 'feature_PCA_fv.mat'], ...
        'feature', 'label');
end

%% load feature
% Alexnet
data_fc = load([dirs.feature, 'feature_', name_layer, '.mat']);
feature_fc = data_fc.feature;
% get transpose for label
label = data_fc.label';
% Alexnet PCA
data_fc = load([dirs.feature, 'feature_PCA_', name_layer, '.mat']);
feature_PCA_fc = data_fc.feature;
% fisher vector
data_fv = load([dirs.feature, 'feature_fv.mat']);
feature_fv = data_fv.feature;
% fisher vector PCA
data_fv = load([dirs.feature, 'feature_PCA_fv.mat']);
feature_PCA_fv = data_fv.feature;
% dimensional reduction
feature_PCA_fc = feature_PCA_fc(:, 1:num_keep);
feature_PCA_fv = feature_PCA_fv(:, 1:num_keep);
%% run multiple times for cross validation
% confusion matrix
cm_fc_lda = zeros(num_class, num_class, num_iteration);
cm_fv_lda = zeros(num_class, num_class, num_iteration);
cm_fc_svm = zeros(num_class, num_class, num_iteration);
cm_fv_svm = zeros(num_class, num_class, num_iteration);
for i = 1:num_iteration
    %% cross validation index
    % can use same random sequence to reproduce the result
    % or generate random
    % sequence every time and get mean result
    % if exist([dirs.feature, 'ind_cv.mat']) ~= 2
    %     ind = crossvalind('Kfold', length(label), num_cv);
    %     save([dirs.feature, 'ind_cv.mat'], 'ind');
    % else
    %     load([dirs.feature, 'ind_cv.mat']);
    % end
    ind = crossvalind('Kfold', length(label), num_cv);
    
    %% LDA result
    [~, cm_fc_lda(:, :, i)] = ...
        LDA_cv(feature_fc, label, ind, num_class, num_cv);
    [~, cm_fv_lda(:, :, i)] = ...
        LDA_cv(feature_fv, label, ind, num_class, num_cv);
    
    %% PCA SVM result
    % run SVM
    [~, cm_fc_svm(:, :, i)] = ...
        SVM_cv(feature_fc, label, ind, num_class, num_cv);
    [~, cm_fv_svm(:, :, i)] = ...
        SVM_cv(feature_fv, label, ind, num_class, num_cv);
end
%% sum of confusion matrix
cm_fc_lda = sum(cm_fc_lda, 3);
cm_fv_lda = sum(cm_fv_lda, 3);
cm_fc_svm = sum(cm_fc_svm, 3);
cm_fv_svm = sum(cm_fv_svm, 3);
%% show the result
% confusion matrix
show_conf_mat_stat(cm_fc_lda)
show_conf_mat_stat(cm_fv_lda)
show_conf_mat_stat(cm_fc_svm)
show_conf_mat_stat(cm_fv_svm)
%% save result
save([dirs.save, 'confusion_matrix_', name_layer, '.mat'], ...
    'cm_fc_lda', 'cm_fv_lda', 'cm_fc_svm', 'cm_fv_svm');