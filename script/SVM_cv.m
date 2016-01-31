function [acc, conf_mat] = SVM_cv(feature, label, ind, num_class, num_cv)
% initialize accuracy array
acc_temp = zeros(1, num_cv);
conf_mat = zeros(num_class, num_class, num_cv);
% record progress
time_rec = 0;
for m = 1:num_cv
    % set testing set and training set
    label_test = (ind == m);
    label_train = ~label_test;
    
    %% run SVM
    %% libsvm training options: linear SVM
    tic;
    opts = '-s 0 -t 0 -c 1/2';    ...
        mdl = ...
        libsvmtrain_ova(label(label_train), feature(label_train, :), opts);
    [conf,acc_temp(m)] = ...
        libsvmpredict_ova(label(label_test), feature(label_test, :), mdl);
    toc;
    % confusion matrix
    conf_mat(:, :, m) = ...
        get_confusion_matrix(label(label_test), ...
        conf, num_class);
    % output the progress
    time_rec = time_rec + 1;
    disp([num2str(time_rec), '/', num2str(num_cv)]);
end
% get mean value to accracy
acc = mean(acc_temp);
conf_mat = sum(conf_mat, 3);