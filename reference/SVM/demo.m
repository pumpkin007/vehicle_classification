clear;
%# laod dataset
S = load('fisheriris');
data = zscore(S.meas);
labels = grp2idx(S.species);

%# cross-validate using one-vs-all approach
opts = '-s 0 -t 2 -c 1 -g 0.25';    %# libsvm training options
nfold = 10;
acc = libsvmcrossval_ova(labels, data, opts, nfold);
fprintf('Cross Validation Accuracy = %.4f%%\n', 100*mean(acc));

%# compute final model over the entire dataset
mdl = libsvmtrain_ova(labels, data, opts);