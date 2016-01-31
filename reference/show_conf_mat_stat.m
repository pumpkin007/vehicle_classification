function s = show_conf_mat_stat(conf_mat)
%% show the stats of confusion matrix
stat = confusionmatStats(squeeze(conf_mat));
% disp(['sens: ', num2str(mean(stat.sensitivity))]);
% disp(['spec: ', num2str(mean(stat.specificity))]);
% disp(['accu: ', num2str(mean(stat.accuracy))]);
% disp(['prec: ', num2str(mean(stat.precision))]);
% disp(['reca: ', num2str(mean(stat.recall))]);
% disp(['F-sc: ', num2str(mean(stat.Fscore))]);
s = zeros(1, 6);
s(1) = mean(stat.sensitivity);
s(2) = mean(stat.specificity);
s(3) = mean(stat.accuracy);
s(4) = mean(stat.precision);
s(5) = mean(stat.recall);
s(6) = mean(stat.Fscore);
s = (round(s*10000))/100;
end