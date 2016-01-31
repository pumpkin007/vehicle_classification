function conf_mat = get_confusion_matrix(target, output, num_class)
conf_mat = zeros(num_class);
for i = 1:length(target)
    conf_mat(target(i), output(i)) = conf_mat(target(i), output(i)) + 1;
end
end