function [] = generate_Alexnet_feature(dirs, name_layer, dim_layer)
%% set path for matcaffe
dir_caffe = '/home/zhou/Documents/caffe-master-20151112/';
addpath([dir_caffe, 'matlab/']);
addpath('../reference/');

%% set parameter
layer_extract = name_layer;
dim_feature = dim_layer;
% set to params
params.layer_extract = layer_extract;
params.dim_feature = dim_feature;

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

feature_save = [dir_save, 'feature_', layer_extract, '.mat'];
name_save = [dir_save, 'namelist.mat'];

%% Add caffe/matlab to you Matlab search PATH to use matcaffe
% if exist('../+caffe', 'dir')
%     addpath('..');
% else
%     error('Please run this demo from caffe/matlab/demo');
% end

% Set caffe mode
use_gpu = 0;
if use_gpu == 1
    caffe.set_mode_gpu();
    gpu_id = 0;  % we will use the first gpu in this demo
    caffe.set_device(gpu_id);
else
    caffe.set_mode_cpu();
end

% Initialize the network using BVLC CaffeNet for image classification
% Weights (parameter) file needs to be downloaded from Model Zoo.
model_dir = [dir_caffe, 'models/bvlc_alexnet/'];
% net_model = [model_dir 'deploy.prototxt'];
net_model = [model_dir 'deploy_single.prototxt'];
net_weights = [model_dir 'bvlc_alexnet.caffemodel'];
phase = 'test'; % run with phase test (so that dropout isn't applied)
if ~exist(net_weights, 'file')
    error('Please download AlexNet from Model Zoo before you run this.');
end

% Initialize a network
net = caffe.Net(net_model, net_weights, phase);

%% for vehicle dataset
dirs.dir_image = dir_passenger;
dirs.list_image = list_passenger;
feature_passenger = extract_feature_caffe(dirs, net, params);
dirs.dir_image = dir_other;
dirs.list_image = list_other;
feature_other = extract_feature_caffe(dirs, net, params);

%% call caffe.reset_all() to reset caffe
caffe.reset_all();

%% for vehicle dataset
feature = [feature_passenger, feature_other]';
label = [ones(1, length(list_passenger)), 2*ones(1, length(list_other))];
name_passenger = {list_passenger(:).name};
name_other = {list_other(:).name};
name = [name_passenger, name_other];
save(feature_save, 'feature', 'label');
save(name_save, 'name');
