%% set directory for all data
% project
dirs.project = fullfile(pwd, '../');
% vehicle data
dirs.vehicle = ...
    '/path/to/dataset/random200/';
dirs.passenger = [dirs.vehicle, 'passenger/'];
dirs.other = [dirs.vehicle, 'other/'];
% feature location
dirs.feature = [dirs.project, 'feature/'];
% result location
dirs.save = [dirs.project, 'result/'];
% caffe directory
dirs.caffe = '/home/zhou/Documents/caffe-master-20151112/';
