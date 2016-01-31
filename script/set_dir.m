%% set directory for all data
% project
dirs.project = fullfile(pwd, '../');
% vehicle data
dirs.vehicle = ...
    '/home/zhou/data/our_vehicle/car_extraction/twoclass/random200/';
dirs.passenger = [dirs.vehicle, 'passenger/'];
dirs.other = [dirs.vehicle, 'other/'];
% feature location
dirs.feature = [dirs.project, 'feature/'];
% result location
dirs.save = [dirs.project, 'result/'];