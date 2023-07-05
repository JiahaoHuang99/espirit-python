% Apply SENSE, GRAPPA, ESPIRiT reconstruction to subsampled data and to build
% a training/testing dataset
% Simulate 2, 4, 6, and 8 times of accelaration factors 


% Created by Chengyan Wang
% 06/15/2019
% ------------------------------------------------------------------------------

clear
clc
close all
%You can use 'delete(myCluster.Jobs)' to remove all jobs created with profile local. To create 'myCluster' use 'myCluster = parcluster('local')'.

myCluster = parcluster('local')
delete(myCluster.Jobs)

%% add paths
addpath(genpath('./utils'));
addpath(genpath('./npy-matlab/npy-matlab'));
%% Load data
PathData = '/home2/HWGroup/wangcy/Data/Calgary/MultiChannel/Train_slc/';%'../../Data/Stanford_Fullysampled_3D_FSE_Knees/slc_2000/';
PathSave = '/home2/HWGroup/wangcy/Data/Calgary/MultiChannel/Brain24_random/';%'../../Data/Stanford_Fullysampled_3D_FSE_Knees/Knee24/';
PathTest = 'Display/';

parfor sub = 1:47 %47
    for slc = 1:100
        % to select one slice data
        FileName = strcat('case_',num2str(sub),'slc_',num2str(slc),'.mat'); %'images_chnl49.mat'
        ParallelRecon(PathData,PathSave,PathTest,FileName,sub,slc); % perform parallel reconstruction
        [num2str(sub),'-',num2str(slc)] % print the subject and slice number
    end
end
