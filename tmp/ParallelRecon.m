% Apply SENSE, GRAPPA, ESPIRiT reconstruction to subsampled data
% Simulate 2, 4, 6, and 8 times of accelaration factors
function isSucessful = ParallelRecon(PathData,PathSave,PathTest,FileName,sub,slc)
%
% ---- inputs ----
% PathData: path to load data

% PathSave: path to save data

% PathTest: path to save jpg images

% FileName: the file name


% ---- outputs ----
% isSucessful: flag of sucessful reconstruction

% ---- example ----
% ParallelRecon(0,0,0,0);

% Created by Chengyan Wang
% 06/06/2019
% ------------------------------------------------------------------------------

%% add paths
% addpath(genpath('./utils'));

%% Load data
if PathData == 0 
    PathData = './Knee/5xCartes_knees_result/e1/';
end
if PathSave == 0 
    PathSave = './Knee/5xCartes_knees_result/e1/';
end
if FileName == 0
    FileName = 'images_chnl49.mat';
end
if PathTest == 0
    PathTest = 'Display/';
end

PathSave0 = 'AF2/';
PathSave1 = 'AF4/';
PathSave2 = 'AF6/';
% PathSave3 = 'AF8/';

DataCoils = load([PathData,FileName]);
DataCoil = DataCoils.images_chnl;
[sx,~,Nc] = size(DataCoil);
% k-space data
%for ind = 1:Nc
%    kspace(:,:,ind) = fft2c(double(DataCoil(:,:,ind)));
%end
kspace = fft2(fftshift(DataCoil));
kspace = fftshift(kspace,1);

eps = 1e-9
DATA = kspace/max(max(max(abs(ifft2c(kspace))))) + eps;
%DATA = kspace;
sy = 256;
newDATA = zeros(sx,sy,Nc);
newDATA(:,20:237,:) = DATA;
DATA = newDATA;
% Predifined parameters
ncalib = 24; % use 24 calibration lines to compute compression
ksize = [6,6]; % ESPIRiT kernel-window-size
eigThresh_k = 0.02; % threshold of eigenvectors in k-space
eigThresh_im = 0.9; % threshold of eigenvectors in image space
nIterCG = 12; % CG recon
isDisplay = 0; % show figures
isDisplay2 = 0; % show figures
isSaveJPG = 0;
isSaveNPY = 1;

% Generate calibration data
calib = crop(DATA,[sx,ncalib,Nc]);
GT = ifft2c(DATA);
%GT_real = real(GT);
%GT_imag = imag(GT);
imgGT = sos(GT); 
%GT = cat(4,GT_real,GT_imag);

%% Create a sampling mask to simulate xR undersampling with autocalibration lines
 R = 2
%R = 2.1; % Accelaration factor
% Save NPY files
PathSaveData = [PathSave,PathSave0];
%%%%%%%%%%%%%%%===============================%%%%%%%%%%%%%%%%%%%%
%mask = mySubsampling(sx,sy,Nc,ncalib,R);
%writeNPY(strcat('/home2/HWGroup/wangcy/Data/Calgary/MultiChannel/Mask/mask_',num2str(R),'.npy'),double(mask));
%%%%%%%%%%%%%%%===============================%%%%%%%%%%%%%%%%%%%%
mask = cs_generate_pattern([sx,sy],R);
mask = fftshift(mask);
mask = permute(mask,[2,1]);
%size(mask)
%size(DATA)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
writeNPY(mask, [strcat('/home2/HWGroup/wangcy/Data/Calgary/MultiChannel/Brain24_random/Mask/mask_',num2str(R),'/mask_',num2str(R),'.npy')]);
DATAc = DATA.*mask; % Subsampling the k-space data
%DATAc_real = real(DATAc);
%DATAc_imag = imag(DATAc);
%DATAc = cat(4,DATAc_real,DATAc_imag);
ZF = ifft2c(DATAc);
imgZF = sos(ZF); 
% Generate sensitivity maps
[maps,weights] = mySensitivityMap(sx,sy,calib,ksize,eigThresh_k,eigThresh_im,isDisplay);
%size(maps)
%maps_new = zeros(size(maps));
%maps_new(:,:,:,1) = real(maps(:,:,:,2));
%maps_new(:,:,:,2) = imag(maps(:,:,:,2));
%maps = maps_new;
% Perform reconstruction
%sensemap = squeeze(maps(:,:,:,2));
%[imgSENSE, ~, ~] = mySenseRecon(sensemap, DATAc, R);
% create and ESPIRiT operator
%ESP = ESPIRiT(maps,weights);
%SNS = ESPIRiT(maps(:,:,:,end));
% [imgkSENSE,imgSENSE,imgkESPIRiT,imgESPIRiT,imgkGRAPPA,imgGRAPPA] = myParallelRecon(DATAc,ESP,SNS,calib,nIterCG);
%[imgkSENSE,imgSENSE] = myParallelRecon(DATAc,ESP,SNS,calib,nIterCG);

% Combine images
% z = [imgZF,abs(imgSENSE),sos(imgESPIRiT), sos(imgGRAPPA),imgGT];
% z_res = [abs(imgZF-imgGT),abs(abs(imgSENSE)-imgGT),abs(sos(imgESPIRiT)-imgGT), abs(sos(imgGRAPPA)-imgGT)];
%[~, ~, ~,~, ~,resL1ESPIRiT,~,~] = myESPIRiT(DATAc,calib,mask,isDisplay)
    
if isSaveNPY
    if ~exist(PathSaveData)
        mkdir(PathSaveData);
    end   
    %writeNPY(DATA, [PathSaveData,strcat('DATA_',num2str(sub),'_',num2str(slc),'.npy')]); % full sample k-space
    %writeNPY(DATAc, [PathSaveData,strcat('DATAc_',num2str(sub),'_',num2str(slc),'.npy')]); % subsample k-space
    %writeNPY(ZF, [PathSaveData,strcat('ZF_',num2str(sub),'_',num2str(slc),'.npy')]); % subsample k-space
    %writeNPY(GT, [PathSaveData,strcat('GT_',num2str(sub),'_',num2str(slc),'.npy')]); % subsample k-space

    writeNPY(imgZF, [PathSaveData,strcat('imgnewZF_',num2str(sub),'_',num2str(slc),'.npy')]);
    % writeNPY(resL1ESPIRiT, [PathSaveData,strcat('imgL1ESPIRiT_',num2str(sub),'_',num2str(slc),'.npy')]);
    % writeNPY(abs(imgSENSE), [PathSaveData,strcat('imgSENSE_',num2str(sub),'_',num2str(slc),'.npy')]);
    %writeNPY(abs(imgSENSE), [PathSaveData,strcat('imgnewSENSE_',num2str(sub),'_',num2str(slc),'.npy')]);
    % writeNPY(sos(imgESPIRiT), [PathSaveData,strcat('imgESPIRiT_',num2str(sub),'_',num2str(slc),'.npy')]);
    % writeNPY(sos(imgGRAPPA), [PathSaveData,strcat('imgGRAPPA_',num2str(sub),'_',num2str(slc),'.npy')]);
    writeNPY(imgGT, [PathSaveData,strcat('imgGT_',num2str(sub),'_',num2str(slc),'.npy')]);
    writeNPY(maps, [PathSaveData,strcat('SensitivityMaps_',num2str(sub),'_',num2str(slc),'.npy')]);
    % writeNPY(weights, [PathSaveData,strcat('SensitivityWeights_',num2str(sub),'_',num2str(slc),'.npy')]);
end

% Display coil images
if isDisplay
    figure, montagesc(abs(ZF),[]); 
    title('magnitude of physical coil images');
    colormap('default'); colorbar;
    figure, montagesc(angle(ZF),[]); 
    title('phase of physical coil images');
    colormap('default'); colorbar;
end

R = 4
%R = 4.7; % Accelaration factor
% Save NPY files
PathSaveData = [PathSave,PathSave1];
%%%%%%%%%%%%%%%===============================%%%%%%%%%%%%%%%%%%%%
%mask = mySubsampling(sx,sy,Nc,ncalib,R);
%%%%%%%%%%%%%%%===============================%%%%%%%%%%%%%%%%%%%%
mask = cs_generate_pattern([sx,sy],R);
mask = fftshift(mask);
mask = permute(mask,[2,1]);
writeNPY(mask, [strcat('/home2/HWGroup/wangcy/Data/Calgary/MultiChannel/Brain24_random/Mask/mask_',num2str(R),'/mask_',num2str(R),'.npy')]);
DATAc = DATA.*mask; % Subsampling the k-space data
%DATAc_real = real(DATAc);
%DATAc_imag = imag(DATAc);
%DATAc = cat(4,DATAc_real,DATAc_imag);
ZF = ifft2c(DATAc);
imgZF = sos(ZF); 
% Generate sensitivity maps
[maps,weights] = mySensitivityMap(sx,sy,calib,ksize,eigThresh_k,eigThresh_im,isDisplay);
%maps_new = zeros(size(maps));
%maps_new(:,:,:,1) = real(maps(:,:,:,1));
%maps_new(:,:,:,2) = imag(maps(:,:,:,2));
%maps = maps_new;

% Perform reconstruction
%sensemap = squeeze(maps(:,:,:,2));
%[imgSENSE, ~, ~] = mySenseRecon(sensemap, DATAc, R);

% create and ESPIRiT operator
%ESP = ESPIRiT(maps,weights);
%SNS = ESPIRiT(maps(:,:,:,end));
% [imgkSENSE,imgSENSE,imgkESPIRiT,imgESPIRiT,imgkGRAPPA,imgGRAPPA] = myParallelRecon(DATAc,ESP,SNS,calib,nIterCG);
%[imgkSENSE,imgSENSE] = myParallelRecon(DATAc,ESP,SNS,calib,nIterCG);

% Combine images
% a = [imgZF,abs(imgSENSE),sos(imgESPIRiT), sos(imgGRAPPA),imgGT];
% a_res = [abs(imgZF-imgGT),abs(abs(imgSENSE)-imgGT),abs(sos(imgESPIRiT)-imgGT), abs(sos(imgGRAPPA)-imgGT)];
%[~, ~, ~,~, ~,resL1ESPIRiT,~,~] = myESPIRiT(DATAc,calib,mask,isDisplay)

if isSaveNPY
    if ~exist(PathSaveData)
        mkdir(PathSaveData);
    end   
    %writeNPY(DATA, [PathSaveData,strcat('DATA_',num2str(sub),'_',num2str(slc),'.npy')]); % full sample k-space
    %writeNPY(DATAc, [PathSaveData,strcat('DATAc_',num2str(sub),'_',num2str(slc),'.npy')]); % subsample k-space
    %writeNPY(ZF, [PathSaveData,strcat('ZF_',num2str(sub),'_',num2str(slc),'.npy')]); % subsample k-space
   % writeNPY(GT, [PathSaveData,strcat('GT_',num2str(sub),'_',num2str(slc),'.npy')]); % subsample k-space
    writeNPY(imgZF, [PathSaveData,strcat('imgnewZF_',num2str(sub),'_',num2str(slc),'.npy')]);
    %writeNPY(resL1ESPIRiT, [PathSaveData,strcat('imgL1ESPIRiT_',num2str(sub),'_',num2str(slc),'.npy')]);  
    % writeNPY(abs(imgSENSE), [PathSaveData,strcat('imgSENSE_',num2str(sub),'_',num2str(slc),'.npy')]);
    %writeNPY(abs(imgSENSE), [PathSaveData,strcat('imgnewSENSE_',num2str(sub),'_',num2str(slc),'.npy')]);
    % writeNPY(sos(imgESPIRiT), [PathSaveData,strcat('imgESPIRiT_',num2str(sub),'_',num2str(slc),'.npy')]);
    % writeNPY(sos(imgGRAPPA), [PathSaveData,strcat('imgGRAPPA_',num2str(sub),'_',num2str(slc),'.npy')]);
    writeNPY(imgGT, [PathSaveData,strcat('imgGT_',num2str(sub),'_',num2str(slc),'.npy')]);
    writeNPY(maps, [PathSaveData,strcat('SensitivityMaps_',num2str(sub),'_',num2str(slc),'.npy')]);
    % writeNPY(weights, [PathSaveData,strcat('SensitivityWeights_',num2str(sub),'_',num2str(slc),'.npy')]);
end


% Display coil images
if isDisplay
    figure, montagesc(abs(ZF),[]); 
    title('magnitude of physical coil images');
    colormap('default'); colorbar;
    figure, montagesc(angle(ZF),[]); 
    title('phase of physical coil images');
    colormap('default'); colorbar;
end


R = 6
%R = 7.9; % Accelaration factor
% Save NPY files
PathSaveData = [PathSave,PathSave2];
%%%%%%%%%%%%%%%===============================%%%%%%%%%%%%%%%%%%%%
%mask = mySubsampling(sx,sy,Nc,ncalib,R);
mask = cs_generate_pattern([sx,sy],R);
mask = fftshift(mask);
mask = permute(mask,[2,1]);
DATAc = DATA.*mask; % Subsampling the k-space data
writeNPY(mask, [strcat('/home2/HWGroup/wangcy/Data/Calgary/MultiChannel/Brain24_random/Mask/mask_',num2str(R),'/mask_',num2str(R),'.npy')]);
%DATAc_real = real(DATAc);
%DATAc_imag = imag(DATAc);
%DATAc = cat(4,DATAc_real,DATAc_imag);
ZF = ifft2c(DATAc);
imgZF = sos(ZF); 
% Generate sensitivity maps
[maps,weights] = mySensitivityMap(sx,sy,calib,ksize,eigThresh_k,eigThresh_im,isDisplay);
%maps_new = zeros(size(maps));
%maps_new(:,:,:,1) = real(maps(:,:,:,2));
%maps_new(:,:,:,2) = imag(maps(:,:,:,2));
%maps = maps_new;
% Perform reconstruction
%[imgSENSE, ~, ~] = mySenseRecon(maps, DATAc, R);

% create and ESPIRiT operator
%ESP = ESPIRiT(maps,weights);
%SNS = ESPIRiT(maps(:,:,:,end));
%[imgkSENSE,imgSENSE] = myParallelRecon(DATAc,ESP,SNS,calib,nIterCG);

% Combine images
% b = [imgZF,abs(imgSENSE),sos(imgESPIRiT), sos(imgGRAPPA),imgGT];
% b_res = [abs(imgZF-imgGT),abs(abs(imgSENSE)-imgGT),abs(sos(imgESPIRiT)-imgGT), abs(sos(imgGRAPPA)-imgGT)];
%[~, ~, ~,~, ~,resL1ESPIRiT,~,~] = myESPIRiT(DATAc,calib,mask,isDisplay)

if isSaveNPY
    if ~exist(PathSaveData)
        mkdir(PathSaveData);
    end   
    %writeNPY(DATA, [PathSaveData,strcat('DATA_',num2str(sub),'_',num2str(slc),'.npy')]); % full sample k-space
    %writeNPY(DATAc, [PathSaveData,strcat('DATAc_',num2str(sub),'_',num2str(slc),'.npy')]); % subsample k-space
    % writeNPY(ZF, [PathSaveData,strcat('ZF_',num2str(sub),'_',num2str(slc),'.npy')]); % subsample k-space
     %writeNPY(GT, [PathSaveData,strcat('GT_',num2str(sub),'_',num2str(slc),'.npy')]); % subsample k-space
    writeNPY(imgZF, [PathSaveData,strcat('imgnewZF_',num2str(sub),'_',num2str(slc),'.npy')]);
    %writeNPY(resL1ESPIRiT, [PathSaveData,strcat('imgL1ESPIRiT_',num2str(sub),'_',num2str(slc),'.npy')]);  
    % writeNPY(abs(imgSENSE), [PathSaveData,strcat('imgSENSE_',num2str(sub),'_',num2str(slc),'.npy')]);
    %writeNPY(abs(imgSENSE), [PathSaveData,strcat('imgnewSENSE_',num2str(sub),'_',num2str(slc),'.npy')]);
    % writeNPY(sos(imgESPIRiT), [PathSaveData,strcat('imgESPIRiT_',num2str(sub),'_',num2str(slc),'.npy')]);
    % writeNPY(sos(imgGRAPPA), [PathSaveData,strcat('imgGRAPPA_',num2str(sub),'_',num2str(slc),'.npy')]);
     writeNPY(imgGT, [PathSaveData,strcat('imgGT_',num2str(sub),'_',num2str(slc),'.npy')]);
    writeNPY(maps, [PathSaveData,strcat('SensitivityMaps_',num2str(sub),'_',num2str(slc),'.npy')]);
    % writeNPY(weights, [PathSaveData,strcat('SensitivityWeights_',num2str(sub),'_',num2str(slc),'.npy')]);
end

% Display coil images
if isDisplay
    figure, montagesc(abs(ZF),[]); 
    title('magnitude of physical coil images');
    colormap('default'); colorbar;
    figure, montagesc(angle(ZF),[]); 
    title('phase of physical coil images');
    colormap('default'); colorbar;
end



isSucessful = 1
