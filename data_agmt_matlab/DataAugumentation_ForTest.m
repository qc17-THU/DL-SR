clc, clear;
addpath(genpath('./XxUtils'));

% ------------------------------------------------------------------------
% DataAugumentation_ForTest: Generate testing data from BioSR dataset
%
% This program is used for generating image patch pairs from BioSR
% dataset(saved as mrc files). Data augumentation oprations include random
% cropping, random angle rotation, vertically flipping and horizontally
% flipping. The augumented data is saved as tiff files in the directory:
% SavePath/
%
% Before running, you should set following parameters first
% Specimen: Select the biological specimen in CCPs, ER, Microtubules,
% F-actin, F-actin_Nonlinear
% SavePath: directory to save augumented data
% DataPath: directory of BioSR dataset, eg: '../dataset/BioSR/F-actin';
% DataFilter: formatting of raw SIM data in BioSR
% GtFilter: formatting of GT-SIM data in BioSR
% RotFlag: 0 for no rotatioin, 1 for random rotation, 2 for flipping and random rotaion
% SaveRawFlag: 0 for only saving WF patches, 1 for saving raw SIM image patches
% SegX: target patch size x
% SegY: target patch size y
% testNum: number of patch pairs for validation
% bg_cut_prct: percentile value of gt background, used for some specific data, eg. CCPs 
% test_cell_list: cells used for generating validation image patch pairs
% test_snr_list: snr levels used for generating validation image patch pairs
%
% Author: Chang Qiao
% Email: qc17@tsinghua.org.cn
% Version: 2022/6/29
% ------------------------------------------------------------------------

%% set path and segmentation parameters
Specimen = 'CCPs'; % CCPs | ER | Microtubules | F-actin | F-actin_Nonlinear

SavePath = ['../dataset/test/' Specimen]; 
DataPath = ['../dataset/BioSR/' Specimen];
DataFilter = 'RawSIMData_level_*.mrc'; % formatting of raw SIM data in BioSR
GtFilter = 'SIM_gt*.mrc'; % formatting of GT-SIM data in BioSR

RotFlag = 2;     % 0 for no rotatioin, 1 for random rotation, 2 for flipping and random rotaion
SaveRawFlag = 1; % 0 for only saving WF patches, 1 for saving raw SIM image patches

SegX = 256; % target patch size x
SegY = 256; % target patch size y
test_num = 150;  % number of patch pairs for each signal level

bg_cut_prct = 0; % percentile value of gt background, used for some specific data, eg. CCPs
test_cell_list = 36:50; % cells used for generating validation image patch pairs
test_snr_list = 1:12; % snr levels used for generating validation image patch pairs

%% initialization
% if processing Non-Linear SIM data
if contains(SavePath,'Nonlinear'), n_perSIM = 25; else, n_perSIM = 9; end

test_cell_num = length(test_cell_list);

save_test_path = [SavePath '/test/'];
save_test_WF_path = [SavePath '/test_wf/'];
save_test_gt_path = [SavePath '/test_gt/'];

if SaveRawFlag == 1
    if exist(save_test_path,'dir'), rmdir(save_test_path,'s'); end
    mkdir(save_test_path);
end
if exist(save_test_WF_path,'dir'), rmdir(save_test_WF_path,'s'); end
if exist(save_test_gt_path,'dir'), rmdir(save_test_gt_path,'s'); end
mkdir(save_test_gt_path), mkdir(save_test_WF_path);


%% Generate test patches
CellList = XxSort(XxDir(DataPath, 'Cell*'));
ImgCount = 0;
% check cell list
nCell = length(CellList);
if sum(test_cell_list > nCell) > 0
    test_cell_list(test_cell_list > nCell) = [];
    test_cell_num = length(test_cell_list);
    warning('The cell list used for validation has been changed!');
end
% check snr list
if strcmp(Specimen, 'ER')
    files_input = XxSort(XxDir([CellList{1} filesep 'RawSIMData'], DataFilter));
else
    files_input = XxSort(XxDir(CellList{1}, DataFilter));
end
snr_num = length(test_snr_list);
if test_snr_list(end) > length(files_input)
    test_snr_list = test_snr_list(1):length(files_input);
    warning(['The SNR range of validation data has been changed into '...
        num2str(test_snr_list(1)) ':' num2str(length(files_input))]);
    snr_num = length(test_snr_list);
end

photon_counts = zeros(snr_num, test_num);
for i = test_cell_list
    fprintf('Generating testing data %d/%d\n',...
        i-test_cell_list(1)+1,test_cell_num);
    
    % read data file list
    if strcmp(Specimen, 'ER')
        files_input = XxSort(XxDir([CellList{i} filesep 'RawSIMData'], DataFilter));
        files_gt = XxSort(XxDir([CellList{i} filesep 'GTSIM'], 'GTSIM*'));
    else
        files_input = XxSort(XxDir(CellList{i}, DataFilter));
        files_gt = XxDir(CellList{i}, GtFilter);
    end
    

    
    % read gt data
    if strcmp(Specimen, 'ER')
        header_gt = XxReadMRCHeader(files_gt{1});
        data_gt = zeros(header_gt(1), header_gt(2), snr_num);
        for j = 1:length(test_snr_list)
            [header_gt, cur_data_gt] = XxReadMRC(files_gt{test_snr_list(j)});
            cur_data_gt = reshape(cur_data_gt, header_gt(1), header_gt(2), header_gt(3));
            cur_data_gt = XxNorm(cur_data_gt,0,100);
            data_gt(:,:,j) = imadjust(cur_data_gt,[bg_cut_prct,1],[]);
        end
    else
        [header_gt, data_gt] = XxReadMRC(files_gt{1});
        data_gt = reshape(data_gt, header_gt(1), header_gt(2), header_gt(3));
        data_gt = XxNorm(data_gt,0,100);
        data_gt = imadjust(data_gt,[bg_cut_prct,1],[]);
    end
    
    % read raw data
    header = XxReadMRCHeader(files_input{1});
    data = zeros(header(1), header(2), header(3), snr_num);
    for j = 1:length(test_snr_list)
        [header, cur_data] = XxReadMRC(files_input{test_snr_list(j)});
        cur_data = reshape(cur_data, header(1), header(2), header(3));
        data(:,:,:,j) = cur_data;
    end
    data = reshape(data, header(1), header(2), header(3) * snr_num);
    
    % perform data augmentation
    [data_seg, data_gt_seg] = XxDataSeg_ForTest(data, data_gt, ...
        ceil(test_num/test_cell_num) , SegX, SegY, RotFlag);
    num_image = size(data_seg, 3) / snr_num / n_perSIM;
    data_seg = reshape(data_seg, SegX, SegY, n_perSIM, snr_num, num_image);
    if strcmp(Specimen, 'ER')
        [Ny_hr, Nx_hr, ~] = size(data_gt_seg);
        data_gt_seg = reshape(data_gt_seg, Ny_hr, Nx_hr, snr_num, num_image);
    end
    
    for n = 1:num_image
        
        ImgCount = ImgCount + 1;
        
        % save GT SIM image
        if ~strcmp(Specimen, 'ER')
            img_gt = uint16(65535 * XxNorm(data_gt_seg(:,:,n)));
            imwrite(img_gt,[save_test_gt_path num2str(ImgCount,'%03d') '.tif']);
        end
        
        for j = 1:length(test_snr_list)
            
            cur_path = [save_test_path filesep 'level_' num2str(test_snr_list(j),'%02d')];
            cur_wf_path = [save_test_WF_path filesep 'level_' num2str(test_snr_list(j),'%02d')];
            if ~exist(cur_path,'dir')
                mkdir(cur_path);
                mkdir(cur_wf_path);
            end
            
            img = uint16(65535 * XxNorm(data_seg(:,:,:,j,n)));
            img_sum = uint16(65535 * XxNorm(squeeze(mean(img, 3))));
            
            % save raw SIM images
            if SaveRawFlag == 1
                im_batch_dir = [cur_path filesep num2str(ImgCount,'%03d') '.tif'];
                XxWriteTiff(img, im_batch_dir);
            end
            
            % save WF image
            imwrite(img_sum,[cur_wf_path filesep num2str(ImgCount,'%03d') '.tif']);
            
            % save GT SIM image for ER
            if strcmp(Specimen, 'ER')
                cur_gt_path = [save_test_gt_path filesep 'level_' num2str(test_snr_list(j),'%02d')];
                if ~exist(cur_gt_path,'dir'), mkdir(cur_gt_path); end
                img_gt = uint16(65535 * XxNorm(data_gt_seg(:,:,j,n)));
                imwrite(img_gt,[cur_gt_path filesep num2str(ImgCount,'%03d') '.tif']);
            end
        end
    end
end

