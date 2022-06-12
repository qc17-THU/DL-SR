clc, clear;
addpath(genpath('./XxUtils'));

% ------------------------------------------------------------------------
% DataAugumentation_ForTrain: Perform data augumentation for BioSR dataset
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
% ValidateNum: number of patch pairs for validation
% TotalSegNum: number of patch pairs for training
% bg_cut_prct: percentile value of gt background, used for some specific data, eg. CCPs
% train_cell_list: cells used for generating training image patch pairs
% train_snr_list: snr levels used for generating training image patch pairs  
% validate_cell_list: cells used for generating validation image patch pairs
% validate_snr_list: snr levels used for generating validation image patch pairs
%
% Author: Chang Qiao
% Email: qc17@mails.tsinghua.edu.cn
% Version: 2022/6/12
% ------------------------------------------------------------------------

%% set path and segmentation parameters
Specimen = 'F-actin'; % CCPs | ER | Microtubules | F-actin | F-actin_Nonlinear

SavePath = ['../dataset/train/' Specimen]; 
DataPath = ['../dataset/BioSR/' Specimen];
DataFilter = 'RawSIMData_level_*.mrc'; % formatting of raw SIM data in BioSR
GtFilter = 'SIM_gt*.mrc'; % formatting of GT-SIM data in BioSR

RotFlag = 2;     % 0 for no rotatioin, 1 for random rotation, 2 for flipping and random rotaion
SaveRawFlag = 1; % 0 for only saving WF patches, 1 for saving raw SIM image patches

SegX = 128; % target patch size x
SegY = 128; % target patch size y
ValidateNum = 2000;  % number of patch pairs for validation
TotalSegNum = 20000; % number of patch pairs for training

bg_cut_prct = 0; % percentile value of gt background, used for some specific data, eg. CCPs
train_cell_list = 6:35; % cells used for generating training image patch pairs
train_snr_list = 1:12;  % snr levels used for generating training image patch pairs 
validate_cell_list = 1:5; % cells used for generating validation image patch pairs
validate_snr_list = 1:12; % snr levels used for generating validation image patch pairs

%% initialization
% if processing Non-Linear SIM data
if contains(SavePath,'Nonlinear'), n_perSIM = 25; else, n_perSIM = 9; end

train_cell_num = length(train_cell_list);
validate_cell_num = length(validate_cell_list);

save_training_path = [SavePath '/training/'];
save_training_WF_path = [SavePath '/training_wf/'];
save_training_gt_path = [SavePath '/training_gt/'];

save_validate_path = [SavePath '/validate/'];
save_validate_WF_path = [SavePath '/validate_wf/'];
save_validate_gt_path = [SavePath '/validate_gt/'];

if SaveRawFlag == 1
    if exist(save_training_path,'dir'), rmdir(save_training_path,'s'); end
    if exist(save_validate_path,'dir'), rmdir(save_validate_path,'s'); end
    mkdir(save_training_path);
    mkdir(save_validate_path);
end
if exist(save_training_WF_path,'dir'), rmdir(save_training_WF_path,'s'); end
if exist(save_validate_WF_path,'dir'), rmdir(save_validate_WF_path,'s'); end
if exist(save_training_gt_path,'dir'), rmdir(save_training_gt_path,'s'); end
if exist(save_validate_gt_path,'dir'), rmdir(save_validate_gt_path,'s'); end
mkdir(save_training_gt_path), mkdir(save_training_WF_path);
mkdir(save_validate_gt_path), mkdir(save_validate_WF_path);


%% Generate validate patches
CellList = XxSort(XxDir(DataPath, 'Cell*'));
ImgCount = 1;
nCell = length(CellList);
if sum(validate_cell_list > nCell) > 0
    validate_cell_list(validate_cell_list > nCell) = [];
    validate_cell_num = length(validate_cell_list);
    warning('The cell list used for validation has been changed!');
end

for i = validate_cell_list
    
    % read data
    if strcmp(Specimen, 'ER')
        files_input = XxSort(XxDir([CellList{i} filesep 'RawSIMData'], DataFilter));
        files_gt = XxSort(XxDir([CellList{i} filesep 'GTSIM'], 'GTSIM*'));
    else
        files_input = XxSort(XxDir(CellList{i}, DataFilter));
        files_gt = XxDir(CellList{i}, GtFilter);
        [header_gt, data_gt] = XxReadMRC(files_gt{1});
        data_gt = reshape(data_gt, header_gt(1), header_gt(2), header_gt(3));
        data_gt = XxNorm(data_gt,0,100);
        data_gt = imadjust(data_gt,[bg_cut_prct,1],[]);
    end
    
    snr_num = length(validate_snr_list);
    if validate_snr_list(end) > length(files_input)
        validate_snr_list = validate_snr_list(1):length(files_input);
        warning(['The SNR range of validation data has been changed into '...
            num2str(validate_snr_list(1)) ':' num2str(length(files_input))]);
        snr_num = length(validate_snr_list);
    end
    
    for j = validate_snr_list
        fprintf('Generating validation data %d/%d, SNR %d/%d\n',...
            i-validate_cell_list(1)+1,validate_cell_num,j,snr_num);
        [header, data] = XxReadMRC(files_input{j});
        data = reshape(data, header(1), header(2), header(3));
        data = XxNorm(data,0,100);
        if strcmp(Specimen, 'ER')
            [header_gt, data_gt] = XxReadMRC(files_gt{j});
            data_gt = reshape(data_gt, header_gt(1), header_gt(2), header_gt(3));
            data_gt = XxNorm(data_gt,0,100);
            data_gt = imadjust(data_gt,[bg_cut_prct,1],[]);
        end
        
        if RotFlag < 2
            [data_seg, data_gt_seg] = XxDataSeg_ForTrain(data, data_gt, ...
                ceil(ValidateNum/validate_cell_num/snr_num) , SegX, SegY, RotFlag);
            num_image = size(data_gt_seg, 3);
        else
            [data_seg, data_gt_seg] = XxDataSeg_ForTrain(data, data_gt, ...
                ceil(ValidateNum/validate_cell_num/snr_num/4) , SegX, SegY, RotFlag);
            % flip axis y
            curdata = flipud(data);
            curdata_gt = flipud(data_gt);
            [data_seg_ap, data_gt_seg_ap] = XxDataSeg_ForTrain(curdata, curdata_gt, ...
                ceil(ValidateNum/validate_cell_num/snr_num/4) , SegX, SegY, RotFlag);
            num_image = size(data_gt_seg_ap, 3);
            data_seg(:, :, end+1:end+num_image*n_perSIM) = data_seg_ap;
            data_gt_seg(:, :, end+1:end+num_image) = data_gt_seg_ap;
            % flip axis x
            curdata = fliplr(data);
            curdata_gt = fliplr(data_gt);
            [data_seg_ap, data_gt_seg_ap] = XxDataSeg_ForTrain(curdata, curdata_gt, ...
                ceil(ValidateNum/validate_cell_num/snr_num/4) , SegX, SegY, RotFlag);
            num_image = size(data_gt_seg_ap, 3);
            data_seg(:, :, end+1:end+num_image*n_perSIM) = data_seg_ap;
            data_gt_seg(:, :, end+1:end+num_image) = data_gt_seg_ap;
            % flip axis x and y
            curdata = flipud(curdata);
            curdata_gt = flipud(curdata_gt);
            [data_seg_ap, data_gt_seg_ap] = XxDataSeg_ForTrain(curdata, curdata_gt, ...
                ceil(ValidateNum/validate_cell_num/snr_num/4) , SegX, SegY, RotFlag);
            num_image = size(data_gt_seg_ap, 3);
            data_seg(:, :, end+1:end+num_image*n_perSIM) = data_seg_ap;
            data_gt_seg(:, :, end+1:end+num_image) = data_gt_seg_ap;
            num_image = size(data_gt_seg, 3);
        end
            
        for n = 1:num_image
            
            img = XxNorm(data_seg(:,:,(n-1)*n_perSIM+1:n*n_perSIM),0,100);
            img = uint16(img * 65535);
            img_sum = uint16(XxNorm(sum(img, 3), 0, 100) * 65535);
            img_gt = uint16(XxNorm(data_gt_seg(:,:,n),0,100) * 65535);         
            
            % save raw SIM images
            if SaveRawFlag == 1
                im_batch_dir = [save_validate_path num2str(ImgCount,'%08d')];
                mkdir(im_batch_dir);
                for nn = 1:n_perSIM
                    imwrite(img(:,:,nn),[im_batch_dir '/' num2str(nn) '.tif']);
                end
            end
            % save WF and gt image
            imwrite(img_sum,[save_validate_WF_path num2str(ImgCount,'%08d') '.tif']);
            imwrite(img_gt,[save_validate_gt_path num2str(ImgCount,'%08d') '.tif']);    
            ImgCount = ImgCount + 1;
        end
    end
end

%% Generate training patches
if sum(train_cell_list > nCell) > 0
    train_cell_list(train_cell_list > nCell) = [];
    train_cell_num = length(train_cell_list);
    warning('The cell list used for training has been changed!');
end
ImgCount = 1;
for i = train_cell_list
    
    % read data
    if strcmp(Specimen, 'ER')
        files_input = XxSort(XxDir([CellList{i} filesep 'RawSIMData'], DataFilter));
        files_gt = XxSort(XxDir([CellList{i} filesep 'GTSIM'], 'GTSIM*'));
    else
        files_input = XxSort(XxDir(CellList{i}, DataFilter));
        files_gt = XxDir(CellList{i}, GtFilter);
        [header_gt, data_gt] = XxReadMRC(files_gt{1});
        data_gt = reshape(data_gt, header_gt(1), header_gt(2), header_gt(3));
        data_gt = XxNorm(data_gt,0,100);
        data_gt = imadjust(data_gt,[bg_cut_prct,1],[]);
    end
    
    snr_num = length(train_snr_list);
    if train_snr_list(end) > length(files_input)
        train_snr_list = train_snr_list(1):length(files_input);
        warning(['The SNR range of training data has been changed into '...
            num2str(train_snr_list(1)) ':' num2str(length(files_input))]);
        snr_num = length(train_snr_list);
    end
    
    for j = train_snr_list
        fprintf('Generating training data %d/%d, SNR %d/%d\n', ...
            i-train_cell_list(1)+1,train_cell_num,j,snr_num);
        
        [header, data] = XxReadMRC(files_input{j});
        data = reshape(data, header(1), header(2), header(3));
        data = XxNorm(data,0,100);
        if strcmp(Specimen, 'ER')
            [header_gt, data_gt] = XxReadMRC(files_gt{j});
            data_gt = reshape(data_gt, header_gt(1), header_gt(2), header_gt(3));
            data_gt = XxNorm(data_gt,0,100);
            data_gt = imadjust(data_gt,[bg_cut_prct,1],[]);
        end
        
        if RotFlag < 2
            [data_seg, data_gt_seg] = XxDataSeg_ForTrain(data, data_gt, ...
                ceil(TotalSegNum/train_cell_num/snr_num) , SegX, SegY, RotFlag);
            num_image = size(data_gt_seg, 3);
        else
            [data_seg, data_gt_seg] = XxDataSeg_ForTrain(data, data_gt, ...
                ceil(TotalSegNum/train_cell_num/snr_num/4) , SegX, SegY, RotFlag);
            % flip axis y
            curdata = flipud(data);
            curdata_gt = flipud(data_gt);
            [data_seg_ap, data_gt_seg_ap] = XxDataSeg_ForTrain(curdata, curdata_gt, ...
                ceil(TotalSegNum/train_cell_num/snr_num/4) , SegX, SegY, RotFlag);
            num_image = size(data_gt_seg_ap, 3);
            data_seg(:, :, end+1:end+num_image*n_perSIM) = data_seg_ap;
            data_gt_seg(:, :, end+1:end+num_image) = data_gt_seg_ap;
            % flip axis x
            curdata = fliplr(data);
            curdata_gt = fliplr(data_gt);
            [data_seg_ap, data_gt_seg_ap] = XxDataSeg_ForTrain(curdata, curdata_gt, ...
                ceil(TotalSegNum/train_cell_num/snr_num/4) , SegX, SegY, RotFlag);
            num_image = size(data_gt_seg_ap, 3);
            data_seg(:, :, end+1:end+num_image*n_perSIM) = data_seg_ap;
            data_gt_seg(:, :, end+1:end+num_image) = data_gt_seg_ap;
            % flip axis x and y
            curdata = flipud(curdata);
            curdata_gt = flipud(curdata_gt);
            [data_seg_ap, data_gt_seg_ap] = XxDataSeg_ForTrain(curdata, curdata_gt, ...
                ceil(TotalSegNum/train_cell_num/snr_num/4) , SegX, SegY, RotFlag);
            num_image = size(data_gt_seg_ap, 3);
            data_seg(:, :, end+1:end+num_image*n_perSIM) = data_seg_ap;
            data_gt_seg(:, :, end+1:end+num_image) = data_gt_seg_ap;
            num_image = size(data_gt_seg, 3);
        end
        
        for n = 1:num_image
            img = XxNorm(data_seg(:,:,(n-1)*n_perSIM+1:n*n_perSIM),0,100);
            img = uint16(img * 65535);
            img_sum = uint16(XxNorm(sum(img, 3), 0, 100) * 65535);
            img_gt = uint16(XxNorm(data_gt_seg(:,:,n),0,100) * 65535); 
            
            % save raw SIM images
            if SaveRawFlag == 1
                im_batch_dir = [save_training_path num2str(ImgCount,'%08d')];
                mkdir(im_batch_dir);
                for nn = 1:n_perSIM
                    imwrite(img(:,:,nn),[im_batch_dir '/' num2str(nn) '.tif']);
                end
            end
            % save WF and gt image
            imwrite(img_sum,[save_training_WF_path num2str(ImgCount,'%08d') '.tif']);
            imwrite(img_gt,[save_training_gt_path num2str(ImgCount,'%08d') '.tif']);
            
            ImgCount = ImgCount + 1;
        end
    end
end

