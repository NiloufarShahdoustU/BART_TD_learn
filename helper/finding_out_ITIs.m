% ::blackrock data "channel" codes::
%1: trial start ::      [1 2 3 4 11 12 13 14] = [Y O R G Yc Oc Rc Gc] 
%2: responded ::        [22]
%3: inflating ::        [23 24] = [start stop]
%4: banked ::           [25]
%5: popped ::           [26]
%6: outcome shown ::    [100 101] = [correct incorrect]
%7: max rt exceeded  :: [127]
%8: trial over ::       [120]


% in this code, I wanted to find out the ITIs across all participants for the BART task. 
% the pdf is saved in the same route as this script. 

clc;
clear;
close all

ptArray =  {'201810','201811','201901','201902', '201902r','201903','201905'...
    ,'201909','201910','201911','201913','201914','201915','202001'...
   ,'202002','202003', '202004', '202005', '202006','202006u','202007','202008','202009'...
   ,'202011','202014','202015','202016', '202105','202107','202110','202114'...
    ,'202117','202118','202201','202202','202205', '202207', '202208', '202209', '202212'...
    ,'202212b','202214', '202215', '202216', '202217', '202302', '202306', '202307',...
    '202308','202311','202314a','202314b','202401','202405', '202406', '202407'...
    '202408', '202409', '202413a', '202413b', '202414', '202416','202417', '202418','202418b',...
    '202421','202422','202501','202503', '202504', '202505'};

plotFlag = false;

trial_end_trig = 120;
trial_start_trig = [1 2 3 4 11 12 13 14];

ITI_patient_mean_all = nan(size(ptArray));

for pt = 1:length(ptArray)

    ptID = ptArray{pt};

    parentDir = ['\\155.100.91.44\d\Data\preProcessed\BART_preprocessed\' ptID '\Data\*.nev']; 

    nevList = dir(parentDir);

    if isempty(nevList)
        warning(['No NEV file found for patient ' ptID]);
        continue
    end

    nevFile = fullfile(nevList(1).folder, nevList(1).name);

    NEV = openNEV(nevFile,'overwrite');
    trigs = NEV.Data.SerialDigitalIO.UnparsedData;
    trigTimes = NEV.Data.SerialDigitalIO.TimeStampSec';

    idx = find(trigs == trial_end_trig);

    % Remove trial_end_trig events that occur at the last element of trigs
    % becAUSE we dont want to keep the last 120 event it's the end of task
    idx(idx == numel(trigs)) = [];
    ITI_end_of_prev_trial_times = trigTimes(idx);

    idx = find(ismember(trigs, trial_start_trig));
    ITI_start_of_next_trial_times = trigTimes(idx);
    ITI_start_of_next_trial_times = ITI_start_of_next_trial_times(2:end);

    nITI = min(length(ITI_start_of_next_trial_times), length(ITI_end_of_prev_trial_times));

    ITI_start_of_next_trial_times = ITI_start_of_next_trial_times(1:nITI);
    ITI_end_of_prev_trial_times = ITI_end_of_prev_trial_times(1:nITI);

    ITI_patient = (ITI_start_of_next_trial_times - ITI_end_of_prev_trial_times)*1000;

    ITI_patient_mean = mean(ITI_patient, 'omitnan');

    ITI_patient_mean_all(pt) = ITI_patient_mean;

end

figure;
bar(ITI_patient_mean_all);
set(gca, 'XTick', 1:length(ptArray), 'XTickLabel', ptArray);
xtickangle(90);
ylabel('ITI patient mean (ms)');
xlabel('ptID');
title('Mean ITI for all patients');
box off;

saveas(gcf, 'ITI_patient_mean_all.pdf');