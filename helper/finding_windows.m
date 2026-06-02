% Outcome-aligned HFA by balloon color for multiple patients
% This script plots mean HFA +/- SEM from -2 sec to outcome time, 0 sec
% using 2x2 subplots for Yellow, Orange, Red, and Gray balloons.
% Gray = triggers 4 and 14.

clear;
clc;
close all;

ptArray =  {'201810','201811','201901','201902', '201902r','201903','201905',...
    '201909','201910','201911','201913','201914','201915','202001',...
    '202002','202003', '202004', '202005', '202006','202006u','202007','202008','202009',...
    '202011','202014','202015','202016', '202105','202107','202110','202114',...
    '202117','202118','202201','202202','202205', '202207', '202208', '202209', '202212',...
    '202212b','202214', '202215', '202216', '202217', '202302', '202306', '202307',...
    '202308','202311','202314a','202314b','202401','202405', '202406', '202407',...
    '202408', '202409', '202413a', '202413b', '202414', '202416','202417', '202418','202418b',...
    '202421','202422','202501','202503', '202504', '202505'};

outputdir = '\\155.100.91.44\d\Code\Rhiannon\BART\0_Nill\Rhia_code\finding_windows\output\';

if ~exist(outputdir, 'dir')
    mkdir(outputdir);
end

pre  = 2;
post = 3;

hgBand = [70 160];

groupNames = {'Yellow','Orange','Red','Gray'};

groupIDs = {
    [1], ...
    [2], ...
    [3], ...
    [4 14] ...
    };

groupColors = {
    [0.9290 0.6940 0.1250], ...
    [0.8500 0.3250 0.0980], ...
    [0.6350 0.0780 0.1840], ...
    [0.4 0.4 0.4] ...
    };

for pp = 1:length(ptArray)

    ptID = ptArray{pp};

    fprintf('\n========================================\n');
    fprintf('Processing patient %d/%d: %s\n', pp, length(ptArray), ptID);
    fprintf('========================================\n');

    try
        dataDir = ['\\155.100.91.44\D\Data\preProcessed\BART_preprocessed\' ptID '\Data\'];
        parentDir = [dataDir '*.nev'];

        nevList = dir(parentDir);

        if isempty(nevList)
            warning('No NEV files found for patient %s. Skipping...', ptID);
            continue;
        end

        [~, nevSortIdx] = sort({nevList.name});
        nevList = nevList(nevSortIdx);

        if length(nevList) > 1
            warning('Many NEV files available for patient %s. Using the first one in the list: %s', ...
                ptID, nevList(1).name);
        end

        nevFile = fullfile(nevList(1).folder, nevList(1).name);
        nevFile(strfind(nevFile,'\')) = '/';

        [trodeLabels,isECoG,~,~,anatomicalLocs,~] = ptTrodesBART_2(ptID);

        selectedChans = find(isECoG);
        trodeLabels_selectedChans = trodeLabels(isECoG);
        anatomicalLocs_selectedChans = anatomicalLocs(isECoG);

        nChans = length(selectedChans);

        if nChans == 0
            warning('No ECoG channels found for patient %s. Skipping...', ptID);
            continue;
        end

        matFile = [dataDir ptID '.bartBHV.mat'];

        if ~exist(matFile, 'file')
            warning('Behavioral file not found for patient %s: %s. Skipping...', ptID, matFile);
            continue;
        end

        BHV = load(matFile);

        if ~isfield(BHV, 'data')
            warning('BHV file for patient %s does not contain data. Skipping...', ptID);
            continue;
        end

        NEV = openNEV(nevFile,'overwrite');

        trigs = NEV.Data.SerialDigitalIO.UnparsedData;
        trigTimes = NEV.Data.SerialDigitalIO.TimeStampSec;

        [nevPath, nevName, ~] = fileparts(nevFile);

        ns2File = fullfile(nevPath,[nevName '.ns2']);

        if ~exist(ns2File, 'file')
            warning('NS2 file not found for patient %s: %s. Skipping...', ptID, ns2File);
            continue;
        end

        NSX = openNSx(ns2File);

        Fs = NSX.MetaTags.SamplingFreq;
        nSamples = size(NSX.Data,2);

        nTime = Fs*(pre+post)+1;
        tSec = linspace(-pre, post, nTime);

        outcomeIdx = sort([find(trigs==25); find(trigs==26)]);
        outcomeTimes = trigTimes(outcomeIdx);
        outcomeType = trigs(outcomeIdx) - 24;

        validBalloonTriggers = [1 2 3 4 14];
        balloonIdx = find(ismember(trigs, validBalloonTriggers));
        balloonTimes = trigTimes(balloonIdx);
        balloonIDs = trigs(balloonIdx);

        nTrials = min([length(outcomeTimes), ...
                       length(balloonTimes), ...
                       length(balloonIDs), ...
                       length(outcomeType), ...
                       length(BHV.data)]);

        outcomeTimes = outcomeTimes(1:nTrials);
        outcomeType  = outcomeType(1:nTrials);
        balloonTimes = balloonTimes(1:nTrials);
        balloonIDs   = balloonIDs(1:nTrials);

        validTrial = false(1,nTrials);

        for tt = 1:nTrials
            startSamp = floor(Fs*outcomeTimes(tt)) - Fs*pre;
            endSamp   = floor(Fs*outcomeTimes(tt)) + Fs*post;

            if startSamp >= 1 && endSamp <= nSamples
                validTrial(tt) = true;
            end
        end

        outcomeTimes = outcomeTimes(validTrial);
        outcomeType  = outcomeType(validTrial);
        balloonTimes = balloonTimes(validTrial);
        balloonIDs   = balloonIDs(validTrial);

        nTrials = length(outcomeTimes);

        if nTrials == 0
            warning('No valid trials remained after checking neural indexing bounds for patient %s. Skipping...', ptID);
            continue;
        end

        [b,a] = butter(4, hgBand/(Fs/2));

        HGmat_outcome = nan(nChans, nTime, nTrials);

        for tt = 1:nTrials

            startSamp = floor(Fs*outcomeTimes(tt)) - Fs*pre;
            endSamp   = floor(Fs*outcomeTimes(tt)) + Fs*post;

            LFPchunk = double(NSX.Data(selectedChans, startSamp:endSamp));

            for ch = 1:nChans
                HGmat_outcome(ch,:,tt) = abs(hilbert(filtfilt(b,a,LFPchunk(ch,:))));
            end
        end

        smoothType = 'movmedian';
        smoothFactor = Fs./5;

        HGmat_outcome = smoothdata(HGmat_outcome, 2, smoothType, smoothFactor);

        outcomeAndCueHGs = struct();
        outcomeAndCueHGs.ptID = ptID;
        outcomeAndCueHGs.outcomeHG = HGmat_outcome;
        outcomeAndCueHGs.tSec = tSec;
        outcomeAndCueHGs.timeWin = [-pre post];
        outcomeAndCueHGs.balloonIDs = balloonIDs;
        outcomeAndCueHGs.outcomeType = outcomeType;
        outcomeAndCueHGs.selectedChans = selectedChans;
        outcomeAndCueHGs.trodeLabels_selectedChans = trodeLabels_selectedChans;
        outcomeAndCueHGs.anatomicalLocs_selectedChans = anatomicalLocs_selectedChans;

        plotMask = tSec >= -2 & tSec <= 0;
        tPlot = tSec(plotMask);

        allLow = [];
        allHigh = [];

        groupMean = cell(1,4);
        groupSEM  = cell(1,4);
        groupN    = zeros(1,4);

        for gg = 1:4

            trialMask = ismember(balloonIDs, groupIDs{gg});
            groupN(gg) = sum(trialMask);

            if groupN(gg) > 0

                tmp = squeeze(mean(HGmat_outcome(:,plotMask,trialMask),1));

                if isvector(tmp)
                    tmp = tmp(:);
                end

                m = mean(tmp, 2, 'omitnan');
                s = std(tmp, 0, 2, 'omitnan') ./ sqrt(size(tmp,2));

                groupMean{gg} = m;
                groupSEM{gg} = s;

                allLow  = [allLow;  m - s];
                allHigh = [allHigh; m + s];

            else
                groupMean{gg} = nan(sum(plotMask),1);
                groupSEM{gg}  = nan(sum(plotMask),1);
            end
        end

        if isempty(allLow) || isempty(allHigh)
            warning('No trials found for requested balloon groups for patient %s. Skipping plot...', ptID);
            continue;
        end

        yMin = min(allLow);
        yMax = max(allHigh);
        yPad = 0.05 * (yMax - yMin + eps);
        ylims = [yMin - yPad, yMax + yPad];

        fig = figure('Color','w','Position',[100 100 1200 900], 'Visible','off');

        for gg = 1:4

            subplot(2,2,gg);
            hold on;

            if groupN(gg) > 0

                m = groupMean{gg};
                s = groupSEM{gg};
                thisColor = groupColors{gg};

                fill([tPlot fliplr(tPlot)], ...
                     [(m-s)' fliplr((m+s)')], ...
                     thisColor, ...
                     'FaceAlpha', 0.25, ...
                     'EdgeColor', 'none');

                plot(tPlot, m, 'Color', thisColor, 'LineWidth', 2.5);

            else
                text(-1, mean(ylims), 'No trials', ...
                    'HorizontalAlignment','center', ...
                    'FontSize',12);
            end

            xline(0,'k--','LineWidth',1.2);

            xlim([-2 0]);
            ylim(ylims);

            xlabel('Time from outcome (s)');
            ylabel('HFA');

            title(sprintf('%s (n = %d trials)', groupNames{gg}, groupN(gg)), ...
                'FontWeight','bold');

            box off;
            set(gca,'FontSize',12,'LineWidth',1.2);
        end

        sgtitle(sprintf('%s Outcome-aligned HFA (-2 to 0 s)', ptID), ...
            'FontSize',16, ...
            'FontWeight','bold');

        pdfFile = fullfile(outputdir, [ptID '_OutcomeAligned_HFA_2x2_YellowOrangeRedGray.pdf']);

        print(fig, pdfFile, '-dpdf', '-bestfit');

        fprintf('PDF saved here:\n%s\n', pdfFile);

        matOutFile = fullfile(outputdir, [ptID '_OutcomeAligned_HFA_data.mat']);
        save(matOutFile, 'outcomeAndCueHGs', 'groupMean', 'groupSEM', 'groupN', '-v7.3');

        fprintf('MAT data saved here:\n%s\n', matOutFile);

        close(fig);
        clear fig;

    catch ME

        warning('Patient %s failed. Moving to next patient...', ptID);
        fprintf('Error message:\n%s\n', ME.message);

        if exist('fig','var') && isgraphics(fig)
            close(fig);
            clear fig;
        end

        continue;
    end
end

fprintf('\nDone processing all patients.\n');