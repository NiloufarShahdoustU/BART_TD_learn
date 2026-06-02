% Outcome-aligned HFA by balloon color for multiple patients
% This script plots mean HFA +/- SEM from -2 sec to +2 sec around outcome time
% It creates one separate figure/PDF for every ECoG channel.
% All PDFs are saved in one single folder.

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
allPDFOutputDir = fullfile(outputdir, 'OutcomeAligned_HFA_all_channels_all_patients');

if ~exist(allPDFOutputDir, 'dir')
    mkdir(allPDFOutputDir);
end

pre  = 2;
post = 2;

hgBand = [70 160];

groupNames = {'yellow','orange','red','gray'};

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

    fprintf('Processing patient %d/%d: %s\n', pp, length(ptArray), ptID);

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

        preSamp = round(Fs * pre);
        postSamp = round(Fs * post);

        nTime = preSamp + postSamp + 1;
        tSec = (-preSamp:postSamp) ./ Fs;

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

        if nTrials == 0
            warning('No usable trials found for patient %s. Skipping...', ptID);
            continue;
        end

        outcomeTimes = outcomeTimes(1:nTrials);
        outcomeType  = outcomeType(1:nTrials);
        balloonTimes = balloonTimes(1:nTrials);
        balloonIDs   = balloonIDs(1:nTrials);

        validTrial = false(1,nTrials);

        for tt = 1:nTrials
            outcomeSamp = floor(Fs * outcomeTimes(tt));
            startSamp = outcomeSamp - preSamp;
            endSamp   = outcomeSamp + postSamp;

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
            warning('No valid outcome-window trials found for patient %s. Skipping...', ptID);
            continue;
        end

        [b,a] = butter(4, hgBand/(Fs/2));

        HGmat_outcome = nan(nChans, nTime, nTrials);

        for tt = 1:nTrials

            outcomeSamp = floor(Fs * outcomeTimes(tt));
            startSamp = outcomeSamp - preSamp;
            endSamp   = outcomeSamp + postSamp;

            LFPchunk = double(NSX.Data(selectedChans, startSamp:endSamp));

            for ch = 1:nChans
                HGmat_outcome(ch,:,tt) = abs(hilbert(filtfilt(b,a,LFPchunk(ch,:))));
            end
        end

        smoothType = 'movmedian';
        smoothFactor = round(Fs ./ 5);

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

        plotMask = tSec >= -2 & tSec <= 2;
        tPlot = tSec(plotMask);

        for ch = 1:nChans

            thisSelectedChan = selectedChans(ch);

            thisTrodeLabel = getTextValue(trodeLabels_selectedChans, ch, sprintf('chan%d', thisSelectedChan));
            thisAnatomicalLoc = getTextValue(anatomicalLocs_selectedChans, ch, 'unknownBrainArea');

            safeTrodeLabel = makeSafeName(thisTrodeLabel);
            safeAnatomicalLoc = makeSafeName(thisAnatomicalLoc);

            allLow = [];
            allHigh = [];

            groupMean = cell(1,4);
            groupSEM  = cell(1,4);
            groupN    = zeros(1,4);

            for gg = 1:4

                trialMask = ismember(balloonIDs, groupIDs{gg});
                groupN(gg) = sum(trialMask);

                if groupN(gg) > 0

                    tmp = squeeze(HGmat_outcome(ch,plotMask,trialMask));

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

            allLow = allLow(isfinite(allLow));
            allHigh = allHigh(isfinite(allHigh));

            if isempty(allLow) || isempty(allHigh)
                ylims = [0 1];
            else
                yMin = min(allLow);
                yMax = max(allHigh);
                yPad = 0.05 * (yMax - yMin + eps);
                ylims = [yMin - yPad, yMax + yPad];
            end

        fig = figure('Color','w','Position',[100 100 1200 900], 'Visible','off');
        hold on;
        
        legendHandles = gobjects(0);
        legendNames = {};
        
        for gg = 1:4
        
            if groupN(gg) > 0
        
                m = groupMean{gg};
                s = groupSEM{gg};
                thisColor = groupColors{gg};
        
                thisColorRGB = thisColor(1:3);
        
                fill([tPlot fliplr(tPlot)], ...
                     [(m-s)' fliplr((m+s)')], ...
                     thisColorRGB, ...
                     'FaceAlpha', 0.2, ...
                     'EdgeColor', 'none', ...
                     'HandleVisibility','off');
        
                h = plot(tPlot, m, ...
                    'Color', thisColorRGB, ...
                    'LineWidth', 0.5);
        
                legendHandles(end+1) = h;
                legendNames{end+1} = sprintf('%s | n = %d trials', groupNames{gg}, groupN(gg));
        
            end
        end
        
        xline(0,'k--','LineWidth',1.2, 'HandleVisibility','off');
        
        xlim([-2 2]);
        ylim(ylims);
        
        xlabel('time from outcome (s)');
        ylabel('HFA');
        
        title(sprintf('%s | %s | %s | outcome-aligned HFA (-2 to +2 s)', ...
            ptID, thisTrodeLabel, thisAnatomicalLoc), ...
            'FontSize',16, ...
            'FontWeight','bold', ...
            'Interpreter','none');
        
        if ~isempty(legendHandles)
            legend(legendHandles, legendNames, ...
                'Location','best', ...
                'Interpreter','none', ...
                'Box','off');
        else
            text(0, mean(ylims), 'No trials', ...
                'HorizontalAlignment','center', ...
                'FontSize',12);
        end
        
        box off;
        set(gca,'FontSize',12,'LineWidth',1.2);
        
        pdfName = sprintf('%s_%s_%s_OutcomeAligned_HFA_minus2_to_plus2.pdf', ...
            ptID, safeTrodeLabel, safeAnatomicalLoc);
        
        pdfFile = fullfile(allPDFOutputDir, pdfName);
        
        print(fig, pdfFile, '-dpdf', '-bestfit');
        
        close(fig);
        clear fig;
        
        fprintf('Saved: %s\n', pdfFile);
        end

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

function txt = getTextValue(x, idx, defaultTxt)

    txt = defaultTxt;

    try
        if iscell(x)
            if idx <= numel(x) && ~isempty(x{idx})
                txt = x{idx};
            end
        elseif isstring(x)
            if idx <= numel(x) && strlength(x(idx)) > 0
                txt = char(x(idx));
            end
        elseif ischar(x)
            txt = x;
        else
            try
                if idx <= numel(x) && ~isempty(x(idx))
                    txt = char(string(x(idx)));
                end
            catch
                txt = defaultTxt;
            end
        end
    catch
        txt = defaultTxt;
    end

    if isempty(txt)
        txt = defaultTxt;
    end

    if isstring(txt)
        txt = char(txt);
    end

    if ~ischar(txt)
        txt = char(string(txt));
    end
end

function safeName = makeSafeName(txt)

    if isempty(txt)
        txt = 'unknown';
    end

    if isstring(txt)
        txt = char(txt);
    end

    if ~ischar(txt)
        txt = char(string(txt));
    end

    safeName = strtrim(txt);
    safeName = regexprep(safeName, '[^\w\-]+', '_');
    safeName = regexprep(safeName, '_+', '_');
    safeName = regexprep(safeName, '^_', '');
    safeName = regexprep(safeName, '_$', '');

    if isempty(safeName)
        safeName = 'unknown';
    end
end