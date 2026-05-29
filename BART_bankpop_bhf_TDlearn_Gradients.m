function [TDdataGradients,outcomeData,cueData,outcomeAndCueHGs] = BART_bankpop_bhf_TDlearn_Gradients(ptID,whichTrialsLM,plotFlag)   % BARTstats
% BART_BANKPOP_BHF_TDLEARN fits BART responses to Q-learning model.
%
%   [TDdata] = BART_bankpop_bhf_TDlearn(ptID) fits BHF data to [RISK SENSITIVE] Q-learning model.
%
%	The output data structure TDdata includes timecourses of [RISK SENSITIVE] Q-learning parameters
%	over trials and fit parameters.
%
%   'whichTrialsLM' input argument specifies, as a string, which trials to
%       evaluate the linear model correlating high gamma activity and RSTD
%       learning model variables. (default: '40:end')
%
%   'whichTrialsTD' input argument specifies, as a string, which trials to
%       include in the RSTD learning model.ning/
%       options are: 'red', 'orange', 'yellow', and 'control' (default: 'all')
%
%   The fourth input argument is a boolean, and specifies whether to
%   plot all significant electrodes. (default: false)

set(0,'defaultfigurerender','painters'); % making sure we can edit figures after saving.

if nargin < 2 || isempty(whichTrialsLM)
    whichTrialsLM = '1:end';
end
if nargin < 3 || isempty(plotFlag)
    plotFlag = false;
end

% author: EHS20181005 / RLC: September 2025

% % % % % % DEBUGGING~~~~~~~~~~~~
 % ptID = '202202';
 % whichRL = 'risksensitive';
 % whichTrialsLM = '1:end';
 % plotFlag = false;
% % % % % ~~~~~~~~~~~~~~~~~~~~~~~

parentDir = ['\\155.100.91.44\D\Data\preProcessed\BART_preprocessed\' ptID '\Data\*.nev'];
nevList = dir(parentDir);
if isempty(nevList)
    error('no nev files found...')
end
[~,nevSortIdx] = sort({nevList.name});
nevList = nevList(nevSortIdx);
if length(nevList)>1
    warning('many nev files available for this patient. Using the first one in the list: %s', nevList(1).name);
end
nevFile = fullfile(nevList(1).folder,nevList(1).name);
nevFile(strfind(nevFile,'\'))='/';
[trodeLabels,isECoG,~,~,anatomicalLocs,~] = ptTrodesBART_2(ptID); % old way of getting brain labels

selectedChans = find(isECoG);
trodeLabels_selectedChans = trodeLabels(isECoG); % Removing NaCs from trodelabels
% data parameters
nChans = length(selectedChans);

% loading behavioral matFile
matFile = ['\\155.100.91.44\D\Data\preProcessed\BART_preprocessed\' ptID '\Data\' ptID '.bartBHV.mat'];
BHV = load(matFile);

% load and define triggers from nevFle
NEV = openNEV(nevFile,'overwrite');
trigs = NEV.Data.SerialDigitalIO.UnparsedData;
trigTimes = NEV.Data.SerialDigitalIO.TimeStampSec;

% load neural data
[nevPath,nevName,~] = fileparts(nevFile);
NSX = openNSx(fullfile(nevPath,[nevName '.ns2']));
Fs = NSX.MetaTags.SamplingFreq;

% timing parameters.
pre = 2;
post = 3;
tSec = linspace(-pre,post,Fs*(pre+post)+1);

% task parameters in chronological order..
% respTimes = trigTimes(trigs==24);
outcomeTimes = trigTimes(trigs==25 | trigs==26);
outcomeType = trigs(sort([find(trigs==25); find(trigs==26)]))-24; % 1 = bank, 2 = pop
validBalloonTriggers = [1 2 3 4 11 12 13 14];
balloonTimes = trigTimes(ismember(trigs,validBalloonTriggers));
% isCTRL = logical([data.is_control]);
balloonIDs = trigs(ismember(trigs,validBalloonTriggers));
% inflateTimes = trigTimes(trigs==23);
% pointsPerTrial = [data.points];

% Make all trial-aligned vectors exactly the same length. This prevents
% trigger 4 / aborted-trial shifts from lowering model fits.
nTrials = min([length(outcomeTimes), length(balloonTimes), length(balloonIDs), length(outcomeType), length(BHV.data)]);
outcomeTimes = outcomeTimes(1:nTrials);
outcomeType = outcomeType(1:nTrials);
balloonTimes = balloonTimes(1:nTrials);
balloonIDs = balloonIDs(1:nTrials);

% HG filter
[b,a] = butter(4,[70 160]/(Fs/2));

% epoching data - makes a smaller array than number of trials for
% whichtrialsTD~='all'

for tt = nTrials:-1:1
    % epoch the data here [channels X samples X trials]
    % % outcome-aligned
    % LFPmat(:,:,tt) = NSX.Data(1:nChans,floor(Fs*outcomeTimes(tt))-Fs*pre:floor(Fs*outcomeTimes(tt))+Fs*post);
    % % balloon-aligned
    % LFPmat1(:,:,tt) = NSX.Data(1:nChans,floor(Fs*balloonTimes(tt))-Fs*pre:floor(Fs*balloonTimes(tt))+Fs*post);


    % ATTENION:
    % the code above is wrong. we should go selectedchans not from 1 to
    % nChans, because nChans is only the length of selected chans

    LFPmat(:,:,tt)  = NSX.Data(selectedChans, floor(Fs*outcomeTimes(tt))-Fs*pre : floor(Fs*outcomeTimes(tt))+Fs*post);
    LFPmat1(:,:,tt) = NSX.Data(selectedChans, floor(Fs*balloonTimes(tt))-Fs*pre : floor(Fs*balloonTimes(tt))+Fs*post);

    % do spectral claculations here
    for ch = 1:nChans
        % outcome-aligned
        HGmat_outcome(ch,:,tt) = abs(hilbert(filtfilt(b,a,double(LFPmat(ch,:,tt)))));
        % balloon-aligned
        HGmat_cue(ch,:,tt) = abs(hilbert(filtfilt(b,a,double(LFPmat1(ch,:,tt)))));
    end

end

% how much smoothing in time? (50 - 100  ms is usually good for BHF)
% Fs = 1000 samples per second
smoothType = 'movmedian';
smoothFactor = Fs./5;

% smoothing high gammma tensors.
HGmat_outcome = smoothdata(HGmat_outcome,2,smoothType,smoothFactor);
HGmat_cue = smoothdata(HGmat_cue,2,smoothType,smoothFactor);

% saving vars
outcomeAndCueHGs.outcomeHG = HGmat_outcome;
outcomeAndCueHGs.cueHG = HGmat_cue;
outcomeAndCueHGs.tSec = tSec;
outcomeAndCueHGs.timeWin = [-pre post];

%save as outcomeAndCueHGs here: (takes about 1.5 mins)
tic
save(fullfile(fileparts(parentDir), [ptID '_outcomeAndCueHGs.mat']), 'outcomeAndCueHGs', '-v7.3');
toc

% defining model data
% ATTENTION; I've changed this to true

% baseline and timing params
% startAnalysis_outcome = 0.1;
% endAnalysis_outcome = 1.1;
% startAnalysis_cue = 0.1;
% endAnalysis_cue = 1.1;

startAnalysis_outcome = 0.1;
endAnalysis_outcome = 0.4;

startAnalysis_cue = 0.1;
endAnalysis_cue = 0.4;

bP_outcome = [-0.7 -0.2]; % one second, starting a second and a half before the outcome
bP_cue = [-0.11 -0.01]; % one second, starting a second and a half before the cue

baselineNorm = true;
if baselineNorm
    outcomeData = squeeze(mean(HGmat_outcome(:,tSec > startAnalysis_outcome & tSec < endAnalysis_outcome,:),2))./squeeze(mean(HGmat_outcome(:,tSec > bP_outcome(1) & tSec < bP_outcome(2),:),2));
    cueData = squeeze(mean(HGmat_cue(:,tSec > startAnalysis_cue & tSec < endAnalysis_cue,:),2))./squeeze(mean(HGmat_cue(:,tSec > bP_cue(1) & tSec < bP_cue(2),:),2));
else
    outcomeData = squeeze(mean(HGmat_outcome(:,tSec > startAnalysis_outcome & tSec < endAnalysis_outcome,:),2));
    cueData = squeeze(mean(HGmat_cue(:,tSec > startAnalysis_cue & tSec < endAnalysis_cue,:),2));
end

%% ~~~~~~~~ NILL!!! This is what we need to replace with your parameters! :)
tic
% fitting the neural data to models (need rstd TD model)



TDdataGradients = TDlearn_rstd_Gradients(ptID,outcomeData,cueData,whichTrialsLM); % ,fileparts(parentDir) % rstd2
elapsedTDlearn = toc;
fprintf('TDlearn_rstd_Gradients.m took %.2f seconds to run...\n', elapsedTDlearn)

% for debugging load: maybe just load here instead of running
% TDlearn_rstd...
%load(sprintf('//155.100.91.44/d/Data/preProcessed/BART_preprocessed/%s/Data/%s_TDdataGradients.mat', ptID, ptID));

%% ~~~~~~~~~~~~~~~~~~~~~~~~

% [EHS::20250729] Adding RSTD visualizations.
% loopng over channels to get indices for significant contacts from ANOVA.
for cc = length(TDdataGradients.neuralFit):-1:1
    % (1) All Trials Cue
    pRSTD_VE_cue(cc) = TDdataGradients.neuralFit(cc).ANOVA_rstd_Cue{2,5}; % all trials VE
    % pRSTD_PE_cue(cc) = TDdata.neuralFit(cc).ANOVA_rstd_Cue{3,5}; % all trials PE (dont need because CUE)
    pRSTD_Color_cue(cc) = TDdataGradients.neuralFit(cc).ANOVA_rstd_Cue{3,5}; % all trials Color
    pRSTD_Type_cue(cc) = TDdataGradients.neuralFit(cc).ANOVA_rstd_Cue{4,5}; % all trials Type

    % (2) Successful Trials Cue
    pRSTD_VE_cueSuccess(cc) = TDdataGradients.neuralFit(cc).ANOVA_rstd_CueSuccess{2,5}; % successful trials VE
    % pRSTD_PE_cueSuccess(cc) = TDdata.neuralFit(cc).ANOVA_rstd_CueSuccess{3,5}; % successful trials PE
    pRSTD_Color_cueSuccess(cc) = TDdataGradients.neuralFit(cc).ANOVA_rstd_CueSuccess{3,5}; % successful trials Color
    pRSTD_Type_cueSuccess(cc) = TDdataGradients.neuralFit(cc).ANOVA_rstd_CueSuccess{4,5}; % successful trials Type

    % (3) All Trials Outcome
    % pRSTD_VE_outcome(cc) = TDdata.neuralFit(cc).ANOVA_rstd_Outcome{2,5}; %  trials VE (dont need VE because outcome)
    pRSTD_PE_outcome(cc) = TDdataGradients.neuralFit(cc).ANOVA_rstd_Outcome{2,5}; %  trials PE
    pRSTD_Color_outcome(cc) = TDdataGradients.neuralFit(cc).ANOVA_rstd_Outcome{3,5}; %  trials Color
    pRSTD_Type_outcome(cc) = TDdataGradients.neuralFit(cc).ANOVA_rstd_Outcome{4,5}; %  trials Type
    pRSTD_Outcome_outcome(cc) = TDdataGradients.neuralFit(cc).ANOVA_rstd_Outcome{5,5}; %  trials Outcome

    % (4) Success Trials Outcome
    % pRSTD_VE_outcomeSuccess(cc) = TDdata.neuralFit(cc).ANOVA_rstd_OutcomeSuccess{2,5}; % successful trials VE
    pRSTD_PE_outcomeSuccess(cc) = TDdataGradients.neuralFit(cc).ANOVA_rstd_OutcomeSuccess{2,5}; % successful trials PE
    pRSTD_Color_outcomeSuccess(cc) = TDdataGradients.neuralFit(cc).ANOVA_rstd_OutcomeSuccess{3,5}; % successful trials Color
    pRSTD_Type_outcomeSuccess(cc) = TDdataGradients.neuralFit(cc).ANOVA_rstd_OutcomeSuccess{4,5}; % successful trials Type

    % just using this loop to put the anatomical labels into TDdata
    try
        TDdataGradients.neuralFit(cc).NMManatomicalLabel = deblank(anatomicalLocs{selectedChans(cc)});
    catch
        TDdataGradients.neuralFit(cc).NMManatomicalLabel = 'could not get labels for this electrode';
    end

end

% significance.
crit = 0.05;

% CUE-ALIGNED
sigRSTD_VE_cue = (pRSTD_VE_cue)<crit;
sigRSTD_Color_cue = (pRSTD_Color_cue)<crit;
sigRSTD_Type_cue = (pRSTD_Type_cue)<crit;
sigRSTD_VE_cueSuccess = (pRSTD_VE_cueSuccess)<crit;
sigRSTD_Color_cueSuccess = (pRSTD_Color_cueSuccess)<crit;
sigRSTD_Type_cueSuccess = (pRSTD_Type_cueSuccess)<crit;

% OUTCOME-ALIGNED
sigRSTD_PE_outcome = (pRSTD_PE_outcome)<crit;
sigRSTD_Color_outcome = (pRSTD_Color_outcome)<crit;
sigRSTD_Type_outcome = (pRSTD_Type_outcome)<crit;
sigRSTD_PE_outcomeSuccess = (pRSTD_PE_outcomeSuccess)<crit;
sigRSTD_Color_outcomeSuccess = (pRSTD_Color_outcomeSuccess)<crit;
sigRSTD_Type_outcomeSuccess = (pRSTD_Type_outcomeSuccess)<crit;

% Create significant vars summary
varNames = {
    'sigRSTD_VE_cue', 'sigRSTD_Color_cue', 'sigRSTD_Type_cue', ...
    'sigRSTD_VE_cueSuccess', 'sigRSTD_Color_cueSuccess', 'sigRSTD_Type_cueSuccess', ...
    'sigRSTD_PE_outcome', 'sigRSTD_Color_outcome', 'sigRSTD_Type_outcome', ...
    'sigRSTD_PE_outcomeSuccess', 'sigRSTD_Color_outcomeSuccess', 'sigRSTD_Type_outcomeSuccess'};
sigVars = {
    sigRSTD_VE_cue, sigRSTD_Color_cue, sigRSTD_Type_cue, ...
    sigRSTD_VE_cueSuccess, sigRSTD_Color_cueSuccess, sigRSTD_Type_cueSuccess, ...
    sigRSTD_PE_outcome, sigRSTD_Color_outcome, sigRSTD_Type_outcome, ...
    sigRSTD_PE_outcomeSuccess, sigRSTD_Color_outcomeSuccess, sigRSTD_Type_outcomeSuccess};
sigCounts = cellfun(@(x) sum(x(:)), sigVars);
summaryTable = table(varNames', sigCounts', ...
    'VariableNames', {'Variable', 'SignificantContactCount'});
disp(summaryTable);

% optimal alphas.
alphaPos = TDdataGradients.positiveBetaMaxIdx;
alphaNeg = TDdataGradients.negativeBetaMaxIdx;

%  decide how many quantiles to look at
% [20220607EHS] (just starting with quartiles)
%ps = [0.25 0.5 0.75];
ps = [0.2 0.4 0.6 0.8];

%% RSTD data
% Reward Quantiles
rstdVQs = quantile(squeeze(TDdataGradients.rstdV(alphaPos,alphaNeg,:)),ps);
rstdPEQs = quantile(squeeze(TDdataGradients.rstdRPE(alphaPos,alphaNeg,:)),ps);

% Active vs Passive Trials
% Keep this aligned with the already-trimmed nTrials. Do NOT drop the last
% element again here; doing so mismatches isActive/color/outcome indices.
balloonIDs = balloonIDs(1:nTrials);
isActive = balloonIDs < 10; % trials 11,12,13,14 are all controls

% Initialize balloonColors as a categorical array
balloonColors = strings(size(balloonIDs));

% Assign colors based on the conditions
balloonColors(balloonIDs == 1 | balloonIDs == 11) = "Y"; % Yellow
balloonColors(balloonIDs == 2 | balloonIDs == 12) = "O"; % Orange
balloonColors(balloonIDs == 3 | balloonIDs == 13) = "R"; % Red
balloonColors(balloonIDs == 4 | balloonIDs == 14) = "G"; % Green
balloonColors = categorical(balloonColors);

% logical index for colors:
redTrials = balloonColors == 'R';
oraTrials = balloonColors == 'O';
yelTrials = balloonColors == 'Y';

% visual params
qCols = viridis(length(ps)+1)./max(max(viridis(length(ps)+1)));
plotALPHA = 0.3;
dotSize = 10;
% colors for regression
% Define color map
colorMap = containers.Map(...
    [1, 2, 3, 4, 11, 12, 13, 14], ...
    {[1 1 0], [1 0.5 0], [1 0 0], [0.5 0.5 0.5], ...        % Normal colors
    [1 1 0], [1 0.5 0], [1 0 0], [0.5 0.5 0.5]});          % Control colors (same fill, different edge)

% figure out which trials are pops
resultsCell = {BHV.data(1:nTrials).result};
isPopped = cellfun(@(x) isequal(x, 'popped'), resultsCell);

% RSTD-specific directory.
saveDir  = '\\155.100.91.44\D\Data\Rhiannon\BART_RLDM_outputs\TDlearn\significantContacts_RSTD_color';

if plotFlag
    % (9) Plotting and looping through ALL channels
    for ch2 = nChans:-1:1
        % % plotting responses for all variables, iff theres significant
        % % encoding of any latenet variable.
        % This code is a little opaque, so here is a layout.
        % rows: 1) rstd expectation, 2) rstd surprise
        % columns: 1) TD trajectory, 2) regression, 3) high gamma quantiles
        % 4) high gamma by color (5) high gamma by trial type

        % setting up figure
        if ishandle(ch2); close(ch2); end
        figure(ch2)

        %% RSTD::
        % plot the RSTD PE quantiles on the behavioral data to illustrate which trials are averaged over.
        trials = 1:length(balloonIDs); % trials needs to be the same length as TDdata
        % ROW 1: Cue-aligned RSTD Value Expectation
        % Column 1: TD trajectory
        subplot(2,5,1)
        hold on
        plot(squeeze(TDdataGradients.rstdV(alphaPos,alphaNeg,:)),'k')
        for q1 = 1:length(ps)+1
            if q1==1
                scatter(trials(squeeze(TDdataGradients.rstdV(alphaPos,alphaNeg,:)<rstdVQs(q1))),squeeze(TDdataGradients.rstdV(alphaPos,alphaNeg,squeeze(TDdataGradients.rstdV(alphaPos,alphaNeg,:)<rstdVQs(q1)))),dotSize,qCols(q1,:),'filled')
            elseif q1==(length(ps)+1)
                scatter(trials(squeeze(TDdataGradients.rstdV(alphaPos,alphaNeg,:)>rstdVQs(q1-1))),squeeze(TDdataGradients.rstdV(alphaPos,alphaNeg,squeeze(TDdataGradients.rstdV(alphaPos,alphaNeg,:)>rstdVQs(q1-1)))),dotSize,qCols(q1,:),'filled')
            else
                scatter(trials(squeeze(TDdataGradients.rstdV(alphaPos,alphaNeg,:)<rstdVQs(q1)) & squeeze(TDdataGradients.rstdV(alphaPos,alphaNeg,:)>rstdVQs(q1-1))),...
                    squeeze(TDdataGradients.rstdV(alphaPos,alphaNeg,squeeze(TDdataGradients.rstdV(alphaPos,alphaNeg,:)<rstdVQs(q1)) & squeeze(TDdataGradients.rstdV(alphaPos,alphaNeg,:)>rstdVQs(q1-1)))),dotSize,qCols(q1,:),'filled')
            end
        end
        hold off
        % axis details.
        axis square
        xlabel('trials')
        ylabel('RSTD VE')

        % Column 2: regression plot for RSTD Value Expectation
        subplot(2,5,2)
        rF = fitted(TDdataGradients.neuralFit(ch2).rstd_CueModel);
        rR = response(TDdataGradients.neuralFit(ch2).rstd_CueModel);
        rLMve = fitlm(rR, rF);
        nTrialsPlot = length(resultsCell); % trial count for plotting already aligned above.
        % plotting
        plt1 = rLMve.plotAdded;
        plt1(1).Visible = 'off';
        plt1(2).Color = 'k';
        plt1(3).Color = 'k';
        hold on;
        x = rLMve.Variables.(rLMve.PredictorNames{1});
        y = rLMve.Variables.(rLMve.ResponseName);
        % Plot each point with trial color coding
        for i = 1:nTrialsPlot
            id = balloonIDs(i);
            c = colorMap(id);
            if id > 10  % Control trial
                scatter(x(i), y(i), 40, ...
                    'MarkerFaceColor', c, ...
                    'MarkerEdgeColor', [0.4 0.4 0.4], ...
                    'LineWidth', 1.0);
            else        % Regular trial
                scatter(x(i), y(i), 40, ...
                    'MarkerFaceColor', c, ...
                    'MarkerEdgeColor', 'none');
            end
        end
        lgd = findobj('type', 'legend'); delete(lgd); % remove legend
        axis square;
        xlabel('RSTD VE');
        ylabel('cue-aligned BHF power (uV)');
        % Get R-squared and p-value
        r2_ve = rLMve.Rsquared.Ordinary;
        pval_ve = rLMve.Coefficients.pValue(2);
        title(sprintf('R^2 = %.3f, p = %.3g', r2_ve, pval_ve));

        % Column 3: plot Quantiles RSTD Value Expectation
        subplot(2,5,3)
        hold on
        for q1 = 1:length(ps)+1
            if q1==1
                nT = sum(squeeze(TDdataGradients.rstdV(alphaPos,alphaNeg,:)<rstdVQs(q1)));
                qDataBar = squeeze(mean(HGmat_cue(ch2,:,squeeze(TDdataGradients.rstdV(alphaPos,alphaNeg,:)<rstdVQs(q1))),3));
                qDataErr = squeeze(std(HGmat_cue(ch2,:,squeeze(TDdataGradients.rstdV(alphaPos,alphaNeg,:)<rstdVQs(q1))),[],3));
            elseif q1==(length(ps)+1)
                nT = sum(squeeze(TDdataGradients.rstdV(alphaPos,alphaNeg,:)>rstdVQs(q1-1)));
                qDataBar = squeeze(mean(HGmat_cue(ch2,:,squeeze(TDdataGradients.rstdV(alphaPos,alphaNeg,:)>rstdVQs(q1-1))),3));
                qDataErr = squeeze(std(HGmat_cue(ch2,:,squeeze(TDdataGradients.rstdV(alphaPos,alphaNeg,:)>rstdVQs(q1-1))),[],3));
            else
                nT = sum(squeeze(TDdataGradients.rstdV(alphaPos,alphaNeg,:)<rstdVQs(q1)) & squeeze(TDdataGradients.rstdV(alphaPos,alphaNeg,:)>rstdVQs(q1-1)));
                qDataBar = squeeze(mean(HGmat_cue(ch2,:,squeeze(TDdataGradients.rstdV(alphaPos,alphaNeg,:)<rstdVQs(q1)) & squeeze(TDdataGradients.rstdV(alphaPos,alphaNeg,:)>rstdVQs(q1-1))),3));
                qDataErr = squeeze(std(HGmat_cue(ch2,:,squeeze(TDdataGradients.rstdV(alphaPos,alphaNeg,:)<rstdVQs(q1)) & squeeze(TDdataGradients.rstdV(alphaPos,alphaNeg,:)>rstdVQs(q1-1))),[],3));
            end
            patch([tSec fliplr(tSec)],[qDataBar+(qDataErr./sqrt(nT)) fliplr(qDataBar-(qDataErr./sqrt(nT)))],qCols(q1,:),'facealpha',plotALPHA,'edgecolor','none')
            plot(tSec,qDataBar,'color',qCols(q1,:))
        end
        hold off

        % plot one deets
        axis tight square
        xlim([-pre+1 post-1])
        xline(0, '--k', 'LineWidth', 1.5);
        xlabel('time relative to cue (s)')
        ylabel('BHF power quantiles')
        title(sprintf('VE: F(%d,%d) = %.2f, p = %.2f',TDdataGradients.neuralFit(ch2).ANOVA_rstd_Cue{2,3},TDdataGradients.neuralFit(ch2).ANOVA_rstd_Cue{2,4},TDdataGradients.neuralFit(ch2).ANOVA_rstd_Cue{2,2},TDdataGradients.neuralFit(ch2).ANOVA_rstd_Cue{2,5}));

        % Column 4: plot colors High Gamma RSTD Cue
        subplot(2,5,4)
        hold on
        % red trials
        nT_red = sum(redTrials);
        cDataBar_red = squeeze(mean(HGmat_cue(ch2,:,redTrials),3)); % squeeze(TDdata.rstdV(alphaPos,alphaNeg,redTrials)
        cDataErr_red = squeeze(std(HGmat_cue(ch2,:,redTrials),[],3));
        patch([tSec fliplr(tSec)],[cDataBar_red+(cDataErr_red./sqrt(nT_red)) fliplr(cDataBar_red-(cDataErr_red./sqrt(nT_red)))],'red','facealpha',plotALPHA,'edgecolor','none')
        plot(tSec,cDataBar_red,'color','red')
        % orange trials
        nT_ora = sum(oraTrials);
        cDataBar_ora = squeeze(mean(HGmat_cue(ch2,:,oraTrials),3));
        cDataErr_ora = squeeze(std(HGmat_cue(ch2,:,oraTrials),[],3));
        patch([tSec fliplr(tSec)],[cDataBar_ora+(cDataErr_ora./sqrt(nT_ora)) fliplr(cDataBar_ora-(cDataErr_ora./sqrt(nT_ora)))],[1, 0.5, 0],'facealpha',plotALPHA,'edgecolor','none')
        plot(tSec,cDataBar_ora,'color',[1, 0.5, 0])
        % yellow trials
        nT_yel = sum(yelTrials);
        cDataBar_yel = squeeze(mean(HGmat_cue(ch2,:,yelTrials),3));
        cDataErr_yel = squeeze(std(HGmat_cue(ch2,:,yelTrials),[],3));
        patch([tSec fliplr(tSec)],[cDataBar_yel+(cDataErr_yel./sqrt(nT_yel)) fliplr(cDataBar_yel-(cDataErr_yel./sqrt(nT_yel)))],'yellow','facealpha',plotALPHA,'edgecolor','none')
        plot(tSec,cDataBar_yel,'color','yellow')
        hold off
        % plot one deets
        axis tight square
        xlim([-pre+1 post-1])
        xline(0, '--k', 'LineWidth', 1.5);
        xlabel('time relative to cue (s)')
        ylabel('BHF power (by trial color)')
        title(sprintf('Color: F(%d,%d) = %.2f, p = %.2f',TDdataGradients.neuralFit(ch2).ANOVA_rstd_Cue{4,3},TDdataGradients.neuralFit(ch2).ANOVA_rstd_Cue{4,4},TDdataGradients.neuralFit(ch2).ANOVA_rstd_Cue{3,2},TDdataGradients.neuralFit(ch2).ANOVA_rstd_Cue{3,5}));

        subplot(2,5,5) % Active/Passive VE BHF
        hold on
        % active trials
        nT_active = sum(isActive);
        cDataBar_active = squeeze(mean(HGmat_cue(ch2,:,isActive),3));
        cDataErr_active = squeeze(std(HGmat_cue(ch2,:,isActive),[],3));
        patch([tSec fliplr(tSec)],[cDataBar_active+(cDataErr_active./sqrt(nT_active)) fliplr(cDataBar_active-(cDataErr_active./sqrt(nT_active)))],[0, 1, 0],'facealpha',plotALPHA,'edgecolor','none')
        plot(tSec,cDataBar_active,'color',[0, 1, 0])
        % passive trials
        nT_pas = sum(~isActive);
        cDataBar_pas = squeeze(mean(HGmat_cue(ch2,:,~isActive),3));
        cDataErr_pas = squeeze(std(HGmat_cue(ch2,:,~isActive),[],3));
        patch([tSec fliplr(tSec)],[cDataBar_pas+(cDataErr_pas./sqrt(nT_pas)) fliplr(cDataBar_pas-(cDataErr_pas./sqrt(nT_pas)))],[0.7059, 0.5490, 0.7843],'facealpha',plotALPHA,'edgecolor','none')
        plot(tSec,cDataBar_pas,'color',[0.7059, 0.5490, 0.7843])
        hold off
        % plot one deets
        axis tight square
        xlim([-pre+1 post-1])
        xline(0, '--k', 'LineWidth', 1.5);
        xlabel('time relative to cue (s)')
        ylabel('BHF power (by trial type)')
        title(sprintf('Type: F(%d,%d) = %.2f, p = %.2f',TDdataGradients.neuralFit(ch2).ANOVA_rstd_Cue{4,3},TDdataGradients.neuralFit(ch2).ANOVA_rstd_Cue{4,4},TDdataGradients.neuralFit(ch2).ANOVA_rstd_Cue{4,2},TDdataGradients.neuralFit(ch2).ANOVA_rstd_Cue{4,5}));

        % ROW 2: RSTD Prediction Error
        subplot(2,5,6)
        hold on
        plot(squeeze(TDdataGradients.rstdRPE(alphaPos,alphaNeg,:)),'k')
        for q1 = 1:length(ps)+1
            if q1==1
                scatter(trials(squeeze(TDdataGradients.rstdRPE(alphaPos,alphaNeg,:)<rstdPEQs(q1))),squeeze(TDdataGradients.rstdRPE(alphaPos,alphaNeg,squeeze(TDdataGradients.rstdRPE(alphaPos,alphaNeg,:)<rstdPEQs(q1)))),dotSize,qCols(q1,:),'filled')
            elseif q1==(length(ps)+1)
                scatter(trials(squeeze(TDdataGradients.rstdRPE(alphaPos,alphaNeg,:)>rstdPEQs(q1-1))),squeeze(TDdataGradients.rstdRPE(alphaPos,alphaNeg,squeeze(TDdataGradients.rstdRPE(alphaPos,alphaNeg,:)>rstdPEQs(q1-1)))),dotSize,qCols(q1,:),'filled')
            else
                scatter(trials(squeeze(TDdataGradients.rstdRPE(alphaPos,alphaNeg,:)<rstdPEQs(q1)) & squeeze(TDdataGradients.rstdRPE(alphaPos,alphaNeg,:)>rstdPEQs(q1-1))),...
                    squeeze(TDdataGradients.rstdRPE(alphaPos,alphaNeg,squeeze(TDdataGradients.rstdRPE(alphaPos,alphaNeg,:)<rstdPEQs(q1)) & squeeze(TDdataGradients.rstdRPE(alphaPos,alphaNeg,:)>rstdPEQs(q1-1)))),dotSize,qCols(q1,:),'filled')
            end
        end
        hold off
        % axis details.
        axis square
        xlabel('trials')
        ylabel('RSTD PE')

        % regression plot for RSTD Prediction Error.
        subplot(2,5,7)
        rF = fitted(TDdataGradients.neuralFit(ch2).rstd_OutcomeModel);
        rR = response(TDdataGradients.neuralFit(ch2).rstd_OutcomeModel);
        rLMpe = fitlm(rR, rF);
        % plotting
        plt1 = rLMpe.plotAdded;
        plt1(1).Visible = 'off';
        plt1(2).Color = 'k';
        plt1(3).Color = 'k';
        hold on;
        x = rLMpe.Variables.(rLMpe.PredictorNames{1});
        y = rLMpe.Variables.(rLMpe.ResponseName);
        % Plot each point with trial color coding
        for i = 1:nTrialsPlot
            id = balloonIDs(i);
            c = colorMap(id);
            if isPopped(i)
                markerShape = '^';  % triangle if popped
            elseif ~isPopped(i)
                markerShape = 'o';  % circle if banked
            end
            if id > 10  % Control trial
                scatter(x(i), y(i), 40, ...
                    'Marker', markerShape, ...
                    'MarkerFaceColor', c, ...
                    'MarkerEdgeColor', [0.4 0.4 0.4], ...
                    'LineWidth', 1.0);
            else        % Regular trial
                scatter(x(i), y(i), 40, ...
                    'Marker', markerShape, ...
                    'MarkerFaceColor', c, ...
                    'MarkerEdgeColor', 'none');
            end
        end
        lgd = findobj('type', 'legend'); delete(lgd); % remove legend
        axis square;
        xlabel('RSTD PE');
        ylabel('outcome-aligned BHF power (uV)');
        % Get R-squared and p-value
        r2_pe = rLMpe.Rsquared.Ordinary;
        pval_pe = rLMpe.Coefficients.pValue(2);
        title(sprintf('R^2 = %.3f, p = %.3g', r2_pe, pval_pe));

        % plot significant RSTD PE example
        subplot(2,5,8)
        hold on
        for q1 = 1:length(ps)+1
            if q1==1
                nT = sum(squeeze(TDdataGradients.rstdRPE(alphaPos,alphaNeg,:)<rstdPEQs(q1)));
                qDataBar = squeeze(mean(HGmat_outcome(ch2,:,squeeze(TDdataGradients.rstdRPE(alphaPos,alphaNeg,:)<rstdPEQs(q1))),3));
                qDataErr = squeeze(std(HGmat_outcome(ch2,:,squeeze(TDdataGradients.rstdRPE(alphaPos,alphaNeg,:)<rstdPEQs(q1))),[],3));
            elseif q1==(length(ps)+1)
                qDataBar = squeeze(mean(HGmat_outcome(ch2,:,squeeze(TDdataGradients.rstdRPE(alphaPos,alphaNeg,:)>rstdPEQs(q1-1))),3));
                qDataErr = squeeze(std(HGmat_outcome(ch2,:,squeeze(TDdataGradients.rstdRPE(alphaPos,alphaNeg,:)>rstdPEQs(q1-1))),[],3));
                nT = sum(squeeze(TDdataGradients.rstdRPE(alphaPos,alphaNeg,:)>rstdPEQs(q1-1)));
            else
                nT = sum(squeeze(TDdataGradients.rstdRPE(alphaPos,alphaNeg,:)<rstdPEQs(q1)) & squeeze(TDdataGradients.rstdRPE(alphaPos,alphaNeg,:)>rstdPEQs(q1-1)));
                qDataBar = squeeze(mean(HGmat_outcome(ch2,:,squeeze(TDdataGradients.rstdRPE(alphaPos,alphaNeg,:)<rstdPEQs(q1)) & squeeze(TDdataGradients.rstdRPE(alphaPos,alphaNeg,:)>rstdPEQs(q1-1))),3));
                qDataErr = squeeze(std(HGmat_outcome(ch2,:,squeeze(TDdataGradients.rstdRPE(alphaPos,alphaNeg,:)<rstdPEQs(q1)) & squeeze(TDdataGradients.rstdRPE(alphaPos,alphaNeg,:)>rstdPEQs(q1-1))),[],3));
            end
            patch([tSec fliplr(tSec)],[qDataBar+(qDataErr./sqrt(nT)) fliplr(qDataBar-(qDataErr./sqrt(nT)))],qCols(q1,:),'facealpha',plotALPHA,'edgecolor','none')
            plot(tSec,qDataBar,'color',qCols(q1,:))
        end
        hold off
        % plot one deets
        axis tight square
        xlim([-pre+1 post-1])
        xline(0, '--k', 'LineWidth', 1.5);
        xlabel('time relative to outcome (s)')
        ylabel('BHF power quantiles')
        title(sprintf('PE: F(%d,%d) = %.2f, p = %.2f',TDdataGradients.neuralFit(ch2).ANOVA_rstd_Outcome{2,3},TDdataGradients.neuralFit(ch2).ANOVA_rstd_Outcome{2,4},TDdataGradients.neuralFit(ch2).ANOVA_rstd_Outcome{2,2},TDdataGradients.neuralFit(ch2).ANOVA_rstd_Outcome{2,5}));

        subplot(2,5,9) % outcome PE BHF
        hold on
        % popped trials
        nT_pop = sum(isPopped);
        cDataBar_pop = squeeze(mean(HGmat_outcome(ch2,:,isPopped),3));
        cDataErr_pop = squeeze(std(HGmat_outcome(ch2,:,isPopped),[],3));
        patch([tSec fliplr(tSec)],[cDataBar_pop+(cDataErr_pop./sqrt(nT_pop)) fliplr(cDataBar_pop-(cDataErr_pop./sqrt(nT_pop)))],'black','facealpha',plotALPHA,'edgecolor','none')
        plot(tSec,cDataBar_pop,'color','black')
        % banked trials
        nT_bank = sum(~isPopped);
        cDataBar_bank = squeeze(mean(HGmat_outcome(ch2,:,~isPopped),3));
        cDataErr_bank = squeeze(std(HGmat_outcome(ch2,:,~isPopped),[],3));
        patch([tSec fliplr(tSec)],[cDataBar_bank+(cDataErr_bank./sqrt(nT_bank)) fliplr(cDataBar_bank-(cDataErr_bank./sqrt(nT_bank)))],'blue','facealpha',plotALPHA,'edgecolor','none')
        plot(tSec,cDataBar_bank,'color','blue')
        hold off
        % plot one deets
        axis tight square
        xlim([-pre+1 post-1])
        xline(0, '--k', 'LineWidth', 1.5);
        xlabel('time relative to outcome (s)')
        ylabel('BHF power (by outcome)')
        title(sprintf('Outcome: F(%d,%d) = %.2f, p = %.2f',TDdataGradients.neuralFit(ch2).ANOVA_rstd_Outcome{5,3},TDdataGradients.neuralFit(ch2).ANOVA_rstd_Outcome{5,4},TDdataGradients.neuralFit(ch2).ANOVA_rstd_Outcome{5,2},TDdataGradients.neuralFit(ch2).ANOVA_rstd_Outcome{5,5}));

        subplot(2,5,10) % Active/Passive PE BHF
        hold on
        % active trials
        nT_active = sum(isActive);
        cDataBar_active = squeeze(mean(HGmat_outcome(ch2,:,isActive),3));
        cDataErr_active = squeeze(std(HGmat_outcome(ch2,:,isActive),[],3));
        patch([tSec fliplr(tSec)],[cDataBar_active+(cDataErr_active./sqrt(nT_active)) fliplr(cDataBar_active-(cDataErr_active./sqrt(nT_active)))],[0, 1, 0],'facealpha',plotALPHA,'edgecolor','none')
        plot(tSec,cDataBar_active,'color',[0, 1, 0])
        % passive trials
        nT_pas = sum(~isActive);
        cDataBar_pas = squeeze(mean(HGmat_outcome(ch2,:,~isActive),3));
        cDataErr_pas = squeeze(std(HGmat_outcome(ch2,:,~isActive),[],3));
        patch([tSec fliplr(tSec)],[cDataBar_pas+(cDataErr_pas./sqrt(nT_pas)) fliplr(cDataBar_pas-(cDataErr_pas./sqrt(nT_pas)))],[0.7059, 0.5490, 0.7843],'facealpha',plotALPHA,'edgecolor','none')
        plot(tSec,cDataBar_pas,'color',[0.7059, 0.5490, 0.7843])
        hold off
        % plot one deets
        axis tight square
        xlim([-pre+1 post-1])
        xline(0, '--k', 'LineWidth', 1.5);
        xlabel('time relative to outcome (s)')
        ylabel('BHF power (by trial type)')
        title(sprintf('Type: F(%d,%d) = %.2f, p = %.2f',TDdataGradients.neuralFit(ch2).ANOVA_rstd_Outcome{4,3},TDdataGradients.neuralFit(ch2).ANOVA_rstd_Outcome{4,4},TDdataGradients.neuralFit(ch2).ANOVA_rstd_Outcome{4,2},TDdataGradients.neuralFit(ch2).ANOVA_rstd_Outcome{4,5}));

        % Getting figure sizing
        set(ch2, 'Units', 'inches');
        set(ch2, 'Position', [1, 1, 11, 8]);  % [x y width height]

        % Set paper size for saving
        set(ch2, 'PaperUnits', 'inches');
        figPos = get(ch2, 'Position');
        set(ch2, 'PaperSize', figPos(3:4));
        set(ch2, 'PaperPosition', [0 0 figPos(3) figPos(4)]);

        % saving
        saveas(ch2,fullfile(saveDir,[ptID '_' deblank(trodeLabels_selectedChans{ch2}) '_' TDdataGradients.neuralFit(ch2).new_trodeLabel '_RSTD_exampleResponses_' smoothType '.pdf']))
        close(ch2)

    end % looping over channels
end % if plotting
end % eof





%% debug


