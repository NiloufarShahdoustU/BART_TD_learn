function [TDdataGradients] = TDlearn_rstd_Gradients(ptID,rewardData,riskData,LMtrials,varargin)
% TDLEARN fits a temporal difference learning RL algorithm to BART choice data.
%
%	Fits neural data variables from both standard and risk-sensistive
%	(asymmetric scaling) temporal difference learning models.
%
%	ptID is a patient identifier that will get BART behavior
%	rewardData is a matrix of neural data or responses for each trial and channel to fit to reward model.
%	riskData is a matrix of neural data or responses for each trial and channel to fit to risk model.
%	startTrial specifies the first ttrial from which iteration begins
%   LMtrials is a matlab formatted string indicating which trials to
%       evaluate a linear model across (e.g. '40:80' or 'end-40:end')
%   TDtrials specifies which trials to evaluate the TD model
%
%   A sixth optional input argument specifies a folder to save temporal
%   difference learning model fit structures.

% author: EHS20201102

if nargin<3
    error('function requires data')
elseif nargin<5
    LMtrials = '40:end';
elseif nargin==6
    saveDir = varargin{1};
end

parentDir = ['\\155.100.91.44\D\Data\preProcessed\BART_preprocessed\' ptID '\Data\*.nev'];
nevList = dir(parentDir);
nevFile = fullfile(nevList.folder,nevList.name);
nevFile(strfind(nevFile,'\'))='/';
[trodeLabels,isECoG,isEEG,isECG,anatomicalLocs,adjacentChanMat] = ptTrodesBART_2(ptID);

% load electrode labels created from BART_BrainRegions:
% (a) with hemipsheres (channel x patient)
regions_wHemi_CoarseStruct = load('\\155.100.91.44\d\Data\Rhiannon\BART_RLDM_outputs\BrainRegions\Brainnetome_Atlas_All_woHemi_Coarse');
% (b) without hemipsheres (channel x patient)
regions_woHemi_CoarseStruct = load('\\155.100.91.44\d\Data\Rhiannon\BART_RLDM_outputs\BrainRegions\Brainnetome_Atlas_All_woHemi_Coarse');
% (c) isECoG logical matrix (channel x patient)
BART_all_isECoG_logical_matrix =  load('\\155.100.91.44\D\Data\Rhiannon\BART_RLDM_outputs\BrainRegions\BART_all_isECoG_logical_matrix.mat');
new_isECoG = BART_all_isECoG_logical_matrix.BART_all_isECoG_logical_matrix;

% finding position of patient in struct.
[ptArray] = BARTnumbers();
ptID_position = contains(ptArray, ptID);
new_trodeLabels = regions_woHemi_CoarseStruct.Brainnetome_Atlas_All_woHemi_Coarse(:,ptID_position);
new_selectedChans = find(new_isECoG(:,ptID_position));
new_trodeLabels_selectedChans = new_trodeLabels(new_isECoG(:,ptID_position)); % Removing NaCs from trodelabels
% data parameters
new_nChans = length(new_selectedChans);

%% reading paramrecovery data
paramRecoveryFolder = '\\155.100.91.44\d\Data\Nill\BART_param_recovery\context_modeling\param_recovery_1_modeling\';
fileName = [ptID '_TDdataParamRecovery.mat'];
fullFilePath = fullfile(paramRecoveryFolder, fileName);
load(fullFilePath);

%%
% initializing bhv output
TDdataGradients.patientID = ptID;

% loading matFile
matFile = ['\\155.100.91.44\D\Data\preProcessed\BART_preprocessed\' ptID '\Data\' ptID '.bartBHV.mat'];
load(matFile)
dataBHV = data;
clear data

% load and define triggers from nevFle
NEV = openNEV(nevFile,'overwrite');
trigs = NEV.Data.SerialDigitalIO.UnparsedData;
trigTimes = NEV.Data.SerialDigitalIO.TimeStampSec;

% [20170713] I made a small error in the handle_input2.m script such that
% sometimes there is an additional 23 after the initial 23 (signifying the
% start of the inflation). The next two lines remove that second 23.
infIdx = trigs==23;
infIdx([diff(infIdx)==0; false] & infIdx==1) = 0;

% trial IDs
isCTRL = logical([dataBHV.is_control]);
balloonIDs = trigs(trigs==1 | trigs==2 | trigs==3 | trigs==11 | trigs==12 | trigs==13 | trigs==14);
outcomeType = trigs(sort([find(trigs==25); find(trigs==26)]))-24;
outcomeTypeCat(outcomeType==1) = {'banked'};
outcomeTypeCat(outcomeType==2) = {'popped'};
categorical(outcomeTypeCat);

% response times
respTimes = trigTimes(trigs==25 | trigs==26);
% trialStarts = trigTimes(trigs<15);
% trialStarts = trialStarts(2:length(respTimes)+1);
% inflateStarts = trigTimes(infIdx);
if ~exist('TDtrials','var')
    nTrials = length(respTimes);
    % adjusting for numbers of trials.
    balloonIDs = balloonIDs(1:nTrials);
    isCTRL = isCTRL(1:nTrials);
else
    nTrials = length(TDtrials);
    % adjusting for numbers of trials.
    balloonIDs = balloonIDs(TDtrials);
    isCTRL = isCTRL(TDtrials);
    outcomeType = outcomeType(TDtrials);
    outcomeTypeCat = outcomeTypeCat(TDtrials);
end

TDdataGradients.nTrials = nTrials;

% risk colormap
cMap(2,:) = [1 0.9 0];
cMap(3,:) = [1 0.5 0];
cMap(4,:) = [1 0 0];
cMap(1,:) = [0.5 0.5 0.5];

% colormap per trial.
balloonColorMap = ones(length(balloonIDs),3)*.5;

% populating balloon color map
for x = 1:3
    balloonColorMap(balloonIDs==x,:) = repmat(cMap(x+1,:),sum(balloonIDs==x),1);
end

% making sure the matrix is oriented correctly:: [channels X trials]
[a,b] = size(riskData);
if isequal(b,nTrials)
    nChannels = a;
    tmpTrials = b;
elseif isequal(a,nTrials)
    riskData = riskData'; % cue-aligned
    rewardData = rewardData'; % outcome-aligned
    nChannels = b;
    tmpTrials = a;
else
    error(['Are you sure the neural data variable is the right size? The script is detecting ' nTrials ' trials, but the data appears to have ' tmpTrials ' trials and ' nChannels ' channels'])
end

% points vector
pointsPerTrial = diff([0 [dataBHV.score]]); %added BHV

%% now fitting risk-sensitive (asymmetric) models.
% 

% Initalize variables.
% Reward = zeros(1,nTrials);
% RewardPE = zeros(1,nTrials);
% Reward(1) = dataBHV(1).points;
% expectedReward = zeros(1,nTrials);
% a = 0.01:0.01:1;		% alphas;
TDdataGradients.a = TDdataParamRecovery.a;
a = TDdataParamRecovery.a;
% 
% for ap = length(a):-1:1
%     for an = length(a):-1:1
%         for t = 2:nTrials
%             % updating reward variable in each trial.
%             if strcmp(dataBHV(t).result,'banked')
%                 Reward(t) = pointsPerTrial(t);			% outcome on current trial.
%             else
%                 Reward(t) = 0;
%             end
% 
%             % updating risk and reward PE
%             RewardPE(t) = Reward(t) - expectedReward(t-1);
% 
%             % updating value in a piece wise fashion with different learning rates for postive and negative RPEs, respectively.
%             if RewardPE(t)>0
%                 expectedReward(t) = expectedReward(t-1) + a(ap)*RewardPE(t);% was t-1, this was wrong.
%             elseif RewardPE(t)<0
%                 expectedReward(t) = expectedReward(t-1) + a(an)*RewardPE(t);
%             else
%                 expectedReward(t) = expectedReward(t-1); % in the case when expectedReward == 0, expectedReward is expectedReward(t-1).
%             end
%         end
% 
%         whichLink = 'logit';
%         % similar but using probit instead of logit. Logit gives the same
%         % answer as above.
%         B = glmfit(expectedReward',categorical(outcomeType),'binomial','link',whichLink);
% 
%         % inverse temperature parameters from glmfit
TDdataGradients.inverseTemperatureRSTD = TDdataParamRecovery.inverseTemperatureRSTD;
% 
%         % TD vars
TDdataGradients.rstdV = TDdataParamRecovery.expectedReward;
TDdataGradients.rstdRPE = TDdataParamRecovery.RPE;
%     end
% end
% 
% % Getting gradient alphas (behavior)
% [~,TDdataGradients.positiveBetaMaxIdx] = max(max(TDdataGradients.inverseTemperatureRSTD,[],1),[],2); % CURRENTLY THIS IS INCORRECT. NEED TO FLIP INDICES
% [~,TDdataGradients.negativeBetaMaxIdx] = max(max(TDdataGradients.inverseTemperatureRSTD,[],2),[],1);
% 
TDdataGradients.positiveBetaMaxIdx = TDdataParamRecovery.bestApIdx;
TDdataGradients.negativeBetaMaxIdx = TDdataParamRecovery.bestAnIdx;

% 
% RSTDplt = false; % plotting learning rates for optimal alphas (behavior)
% if RSTDplt
%     % plotting the two alphas against each other and picking maxes?
%     figure(314)
% 
%     subplot(3,2,1)
%     hold on
%     imagesc(a,a,TDdataGradients.inverseTemperatureRSTD,[0 max(max(TDdataGradients.inverseTemperatureRSTD))]) % updated to inverseTemperatureRSTD from inverseTemperature
%     scatter(a(TDdataGradients.positiveBetaMaxIdx),a(TDdataGradients.negativeBetaMaxIdx),10,'k')
%     hold off
%     axis xy tight square
%     cb = colorbar();
%     ylabel(cb,'inverse temperature','Rotation',270)
%     xlabel('alpha_+')
%     ylabel('alpha_-')
%     title(ptID)
% 
%     subplot(3,2,3)
%     imagesc(1:nTrials,a,squeeze(TDdataGradients.rstdV(TDdataGradients.positiveBetaMaxIdx,:,:)))
%     axis xy square
%     xlabel('trials')
%     ylabel('alpha_+')
%     title('reward expectation for positive PEs')
% 
%     subplot(3,2,4)
%     imagesc(1:nTrials,a,squeeze(TDdataGradients.rstdV(:,TDdataGradients.negativeBetaMaxIdx,:)))
%     axis xy square
%     xlabel('trials')
%     ylabel('alpha_-')
%     title('reward expectation for negative PEs')
% 
%     subplot(3,2,5)
%     plot(1:nTrials,squeeze(TDdataGradients.rstdV(TDdataGradients.positiveBetaMaxIdx,TDdataGradients.negativeBetaMaxIdx,:)));
%     axis xy square
%     xlabel('trials')
%     ylabel('Value')
%     title('Reward expectation incorporating both VEs')
% 
%     subplot(3,2,6)
%     plot(1:nTrials,squeeze(TDdataGradients.rstdRPE(TDdataGradients.positiveBetaMaxIdx,TDdataGradients.negativeBetaMaxIdx,:)));
%     axis xy square
%     xlabel('trials')
%     ylabel('PE')
%     title('Reward surprise incorporating both PEs')
% 
%     halfMaximize(314,'page')
%     saveas(314,fullfile('\\155.100.91.44\d\Data\Rhiannon\BART_RLDM_outputs\TDlearn\riskSensitivity\',[ptID '_positive_negativePEs.pdf'])) % for elliot
%     close(314)
% end

%% NEURAL LOOP OVER CHANNELS FOR GRADIENT DATA

    % TD model variables to add to table
    % Assign colors based on the conditions
    % Trial Color
    balloonColors = categorical(zeros(size(balloonIDs)));
    balloonColors(balloonIDs == 1 | balloonIDs == 11) = 'Y'; % Yellow
    balloonColors(balloonIDs == 2 | balloonIDs == 12) = 'O'; % Orange
    balloonColors(balloonIDs == 3 | balloonIDs == 13) = 'R'; % Red
    balloonColors(balloonIDs == 4 | balloonIDs == 14) = 'G'; % Green
    TrialColor = balloonColors; % Y O R G
    TrialColor = categorical(TrialColor);
    % Trial Type
    isActive = balloonIDs < 10; % trials 11,12,13,14 are all controls
    TrialType = isActive; % 1 = active, 0 = passive
    TrialType = categorical(TrialType);
    % Outcome Type
    OutcomeType = outcomeTypeCat';
    OutcomeType = categorical(OutcomeType);

    %% fitting temporal difference learning models to NEURAL DATA
    % fitting the model for each channel for the alpha.
    for chz = nChannels:-1:1
        updateUser('assessing RSTD model encoding across learning rates on channel ',chz,10,nChannels)
        skipFactor = 2;
        for ap = length(a):-skipFactor:1
            for an = length(a):-skipFactor:1
                % reward ECoG (outcome aligned)
                tmpTbl_outcomeAligned = array2table(rewardData','VariableNames',trodeLabels(isECoG)'); % HG outcome data
                % risk ECOG (cue aligned)
                tmpTbl_cueAligned = array2table(riskData','VariableNames',trodeLabels(isECoG)'); % HG cue data

                % disp(size(squeeze(TDdataGradients.rstdV(ap,an,:))));
                % disp(size(squeeze(TDdataGradients.rstdRPE(ap,an,:))));
                % disp(size(TrialColor));
                % disp(size(TrialType));
                % disp(size(OutcomeType));

                % RSTD model variables (temp tables are all behavior.. best VE and PE for behav)
                tmpTbl_rstd = table(squeeze(TDdataGradients.rstdV(ap,an,:)),...
                                    squeeze(TDdataGradients.rstdRPE(ap,an,:)),...
                                    TrialColor,TrialType,OutcomeType,...
                                   'VariableNames',{'RSTD_VE','RSTD_PE','TrialColor', 'TrialType', 'Outcome'});

                % fixed trial colors
                tmpTbl_rstd.TrialColor = categorical(string(tmpTbl_rstd.TrialColor));
                tmpTbl_rstd.TrialColor = removecats(tmpTbl_rstd.TrialColor);
                categories(tmpTbl_rstd.TrialColor);
                tmpTbl_rstd.TrialType = categorical(isActive, [1 0], {'active', 'passive'});
                tmpTbl_rstd.TrialType = reordercats(tmpTbl_rstd.TrialType, {'passive', 'active'});

                % FINAL TABLES
                rstdTbl_outcome = cat(2,tmpTbl_outcomeAligned,tmpTbl_rstd); % all trials (outcome-aligned)
                rstdTbl_cue = cat(2,tmpTbl_cueAligned,tmpTbl_rstd); % all trials (cue-aligned)
                rstdTbl_outcome_success = rstdTbl_outcome(rstdTbl_outcome.Outcome == "banked", :); % only successful trials
                rstdTbl_cue_success = rstdTbl_cue(rstdTbl_cue.Outcome == "banked", :); % only successful trials

                TDdataGradients.neuralFit(chz).bestAlphaPositive = a(TDdataGradients.positiveBetaMaxIdx);
                TDdataGradients.neuralFit(chz).bestAlphaNegative = a(TDdataGradients.negativeBetaMaxIdx);
                TDdataGradients.neuralFit(chz).trodeLabel = tmpTbl_outcomeAligned.Properties.VariableNames{chz};
                TDdataGradients.neuralFit(chz).new_trodeLabel = new_trodeLabels_selectedChans{chz}; % adding new labels!

                % for RSTD model (using behavioral alphas...for this as
                % well)
                try
                    TDdataGradients.neuralFit(chz).rstd_OutcomeModel = fitglme(rstdTbl_outcome,[TDdataGradients.neuralFit(chz).trodeLabel ' ~ RSTD_PE + TrialColor + TrialType + Outcome']); % ALL TRIALS
                    TDdataGradients.neuralFit(chz).rstd_OutcomeSuccessModel = fitglme(rstdTbl_outcome_success,[TDdataGradients.neuralFit(chz).trodeLabel ' ~ RSTD_PE + TrialColor + TrialType']); % SUCCESSFUL (BANKED) TRIALS
                    TDdataGradients.neuralFit(chz).rstd_CueModel = fitglme(rstdTbl_cue,[TDdataGradients.neuralFit(chz).trodeLabel ' ~ RSTD_VE + TrialColor + TrialType']); % ALL TRIALS
                    TDdataGradients.neuralFit(chz).rstd_CueSuccessModel = fitglme(rstdTbl_cue_success,[TDdataGradients.neuralFit(chz).trodeLabel ' ~ RSTD_VE + TrialColor + TrialType']); % SUCCESSFUL (BANKED) TRIALS

                catch ME
                    TDdataGradients.neuralFit(chz).rstd_OutcomeModel = ME.identifier;
                    TDdataGradients.neuralFit(chz).rstd_OutcomeSuccessModel = ME.identifier;
                    TDdataGradients.neuralFit(chz).rstd_CueModel = ME.identifier;
                    TDdataGradients.neuralFit(chz).rstd_CueSuccessModel = ME.identifier;
                end

                % run anova on each model and save

                % disp(TDdataGradients.neuralFit(chz).rstd_OutcomeModel);
                TDdataGradients.neuralFit(chz).ANOVA_rstd_Outcome = anova(TDdataGradients.neuralFit(chz).rstd_OutcomeModel);
                TDdataGradients.neuralFit(chz).ANOVA_rstd_OutcomeSuccess = anova(TDdataGradients.neuralFit(chz).rstd_OutcomeSuccessModel);
                TDdataGradients.neuralFit(chz).ANOVA_rstd_Cue = anova(TDdataGradients.neuralFit(chz).rstd_CueModel);
                TDdataGradients.neuralFit(chz).ANOVA_rstd_CueSuccess = anova(TDdataGradients.neuralFit(chz).rstd_CueSuccessModel);

                % Saving log likelihood landscapes per electrode.
                TDdataGradients.neuralFit(chz).LLimg_outcome(ap,an) = TDdataGradients.neuralFit(chz).rstd_OutcomeModel.LogLikelihood;
                TDdataGradients.neuralFit(chz).LLimg_outcomeSuccess(ap,an) = TDdataGradients.neuralFit(chz).rstd_OutcomeSuccessModel.LogLikelihood;
                TDdataGradients.neuralFit(chz).LLimg_cue(ap,an) = TDdataGradients.neuralFit(chz).rstd_CueModel.LogLikelihood;
                TDdataGradients.neuralFit(chz).LLimg_cueSuccess(ap,an) = TDdataGradients.neuralFit(chz).rstd_CueSuccessModel.LogLikelihood;

                % saving R-squared landscapes per electrode.
                TDdataGradients.neuralFit(chz).R2img_outcome(ap,an) = TDdataGradients.neuralFit(chz).rstd_OutcomeModel.Rsquared.Adjusted;
                TDdataGradients.neuralFit(chz).R2img_outcomeSuccess(ap,an) = TDdataGradients.neuralFit(chz).rstd_OutcomeSuccessModel.Rsquared.Adjusted;
                TDdataGradients.neuralFit(chz).R2img_cue(ap,an) = TDdataGradients.neuralFit(chz).rstd_CueModel.Rsquared.Adjusted;
                TDdataGradients.neuralFit(chz).R2img_cueSuccess(ap,an) = TDdataGradients.neuralFit(chz).rstd_CueSuccessModel.Rsquared.Adjusted;

            end
        end

        % saving figures for significant models
        % if (TDdata.neuralFit(chz).rstdExpectationModelPOPCTRL.anova{end,end}<0.05 || TDdata.neuralFit(chz).rstdSurpriseModelPOPCTRL.anova{end,end}<0.05)

        plotFlag = true; % do you want to plot the figures

        if plotFlag
            % figure to visualize best alphas & best alpha ratio.
            figure(chz*1000)

            % RPE plots
            subplot(2,2,1)
            imagesc(a(end:-skipFactor:1),a(end:-skipFactor:1),TDdataGradients.neuralFit(chz).LLimg_outcome(end:-skipFactor:1,end:-skipFactor:1));
            axis xy square
            xlabel('positive alpha')
            ylabel('negative alpha')
            title(['value [LL] -- ' TDdataGradients.neuralFit(chz).trodeLabel])
            colorbar

            subplot(2,2,3)
            imagesc(a(end:-skipFactor:1),a(end:-skipFactor:1),TDdataGradients.neuralFit(chz).R2img_outcome(end:-skipFactor:1,end:-skipFactor:1));
            axis xy square
            xlabel('positive alpha')
            ylabel('negative alpha')
            title('value [R^2]')
            colorbar

            % value plots.
            subplot(2,2,2)
            imagesc(a(end:-skipFactor:1),a(end:-skipFactor:1),TDdataGradients.neuralFit(chz).LLimg_cue(end:-skipFactor:1,end:-skipFactor:1));
            axis xy square
            xlabel('positive alpha')
            ylabel('negative alpha')
            title('RPE [LL]')
            colorbar

            subplot(2,2,4)
            imagesc(a(end:-skipFactor:1),a(end:-skipFactor:1),TDdataGradients.neuralFit(chz).R2img_cue(end:-skipFactor:1,end:-skipFactor:1));
            axis xy square
            xlabel('positive alpha')
            ylabel('negative alpha')
            title('RPE [R^2]')
            colorbar

            % saving figures with significant models.
            halfMaximize(chz*1000,'page')
            % saveas(chz*1000,sprintf('\\155.100.91.44\d\Data\Rhiannon\BART_RLDM_outputs\RSTD\RSTD_neuralFits\pt%s_%s_%s_RSTD_modelFitLandscapes.pdf',ptID,TDdata.neuralFit(chz).trodeLabel, TDdata.neuralFit(chz).new_trodeLabel))
             saveas(chz*1000,fullfile('\\155.100.91.44\d\Data\Rhiannon\BART_RLDM_outputs\RSTD\RSTD_neuralFits\',[ptID '_' TDdataGradients.neuralFit(chz).trodeLabel '_' TDdataGradients.neuralFit(chz).new_trodeLabel '_RSTD_modelFitLandscapes.pdf']))
            close(chz*1000)

        end % if plot  

    end % for chans

%end % eof

 % learning rate gradients
    % save TDdata by patient here:
    tic
    [pDir] = fileparts(parentDir);
    save(fullfile(pDir,[ptID '_TDdataGradients.mat']),'-v7.3')
    toc
    fprintf('\n\nSaving TDdata struct took %.2f minutes...', toc / 60);
