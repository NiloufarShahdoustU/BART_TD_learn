function [TDdataGradients] = TDlearn_rstd_Gradients(ptID,outcomeData,cueData,LMtrials,varargin)
% TDLEARN fits a temporal difference learning RL algorithm to BART choice data.
%
%   Fits neural data variables from both standard and risk-sensistive
%   (asymmetric scaling) temporal difference learning models.
%
%   ptID is a patient identifier that will get BART behavior
%   outcomeData is a matrix of neural data or responses for each trial and channel to fit to reward model.
%   cueData is a matrix of neural data or responses for each trial and channel to fit to risk model.
%   startTrial specifies the first ttrial from which iteration begins
%   LMtrials is a matlab formatted string indicating which trials to
%       evaluate a linear model across (e.g. '40:80' or 'end-40:end')
%   TDtrials specifies which trials to evaluate the TD model
%
%   A sixth optional input argument specifies a folder to save temporal
%   difference learning model fit structures.

% author: EHS20201102
% Fixed version: no local/subfunctions, to avoid MATLAB parser issues.
% Visualization section is intentionally kept unchanged.

if nargin < 3
    error('function requires data')
end
if nargin < 4 || isempty(LMtrials)
    LMtrials = '1:end';
end
if nargin >= 5 && ~isempty(varargin{1})
    saveDir = varargin{1}; 
end

parentDir = ['\\155.100.91.44\D\Data\preProcessed\BART_preprocessed\' ptID '\Data\*.nev'];
nevList = dir(parentDir);
if isempty(nevList)
    error('No NEV files found for patient %s.', ptID);
end
[~,nevSortIdx] = sort({nevList.name});
nevList = nevList(nevSortIdx);
if numel(nevList) > 1
    warning('Multiple NEV files found for %s. Using the first one in dir order: %s', ptID, nevList(1).name);
end
nevFile = fullfile(nevList(1).folder,nevList(1).name);
nevFile(strfind(nevFile,'\'))='/';

[trodeLabels,isECoG,isEEG,isECG,anatomicalLocs,adjacentChanMat] = ptTrodesBART_2(ptID); 
selectedChans = find(isECoG); 
trodeLabels_selectedChans = trodeLabels(isECoG);

% load electrode labels created from BART_BrainRegions:
regions_wHemi_CoarseStruct = load('\\155.100.91.44\d\Data\Rhiannon\BART_RLDM_outputs\BrainRegions\Brainnetome_Atlas_All_woHemi_Coarse'); 
regions_woHemi_CoarseStruct = load('\\155.100.91.44\d\Data\Rhiannon\BART_RLDM_outputs\BrainRegions\Brainnetome_Atlas_All_woHemi_Coarse');
BART_all_isECoG_logical_matrix =  load('\\155.100.91.44\D\Data\Rhiannon\BART_RLDM_outputs\BrainRegions\BART_all_isECoG_logical_matrix.mat');
new_isECoG = BART_all_isECoG_logical_matrix.BART_all_isECoG_logical_matrix;

% finding position of patient in struct.
[ptArray] = BARTnumbers();
ptID_position = contains(ptArray, ptID);
new_trodeLabels = regions_woHemi_CoarseStruct.Brainnetome_Atlas_All_woHemi_Coarse(:,ptID_position);
new_selectedChans = find(new_isECoG(:,ptID_position)); 
new_trodeLabels_selectedChans = new_trodeLabels(new_isECoG(:,ptID_position)); % Removing NaCs from trodelabels
new_nChans = length(new_selectedChans); 

%% reading paramrecovery data
paramRecoveryFolder = '\\155.100.91.44\d\Data\Nill\BART_param_recovery\context_modeling\param_recovery_1_modeling\';
fileName = [ptID '_TDdataParamRecovery.mat'];
fullFilePath = fullfile(paramRecoveryFolder, fileName);
load(fullFilePath);

%% initializing bhv output
TDdataGradients.patientID = ptID;
TDdataGradients.LMtrials = LMtrials;

% loading matFile
matFile = ['\\155.100.91.44\D\Data\preProcessed\BART_preprocessed\' ptID '\Data\' ptID '.bartBHV.mat'];
load(matFile)
dataBHV = data;
clear data

% load and define triggers from nevFile
NEV = openNEV(nevFile,'overwrite');
trigs = NEV.Data.SerialDigitalIO.UnparsedData;
trigTimes = NEV.Data.SerialDigitalIO.TimeStampSec;

% [20170713] duplicate 23 trigger cleanup retained for compatibility.
infIdx = trigs==23;
infIdx([diff(infIdx)==0; false] & infIdx==1) = 0; 

% trial IDs
% FIX: include trigger 4 in balloonIDs because the wrapper includes trigger 4 in balloonTimes.
% If trigger 4 exists and is missing here, neural epochs and labels can shift.
validBalloonTriggers = [1 2 3 4 11 12 13 14];
isCTRL = logical([dataBHV.is_control]);
balloonIDs = trigs(ismember(trigs,validBalloonTriggers));
outcomeType = trigs(sort([find(trigs==25); find(trigs==26)]))-24;
outcomeTypeCat = cell(1,numel(outcomeType));
outcomeTypeCat(outcomeType==1) = {'banked'};
outcomeTypeCat(outcomeType==2) = {'popped'};

% response times
respTimes = trigTimes(trigs==25 | trigs==26);

if ~exist('TDtrials','var')
    nTrials = min([length(respTimes), length(balloonIDs), length(isCTRL), length(outcomeTypeCat)]);
    if nTrials < length(respTimes) || nTrials < length(balloonIDs) || nTrials < length(isCTRL) || nTrials < length(outcomeTypeCat)
        warning('Trial count mismatch for %s. Trimming all trial vectors to %d aligned trials.', ptID, nTrials);
    end
    balloonIDs = balloonIDs(1:nTrials);
    isCTRL = isCTRL(1:nTrials); 
    outcomeType = outcomeType(1:nTrials); 
    outcomeTypeCat = outcomeTypeCat(1:nTrials);
else
    nTrials = length(TDtrials);
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
[ra,rb] = size(cueData);
if isequal(rb,nTrials)
    nChannels = ra;
    tmpTrials = rb;
elseif isequal(ra,nTrials)
    cueData = cueData'; % cue-aligned
    outcomeData = outcomeData'; % outcome-aligned
    nChannels = rb;
    tmpTrials = ra;
else
    % Allow a safe trim when the only issue is an extra/missing aborted trial.
    candidateTrials = [ra rb];
    candidateTrials = candidateTrials(candidateTrials <= nTrials);
    if isempty(candidateTrials)
        error('Neural data size does not match trigger-derived trial count. nTrials=%d, size(cueData)=[%d %d].', nTrials, ra, rb);
    end
    tmpTrials = max(candidateTrials);
    warning('Neural data has %d trials but trigger-derived nTrials is %d. Trimming all trial variables to %d.', tmpTrials, nTrials, tmpTrials);
    if rb == tmpTrials
        nChannels = ra;
    else
        cueData = cueData';
        outcomeData = outcomeData';
        nChannels = rb;
    end
    nTrials = tmpTrials;
    balloonIDs = balloonIDs(1:nTrials);
    outcomeTypeCat = outcomeTypeCat(1:nTrials);
end

outcomeData = outcomeData(:,1:nTrials);
cueData = cueData(:,1:nTrials);
TDdataGradients.nTrials = nTrials;

% points vector
pointsPerTrial = diff([0 [dataBHV.score]]); 

%% now fitting risk-sensitive (asymmetric) models.
TDdataGradients.a = TDdataParamRecovery.a;
a = TDdataParamRecovery.a;
TDdataGradients.inverseTemperatureRSTD = TDdataParamRecovery.inverseTemperatureRSTD;
TDdataGradients.rstdV = TDdataParamRecovery.expectedReward;
TDdataGradients.rstdRPE = TDdataParamRecovery.RPE;
TDdataGradients.positiveBetaMaxIdx = TDdataParamRecovery.bestApIdx;
TDdataGradients.negativeBetaMaxIdx = TDdataParamRecovery.bestAnIdx;

% Validate and align TD variables.
if ndims(TDdataGradients.rstdV) ~= 3
    error('TDdataParamRecovery.expectedReward must be a 3-D matrix [alphaPos x alphaNeg x trials].');
end
if ndims(TDdataGradients.rstdRPE) ~= 3
    error('TDdataParamRecovery.RPE must be a 3-D matrix [alphaPos x alphaNeg x trials].');
end

nV = size(TDdataGradients.rstdV,3);
nPE = size(TDdataGradients.rstdRPE,3);

if nV == nTrials - 1
    warning('expectedReward has one fewer trial than behavior/neural data. Padding trial 1 with NaN and keeping trials 2:end aligned.');
    tmpV = TDdataGradients.rstdV;
    TDdataGradients.rstdV = nan(size(tmpV,1),size(tmpV,2),nTrials);
    TDdataGradients.rstdV(:,:,2:nTrials) = tmpV;
elseif nV > nTrials
    warning('expectedReward has %d trials but behavior/neural data has %d. Trimming expectedReward.', nV, nTrials);
    TDdataGradients.rstdV = TDdataGradients.rstdV(:,:,1:nTrials);
end

if nPE == nTrials - 1
    warning('RPE has one fewer trial than behavior/neural data. Padding trial 1 with NaN and keeping trials 2:end aligned.');
    tmpPE = TDdataGradients.rstdRPE;
    TDdataGradients.rstdRPE = nan(size(tmpPE,1),size(tmpPE,2),nTrials);
    TDdataGradients.rstdRPE(:,:,2:nTrials) = tmpPE;
elseif nPE > nTrials
    warning('RPE has %d trials but behavior/neural data has %d. Trimming RPE.', nPE, nTrials);
    TDdataGradients.rstdRPE = TDdataGradients.rstdRPE(:,:,1:nTrials);
end

commonTDTrials = min([size(TDdataGradients.rstdV,3), size(TDdataGradients.rstdRPE,3), nTrials]);
if commonTDTrials < nTrials
    warning('TD variables are shorter than behavior/neural data. Trimming all trial variables to %d.', commonTDTrials);
end
TDdataGradients.rstdV = TDdataGradients.rstdV(:,:,1:commonTDTrials);
TDdataGradients.rstdRPE = TDdataGradients.rstdRPE(:,:,1:commonTDTrials);
nTrials = commonTDTrials;
TDdataGradients.nTrials = nTrials;
outcomeData = outcomeData(:,1:nTrials);
cueData = cueData(:,1:nTrials);
balloonIDs = balloonIDs(1:nTrials);
outcomeTypeCat = outcomeTypeCat(1:nTrials);

fprintf('\nTD variable checks for %s:\n', ptID);
fprintf('  nTrials used = %d\n', nTrials);
fprintf('  size(rstdV)   = [%s]\n', num2str(size(TDdataGradients.rstdV)));
fprintf('  size(rstdRPE) = [%s]\n', num2str(size(TDdataGradients.rstdRPE)));
fprintf('  NaNs in rstdV   = %d\n', sum(isnan(TDdataGradients.rstdV(:))));
fprintf('  NaNs in rstdRPE = %d\n', sum(isnan(TDdataGradients.rstdRPE(:))));

% Apply requested trial window. Original code overwrote/ignored LMtrials.
allTrials = 1:nTrials;
try
    LMidx = eval(['allTrials(' LMtrials ')']);
catch ME
    error('Could not evaluate LMtrials = %s. Original error: %s', LMtrials, ME.message);
end
LMidx = LMidx(:);
LMidx = LMidx(LMidx >= 1 & LMidx <= nTrials);
LMidx = unique(LMidx,'stable');
if isempty(LMidx)
    error('LMtrials = %s returned no valid trials.', LMtrials);
end
TDdataGradients.LMidx = LMidx;

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

% Build neural tables once. Use safe valid MATLAB variable names to avoid formula errors.
rawTrodeLabels = trodeLabels_selectedChans(:)';
safeTrodeLabels = matlab.lang.makeUniqueStrings(matlab.lang.makeValidName(rawTrodeLabels));
tmpTbl_outcomeAligned = array2table(outcomeData','VariableNames',safeTrodeLabels); % HG outcome data
tmpTbl_cueAligned = array2table(cueData','VariableNames',safeTrodeLabels); % HG cue data

%% fitting temporal difference learning models to NEURAL DATA
% fitting the model for each channel for the alpha.
skipFactor = 5;
bestAp = TDdataGradients.positiveBetaMaxIdx;
bestAn = TDdataGradients.negativeBetaMaxIdx;

for chz = nChannels:-1:1
    updateUser('assessing RSTD model encoding across learning rates on channel ',chz,10,nChannels)

    TDdataGradients.neuralFit(chz).bestAlphaPositive = a(bestAp);
    TDdataGradients.neuralFit(chz).bestAlphaNegative = a(bestAn);
    TDdataGradients.neuralFit(chz).trodeLabel = tmpTbl_outcomeAligned.Properties.VariableNames{chz};
    TDdataGradients.neuralFit(chz).rawTrodeLabel = rawTrodeLabels{chz};
    if chz <= numel(new_trodeLabels_selectedChans) && ~isempty(new_trodeLabels_selectedChans{chz})
        TDdataGradients.neuralFit(chz).new_trodeLabel = new_trodeLabels_selectedChans{chz}; % adding new labels!
    else
        TDdataGradients.neuralFit(chz).new_trodeLabel = 'unknownRegion';
    end

    TDdataGradients.neuralFit(chz).LLimg_outcome = nan(length(a),length(a));
    TDdataGradients.neuralFit(chz).LLimg_outcomeSuccess = nan(length(a),length(a));
    TDdataGradients.neuralFit(chz).LLimg_cue = nan(length(a),length(a));
    TDdataGradients.neuralFit(chz).LLimg_cueSuccess = nan(length(a),length(a));
    TDdataGradients.neuralFit(chz).R2img_outcome = nan(length(a),length(a));
    TDdataGradients.neuralFit(chz).R2img_outcomeSuccess = nan(length(a),length(a));
    TDdataGradients.neuralFit(chz).R2img_cue = nan(length(a),length(a));
    TDdataGradients.neuralFit(chz).R2img_cueSuccess = nan(length(a),length(a));

    for ap = length(a):-skipFactor:1
        for an = length(a):-skipFactor:1

            responseName = TDdataGradients.neuralFit(chz).trodeLabel;

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
            tmpTbl_rstd.Outcome = removecats(categorical(tmpTbl_rstd.Outcome));

            % FINAL TABLES
            rstdTbl_outcome = cat(2,tmpTbl_outcomeAligned,tmpTbl_rstd); % all trials (outcome-aligned)
            rstdTbl_cue = cat(2,tmpTbl_cueAligned,tmpTbl_rstd); % all trials (cue-aligned)

            % FIX: Apply LMtrials for real.
            rstdTbl_outcome = rstdTbl_outcome(LMidx,:);
            rstdTbl_cue = rstdTbl_cue(LMidx,:);

            % FIX: remove rows with NaN/Inf TD variables or neural response.
            validOutcome = isfinite(rstdTbl_outcome.(responseName)) & isfinite(rstdTbl_outcome.RSTD_PE) & ...
                           ~isundefined(rstdTbl_outcome.TrialColor) & ~isundefined(rstdTbl_outcome.TrialType) & ~isundefined(rstdTbl_outcome.Outcome);
            validCue = isfinite(rstdTbl_cue.(responseName)) & isfinite(rstdTbl_cue.RSTD_VE) & ...
                       ~isundefined(rstdTbl_cue.TrialColor) & ~isundefined(rstdTbl_cue.TrialType) & ~isundefined(rstdTbl_cue.Outcome);
            rstdTbl_outcome = rstdTbl_outcome(validOutcome,:);
            rstdTbl_cue = rstdTbl_cue(validCue,:);

            rstdTbl_outcome_success = rstdTbl_outcome(rstdTbl_outcome.Outcome == "banked", :); % only successful trials
            rstdTbl_cue_success = rstdTbl_cue(rstdTbl_cue.Outcome == "banked", :); % only successful trials

            % Defaults for failed models.
            LL_outcome = nan; LL_outcomeSuccess = nan; LL_cue = nan; LL_cueSuccess = nan;
            R2_outcome = nan; R2_outcomeSuccess = nan; R2_cue = nan; R2_cueSuccess = nan;

            try
                if height(rstdTbl_outcome) >= 5
                    tmpMdl = fitglme(rstdTbl_outcome,[responseName ' ~ RSTD_PE + TrialColor + TrialType + Outcome']); % ALL TRIALS
                    LL_outcome = tmpMdl.LogLikelihood;
                    try
                        R2_outcome = tmpMdl.Rsquared.Adjusted;
                    catch
                        R2_outcome = tmpMdl.Rsquared.Ordinary;
                    end
                end
            catch ME
                warning('Outcome model failed for %s ap=%d an=%d: %s', responseName, ap, an, ME.message);
            end

            try
                if height(rstdTbl_outcome_success) >= 5
                    tmpMdl = fitglme(rstdTbl_outcome_success,[responseName ' ~ RSTD_PE + TrialColor + TrialType']); % SUCCESSFUL (BANKED) TRIALS
                    LL_outcomeSuccess = tmpMdl.LogLikelihood;
                    try
                        R2_outcomeSuccess = tmpMdl.Rsquared.Adjusted;
                    catch
                        R2_outcomeSuccess = tmpMdl.Rsquared.Ordinary;
                    end
                end
            catch ME
                warning('Outcome success model failed for %s ap=%d an=%d: %s', responseName, ap, an, ME.message);
            end

            try
                if height(rstdTbl_cue) >= 5
                    tmpMdl = fitglme(rstdTbl_cue,[responseName ' ~ RSTD_VE + TrialColor + TrialType']); % ALL TRIALS
                    LL_cue = tmpMdl.LogLikelihood;
                    try
                        R2_cue = tmpMdl.Rsquared.Adjusted;
                    catch
                        R2_cue = tmpMdl.Rsquared.Ordinary;
                    end
                end
            catch ME
                warning('Cue model failed for %s ap=%d an=%d: %s', responseName, ap, an, ME.message);
            end

            try
                if height(rstdTbl_cue_success) >= 5
                    tmpMdl = fitglme(rstdTbl_cue_success,[responseName ' ~ RSTD_VE + TrialColor + TrialType']); % SUCCESSFUL (BANKED) TRIALS
                    LL_cueSuccess = tmpMdl.LogLikelihood;
                    try
                        R2_cueSuccess = tmpMdl.Rsquared.Adjusted;
                    catch
                        R2_cueSuccess = tmpMdl.Rsquared.Ordinary;
                    end
                end
            catch ME
                warning('Cue success model failed for %s ap=%d an=%d: %s', responseName, ap, an, ME.message);
            end

            % Saving log likelihood landscapes per electrode.
            TDdataGradients.neuralFit(chz).LLimg_outcome(ap,an) = LL_outcome;
            TDdataGradients.neuralFit(chz).LLimg_outcomeSuccess(ap,an) = LL_outcomeSuccess;
            TDdataGradients.neuralFit(chz).LLimg_cue(ap,an) = LL_cue;
            TDdataGradients.neuralFit(chz).LLimg_cueSuccess(ap,an) = LL_cueSuccess;

            % saving R-squared landscapes per electrode.
            TDdataGradients.neuralFit(chz).R2img_outcome(ap,an) = R2_outcome;
            TDdataGradients.neuralFit(chz).R2img_outcomeSuccess(ap,an) = R2_outcomeSuccess;
            TDdataGradients.neuralFit(chz).R2img_cue(ap,an) = R2_cue;
            TDdataGradients.neuralFit(chz).R2img_cueSuccess(ap,an) = R2_cueSuccess;

        end
    end

    % FIX: save final models/ANOVA at behavioral best alpha, not the last alpha pair in the loop.
    responseName = TDdataGradients.neuralFit(chz).trodeLabel;
    tmpTbl_rstd = table(squeeze(TDdataGradients.rstdV(bestAp,bestAn,:)),...
                        squeeze(TDdataGradients.rstdRPE(bestAp,bestAn,:)),...
                        TrialColor,TrialType,OutcomeType,...
                       'VariableNames',{'RSTD_VE','RSTD_PE','TrialColor', 'TrialType', 'Outcome'});
    tmpTbl_rstd.TrialColor = categorical(string(tmpTbl_rstd.TrialColor));
    tmpTbl_rstd.TrialColor = removecats(tmpTbl_rstd.TrialColor);
    tmpTbl_rstd.TrialType = categorical(isActive, [1 0], {'active', 'passive'});
    tmpTbl_rstd.TrialType = reordercats(tmpTbl_rstd.TrialType, {'passive', 'active'});
    tmpTbl_rstd.Outcome = removecats(categorical(tmpTbl_rstd.Outcome));

    rstdTbl_outcome = cat(2,tmpTbl_outcomeAligned,tmpTbl_rstd);
    rstdTbl_cue = cat(2,tmpTbl_cueAligned,tmpTbl_rstd);
    rstdTbl_outcome = rstdTbl_outcome(LMidx,:);
    rstdTbl_cue = rstdTbl_cue(LMidx,:);

    validOutcome = isfinite(rstdTbl_outcome.(responseName)) & isfinite(rstdTbl_outcome.RSTD_PE) & ...
                   ~isundefined(rstdTbl_outcome.TrialColor) & ~isundefined(rstdTbl_outcome.TrialType) & ~isundefined(rstdTbl_outcome.Outcome);
    validCue = isfinite(rstdTbl_cue.(responseName)) & isfinite(rstdTbl_cue.RSTD_VE) & ...
               ~isundefined(rstdTbl_cue.TrialColor) & ~isundefined(rstdTbl_cue.TrialType) & ~isundefined(rstdTbl_cue.Outcome);
    rstdTbl_outcome = rstdTbl_outcome(validOutcome,:);
    rstdTbl_cue = rstdTbl_cue(validCue,:);
    rstdTbl_outcome_success = rstdTbl_outcome(rstdTbl_outcome.Outcome == "banked", :);
    rstdTbl_cue_success = rstdTbl_cue(rstdTbl_cue.Outcome == "banked", :);

    dummyAnova = array2table(nan(5,5),'VariableNames',{'Col1','Col2','Col3','Col4','Col5'});
    TDdataGradients.neuralFit(chz).rstd_OutcomeModel = [];
    TDdataGradients.neuralFit(chz).rstd_OutcomeSuccessModel = [];
    TDdataGradients.neuralFit(chz).rstd_CueModel = [];
    TDdataGradients.neuralFit(chz).rstd_CueSuccessModel = [];
    TDdataGradients.neuralFit(chz).ANOVA_rstd_Outcome = dummyAnova;
    TDdataGradients.neuralFit(chz).ANOVA_rstd_OutcomeSuccess = dummyAnova;
    TDdataGradients.neuralFit(chz).ANOVA_rstd_Cue = dummyAnova;
    TDdataGradients.neuralFit(chz).ANOVA_rstd_CueSuccess = dummyAnova;

    try
        if height(rstdTbl_outcome) >= 5
            TDdataGradients.neuralFit(chz).rstd_OutcomeModel = fitglme(rstdTbl_outcome,[responseName ' ~ RSTD_PE + TrialColor + TrialType + Outcome']);
            TDdataGradients.neuralFit(chz).ANOVA_rstd_Outcome = anova(TDdataGradients.neuralFit(chz).rstd_OutcomeModel);
        end
    catch ME
        warning('Best-alpha outcome model failed for %s: %s', responseName, ME.message);
    end
    try
        if height(rstdTbl_outcome_success) >= 5
            TDdataGradients.neuralFit(chz).rstd_OutcomeSuccessModel = fitglme(rstdTbl_outcome_success,[responseName ' ~ RSTD_PE + TrialColor + TrialType']);
            TDdataGradients.neuralFit(chz).ANOVA_rstd_OutcomeSuccess = anova(TDdataGradients.neuralFit(chz).rstd_OutcomeSuccessModel);
        end
    catch ME
        warning('Best-alpha outcome success model failed for %s: %s', responseName, ME.message);
    end
    try
        if height(rstdTbl_cue) >= 5
            TDdataGradients.neuralFit(chz).rstd_CueModel = fitglme(rstdTbl_cue,[responseName ' ~ RSTD_VE + TrialColor + TrialType']);
            TDdataGradients.neuralFit(chz).ANOVA_rstd_Cue = anova(TDdataGradients.neuralFit(chz).rstd_CueModel);
        end
    catch ME
        warning('Best-alpha cue model failed for %s: %s', responseName, ME.message);
    end
    try
        if height(rstdTbl_cue_success) >= 5
            TDdataGradients.neuralFit(chz).rstd_CueSuccessModel = fitglme(rstdTbl_cue_success,[responseName ' ~ RSTD_VE + TrialColor + TrialType']);
            TDdataGradients.neuralFit(chz).ANOVA_rstd_CueSuccess = anova(TDdataGradients.neuralFit(chz).rstd_CueSuccessModel);
        end
    catch ME
        warning('Best-alpha cue success model failed for %s: %s', responseName, ME.message);
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

% learning rate gradients
% save TDdata by patient here:
tic
[pDir] = fileparts(parentDir);
save(fullfile(pDir,[ptID '_TDdataGradients.mat']),'TDdataGradients','-v7.3')
toc
fprintf('\n\nSaving TDdata struct took %.2f minutes...', toc / 60);
