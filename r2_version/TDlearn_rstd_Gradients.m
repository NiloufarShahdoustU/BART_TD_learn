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
warning('off','all');
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
TDdataGradients.a = TDdataParamRecovery.a(:)';
a = TDdataGradients.a;
TDdataGradients.inverseTemperatureRSTD = TDdataParamRecovery.inverseTemperatureRSTD;
TDdataGradients.rstdV = TDdataParamRecovery.expectedReward;
TDdataGradients.rstdRPE = TDdataParamRecovery.RPE;

% -------------------------------------------------------------------------
% FIX FOR BOUNDARY ALPHAS
% -------------------------------------------------------------------------
% Problem: TDdataParamRecovery.bestApIdx/bestAnIdx can land on the edge of
% the alpha grid (near 0 or 1). If we copy those directly into neural
% modeling, all downstream neural summaries inherit boundary alphas even
% when the likelihood surface is flat/noisy.
%
% Fix: choose the final alpha pair from the behavioral fit landscape after:
%   1) excluding extreme grid edges, and
%   2) adding a weak Beta(2,2)-like prior that favors identifiable interior
%      solutions without changing the TD trajectories themselves.
%
% This keeps the neural test non-circular: neural activity is NOT used to
% choose the final alpha pair used for the ANOVA models.
rawBestAp = TDdataParamRecovery.bestApIdx;
rawBestAn = TDdataParamRecovery.bestAnIdx;
rawBestAp = max(1,min(numel(a),rawBestAp));
rawBestAn = max(1,min(numel(a),rawBestAn));

alphaLowerBound = 0.05;     % change to 0.10 if you want stricter trimming
alphaUpperBound = 0.95;     % change to 0.90 if you want stricter trimming
alphaPriorStrength = 3;     % 0 = only trim edges; higher = stronger interior pull

alphaInteriorMask = (a >= alphaLowerBound) & (a <= alphaUpperBound);
if sum(alphaInteriorMask) < 2
    warning('Alpha grid has too few values inside [%.2f %.2f]. Using full grid.', alphaLowerBound, alphaUpperBound);
    alphaInteriorMask = true(size(a));
end
alphaPairMask = alphaInteriorMask(:) & alphaInteriorMask(:)';
[APgrid,ANgrid] = ndgrid(a,a);
alphaPrior = log(max(APgrid .* (1-APgrid), realmin)) + log(max(ANgrid .* (1-ANgrid), realmin));

alphaScoreRaw = [];
alphaScoreName = 'none';
if isfield(TDdataParamRecovery,'fitScoreRSTD') && isequal(size(TDdataParamRecovery.fitScoreRSTD),[numel(a) numel(a)])
    alphaScoreRaw = TDdataParamRecovery.fitScoreRSTD;
    alphaScoreName = 'fitScoreRSTD';
elseif isfield(TDdataParamRecovery,'logLikRSTD') && isequal(size(TDdataParamRecovery.logLikRSTD),[numel(a) numel(a)])
    alphaScoreRaw = TDdataParamRecovery.logLikRSTD;
    alphaScoreName = 'logLikRSTD';
elseif isfield(TDdataParamRecovery,'inverseTemperatureRSTD') && isequal(size(TDdataParamRecovery.inverseTemperatureRSTD),[numel(a) numel(a)])
    alphaScoreRaw = TDdataParamRecovery.inverseTemperatureRSTD;
    alphaScoreName = 'inverseTemperatureRSTD';
end

if isempty(alphaScoreRaw)
    % Fallback: keep the raw best pair, but clamp it into the interior mask.
    interiorIdx = find(alphaInteriorMask);
    [~,tmpAp] = min(abs(a(interiorIdx)-a(rawBestAp)));
    [~,tmpAn] = min(abs(a(interiorIdx)-a(rawBestAn)));
    bestAp = interiorIdx(tmpAp);
    bestAn = interiorIdx(tmpAn);
    alphaScoreReg = nan(numel(a),numel(a));
    warning('No 2-D RSTD fit landscape found. Clamping raw alpha indices into the interior range only.');
else
    alphaScoreReg = double(alphaScoreRaw);
    alphaScoreReg(~isfinite(alphaScoreReg)) = nan;
    alphaScoreReg(~alphaPairMask) = nan;
    alphaScoreReg = alphaScoreReg + alphaPriorStrength .* alphaPrior;

    validScore = isfinite(alphaScoreReg(:));
    if any(validScore)
        validLinearIdx = find(validScore);
        [~,tmpBest] = max(alphaScoreReg(validScore));
        bestLinearIdx = validLinearIdx(tmpBest);
        [bestAp,bestAn] = ind2sub(size(alphaScoreReg),bestLinearIdx);
    else
        interiorIdx = find(alphaInteriorMask);
        [~,tmpAp] = min(abs(a(interiorIdx)-a(rawBestAp)));
        [~,tmpAn] = min(abs(a(interiorIdx)-a(rawBestAn)));
        bestAp = interiorIdx(tmpAp);
        bestAn = interiorIdx(tmpAn);
        warning('Regularized alpha landscape had no finite values. Clamping raw alpha indices into the interior range only.');
    end
end

TDdataGradients.positiveBetaMaxIdx_raw = rawBestAp;
TDdataGradients.negativeBetaMaxIdx_raw = rawBestAn;
TDdataGradients.positiveBetaMaxIdx = bestAp;
TDdataGradients.negativeBetaMaxIdx = bestAn;
TDdataGradients.bestAlphaPositive = a(bestAp);
TDdataGradients.bestAlphaNegative = a(bestAn);
TDdataGradients.alphaSelection.alphaLowerBound = alphaLowerBound;
TDdataGradients.alphaSelection.alphaUpperBound = alphaUpperBound;
TDdataGradients.alphaSelection.alphaPriorStrength = alphaPriorStrength;
TDdataGradients.alphaSelection.scoreName = alphaScoreName;
TDdataGradients.alphaSelection.rawAlphaPositive = a(rawBestAp);
TDdataGradients.alphaSelection.rawAlphaNegative = a(rawBestAn);
TDdataGradients.alphaSelection.regularizedAlphaPositive = a(bestAp);
TDdataGradients.alphaSelection.regularizedAlphaNegative = a(bestAn);
TDdataGradients.alphaSelection.regularizedScore = alphaScoreReg;

fprintf('\nAlpha selection for %s:\n', ptID);
fprintf('  raw behavioral alpha+ = %.3f, alpha- = %.3f\n', a(rawBestAp), a(rawBestAn));
fprintf('  regularized alpha+    = %.3f, alpha- = %.3f\n', a(bestAp), a(bestAn));
fprintf('  alpha score source    = %s\n', alphaScoreName);


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

% Align points vector to the exact trial set used for neural modeling.
% pointsPerTrial is used only as a nuisance covariate in the neural model.
pointsPerTrial = pointsPerTrial(:);
if numel(pointsPerTrial) < nTrials
    warning('pointsPerTrial has %d trials but neural model uses %d. Padding missing values with NaN.', numel(pointsPerTrial), nTrials);
    pointsPerTrial(end+1:nTrials,1) = nan;
elseif numel(pointsPerTrial) > nTrials
    pointsPerTrial = pointsPerTrial(1:nTrials);
end

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

% Nuisance covariates for neural models.
% TrialNumberZ controls slow drift/fatigue/learning over the session.
% AbsPointsPerTrialZ controls reward magnitude at outcome without duplicating
% the sign already captured by Outcome (banked vs popped).
TrialNumber = (1:nTrials)';
TrialNumberZ = TrialNumber;
if std(TrialNumberZ) > 0
    TrialNumberZ = (TrialNumberZ - mean(TrialNumberZ)) ./ std(TrialNumberZ);
else
    TrialNumberZ = zeros(size(TrialNumberZ));
end

PointsPerTrial = pointsPerTrial(:);
AbsPointsPerTrial = abs(PointsPerTrial);
AbsPointsPerTrialZ = AbsPointsPerTrial;
validPtsForZ = isfinite(AbsPointsPerTrialZ);
if any(validPtsForZ)
    ptsMean = mean(AbsPointsPerTrialZ(validPtsForZ));
    ptsStd = std(AbsPointsPerTrialZ(validPtsForZ));
    if ptsStd > 0
        AbsPointsPerTrialZ(validPtsForZ) = (AbsPointsPerTrialZ(validPtsForZ) - ptsMean) ./ ptsStd;
    else
        AbsPointsPerTrialZ(validPtsForZ) = 0;
    end
end
AbsPointsPerTrialZ(~validPtsForZ) = nan;

TDdataGradients.modelCovariates.TrialNumber = TrialNumber;
TDdataGradients.modelCovariates.TrialNumberZ = TrialNumberZ;
TDdataGradients.modelCovariates.PointsPerTrial = PointsPerTrial;
TDdataGradients.modelCovariates.AbsPointsPerTrial = AbsPointsPerTrial;
TDdataGradients.modelCovariates.AbsPointsPerTrialZ = AbsPointsPerTrialZ;

TDdataGradients.modelFormulas.outcomeAll = 'HG ~ RSTD_PE_pos + RSTD_PE_neg + TrialColor + Outcome + TrialNumberZ + AbsPointsPerTrialZ';
TDdataGradients.modelFormulas.outcomeSuccess = 'HG ~ RSTD_PE_pos + RSTD_PE_neg + TrialColor + TrialNumberZ + AbsPointsPerTrialZ';
TDdataGradients.modelFormulas.cueAll = 'HG ~ RSTD_VE + TrialColor + TrialNumberZ';
TDdataGradients.modelFormulas.cueSuccess = 'HG ~ RSTD_VE + TrialColor + TrialNumberZ';

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
            tmpVE = squeeze(TDdataGradients.rstdV(ap,an,:));
            tmpPE = squeeze(TDdataGradients.rstdRPE(ap,an,:));
            tmpVE = tmpVE(:);
            tmpPE = tmpPE(:);

            % Split signed RPE into separate positive and negative components.
            % RSTD_PE_pos = positive PE magnitude; zero otherwise.
            % RSTD_PE_neg = negative PE magnitude; zero otherwise.
            RSTD_PE_pos = max(tmpPE,0);
            RSTD_PE_neg = max(-tmpPE,0);

            tmpTbl_rstd = table(tmpVE,...
                                tmpPE,...
                                RSTD_PE_pos,...
                                RSTD_PE_neg,...
                                TrialColor,TrialType,OutcomeType,...
                                TrialNumber,TrialNumberZ,...
                                PointsPerTrial,AbsPointsPerTrial,AbsPointsPerTrialZ,...
                               'VariableNames',{'RSTD_VE','RSTD_PE','RSTD_PE_pos','RSTD_PE_neg','TrialColor', 'TrialType', 'Outcome',...
                                                'TrialNumber','TrialNumberZ','PointsPerTrial','AbsPointsPerTrial','AbsPointsPerTrialZ'});

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
            validOutcome = isfinite(rstdTbl_outcome.(responseName)) & isfinite(rstdTbl_outcome.RSTD_PE_pos) & isfinite(rstdTbl_outcome.RSTD_PE_neg) & ...
                           isfinite(rstdTbl_outcome.TrialNumberZ) & isfinite(rstdTbl_outcome.AbsPointsPerTrialZ) & ...
                           ~isundefined(rstdTbl_outcome.TrialColor) & ~isundefined(rstdTbl_outcome.TrialType) & ~isundefined(rstdTbl_outcome.Outcome);
            validCue = isfinite(rstdTbl_cue.(responseName)) & isfinite(rstdTbl_cue.RSTD_VE) & ...
                       isfinite(rstdTbl_cue.TrialNumberZ) & ...
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
                    tmpMdl = fitglme(rstdTbl_outcome,[responseName ' ~ RSTD_PE_pos + RSTD_PE_neg + TrialColor + Outcome + TrialNumberZ + AbsPointsPerTrialZ']); % ALL TRIALS
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
                    tmpMdl = fitglme(rstdTbl_outcome_success,[responseName ' ~ RSTD_PE_pos + RSTD_PE_neg + TrialColor + TrialNumberZ + AbsPointsPerTrialZ']); % SUCCESSFUL (BANKED) TRIALS
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
                    tmpMdl = fitglme(rstdTbl_cue,[responseName ' ~ RSTD_VE + TrialColor + TrialNumberZ']); % ALL TRIALS
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
                    tmpMdl = fitglme(rstdTbl_cue_success,[responseName ' ~ RSTD_VE + TrialColor + TrialNumberZ']); % SUCCESSFUL (BANKED) TRIALS
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

    % Optional diagnostic only: where would the neural LL landscape peak?
    % This is stored for visualization/diagnosis, but NOT used for the final
    % ANOVA models below to avoid selecting alpha and testing neural encoding
    % with the same neural data.
    neuralScore = nan(size(TDdataGradients.neuralFit(chz).LLimg_outcome));
    neuralStack = cat(3,TDdataGradients.neuralFit(chz).LLimg_outcome,TDdataGradients.neuralFit(chz).LLimg_cue);
    neuralValid = isfinite(neuralStack);
    neuralStackZero = neuralStack;
    neuralStackZero(~neuralValid) = 0;
    neuralDenom = sum(neuralValid,3);
    neuralNumer = sum(neuralStackZero,3);
    neuralScore(neuralDenom > 0) = neuralNumer(neuralDenom > 0) ./ neuralDenom(neuralDenom > 0);
    neuralScore(~alphaPairMask) = nan;
    neuralScoreReg = neuralScore + alphaPriorStrength .* alphaPrior;
    validNeuralScore = isfinite(neuralScoreReg(:));
    if any(validNeuralScore)
        validNeuralIdx = find(validNeuralScore);
        [~,tmpNeuralBest] = max(neuralScoreReg(validNeuralScore));
        neuralBestLinearIdx = validNeuralIdx(tmpNeuralBest);
        [neuralBestAp,neuralBestAn] = ind2sub(size(neuralScoreReg),neuralBestLinearIdx);
        TDdataGradients.neuralFit(chz).diagnosticNeuralBestApIdx = neuralBestAp;
        TDdataGradients.neuralFit(chz).diagnosticNeuralBestAnIdx = neuralBestAn;
        TDdataGradients.neuralFit(chz).diagnosticNeuralBestAlphaPositive = a(neuralBestAp);
        TDdataGradients.neuralFit(chz).diagnosticNeuralBestAlphaNegative = a(neuralBestAn);
    else
        TDdataGradients.neuralFit(chz).diagnosticNeuralBestApIdx = nan;
        TDdataGradients.neuralFit(chz).diagnosticNeuralBestAnIdx = nan;
        TDdataGradients.neuralFit(chz).diagnosticNeuralBestAlphaPositive = nan;
        TDdataGradients.neuralFit(chz).diagnosticNeuralBestAlphaNegative = nan;
    end

    % FIX: save final models/ANOVA at the regularized behavioral alpha pair, not the last alpha pair in the loop.
    responseName = TDdataGradients.neuralFit(chz).trodeLabel;
    tmpVE = squeeze(TDdataGradients.rstdV(bestAp,bestAn,:));
    tmpPE = squeeze(TDdataGradients.rstdRPE(bestAp,bestAn,:));
    tmpVE = tmpVE(:);
    tmpPE = tmpPE(:);

    % Split signed RPE into separate positive and negative components.
    % RSTD_PE_pos = positive PE magnitude; zero otherwise.
    % RSTD_PE_neg = negative PE magnitude; zero otherwise.
    RSTD_PE_pos = max(tmpPE,0);
    RSTD_PE_neg = max(-tmpPE,0);

    tmpTbl_rstd = table(tmpVE,...
                        tmpPE,...
                        RSTD_PE_pos,...
                        RSTD_PE_neg,...
                        TrialColor,TrialType,OutcomeType,...
                        TrialNumber,TrialNumberZ,...
                        PointsPerTrial,AbsPointsPerTrial,AbsPointsPerTrialZ,...
                       'VariableNames',{'RSTD_VE','RSTD_PE','RSTD_PE_pos','RSTD_PE_neg','TrialColor', 'TrialType', 'Outcome',...
                                        'TrialNumber','TrialNumberZ','PointsPerTrial','AbsPointsPerTrial','AbsPointsPerTrialZ'});
    tmpTbl_rstd.TrialColor = categorical(string(tmpTbl_rstd.TrialColor));
    tmpTbl_rstd.TrialColor = removecats(tmpTbl_rstd.TrialColor);
    tmpTbl_rstd.TrialType = categorical(isActive, [1 0], {'active', 'passive'});
    tmpTbl_rstd.TrialType = reordercats(tmpTbl_rstd.TrialType, {'passive', 'active'});
    tmpTbl_rstd.Outcome = removecats(categorical(tmpTbl_rstd.Outcome));

    rstdTbl_outcome = cat(2,tmpTbl_outcomeAligned,tmpTbl_rstd);
    rstdTbl_cue = cat(2,tmpTbl_cueAligned,tmpTbl_rstd);
    rstdTbl_outcome = rstdTbl_outcome(LMidx,:);
    rstdTbl_cue = rstdTbl_cue(LMidx,:);

    validOutcome = isfinite(rstdTbl_outcome.(responseName)) & isfinite(rstdTbl_outcome.RSTD_PE_pos) & isfinite(rstdTbl_outcome.RSTD_PE_neg) & ...
                   isfinite(rstdTbl_outcome.TrialNumberZ) & isfinite(rstdTbl_outcome.AbsPointsPerTrialZ) & ...
                   ~isundefined(rstdTbl_outcome.TrialColor) & ~isundefined(rstdTbl_outcome.TrialType) & ~isundefined(rstdTbl_outcome.Outcome);
    validCue = isfinite(rstdTbl_cue.(responseName)) & isfinite(rstdTbl_cue.RSTD_VE) & ...
               isfinite(rstdTbl_cue.TrialNumberZ) & ...
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
            TDdataGradients.neuralFit(chz).rstd_OutcomeModel = fitglme(rstdTbl_outcome,[responseName ' ~ RSTD_PE_pos + RSTD_PE_neg + TrialColor + Outcome + TrialNumberZ + AbsPointsPerTrialZ']);
            TDdataGradients.neuralFit(chz).ANOVA_rstd_Outcome = anova(TDdataGradients.neuralFit(chz).rstd_OutcomeModel);
        end
    catch ME
        warning('Best-alpha outcome model failed for %s: %s', responseName, ME.message);
    end
    try
        if height(rstdTbl_outcome_success) >= 5
            TDdataGradients.neuralFit(chz).rstd_OutcomeSuccessModel = fitglme(rstdTbl_outcome_success,[responseName ' ~ RSTD_PE_pos + RSTD_PE_neg + TrialColor + TrialNumberZ + AbsPointsPerTrialZ']);
            TDdataGradients.neuralFit(chz).ANOVA_rstd_OutcomeSuccess = anova(TDdataGradients.neuralFit(chz).rstd_OutcomeSuccessModel);
        end
    catch ME
        warning('Best-alpha outcome success model failed for %s: %s', responseName, ME.message);
    end
    try
        if height(rstdTbl_cue) >= 5
            TDdataGradients.neuralFit(chz).rstd_CueModel = fitglme(rstdTbl_cue,[responseName ' ~ RSTD_VE + TrialColor + TrialNumberZ']);
            TDdataGradients.neuralFit(chz).ANOVA_rstd_Cue = anova(TDdataGradients.neuralFit(chz).rstd_CueModel);
        end
    catch ME
        warning('Best-alpha cue model failed for %s: %s', responseName, ME.message);
    end
    try
        if height(rstdTbl_cue_success) >= 5
            TDdataGradients.neuralFit(chz).rstd_CueSuccessModel = fitglme(rstdTbl_cue_success,[responseName ' ~ RSTD_VE + TrialColor + TrialNumberZ']);
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

        % PDF filename/path. Use the same text as the figure title.
        pdfName = [ptID '_' TDdataGradients.neuralFit(chz).trodeLabel '_' ...
                   TDdataGradients.neuralFit(chz).new_trodeLabel '_RSTD_modelFitLandscapes.pdf'];
        pdfPath = fullfile('\\155.100.91.44\d\Data\Rhiannon\BART_RLDM_outputs\RSTD\RSTD_neuralFits\', pdfName);

        % Put the full figure title exactly the same as the PDF name.
        set(gcf,'Name',pdfName,'NumberTitle','off')
        sgtitle(pdfName,'Interpreter','none')

        % saveas(chz*1000,sprintf('\\155.100.91.44\d\Data\Rhiannon\BART_RLDM_outputs\RSTD\RSTD_neuralFits\pt%s_%s_%s_RSTD_modelFitLandscapes.pdf',ptID,TDdata.neuralFit(chz).trodeLabel, TDdata.neuralFit(chz).new_trodeLabel))
        saveas(chz*1000,pdfPath)
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
