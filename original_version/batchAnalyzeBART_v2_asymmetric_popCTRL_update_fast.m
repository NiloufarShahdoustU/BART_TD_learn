function batchAnalyzeBART_v2_asymmetric_popCTRL_update_fast(analysis)
% BATCHANALYZEBART carries out BART analyses for all patients
%
%   batchAnalyzeBART(analysis,signal) will run the specified analysis, on
%   the signal specified.
%
%   signal can either be 'BHF' or 'loS' which indicate the broadband high
%   frequency signal (70 - 150 Hz) or the spectrogram from 1 - 50 Hz,
%   respectively.
%
%   Available analyses:
%       - 'behavior' - behavior figures and stats
%       - 'balloonSizeLMM' - balloon size LMM
%       - 'postOutcomeBalloonSizeLMM' - post outcome balloon size LMM
%       - 'balloonSizePlots' - balloon size BHF plots
%       - 'visLMMs' - summarize linear models across patients.

% author: EHS20190308
%Edited: RC20220215

close all;

sigBHFEEGcorr = struct();
unitIDs_Pts = {};
clusterTbl = [];
sigContacts_allPts = false(0,4);
moreImpulsive = false(0,1);
MNItrodeLocs_allPTs = [];
trodeLabels_allPts = {''};
anatomicalLocs_allPts = {''};
ptIDs_all = {''};
leftHemiContacts_allPts = [];

noWMpts = {};

[ptArray,bhvStruct,hazEEG] = BARTnumbers;
% looping over patients
%ptArray = {'202003'} %for debugging
nPts = length(ptArray);

%TODO:: removing patients without EEG for EEG batch analyses
set(0,'defaultfigurerenderer','painters')

% Only those patients who have EEG or micros.
pts = 1%:nPts; % UPDATE THIS TO RUN THROUGH ALL PATIENTS

% 26-71 Mark's computer 
% 1-25 Rhiannon's computer
 
% color map
ptMap = cbrewer('qual','Set3',nPts);

% looping over patients.
for pt = fliplr(pts)
    % clearing some variables so they aren't accidentally combined across patients.
    clear sigContacts MNItrodeLocs sEEGverts els els2 ChanMap ElecXYZMNIProj ElecAtlasProjRaw labelsFromBoth

    % which patient?
    ptID = ptArray{pt};

    % which directory
    % BARTdir = (['\\155.100.91.44\D\Data\preProcessed\BART_preprocessed\' ptID '\Data']);
    % bhvStructdir = (['\\155.100.91.44\D\Data\Rhiannon\BART_RLDM_outputs\bhvStruct\' ptID '.bhvStruct.mat'])

    % which variable to analyze.
    if strcmp('behavior',analysis)
        % if you want to plot behavioral analyses again.....
        % [bhvStruct(pt)] = BARTbehavior(ptID,false);

        % if you want to load the bhvStructs that are already plotted....
        % [bhvStruct(pt)] = load((['\\155.100.91.44\D\Data\Rhiannon\BART_RLDM_outputs\bhvStruct\' ptID '.bhvStruct.mat']))

        % some variables used in the across subjects section...
        redMeans(pt) = median([bhvStruct(pt).redPoints]);
        orangeMeans(pt) = median([bhvStruct(pt).orangePoints]);
        yellowMeans(pt) = median([bhvStruct(pt).yellowPoints]);
        redSTD(pt) = std([bhvStruct(pt).redPoints],1);
        orangeSTD(pt) = std([bhvStruct(pt).orangePoints],1);
        yellowSTD(pt) = std([bhvStruct(pt).yellowPoints],1);


        % we wont run this part here. 
    elseif strcmp('TDlearnBHV',analysis)
        % if you want to fit temporal difference learning models.
        % there are some other interesting parameters to study in there,
        % such as learning rates (alphas), and quantile/expectile distRL
        % whichRL = 'risksensitive'; % if running rstd TD
        whichRL = 'vanilla'; % if running standard TD
        [TDdata(pt),bestAlpha(pt),bestAlphaRisk(pt),~] = BART_behavior_TDlearn(ptID,whichRL);

        if strcmp(whichRL,'vanilla')
            % figuring out the optimal learning rate.
            [~,tmpidx] = max(TDdata(pt).inverseTemperature(1:end-10));
            bestAlpha(pt) = TDdata(pt).a(tmpidx);

            % figuring out the optimal learning rate.
            [~,tmpidxR] = max(TDdata(pt).inverseTemperatureRisk(1:end-10));
            bestAlphaRisk(pt) = TDdata(pt).a(tmpidxR);

            % plotting a surface for alpha and beta
            alphaPlot = true;
            if alphaPlot 

                % [X,Y] = meshgrid(TDdata(pt).a,TDdata(pt).inverseTemperature);
                figure(pt)

                subplot(1,2,1)
                hold on
                imagesc(TDdata(pt).a,TDdata(pt).inverseTemperature,TDdata(pt).a'*TDdata(pt).inverseTemperature)
                %             plot(TDdata(pt).a,TDdata(pt).inverseTemperature)
                %             scatter(TDdata(pt).a(tmpidx),TDdata(pt).inverseTemperature(tmpidx),20,["black"])
                hold off
                xlabel('alpha')
                ylabel('beta')
                title('reward')
                axis tight

                subplot(1,2,2)
                hold on
                imagesc(TDdata(pt).a,TDdata(pt).inverseTemperatureRisk,TDdata(pt).a'*TDdata(pt).inverseTemperatureRisk)
                %             scatter(TDdata(pt).a(tmpidxR),TDdata(pt).inverseTemperatureRisk(tmpidxR),20,["black"])
                hold off
                xlabel('alpha')
                ylabel('beta')
                title('risk')
                axis tight
                suptitle('learning rate parameter estimation for:')

                saveas(pt,['\\155.100.91.44\D\Data\Rhiannon\BART_RLDM_outputs\BatchAnalyze_Controls\TDlearnNeural\' ptID '_learningRateParamEstimation_' whichRL 'TDmodel_asymmetricCTRL.pdf'])
                close(pt)
            end
        end

    elseif strcmp('TDlearnNeural',analysis)
        % if you want to fit high gamma responses to the trial by trial
        % estimates of value and RPE from the temporal difference learning
        % models.

        % fitting the temporal difference learning models
        whichTrialsLM = '1:end'; % Look at first 50 trials ("1:50"), last 50 trials ("end - 50:end"), blocks of trials (e.g., "40:80").
        optimalAlphas = false; % true if want best learning rate, false to run gradients

         if optimalAlphas
             [TDdata(pt), ~, ~, rewardAndRiskHGs(pt)] = BART_bankpop_bhf_TDlearn2(ptID,whichTrialsLM,false); % if plt true then this plots high gamma examples.
         elseif ~optimalAlphas
            [TDdataGradients(pt), ~, ~, rewardAndRiskHGs(pt)] = BART_bankpop_bhf_TDlearn_Gradients(ptID,whichTrialsLM,false); % if plt true then this plots high gamma examples.
         end

        % impulsivity classification: KLD yellow balloons.
        riskaversionKLD_yellows = [bhvStruct.impulsivityKLD_yellows];
        riskaversionMetric = -riskaversionKLD_yellows;

        %% ~~~~~~ SAVING VARIABLES ACROSS 'TRODES ~~~~~~~~~~~~~~~~~~~~~~~~~
        % concatenating the significant contact matrices accross subjects
        % [20201105] now only incuding significant contacts closer to
        % gray matter.
        % which variables to save:
        saveDir  = '\\155.100.91.44\D\Data\Rhiannon\BART_RLDM_outputs\BatchAnalyze\20250403fast';
        % moreImpulsive =
        % cat(1,moreImpulsive,repmat(impulsiveChoosers,length(sigContacts),1));c
        %  ptIDs_all = cat(1,ptIDs_all,repmat(ptID,length(logical(leftHemiContacts)),1));
%         if pt == 1
            % saving
%             save(fullfile(saveDir,'TDneuralFits_asymmetricPOPCTRL_RSTD.mat'),'TDdata','rewardAndRiskHGs','-v7.3')
%         end

    end
end

 keyboard % just want to run TDdata not rest of analysis.

% ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
% ^^^^^^^^^^^^^^^^^^^^^^^^^^^ WITHIN SUBJECT ANALYSES ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
% ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~ ACROSS SUBJECT ANALYSES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if strcmp('behavior',analysis)

    for bhvzz=1
        % calculate mean ctrl and passive trials
        for k = 1:length(ptArray)
            passivectrls(k) =sum([bhvStruct(k).nCtrls])-([bhvStruct(k).nGreyCtrls])
        end
        mean(passivectrls);
        std(passivectrls);

        %% across patients behavior
        set(0,'defaultfigurerenderer','painters') % vectors

        % impulsivity classification: KLD yellow balloons.
        riskaversionKLD_yellows = [bhvStruct.impulsivityKLD_yellows]; % get kld fro bhvstruct
        riskaversionKLD_yellows = -(riskaversionKLD_yellows) % make it log negative
        riskaversionMetric = riskaversionKLD_yellows;

        load("\\155.100.91.44\d\Code\Rhiannon\BART\riskSensitivity_TD_bhv.mat") % load RSTDmetric that we derived from task.

        % (1) behavioral descriptive statistics for paper
        mean([bhvStruct(:).totalTrials])
        std([bhvStruct(:).totalTrials])
        length([bhvStruct(:).greyCtrlITs])
        std([bhvStruct(:).nCtrls])
        mean([bhvStruct(:).totalTrials])
        std([bhvStruct(:).totalTrials])

        yA = [bhvStruct.yellowAccuracy];         mean(yA); std(yA)
        oA = [bhvStruct.orangeAccuracy];         mean(oA); std(oA)
        rA = [bhvStruct.redAccuracy];            mean(rA); std(rA)
        tA = [bhvStruct.accuracyTot];            mean(tA); std(tA)

        % Accuracy anova
        X_bColor_Acc = [yA' oA' rA'];
        [pX,tblX,statsX] = anova1(X_bColor_Acc)
        [c,~,~,gnames] = multcompare(statsX)

          % points by color
        for k = 1:length(bhvStruct)
            rP(:,k) = sum([bhvStruct(k).redPoints]);
            oP(:,k) = sum([bhvStruct(k).orangePoints]);
            yP(:,k) = sum([bhvStruct(k).yellowPoints]);
        end

        % Points anova
        X_bColor_Points = [yP' oP' rP'];
        [pX,tblX,statsX] = anova1(X_bColor_Points)
        [c,~,~,gnames] = multcompare(statsX)
  
keyboard
        % Figure 1: Temporal Difference Paper. Behavior.
        figure(1)
        BOXPLOT = true;

        % colors for MI and LI individuals
        sage_green = [119, 171, 86] / 255; % RGB values for sage green ICs
        muted_purple = [79, 49, 170] / 255; % RGB values for muted purple ~ICs

        % plotting impulsivity metric
        subplot(3,3,1)
        hold on
        line([0 nPts],[median(riskaversionKLD_yellows) median(riskaversionKLD_yellows)],'linestyle','-','color',rgb('indigo'))
        text(nPts-10,median(riskaversionKLD_yellows)+1,'median split','color',rgb('indigo'))
        scatter(find(medSplit),riskaversionKLD_yellows(medSplit),40,sage_green,'filled','^')
        scatter(find(~medSplit),riskaversionKLD_yellows(~medSplit),40,muted_purple,'filled')
        hold off
        xlim([0 nPts]) %from .7
        ylabel('impulsivityKLD_yellows', 'fontsize', 12)
        %yticks(1:1:7)
        %ylim([1 7]) % for apZVals
        xlabel('patient number')
        %title('impulsivity: - low: o  - high: ^')
        axis square

        % histogram of impulsivity metric (impulsivityKLD_yellows)
        subplot(3,3,2)
        hold on
        Hm = histfit((riskaversionKLD_yellows(impulsiveChoosers)),8);
        Hl = histfit((riskaversionKLD_yellows(~impulsiveChoosers)),8);
        Hm(1).FaceColor = sage_green;
        Hm(1).EdgeColor = sage_green;
        Hm(1).LineWidth = 1.5;
        Hm(1).FaceAlpha = 0.4;
        Hm(2).Color = sage_green;
        Hl(1).FaceColor = muted_purple;
        Hl(1).EdgeColor = muted_purple;
        Hl(1).LineWidth = 1.5;
        Hl(1).FaceAlpha = 0.4;
        Hl(2).Color = muted_purple;
        hold off
        view([90 -90])
        xlim([0.5 7.5])
        xticks(0.5:1:7.5)
        ylim([0 11])
        ylabel('subject count', 'fontsize', 12)
        axis square

        subplot(3,3,3)
        hold on
        if BOXPLOT
            betterBoxplot(1,yA,rgb('gold'),40,'o',1,false)
            %betterBoxplot(1.5,yA(~medSplit),rgb('gold'),20,'o',1,false)
            betterBoxplot(2,oA,rgb('orangered'),40,'o',1,false)
            %   betterBoxplot(2.5,oA(~medSplit),rgb('orangered'),20,'o',1,false)
            betterBoxplot(3,rA,rgb('red'),40,'o',1,false)
            % betterBoxplot(3.5,rA(~medSplit),rgb('red'),20,'o',1,false)
        else
            scatter(ones(1,sum(medSplit)),yA(medSplit),20,rgb('gold'),'filled','^')
            %  scatter(1.3*ones(1,sum(~medSplit)),yA(~medSplit),10,rgb('gold'),'filled')
            scatter(2*ones(1,sum(medSplit)),oA(medSplit),10,rgb('orangered'),'filled','^')
            % scatter(2.3*ones(1,sum(~medSplit)),oA(~medSplit),10,rgb('orangered'),'filled')
            scatter(3*ones(1,sum(medSplit)),rA(medSplit),10,rgb('red'),'filled','^')
            % scatter(3.3*ones(1,sum(~medSplit)),rA(~medSplit),10,rgb('red'),'filled')
        end
        hold off
        xticks([1 2 3])
        set(gca,'XTickLabel',{'Y','O','R'})
        xlim([0.5 3.5])
        ylabel('balloon accuracy (%)', 'fontsize', 12)
        ylim([20 110])
        axis square

        % plot point accumulation
        % accuracy grouped by impulsivtiy metrics
        fP = [bhvStruct.freePoints];
        cP = [bhvStruct.ctrlPoints];
        aTs = [bhvStruct.nActiveTrials];
        fITs = [bhvStruct.freeITs];
        fITs = [bhvStruct.freeITs];

        for p = 1:length(bhvStruct)
            yITs(p) = mean([bhvStruct(p).yellowITs]);
        end

        mean(yITs(medSplit))
        mean(yITs(~medSplit))

        [pointPactiveTrials_Red,pointHactiveTrials_Red,pointSTATSactiveTrials_Red] = ranksum(rP(medSplit)./aTs(medSplit),rP(~medSplit)./aTs(~medSplit))
        [pointPactiveTrials_Red,pointHactiveTrials_Red,pointSTATSactiveTrials_Red] = ranksum(rP(medSplit),rP(~medSplit))
        [pointPactiveTrials_Orange,pointHactiveTrials_Orange,pointSTATSactiveTrials_Orange] = ranksum(oP(medSplit)./aTs(medSplit),oP(~medSplit)./aTs(~medSplit))
        [pointPactiveTrials_Orange,pointHactiveTrials_Orange,pointSTATSactiveTrials_Orange] = ranksum(oP(medSplit),oP(~medSplit))
        [pointPactiveTrials_Yellow,pointHactiveTrials_Yellow,pointSTATSactiveTrials_Yellow] = ranksum(yP(medSplit)./aTs(medSplit),yP(~medSplit)./aTs(~medSplit))
        [pointPactiveTrials_Yellow,pointHactiveTrials_Yellow,pointSTATSactiveTrials_Yellow] = ranksum(yP(medSplit),yP(~medSplit))

        [P_fITs,H_fITs,STATS_fITs] = ranksum(fITs(medSplit),fITs(~medSplit)) % free ITs between MI/LI choosers

        subplot(3,3,4)
        hold on
        betterBoxplot(1,yP./aTs, rgb('gold'),40,'^',1,false)
        % betterBoxplot(1.5,yP(~medSplit)./aTs(~medSplit), rgb('gold'),40,'o',1,false)
        betterBoxplot(2.0,oP./aTs, rgb('orangered'),40,'^',1,false)
        % betterBoxplot(2.5,oP(~medSplit)./aTs(~medSplit), rgb('orangered'),40,'o',1,false)
        betterBoxplot(3.0,rP./aTs, rgb('red'),40,'^',1,false)
        % betterBoxplot(3.5,rP(~medSplit)./aTs(~medSplit), rgb('red'),40,'o',1,false)
        hold off
        xticks([1 2 3])
        xlim([0.5 3.5])
        ylim([1000 9000])
        ylabel('total points per active trial')
        xticks([1 2 3])
        set(gca,'XTickLabel',{'Y','O','R'})
        %title(sprintf('ranksum, U(%d,%d) = %.2f, p = %.2f',sum(medSplit),sum(~medSplit),pointSTATS.ranksum,pointP))
        axis square

        % for KLD metric
        % [pointPactiveTrials,pointHactiveTrials,pointSTATSactiveTrials] = ranksum(fP(medSplit)./aTs(medSplit),fP(~medSplit)./aTs(~medSplit))
        % [pointP,pointH,pointSTATS] = ranksum(fP(medSplit),fP(~medSplit))

        % for impulsivityKLD_yellows metric
        % [pointPactiveTrials_z,pointHactiveTrials_z,pointSTATSactiveTrials_z] = ranksum(fP(medSplit)./aTs(medSplit),fP(~medSplit)./aTs(~medSplit))
        % [pointP_z,pointH_z,pointSTATS_z] = ranksum(fP(medSplit),fP(~medSplit))

        % Total Points that we arent plotting in fig 1 anymore.
        subplot(3,3,5)
        hold on
        if BOXPLOT
            betterBoxplot(0.5,fP(medSplit)./aTs(medSplit), rgb('black'),20,'^',1,false)
            betterBoxplot(1.5,fP(~medSplit)./aTs(~medSplit), rgb('black'),20,'o',1,false)
        else
            scatter(0.5*ones(1,sum(medSplit)),fP(medSplit)./aTs(medSplit),20,rgb('black'),'filled','^')
            scatter(1.5*ones(1,sum(~medSplit)),fP(~medSplit)./aTs(~medSplit),20,rgb('black'),'filled')
        end
        hold off
        set(gca,'XTickLabel',{'','more impulsive','','less impulsive',''})
        xlim([0 2])
        ylim([5000 12000])
        ylabel('total points per active trial')
        xticks([.5 1.5])
        set(gca,'XTickLabel',{'MI', 'LI'})
        %title(sprintf('ranksum, U(%d,%d) = %.2f, p = %.2f',sum(medSplit),sum(~medSplit),pointSTATS.ranksum,pointP))
        axis square

        % linear model for rstd, and KLD
        subplot(3,3,6)
        fprintf('\nKLD_Y vs. rstd metric\n')
        lm = fitlm(riskaversionMetric,RSTDmetric);
        h = lm.plot;
        h(1).Marker = '.';
        h(1).MarkerEdgeColor = 'k';
        h(1).MarkerSize = 15;
        xlabel('impulsivity Metric')
        ylabel('RSTD metric')
        subtitle(sprintf('p = %2f', lm.ModelFitVsNullModel.Pvalue))
        legend off
        axis tight square

        % any correlations between accuracy and rstd across subs??
        fprintf('\nRSTD_metric vs. total accuracy\n')
        lm = fitlm(RSTDmetric,tA)
        subplot(3,3,7)
        h = lm.plot;
        h(1).Marker = '.';
        h(1).MarkerEdgeColor = 'k';
        h(1).MarkerSize = 15;
        xlabel('RSTD_metric')
        ylabel('total accuracy')
        subtitle(sprintf('p = %2f', lm.ModelFitVsNullModel.Pvalue))
        legend off
        axis tight square

        % any correlations between points and impulsivity across subs??
        fprintf('\nKLD_Y vs. points per trial\n')
        lm = fitlm(riskaversionMetric,fP./aTs)
        subplot(3,3,8)
        h = lm.plot;
        h(1).Marker = '.';
        h(1).MarkerEdgeColor = 'k';
        h(1).MarkerSize = 15;
        xlabel('impulsivity Metric')
        ylabel('points per active trial')
        subtitle(sprintf('p = %2f', lm.ModelFitVsNullModel.Pvalue))
        legend off
        axis tight square

        % any correlations between accuracy and impulsivity across subs??
        fprintf('\nRSTDmetric vs. points per trial\n')
        lm = fitlm(RSTDmetric,fP./aTs)
        subplot(3,3,5)
        h = lm.plot;
        h(1).Marker = '.';
        h(1).MarkerEdgeColor = 'k';
        h(1).MarkerSize = 15;
        xlabel('RSTDmetric')
        ylabel('points per active trial')
        subtitle(sprintf('p = %2f', lm.ModelFitVsNullModel.Pvalue))
        legend off
        axis tight square

        % any correlations between accuracy and impulsivity across subs??
        fprintf('\nactive/total accuracy vs. points per trial\n')
        lm = fitlm(tA,fP./aTs)
        subplot(3,3,9)
        h = lm.plot;
        h(1).Marker = '.';
        h(1).MarkerEdgeColor = 'k';
        h(1).MarkerSize = 15;
        xlabel('total accuracy')
        ylabel('points per active trial')
        subtitle(sprintf('p = %2f', lm.ModelFitVsNullModel.Pvalue))
        legend off
        axis tight square
        maximize(1)

%         % calcualting impulsivity based on point/accuracy tradeoff.
%         coeffs = polyfit(tA, fP./aTs, 1);
%         y = polyval(coeffs,fP./aTs)
%         [list_metric, line_rank] = sort(y)
%         yP = [bhvStruct.yellowPoints];
%         yITs = [bhvStruct.yellowITs]; yellowITs
%         metric =  (tA + fP./aTs)/2
%         metric =  (yA + yP./yITs)/2

        % average x predicted y predict

%         m = coeffs(1);
%         b = coeffs(2);
%         % Calculate distances
%         distances = abs(m * tA - fP./aTs + b) / sqrt(m^2 + 1);
%         % Example metric: weighted sum of x, y, and distance
%         alpha = 0.3;  % weight for x
%         beta = 0.3;   % weight for y
%         gamma = 0.4;  % weight for distance
%         metric = alpha * tA + beta * fP./aTs + gamma * distances;


        % saving behavioral figure
        halfMaximize(1,'left')

        saveas(1,'\\155.100.91.44\D\Data\Rhiannon\BART_RLDM_outputs\BatchAnalyze_Controls\Behavior\BARTacrossPtBehaviorCTRL_regs.pdf') %added new

        % SUPPLEMENTARY FIGURE
        % color points and accuracy by yellowKLD
        figure(2222)
        %red points
        subplot(3,3,1)
        lm = fitlm(riskaversionKLD_yellows,rP) % NS
        h = lm.plot;
        h(1).Marker = '.';
        h(1).MarkerEdgeColor = 'k';
        h(1).MarkerSize = 15;
        xlabel('impulsivity metric')
        ylabel('red points')
        subtitle(sprintf('p = %.6f', lm.ModelFitVsNullModel.Pvalue))
        legend off
        axis tight square

        % orange points
        subplot(3,3,2)
        lm = fitlm(riskaversionKLD_yellows,oP) % NS
        h = lm.plot;
        h(1).Marker = '.';
        h(1).MarkerEdgeColor = 'k';
        h(1).MarkerSize = 15;
        xlabel('impulsivity metric')
        ylabel('orange points')
        subtitle(sprintf('p = %.6f', lm.ModelFitVsNullModel.Pvalue))
        legend off
        axis tight square

        % yellow points
        subplot(3,3,3)
        lm = fitlm(riskaversionKLD_yellows,yP) % NS
        h = lm.plot;
        h(1).Marker = '.';
        h(1).MarkerEdgeColor = 'k';
        h(1).MarkerSize = 15;
        xlabel('impulsivity metric')
        ylabel('yellow points')
        subtitle(sprintf('p = %.6f', lm.ModelFitVsNullModel.Pvalue))
        legend off
        axis tight square

        % red accuracy
        subplot(3,3,4)
        lm = fitlm(riskaversionKLD_yellows,rA) % NS
        h = lm.plot;
        h(1).Marker = '.';
        h(1).MarkerEdgeColor = 'k';
        h(1).MarkerSize = 15;
        xlabel('impulsivity metric')
        ylabel('red accuracy (%)')
        subtitle(sprintf('p = %.6f', lm.ModelFitVsNullModel.Pvalue))
        legend off
        axis tight square
        ax3 = gca;
        ylim(ax3, [30 100]);
        yticks(ax3, 30:10:100);

        % orange accuracy
        subplot(3,3,5)
        lm = fitlm(riskaversionKLD_yellows,oA) % NS
        h = lm.plot;
        h(1).Marker = '.';
        h(1).MarkerEdgeColor = 'k';
        h(1).MarkerSize = 15;
        xlabel('impulsivity metric')
        ylabel('orange accuracy (%)')
        subtitle(sprintf('p = %.6f', lm.ModelFitVsNullModel.Pvalue))
        legend off
        axis tight square
        ax3 = gca;
        ylim(ax3, [30 100]);
        yticks(ax3, 30:10:100);

        % yellow accuracy
        subplot(3,3,6)
        lm = fitlm(riskaversionKLD_yellows,yA) % NS
        h = lm.plot;
        h(1).Marker = '.';
        h(1).MarkerEdgeColor = 'k';
        h(1).MarkerSize = 15;
        xlabel('impulsivity metric')
        ylabel('yellow accuracy (%)')
        subtitle(sprintf('p = %.6f', lm.ModelFitVsNullModel.Pvalue))
        legend off
        axis tight square
        ax3 = gca;
        ylim(ax3, [30 100]);
        yticks(ax3, 30:10:100);

        %red accuracy vs points
        subplot(3,3,7)
        lm = fitlm(rA,rP) % NS
        h = lm.plot;
        h(1).Marker = '.';
        h(1).MarkerEdgeColor = 'k';
        h(1).MarkerSize = 15;
        xlabel('red accuracy (%)')
        ylabel('red points')
        subtitle(sprintf('p = %.6f', lm.ModelFitVsNullModel.Pvalue))
        legend off
        axis tight square

        % orange accuracy vs points
        subplot(3,3,8)
        lm = fitlm(oA,oP) % NS
        h = lm.plot;
        h(1).Marker = '.';
        h(1).MarkerEdgeColor = 'k';
        h(1).MarkerSize = 15;
        xlabel('orange accuracy (%)')
        ylabel('orange points')
        subtitle(sprintf('p = %.6f', lm.ModelFitVsNullModel.Pvalue))
        legend off
        axis tight square

        % yellow accuracy vs points
        subplot(3,3,9)
        lm = fitlm(yA,yP) % NS
        h = lm.plot;
        h(1).Marker = '.';
        h(1).MarkerEdgeColor = 'k';
        h(1).MarkerSize = 15;
        xlabel('yellow accuracy (%)')
        ylabel('yellow points')
        subtitle(sprintf('p = %.6f', lm.ModelFitVsNullModel.Pvalue))
        legend off
        axis tight square
        
        saveas(2222,'\\155.100.91.44\D\Data\Rhiannon\BART_RLDM_outputs\BatchAnalyze_Controls\Behavior\regressions_acc_points_IM_allPts.pdf')



        %% Additional regression models
        fprintf('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n LINEAR MODELS FOR IMPULSIVITY METRIC PREDICTIONS OF TASK ELEMENTS \n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
        figure(2)
        % any correlations between accuracy and impulsivity across subs??
        fprintf('\ntotal accuracy vs. Ymetric\n')
        lm = fitlm(riskaversionKLD_yellows,tA)
        lm = fitlm(riskaversionKLD_yellows,yA)

        subplot(2,2,1)
        h = lm.plot;
        h(1).Marker = '.';
        h(1).MarkerEdgeColor = 'k';
        h(1).MarkerSize = 15;
        xlabel('impulsivity metric')
        ylabel('total accuracy')
        subtitle(sprintf('p = %s', lm.ModelFitVsNullModel.Pvalue))
        legend off
        axis tight square

        % any correlations between accuracy and impulsivity across subs??
        fprintf('\npoints per trial vs. impulsivty metric\n')
        lm = fitlm(riskaversionKLD_yellows,fP./aTs)
        subplot(2,2,2)
        h = lm.plot;
        h(1).Marker = '.';
        h(1).MarkerEdgeColor = 'k';
        h(1).MarkerSize = 15;
        xlabel('impulsivityKLD_yellows')
        ylabel('points per active trial')
        subtitle(sprintf('p = %s', lm.ModelFitVsNullModel.Pvalue))
        legend off
        axis tight square

        % any correlations between accuracy and impulsivity across subs??
        fprintf('\ntotal points vs. impulsivty metric\n')
        lm = fitlm(riskaversionKLD_yellows,fP)
        subplot(2,2,4)
        h = lm.plot;
        h(1).Marker = '.';
        h(1).MarkerEdgeColor = 'k';
        h(1).MarkerSize = 15;
        xlabel('impulsivityKLD_yellows')
        ylabel('total points')
        subtitle(sprintf('p = %s', lm.ModelFitVsNullModel.Pvalue))
        legend off
        axis tight square

        % any correlations between accuracy and impulsivity across subs??
        fprintf('\ntotal points vs. impulsivty metric\n')
        lm = fitlm(RSTDmetric,fP)
        subplot(2,2,1)
        h = lm.plot;
        h(1).Marker = '.';
        h(1).MarkerEdgeColor = 'k';
        h(1).MarkerSize = 15;
        xlabel('rstdMetric')
        ylabel('total points')
        subtitle(sprintf('p = %s', lm.ModelFitVsNullModel.Pvalue))
        legend off
        axis tight square

        % any correlations between accuracy and impulsivity across subs??
        fprintf('\ntotal accuracy vs. points per trial\n')
        lm = fitlm(tA,fP./aTs);
        subplot(2,2,3)
        h = lm.plot;
        h(1).Marker = '.';
        h(1).MarkerEdgeColor = 'k';
        h(1).MarkerSize = 15;
        xlabel('total accuracy')
        ylabel('points per active trial')
        subtitle(sprintf('p = %s', lm.ModelFitVsNullModel.Pvalue))
        legend off
        axis tight square
        maximize(2)
        saveas(2,'\\155.100.91.44\D\Data\Rhiannon\BART_RLDM_outputs\BatchAnalyze_Controls\Behavior\regressions_YMetricAndTaskPerformancePredictions_allPts.pdf')

        % additional models for paper
        %    lm = fitlm(impulsivityMetric,apZs) % IM by PA_diff
        %   lm = fitlm(tA,apZs) % accuracy by PA_diff
        %   lm = fitlm(apZs,fP./aTs) % activeTrialPoints by PA_diff

        % summarizing balloon distributions across subjects, sorted by
        % impulsivity metric.
        [~,sortedImpulsivityIdcs] = sort(riskaversionMetric);
        figure
        hold on
        % starting with the error
        scatter(sortedImpulsivityIdcs,redMeans-redSTD,5,rgb('red'),'+');
        scatter(sortedImpulsivityIdcs,orangeMeans-orangeSTD,5,rgb('orangered'),'+');
        scatter(sortedImpulsivityIdcs,yellowMeans-yellowSTD,5,rgb('gold'),'+');

        scatter(sortedImpulsivityIdcs,redMeans+redSTD,5,rgb('red'),'+');
        scatter(sortedImpulsivityIdcs,orangeMeans+orangeSTD,5,rgb('orangered'),'+');
        scatter(sortedImpulsivityIdcs,yellowMeans+yellowSTD,5,rgb('gold'),'+');

        % and the means for each group.
        scatter(sortedImpulsivityIdcs(medSplit),redMeans(medSplit),10,rgb('red'),'filled','^');
        scatter(sortedImpulsivityIdcs(medSplit),orangeMeans(medSplit),10,rgb('orangered'),'filled','^');
        scatter(sortedImpulsivityIdcs(medSplit),yellowMeans(medSplit),10,rgb('gold'),'filled','^');
        scatter(sortedImpulsivityIdcs(~medSplit),redMeans(~medSplit),10,rgb('red'),'filled');
        scatter(sortedImpulsivityIdcs(~medSplit),orangeMeans(~medSplit),10,rgb('orangered'),'filled');
        scatter(sortedImpulsivityIdcs(~medSplit),yellowMeans(~medSplit),10,rgb('gold'),'filled');

        hold off
        saveas(gcf,'\\155.100.91.44\D\Data\Rhiannon\BART_RLDM_outputs\BatchAnalyze_Controls\Behavior\balloonPointSummariesAllPts_71.pdf')


        %% Regression models looking at color zVals (This doesnt work now.. because IM is the apZVals)

        tmps_R = [bhvStruct.activeVsPassiveStats_R];
        apZs_R = [tmps_R.zVal];
        tmps_O = [bhvStruct.activeVsPassiveStats_O];
        apZs_O = [tmps_O.zVal];
        tmps_Y = [bhvStruct.activeVsPassiveStats_Y];
        apZs_Y = [tmps_Y.zVal];

        figure(3)
        fprintf('\nzVals Red vs. impulsivty metric\n')
        lm = fitlm(apZs_R,riskaversionMetric)
        subplot(2,2,1)
        h = lm.plot;
        h(1).Marker = '.';
        h(1).MarkerEdgeColor = 'k';
        h(1).MarkerSize = 15;
        xlabel('zVals active red')
        ylabel('impulsivity metric')
        subtitle(sprintf('p = %s', lm.ModelFitVsNullModel.Pvalue))
        legend off
        axis tight square

        fprintf('\nzVals Orange vs. impulsivty metric\n')
        lm = fitlm(apZs_O,riskaversionMetric)
        subplot(2,2,2)
        h = lm.plot;
        h(1).Marker = '.';
        h(1).MarkerEdgeColor = 'k';
        h(1).MarkerSize = 15;
        xlabel('zVals active orange')
        ylabel('impulsivity metric')
        subtitle(sprintf('p = %s', lm.ModelFitVsNullModel.Pvalue))
        legend off
        axis tight square

        fprintf('\nzVals Yellow vs. impulsivty metric\n')
        lm = fitlm(apZs_Y,riskaversionMetric)
        subplot(2,2,3)
        h = lm.plot;
        h(1).Marker = '.';
        h(1).MarkerEdgeColor = 'k';
        h(1).MarkerSize = 15;
        xlabel('zVals active yellow')
        ylabel('impulsivity metric')
        subtitle(sprintf('p = %s', lm.ModelFitVsNullModel.Pvalue))
        legend off
        axis tight square

        saveas(3,'\\155.100.91.44\D\Data\Rhiannon\BART_RLDM_outputs\BatchAnalyze_Controls\Behavior\regression_zVals_TrialType.pdf')

        % Plotting Optimal Performance
        for k = 1:length(bhvStruct)
            passiveCtrlITs(:,k) = mean([bhvStruct(k).passiveCtrlITs]); % mean ITs for each color by pt
            redPassiveCtrlITs(:,k) = mean([bhvStruct(k).redPassiveCtrlITs]);
            orangePassiveCtrlITs(:,k) = mean([bhvStruct(k).orangePassiveCtrlITs]);
            yellowPassiveCtrlITs(:,k) = mean([bhvStruct(k).yellowPassiveCtrlITs]);
            redbankedITs(:,k) = mean([bhvStruct(k).redbankedITs]); % banked active trials by color.
            orangebankedITs(:,k) = mean([bhvStruct(k).orangebankedITs]);
            yellowbankedITs(:,k) = mean([bhvStruct(k).yellowbankedITs]);
        end

        % (1) Calculate outer layers of circle (mean overall)
        mean_redPassiveCtrlITs = mean(redPassiveCtrlITs);
        mean_orangePassiveCtrlITs = mean(orangePassiveCtrlITs);
        mean_yellowPassiveCtrlITs = mean(yellowPassiveCtrlITs);
        % (2) Calculate inner layer of circle (mean MI) of active responses
        meanMI_redbankedITs = mean(redbankedITs(impulsiveChoosers));
        meanMI_orangebankedITs = mean(orangebankedITs(impulsiveChoosers));
        meanMI_yellowbankedITs = mean(yellowbankedITs(impulsiveChoosers));
        % (3) Calculate inner layer of circle (mean LI)
        meanLI_redbankedITs = mean(redbankedITs(~impulsiveChoosers));
        meanLI_orangebankedITs = mean(orangebankedITs(~impulsiveChoosers));
        meanLI_yellowbankedITs = mean(yellowbankedITs(~impulsiveChoosers));

        % make data relative for visualization

        % Relative Data
        mean_redPassiveCtrlITs_relative = mean_redPassiveCtrlITs/mean_redPassiveCtrlITs*100; % Optimal Red
        meanMI_redPassiveCtrlITs_relative  = meanMI_redbankedITs/mean_redPassiveCtrlITs*100; % MI Red
        meanLI_redPassiveCtrlITs_relative =  meanLI_redbankedITs/mean_redPassiveCtrlITs*100; % LI Red
        mean_orangePassiveCtrlITs_relative = mean_orangePassiveCtrlITs/mean_orangePassiveCtrlITs*100; % Optimal Orange
        meanMI_orangePassiveCtrlITs_relative  = meanMI_orangebankedITs/mean_orangePassiveCtrlITs*100; % MI Orange
        meanLI_orangePassiveCtrlITs_relative =  meanLI_orangebankedITs/mean_orangePassiveCtrlITs*100; % LI Orange
        mean_yellowPassiveCtrlITs_relative = mean_yellowPassiveCtrlITs/mean_yellowPassiveCtrlITs*100; % Optimal Yellow
        meanMI_yellowPassiveCtrlITs_relative  = meanMI_yellowbankedITs/mean_yellowPassiveCtrlITs*100; % MI Orange
        meanLI_yellowPassiveCtrlITs_relative =  meanLI_yellowbankedITs/mean_yellowPassiveCtrlITs*100; % LI Orange

        % Create the data array
        data = [
            mean_redPassiveCtrlITs_relative, meanMI_redPassiveCtrlITs_relative, meanLI_redPassiveCtrlITs_relative;
            mean_orangePassiveCtrlITs_relative, meanMI_orangePassiveCtrlITs_relative, meanLI_orangePassiveCtrlITs_relative;
            mean_yellowPassiveCtrlITs_relative, meanMI_yellowPassiveCtrlITs_relative, meanLI_yellowPassiveCtrlITs_relative];

        labels = {'Red Balloons', 'Orange Balloons', 'Yellow Balloons'};
        b = bar(data);
        % Get the number of groups (bars)
        numGroups = size(data, 1);
        colors = [0.68, 0.85, 0.90; sage_green;muted_purple];
        for k = 1:numGroups
            b(k).FaceColor = colors(k, :);
        end
        set(gca, 'xticklabel', labels);
        xlabel('Balloon Color');
        ylabel('Relative Performance (%)');
        title('Relative Performance by Balloon Color and Impulsivity Group');
        legend('Optimal Performance', 'MI Performance', 'LI Performance', 'Location', 'Best');
        saveas(3,'\\155.100.91.44\D\Data\Rhiannon\BART_RLDM_outputs\BatchAnalyze_Controls\Behavior\optimalPerformanceByImpulsivity.pdf')

        % statistically better performance? In Yellow....
        [optimalRed_P,optimalRed_H,optimalRed_STATS] = ranksum(redbankedITs(medSplit),redbankedITs(~medSplit))
        [optimalOrange_P,optimalOrange_H,optimalOrange_STATS] = ranksum(orangebankedITs(medSplit),orangebankedITs(~medSplit))
        [optimalYellow_P,optimalYellow_H,optimalYellow_STATS] = ranksum(yellowbankedITs(medSplit),yellowbankedITs(~medSplit))

        % Plotting trajectories from optimal Performance (cumsum)

        for k = 1:length(bhvStruct)
            optimalPerformanceIndex_Red(k,:) = mean([bhvStruct(k).optimalPerformanceIndex_Red]);
            optimalPerformanceIndex_Orange(k,:) = mean([bhvStruct(k).optimalPerformanceIndex_Orange]);
            optimalPerformanceIndex_Yellow(k,:) = mean([bhvStruct(k).optimalPerformanceIndex_Yellow]);
            cumulativeRedIT(k,:) = mean([bhvStruct(k).cumulativeRedIT]);
            cumulativeOrangeIT(k,:) = mean([bhvStruct(k).cumulativeOrangeIT]);
            cumulativeYellowIT(k,:) = mean([bhvStruct(k).cumulativeYellowIT]);
        end

        % means
        mean_optimalPerformanceIndex_Red = mean(optimalPerformanceIndex_Red) % is this right? Seems low for yellow
        mean_optimalPerformanceIndex_Orange = mean(optimalPerformanceIndex_Orange)
        mean_optimalPerformanceIndex_Yellow = mean(optimalPerformanceIndex_Yellow)

        cumulativeRedIT_MI = mean(cumulativeRedIT(impulsiveChoosers));
        cumulativeRedIT_LI = mean(cumulativeRedIT(~impulsiveChoosers));
        cumulativeOrangeIT_MI = mean(cumulativeOrangeIT(impulsiveChoosers));
        cumulativeOrangeIT_LI = mean(cumulativeOrangeIT(~impulsiveChoosers));
        cumulativeYellowIT_MI = mean(cumulativeYellowIT(impulsiveChoosers));
        cumulativeYellowIT_LI = mean(cumulativeYellowIT(~impulsiveChoosers));

        % Create the data array
        cumsum_data = [mean_optimalPerformanceIndex_Red, cumulativeRedIT_MI, cumulativeRedIT_LI;...
            mean_optimalPerformanceIndex_Orange, cumulativeOrangeIT_MI, cumulativeOrangeIT_LI;...
            mean_optimalPerformanceIndex_Yellow,cumulativeYellowIT_MI, cumulativeYellowIT_LI];

        labels = {'Red Balloons', 'Orange Balloons', 'Yellow Balloons'};
        b = bar(cumsum_data);
        % Get the number of groups (bars)
        numGroups = size(cumsum_data, 1);
        colors = [0.68, 0.85, 0.90; sage_green;muted_purple];
        for k = 1:numGroups
            b(k).FaceColor = colors(k, :);
        end
        set(gca, 'xticklabel', labels);
        xlabel('Balloon Color');
        ylabel('Cumulative Performance (%)');
        title('Relative Performance by Balloon Color and Impulsivity Group');
        legend('Optimal Performance', 'MI Performance', 'LI Performance', 'Location', 'Best');
    end

elseif strcmp('TDlearnBHV',analysis)

    for tdbhvzzz=1
        %% KLD impulsivity definition
        % across subject analysis of BART behavior with asymmetric learning rates.
        % median split in KLD metric
        riskaversionMetric = ([bhvStruct(:).impulsivityKLD_yellows]);
        riskaversionMetric = -riskaversionMetric;
        
        %         % Gaussian mixture modeling on KLD metric.
%         GM = fitgmdist(impulsivityMetric',2);
%         idcs = cluster(GM,impulsivityMetric');
%         tmp = logical(idcs-1);

        % [20220627EHS] since the GMM doens't classify in order of means, we
        % have to do that here...
        muA = mean(riskaversionMetric(tmp));
        muB = mean(riskaversionMetric(~tmp));
        if muA>muB
            impulsiveChoosers = tmp;
        else
            impulsiveChoosers = ~tmp;
        end
        medSplit = impulsiveChoosers;

        %% plotting
        % plotting TD learning rates by risk aversion or impulsivity metric.
        if strcmp(whichRL,'risksensitive')
            % comparing risk sensistivity measures vs RSTD models.
            
            % do this for gradient alphas
            for pz = 1:length(TDdata)
                RSTDmetric(pz) = (TDdata(pz).a(TDdata(pz).negativeBetaMaxIdx)-TDdata(pz).a(TDdata(pz).positiveBetaMaxIdx))/(TDdata(pz).a(TDdata(pz).positiveBetaMaxIdx)+TDdata(pz).a(TDdata(pz).negativeBetaMaxIdx));
                negAlpha(pz) = TDdata(pz).a(TDdata(pz).negativeBetaMaxIdx);
                posAlpha(pz) = TDdata(pz).a(TDdata(pz).positiveBetaMaxIdx);
            end
            
            % linear regression
            figure(222)

            subplot(2,2,2)
            lm = fitlm(riskaversionMetric,RSTDmetric)
            h = lm.plot;
            h(1).Marker = '.';
            h(1).MarkerEdgeColor = 'k';
            h(1).MarkerSize = 15;
            xlabel('risk aversion')
            ylabel('risk sensitivity')
            subtitle(sprintf('p = %2f', lm.ModelFitVsNullModel.Pvalue))
            legend off
            axis tight square
            ax = gca;
            ylim(ax, [-1 1]);
            yticks(ax, -1:0.2:1);
           
            % regression plot
            subplot(2,2,3)
            lm = fitlm(riskaversionMetric,negAlpha)
            h = lm.plot;
            h(1).Marker = '.';
            h(1).MarkerEdgeColor = 'k';
            h(1).MarkerSize = 15;
            xlabel('impulsivity metric')
            ylabel('negative Alphas')
            subtitle(sprintf('p = %2f', lm.ModelFitVsNullModel.Pvalue))
            legend off
            axis tight square
            ax2 = gca;
            ylim(ax2, [0 1]);  
            yticks(ax2, 0:0.1:1); 

            subplot(2,2,4)
            lm = fitlm(riskaversionMetric,posAlpha)
            h = lm.plot;
            h(1).Marker = '.';
            h(1).MarkerEdgeColor = 'k';
            h(1).MarkerSize = 15;
            xlabel('yellow balloon impulsivity metric -log10(KLD(active||passive))')
            ylabel('positive Alphas')
            subtitle(sprintf('p = %2f', lm.ModelFitVsNullModel.Pvalue))
            legend off
            axis tight square
            ax3 = gca;
            ylim(ax3, [0 1]);
            yticks(ax3, 0:0.1:1);

            saveas(222, '//155.100.91.44/d/Data/Rhiannon/BART_RLDM_outputs/TDlearn/trstdOptimalAlphas_btwnICs.pdf')

            figure(3)
            [p,h,stats] = ranksum(negAlpha,posAlpha) % sig
            [p,h,stats] = ranksum(bestAlphaRewardTD,bestAlphaRiskTD) % sig
            betterBoxplot(1,negAlpha)
            betterBoxplot(2,posAlpha)
            betterBoxplot(3,bestAlphaRewardTD)
            betterBoxplot(4,bestAlphaRiskTD)
            set(gca, 'XTick', [1 2 3 4], 'XTickLabel', {'negative', 'positive', 'reward', 'risk'})
            %xlabel('Group')
            ylabel('optimal alpha value')
            xlim([0.5 4.5])
            
            axis square
            dots = findall(gca, 'Type', 'Line', 'Marker', '.');  % Finds all dot markers
            set(dots, 'MarkerSize', 70)
            saveas(3, '//155.100.91.44/d/Data/Rhiannon/BART_RLDM_outputs/TDlearn/OptimalAlphas_allModels.pdf')
            close(3);

            [pNeg,H,stats] = ranksum(negAlpha(impulsiveChoosers),negAlpha(~impulsiveChoosers)) % NS
            [pPos,HPos,statsPos] = ranksum(posAlpha(impulsiveChoosers),posAlpha(~impulsiveChoosers)) % NS
            [pRSTD,H,stats] = signrank(negAlpha,posAlpha) % sig diff between MI positive and negative alphas
            [pRSTDLI,HPos,statsPos] = signrank(negAlpha(~impulsiveChoosers),posAlpha(~impulsiveChoosers)) % NS between LI positive and negative alphas
            iclm = fitlm(riskaversionMetric,negAlpha) % significant
            iclm = fitlm(riskaversionMetric,posAlpha)

            lm = fitlm(riskaversionMetric,bestAlpha) % NS
            lm = fitlm(riskaversionMetric,bestAlphaRisk)% NS
            [p,h,stats] = signrank(bestAlphaRisk, bestAlpha); % SIG

            [pRew,HRew,statsRew] = ranksum(bestAlpha(impulsiveChoosers),bestAlpha(~impulsiveChoosers)) % NS
            [pRisk,HRisk,statsRisk] = ranksum(bestAlphaRisk(impulsiveChoosers),bestAlphaRisk(~impulsiveChoosers)) % NS
            [pRew,HRew,statsRew] = signrank(bestAlphaRewardTD,bestAlphaRiskTD) % NS


            figure(4)
            subplot(2,2,1)
            lm = fitlm(riskaversionMetric,bestAlphaRewardTD)
            h = lm.plot;
            h(1).Marker = '.';
            h(1).MarkerEdgeColor = 'k';
            h(1).MarkerSize = 15;
            xlabel('impulsivity metric')
            ylabel('reward alpha')
            subtitle(sprintf('p = %2f', lm.ModelFitVsNullModel.Pvalue))
            legend off
            axis tight square
            ax3 = gca;
            ylim(ax3, [0 1]);
            yticks(ax3, 0:0.1:1);

            % regression plot
            subplot(2,2,2)
            lm = fitlm(riskaversionMetric,bestAlphaRiskTD)
            h = lm.plot;
            h(1).Marker = '.';
            h(1).MarkerEdgeColor = 'k';
            h(1).MarkerSize = 15;
            xlabel('impulsivity metric')
            ylabel('risk alpha')
            subtitle(sprintf('p = %2f', lm.ModelFitVsNullModel.Pvalue))
            legend off
            axis tight square
            ax3 = gca;
            ylim(ax3, [0 1]);
            yticks(ax3, 0:0.1:1);

            %make logical >.1
            bestAlphaRewardTDLogical = bestAlphaRewardTD > .01
            bestAlphaRiskTDLogical = bestAlphaRiskTD > .01

            subplot(2,2,3)
            lm = fitlm(riskaversionKLD_yellows(bestAlphaRewardTDLogical),bestAlphaRewardTD(bestAlphaRewardTDLogical))
            h = lm.plot;
            h(1).Marker = '.';
            h(1).MarkerEdgeColor = 'k';
            h(1).MarkerSize = 15;
            xlabel('impulsivity metric')
            ylabel('reward alpha (> .01)')
            subtitle(sprintf('p = %2f', lm.ModelFitVsNullModel.Pvalue))
            legend off
            axis tight square
            ax3 = gca;
            ylim(ax3, [0 1]);
            yticks(ax3, 0:0.1:1);

            % regression plot
            subplot(2,2,4)
            lm = fitlm(riskaversionKLD_yellows(bestAlphaRiskTDLogical),bestAlphaRiskTD(bestAlphaRiskTDLogical))
            h = lm.plot;
            h(1).Marker = '.';
            h(1).MarkerEdgeColor = 'k';
            h(1).MarkerSize = 15;
            xlabel('impulsivity metric')
            ylabel('risk alpha (> .01)')
            subtitle(sprintf('p = %2f', lm.ModelFitVsNullModel.Pvalue))
            legend off
            axis tight square
            ax3 = gca;
            ylim(ax3, [0 1]);
            yticks(ax3, 0:0.1:1);

            saveas(4, '//155.100.91.44/d/Data/Rhiannon/BART_RLDM_outputs/TDlearn/OptimalAlphas_Simple.pdf')

        elseif strcmp(whichRL,'vanilla')

            bestAlphaRewardTD = bestAlpha;
            bestAlphaRiskTD = bestAlphaRisk;

            % quick stats
            [pBank,H,stats] = ranksum(bestAlphaRewardTD(impulsiveChoosers),bestAlphaRewardTD(~impulsiveChoosers))
            [pRisk,HRisk,statsRisk] = ranksum(bestAlphaRiskTD(impulsiveChoosers),bestAlphaRiskTD(~impulsiveChoosers))

            figure(777)
            % linear regression between simple and reward
            subplot(2,2,1)
            iclm = fitlm(riskaversionMetric,bestAlphaRewardTD)
            h = iclm.plot;
            h(1).Marker = '.';
            h(1).MarkerEdgeColor = 'k';
            h(1).MarkerSize = 15;
            xlabel('yellow balloon impulsivity metric -log10(KLD(active||passive))')
            ylabel('optimal alpha - reward')
            subtitle(sprintf('p = %2f', iclm.ModelFitVsNullModel.Pvalue))
            legend off
            axis tight square

            % linear regression between simple and risk
            subplot(2,2,2)
            iclm = fitlm(riskaversionMetric,bestAlphaRiskTD)
            h = iclm.plot;
            h(1).Marker = '.';
            h(1).MarkerEdgeColor = 'k';
            h(1).MarkerSize = 15;
            xlabel('yellow balloon impulsivity metric -log10(KLD(active||passive))')
            ylabel('optimal alpha - risk')
            subtitle(sprintf('p = %2f', iclm.ModelFitVsNullModel.Pvalue))
            legend off
            axis tight square

           % linear regression between rstd and reward
            subplot(2,2,3)
            iclm = fitlm(RSTDmetric,bestAlphaRewardTD)
            h = iclm.plot;
            h(1).Marker = '.';
            h(1).MarkerEdgeColor = 'k';
            h(1).MarkerSize = 15;
            xlabel('risk sensitivity metric (ratio)')
            ylabel('optimal alpha - reward')
            subtitle(sprintf('p = %2f', iclm.ModelFitVsNullModel.Pvalue))
            legend off
            axis tight square

            % linear regression between rstd and risk
            subplot(2,2,4)
            iclm = fitlm(RSTDmetric,bestAlphaRiskTD)
            h = iclm.plot;
            h(1).Marker = '.';
            h(1).MarkerEdgeColor = 'k';
            h(1).MarkerSize = 15;
            xlabel('risk sensitivity metric (ratio)')
            ylabel('optimal alpha - risk')
            subtitle(sprintf('p = %2f', iclm.ModelFitVsNullModel.Pvalue))
            legend off
            axis tight square
            maximize(777)
            saveas(777,'\\155.100.91.44\D\Data\Rhiannon\BART_RLDM_outputs\TDlearn\simpleTDalphaRegressions.pdf')
            
            % median split in KLD metric
            [~,tmp] = sort(riskaversionMetric);
            [~,sortedImpulsivityIdcs] = sort(tmp);

            % accuracy across balloon colors, grouped by impulsivtiy metrics
            yA = [bhvStruct.yellowAccuracy];
            oA = [bhvStruct.orangeAccuracy];
            rA = [bhvStruct.redAccuracy];
            tA = [bhvStruct.accuracyTot];

            % anova
            X = [yA' oA' rA'];
            [pX,tblX,statsX] = anova1(X)
            [c,~,~,gnames] = multcompare(statsX)

            % plot point accumulation
            % accuracy grouped by impulsivtiy metrics
            fP = [bhvStruct.freePoints];
            cP = [bhvStruct.ctrlPoints];
            aTs = [bhvStruct.nActiveTrials];
            [pointPactiveTrials,pointHactiveTrials,pointSTATSactiveTrials] = ranksum(fP(medSplit)./aTs(medSplit),fP(~medSplit)./aTs(~medSplit))
            [pointP,pointH,pointSTATS] = ranksum(fP(medSplit),fP(~medSplit))

            %% [EHS20220222] Hypothesis: sig.dif. corr strength btw Vrisk/Vreward in two pt groups
            for p2 = 1:nPts
                for al = 1:length(TDdata(p2).valueVsRiskEstimateLM)
                    % TDdata.rewardVsRiskPElm.PEmodel = PElm;
                    % TDdata.valueVsRiskEstimateLM.valueVsRiskModel = Vlm;

                    tValsV(p2,al) = TDdata(p2).valueVsRiskEstimateLM.valueVsRiskModel.Coefficients{2,3};
                    RsqV(p2,al) = TDdata(p2).valueVsRiskEstimateLM.valueVsRiskModel.Rsquared.ordinary;

                    tValsPE(p2,al) = TDdata(p2).rewardVsRiskPElm.PEmodel.Coefficients{2,3};
                    RsqPE(p2,al) = TDdata(p2).rewardVsRiskPElm.PEmodel.Rsquared.ordinary;

                end
            end

            %% [EHS20220429] some plotting code for less relevant anaylses...
            % going to turn it off for now...
            distAnalysis = true;
            if distAnalysis
                % hypothesis testing
                % 1) hypothesis: slopes of reward vs risk models are
                % greater/lessthan for more/less impulsive
                [pTV,HTV,statsTV] = ranksum(tValsV(impulsiveChoosers),tValsV(~impulsiveChoosers)) % SIGNIFICANT
                % 2) hypothesis: explained variance of reward vs risk models are
                % greater/lessthan for more/less impulsive
                [pRV,HRV,statsRV] = ranksum(RsqV(impulsiveChoosers),RsqV(~impulsiveChoosers)) % NS
                % [EHS20220222] both trending in opposite directions, neither signficant

                % hypothesis testing [20220608] same thing for PEs
                % 1) hypothesis: slopes of reward vs risk models are
                % greater/lessthan for more/less impulsive
                [pTPE,HTPE,statsTPE] = ranksum(tValsPE(impulsiveChoosers),tValsPE(~impulsiveChoosers)) % SIGNIFICANT
                % 2) hypothesis: explained variance of reward vs risk models are
                % greater/lessthan for more/less impulsive
                [pRPE,HRPE,statsRPE] = ranksum(RsqPE(impulsiveChoosers),RsqPE(~impulsiveChoosers)) % SIGNIFICANT
                % [EHS20220222] both trending in opposite directions, neither signficant

                % colors for MI and LI individuals
                sage_green = [119, 171, 86] / 255; % RGB values for sage green ICs
                muted_purple = [79, 49, 170] / 255; % RGB values for muted purple ~ICs

                % visualization
                figure(2222)
                % plotting tvalues for risk reward
                subplot(2,2,1)
                subtitle(sprintf('Value Slopes btwn MI/LI: %.4f', pTV))
                hold on
                th2 = histfit(tValsV(~impulsiveChoosers),8);
                th1 = histfit(tValsV(impulsiveChoosers),8);
                th1(2).Color = sage_green;
                th2(2).Color = muted_purple;
                th1(1).FaceColor = sage_green;
                th2(1).FaceColor = muted_purple;
                th1(1).FaceAlpha = 0.5;
                th2(1).FaceAlpha = 0.5;
                hold off
                axis square
                legend('more impulsive','fit','less impulsive','fit')
                xlabel('value estimate correaltion strength (t-value)')

                % plotting r-squares for value
                subplot(2,2,2)
                subtitle(sprintf('Rsq for value btwn MI/LI: %.4f', pRV))
                hold on
                rh2 = histfit(RsqV(~impulsiveChoosers),8);
                rh1 = histfit(RsqV(impulsiveChoosers),8);
                rh1(2).Color = sage_green;
                rh2(2).Color = muted_purple;
                rh1(1).FaceColor = sage_green;
                rh2(1).FaceColor = muted_purple;
                rh1(1).FaceAlpha = 0.5;
                rh2(1).FaceAlpha = 0.5;
                hold off
                axis square
                xlabel('value estimate correlation Rsquared')

                % plotting tvalues for risk reward
                subplot(2,2,3)
                subtitle(sprintf('PE Slopes btwn MI/LI: %.4f', pTPE))
                hold on
                th2 = histfit(tValsPE(~impulsiveChoosers),8);
                th1 = histfit(tValsPE(impulsiveChoosers),8);
                th1(2).Color = sage_green;
                th2(2).Color = muted_purple;
                th1(1).FaceColor = sage_green;
                th2(1).FaceColor = muted_purple;
                th1(1).FaceAlpha = 0.5;
                th2(1).FaceAlpha = 0.5;
                hold off
                axis square
                legend('more impulsive','fit','less impulsive','fit')
                xlabel('value estimate correaltion strength (t-value)')

                % plotting r-squares for value
                subplot(2,2,4)
                subtitle(sprintf('Rsq for PE btwn MI/LI: %.4f', pRPE))
                hold on
                rh2 = histfit(RsqPE(~impulsiveChoosers),8);
                rh1 = histfit(RsqPE(impulsiveChoosers),8);
                rh1(2).Color = sage_green;
                rh2(2).Color = muted_purple;
                rh1(1).FaceColor = sage_green;
                rh2(1).FaceColor = muted_purple;
                rh1(1).FaceAlpha = 0.5;
                rh2(1).FaceAlpha = 0.5;
                hold off
                axis square
                xlabel('value estimate correlation Rsquared')

                saveas(2222,'\\155.100.91.44\D\Data\Rhiannon\BART_RLDM_outputs\BatchAnalyze_Controls\Behavior\slopes_Rsq_ValuePE.pdf')


                %% redo with optimal alphas...

                % saving histogram data across subjects
                % first define edges
                RPEedges = min(min([TDdata.RewardPE])):25:max(max([TDdata.RewardPE]));
                XPEedges = min(min([TDdata.RiskPE])):0.1:max(max([TDdata.RiskPE]));
                Vedges = min(min([TDdata.V])):10:max(max([TDdata.V]));
                Xedges = min(min([TDdata.Vrisk])):0.1:max(max([TDdata.Vrisk]));
                % then calculate histos for each pt.
                for p2 = 1:nPts
                    hRPE(p2,:) = histcounts(TDdata(p2).RewardPE,RPEedges);
                    hXPE(p2,:) = histcounts(TDdata(p2).RiskPE,XPEedges);
                    hV(p2,:) = histcounts(TDdata(p2).V,Vedges);
                    hX(p2,:) = histcounts(TDdata(p2).Vrisk,Xedges);
                end

                % plot mean distributions for each subject. [need to specify edges?]
                figure(2)
                subplot(2,2,1)
                hold on
                patch([RPEedges(2:end)+RPEedges(1:(end-1))/2 fliplr((RPEedges(2:end)+RPEedges(1:(end-1))/2))],[mean(hRPE(impulsiveChoosers,:))+(std(hRPE(impulsiveChoosers,:))./sqrt(sum(impulsiveChoosers)))  fliplr(mean(hRPE(impulsiveChoosers,:))-(std(hRPE(impulsiveChoosers,:)))./sqrt(sum(impulsiveChoosers)))],[1 0 0],'facealpha',0.5,'edgecolor','none')
                line(RPEedges(2:end)+RPEedges(1:(end-1))/2,mean(hRPE(impulsiveChoosers,:)),'color','r')
                patch([RPEedges(2:end)+RPEedges(1:(end-1))/2 fliplr((RPEedges(2:end)+RPEedges(1:(end-1))/2))],[mean(hRPE(~impulsiveChoosers,:))+(std(hRPE(~impulsiveChoosers,:))./sqrt(sum(impulsiveChoosers)))  fliplr(mean(hRPE(~impulsiveChoosers,:))-(std(hRPE(~impulsiveChoosers,:)))./sqrt(sum(impulsiveChoosers)))],[0 0 1],'facealpha',0.5,'edgecolor','none')
                line(RPEedges(2:end)+RPEedges(1:(end-1))/2,mean(hRPE(~impulsiveChoosers,:)),'color','b')
                % Dummy plot objects for the legend
                h1 = plot(NaN, NaN, 'r'); % Red for more impulsive
                h2 = plot(NaN, NaN, 'b'); % Blue for less impulsive
                % Add legend
                legend([h1, h2], 'more impulsive', 'less impulsive', 'Location', 'northeast');
                hold off
                title('RPE distributions', 'fontsize', 12)
                axis square tight

                subplot(2,2,2)
                hold on
                patch([XPEedges(2:end)+XPEedges(1:(end-1))/2 fliplr((XPEedges(2:end)+XPEedges(1:(end-1))/2))],[mean(hXPE(impulsiveChoosers,:))+(std(hXPE(impulsiveChoosers,:))./sqrt(sum(impulsiveChoosers)))  fliplr(mean(hXPE(impulsiveChoosers,:))-(std(hXPE(impulsiveChoosers,:)))./sqrt(sum(impulsiveChoosers)))],[1 0 0],'facealpha',0.5,'edgecolor','none')
                line(XPEedges(2:end)+XPEedges(1:(end-1))/2,mean(hXPE(impulsiveChoosers,:)),'color','r')
                patch([XPEedges(2:end)+XPEedges(1:(end-1))/2 fliplr((XPEedges(2:end)+XPEedges(1:(end-1))/2))],[mean(hXPE(~impulsiveChoosers,:))+(std(hXPE(~impulsiveChoosers,:))./sqrt(sum(impulsiveChoosers)))  fliplr(mean(hXPE(~impulsiveChoosers,:))-(std(hXPE(~impulsiveChoosers,:)))./sqrt(sum(impulsiveChoosers)))],[0 0 1],'facealpha',0.5,'edgecolor','none')
                line(XPEedges(2:end)+XPEedges(1:(end-1))/2,mean(hXPE(~impulsiveChoosers,:)),'color','b')
                hold off
                title('risk PE distributions')
                axis square tight

                subplot(2,2,3)
                hold on
                patch([Vedges(2:end)+Vedges(1:(end-1))/2 fliplr((Vedges(2:end)+Vedges(1:(end-1))/2))],[mean(hV(impulsiveChoosers,:))+(std(hV(impulsiveChoosers,:))./sqrt(sum(impulsiveChoosers)))  fliplr(mean(hV(impulsiveChoosers,:))-(std(hV(impulsiveChoosers,:)))./sqrt(sum(impulsiveChoosers)))],[1 0 0],'facealpha',0.5,'edgecolor','none')
                line(Vedges(2:end)+Vedges(1:(end-1))/2,mean(hV(impulsiveChoosers,:)),'color','r')
                patch([Vedges(2:end)+Vedges(1:(end-1))/2 fliplr((Vedges(2:end)+Vedges(1:(end-1))/2))],[mean(hV(~impulsiveChoosers,:))+(std(hV(~impulsiveChoosers,:))./sqrt(sum(impulsiveChoosers)))  fliplr(mean(hV(~impulsiveChoosers,:))-(std(hV(~impulsiveChoosers,:)))./sqrt(sum(impulsiveChoosers)))],[0 0 1],'facealpha',0.5,'edgecolor','none')
                line(Vedges(2:end)+Vedges(1:(end-1))/2,mean(hV(~impulsiveChoosers,:)),'color','b')
                hold off
                title('Value distributions')
                axis square tight

                subplot(2,2,4)
                hold on
                patch([Xedges(2:end)+Xedges(1:(end-1))/2 fliplr((Xedges(2:end)+Xedges(1:(end-1))/2))],[mean(hX(impulsiveChoosers,:))+(std(hX(impulsiveChoosers,:))./sqrt(sum(impulsiveChoosers)))  fliplr(mean(hX(impulsiveChoosers,:))-(std(hX(impulsiveChoosers,:)))./sqrt(sum(impulsiveChoosers)))],[1 0 0],'facealpha',0.5,'edgecolor','none')
                line(Xedges(2:end)+Xedges(1:(end-1))/2,mean(hX(impulsiveChoosers,:)),'color','r')
                patch([Xedges(2:end)+Xedges(1:(end-1))/2 fliplr((Xedges(2:end)+Xedges(1:(end-1))/2))],[mean(hX(~impulsiveChoosers,:))+(std(hX(~impulsiveChoosers,:))./sqrt(sum(impulsiveChoosers)))  fliplr(mean(hX(~impulsiveChoosers,:))-(std(hX(~impulsiveChoosers,:)))./sqrt(sum(impulsiveChoosers)))],[0 0 1],'facealpha',0.5,'edgecolor','none')
                line(Xedges(2:end)+Xedges(1:(end-1))/2,mean(hX(~impulsiveChoosers,:)),'color','b')
                hold off
                title('risk distributions')
                axis square tight
                maximize(2)
                %TODO:: plot 2-D distributions of risk vs reward PEs for
                %subsets of trial types.

                subtitle('distributions for each patient category')
                saveas(gcf,['\\155.100.91.44\D\Data\Rhiannon\BART_RLDM_outputs\BatchAnalyze_Controls\Behavior\allPts_BARTbehaviorDistributions_' whichRL '.pdf']);
            end

        end
    end

elseif strcmp('TDlearnNeural',analysis)

    % Loading in TDdata for each patient
mainPath = '\\155.100.91.44\D\Data\preProcessed\BART_preprocessed\';
neuralData_All = {};

keyboard

for p = 1:length(ptArray)
    ptID = ptArray{p};
    dataPath = fullfile(mainPath, ptID, 'Data');
    fileInfo = dir(fullfile(dataPath, '*TDdataGradients.mat'));
    
    if ~isempty(fileInfo)
        fileToLoad = fullfile(dataPath, fileInfo(1).name);
        data = load(fileToLoad);
        neuralData_All{p} = data;
    else
        warning('No TDdataGradients.mat file found for %s', ptID);
        neuralData_All{p} = [];
    end
end

load('riskSensitivity_TD_bhv.mat'); % to get RSTDmetric from behavior

%% GRADIENT ALPHAS
    % % Get neural RSTD metric and R2 values: create matrices with the right height and width
    max_positiveVE = cell(nPts,1); % CUE
    rstd_positiveVE_Index = cell(nPts,1);
    max_negativeVE = cell(nPts,1);
    rstd_negativeVE_Index = cell(nPts,1);
    max_positivePE = cell(nPts,1); % OUTCOME
    rstd_positivePE_Index = cell(nPts,1);
    max_negativePE = cell(nPts,1);
    rstd_negativePE_Index = cell(nPts,1);
    rstd_positiveVE = cell(nPts,1); % CUE
    rstd_negativeVE = cell(nPts,1);
    rstd_positivePE = cell(nPts,1); % OUTCOME
    rstd_negativePE = cell(nPts,1);
    RSTD_metric_VE_neural = cell(nPts,1); % CUE
    RSTD_metric_VE_neural_NORM = cell(nPts,1);
    RSTD_metric_PE_neural = cell(nPts,1); % OUTCOME
    RSTD_metric_PE_neural_NORM = cell(nPts,1);
    
    % % loop through neural fits and get the rstd values and calculate rstd metric
    for k = 1:nPts
        nFits = length(neuralData_All{k}.TDdataGradients.neuralFit);
         max_positiveVE_k = zeros(1, nFits);
        rstd_positiveVE_Index_k = zeros(1, nFits);
        max_negativeVE_k = zeros(1, nFits);
        rstd_negativeVE_Index_k = zeros(1, nFits);
        max_positivePE_k = zeros(1, nFits);
        rstd_positivePE_Index_k = zeros(1, nFits);
        max_negativePE_k = zeros(1, nFits);
        rstd_negativePE_Index_k = zeros(1, nFits);
        rstd_positiveVE_k = zeros(1, nFits);
        rstd_negativeVE_k = zeros(1, nFits);
        rstd_positivePE_k = zeros(1, nFits);
        rstd_negativePE_k = zeros(1, nFits);
        RSTD_VE_metric_k = zeros(1, nFits);
        RSTD_PE_metric_k = zeros(1, nFits);

        for j = 1:nFits
            % CUE_ALIGNED MODEL
            [max_positiveVE_k(j), rstd_positiveVE_Index_k(j)] =  min(min(neuralData_All{k}.TDdataGradients.neuralFit(j).LLimg_cue,[],1),[],2);
            [max_negativeVE_k(j), rstd_negativeVE_Index_k(j)] = min(min(neuralData_All{k}.TDdataGradients.neuralFit(j).LLimg_cue,[],2),[],1);
            rstd_positiveVE_k(j) = neuralData_All{k}.a(rstd_positiveVE_Index_k(j));
            rstd_negativeVE_k(j) = neuralData_All{k}.a(rstd_negativeVE_Index_k(j));
            % OUTCOME-ALGNED MODEL
            [max_positivePE_k(j), rstd_positivePE_Index_k(j)] =  min(min(neuralData_All{k}.TDdataGradients.neuralFit(j).LLimg_outcome,[],1),[],2);
            [max_negativePE_k(j), rstd_negativePE_Index_k(j)] = min(min(neuralData_All{k}.TDdataGradients.neuralFit(j).LLimg_outcome,[],2),[],1);
            rstd_positivePE_k(j) = neuralData_All{k}.a(rstd_positivePE_Index_k(j));
            rstd_negativePE_k(j) = neuralData_All{k}.a(rstd_negativePE_Index_k(j));
            % rstd metric: ratio of pos/neg
            RSTD_VE_metric_k(j) = (rstd_negativeVE_k(j) - rstd_positiveVE_k(j)) / (rstd_negativeVE_k(j) + rstd_positiveVE_k(j)); % CUE
            RSTD_PE_metric_k(j) = (rstd_negativePE_k(j) - rstd_positivePE_k(j)) / (rstd_negativePE_k(j) + rstd_positivePE_k(j)); % OUTCOME
        end
        
        % CUE_ALIGNED
        max_positiveVE{k} = max_positiveVE_k;
        rstd_positiveVE_Index{k} = rstd_positiveVE_Index_k;
        max_negativeVE{k} = max_negativeVE_k;
        rstd_negativeVE_Index{k} = rstd_negativeVE_Index_k;
        rstd_positiveVE{k} = rstd_positiveVE_k;
        rstd_negativeVE{k} = rstd_negativeVE_k;
        % OUTCOME_ALIGNED
        max_positivePE{k} = max_positivePE_k;
        rstd_positivePE_Index{k} = rstd_positivePE_Index_k;
        max_negativePE{k} = max_negativePE_k;
        rstd_negativePE_Index{k} = rstd_negativePE_Index_k;
        rstd_positivePE{k} = rstd_positivePE_k;
        rstd_negativePE{k} = rstd_negativePE_k;

        RSTD_metric_VE_neural{k} = RSTD_VE_metric_k;
        RSTD_metric_PE_neural{k} = RSTD_PE_metric_k;
        RSTD_metric_VE_neural_NORM{k} = RSTD_metric_VE_neural{k,:}/RSTDmetric(k);
        RSTD_metric_PE_neural_NORM{k} = RSTD_metric_PE_neural{k,:}/RSTDmetric(k);
    end

    % Creating 71x119 structures
    nPts = 71;
    maxLength = 119;
     max_positiveVE_expanded     = cell(nPts, maxLength);
    rstd_positiveVE_Index_expanded = cell(nPts, maxLength);
    max_negativeVE_expanded     = cell(nPts, maxLength);
    rstd_negativeVE_Index_expanded = cell(nPts, maxLength);
    rstd_positiveVE_expanded    = cell(nPts, maxLength);
    rstd_negativeVE_expanded    = cell(nPts, maxLength);
    RSTD_metric_VE_neural_expanded = cell(nPts, maxLength);
    RSTD_metric_VE_neural_NORM_expanded = cell(nPts, maxLength);
    
    max_positivePE_expanded     = cell(nPts, maxLength);
    rstd_positivePE_Index_expanded = cell(nPts, maxLength);
    max_negativePE_expanded     = cell(nPts, maxLength);
    rstd_negativePE_Index_expanded = cell(nPts, maxLength);
    rstd_positivePE_expanded    = cell(nPts, maxLength);
    rstd_negativePE_expanded    = cell(nPts, maxLength);
    RSTD_metric_PE_neural_expanded = cell(nPts, maxLength);
    RSTD_metric_PE_neural_NORM_expanded = cell(nPts, maxLength);
    for k = 1:nPts
        currLen = length(max_positivePE(k));
        for j = 1:currLen
            max_positiveVE_expanded{k, j}      = max_positiveVE{k}(j);
            rstd_positiveVE_Index_expanded{k, j} = rstd_positiveVE_Index{k}(j);
            max_negativeVE_expanded{k, j}      = max_negativeVE{k}(j);
            rstd_negativeVE_Index_expanded{k, j} = rstd_negativeVE_Index{k}(j);
            rstd_positiveVE_expanded{k, j}     = rstd_positiveVE{k}(j);
            rstd_negativeVE_expanded{k, j}     = rstd_negativeVE{k}(j);
            RSTD_metric_VE_neural_expanded{k, j}  = RSTD_metric_VE_neural{k}(j);
            RSTD_metric_VE_neural_NORM_expanded{k, j}  = RSTD_metric_VE_neural_NORM{k}(j);

             max_positivePE_expanded{k, j}      = max_positivePE{k}(j);
            rstd_positivePE_Index_expanded{k, j} = rstd_positivePE_Index{k}(j);
            max_negativePE_expanded{k, j}      = max_negativePE{k}(j);
            rstd_negativePE_Index_expanded{k, j} = rstd_negativePE_Index{k}(j);
            rstd_positivePE_expanded{k, j}     = rstd_positivePE{k}(j);
            rstd_negativePE_expanded{k, j}     = rstd_negativePE{k}(j);
            RSTD_metric_PE_neural_expanded{k, j}  = RSTD_metric_VE_neural{k}(j);
            RSTD_metric_PE_neural_NORM_expanded{k, j}  = RSTD_metric_VE_neural_NORM{k}(j);
        end
    end

    % Creating 1 column cells
    max_positiveVE_long = reshape(max_positiveVE_expanded.', [], 1); %vertcat(max_positivePE_expanded{:});
    rstd_positiveVE_Index_long = reshape(rstd_positiveVE_Index_expanded.', [], 1);
    max_negativeVE_long = reshape(max_negativeVE_expanded.', [], 1);
    rstd_negativeVE_Index_long = reshape(rstd_negativeVE_Index_expanded.', [], 1);
    rstd_positiveVE_long  = reshape(rstd_positiveVE_expanded.', [], 1);
    rstd_negativeVE_long = reshape(rstd_negativeVE_expanded.', [], 1);
    RSTD_metric_VE_neural_long = reshape(RSTD_metric_VE_neural_expanded.', [], 1);
    RSTD_metric_VE_neural_NORM_long = reshape(RSTD_metric_VE_neural_NORM_expanded.', [], 1);

    max_positivePE_long = reshape(max_positivePE_expanded.', [], 1); %vertcat(max_positivePE_expanded{:});
    rstd_positivePE_Index_long = reshape(rstd_positivePE_Index_expanded.', [], 1);
    max_negativePE_long = reshape(max_negativePE_expanded.', [], 1);
    rstd_negativePE_Index_long = reshape(rstd_negativePE_Index_expanded.', [], 1);
    rstd_positivePE_long  = reshape(rstd_positivePE_expanded.', [], 1);
    rstd_negativePE_long = reshape(rstd_negativePE_expanded.', [], 1);
    RSTD_metric_PE_neural_long = reshape(RSTD_metric_PE_neural_expanded.', [], 1);
    RSTD_metric_PE_neural_NORM_long = reshape(RSTD_metric_PE_neural_NORM_expanded.', [], 1);

    % drop empties from these columns....
    max_positiveVE_long = max_positiveVE_long(~cellfun('isempty', max_positiveVE_long));
    rstd_positiveVE_Index_long = rstd_positiveVE_Index_long(~cellfun('isempty', rstd_positiveVE_Index_long));
    max_negativePE_long = max_negativeVE_long(~cellfun('isempty', max_negativeVE_long));
    rstd_negativeVE_Index_long = rstd_negativeVE_Index_long(~cellfun('isempty', rstd_negativeVE_Index_long));
    rstd_positiveVE_long = rstd_positiveVE_long(~cellfun('isempty', rstd_positiveVE_long));
    rstd_negativeVE_long = rstd_negativeVE_long(~cellfun('isempty', rstd_negativeVE_long));
    RSTD_metric_VE_neural_long = RSTD_metric_VE_neural_long(~cellfun('isempty', RSTD_metric_VE_neural_long));
    RSTD_metric_VE_neural_NORM_long = RSTD_metric_VE_neural_NORM_long(~cellfun('isempty', RSTD_metric_VE_neural_NORM_long));

    max_positivePE_long = max_positivePE_long(~cellfun('isempty', max_positivePE_long));
    rstd_positivePE_Index_long = rstd_positivePE_Index_long(~cellfun('isempty', rstd_positivePE_Index_long));
    max_negativePE_long = max_negativePE_long(~cellfun('isempty', max_negativePE_long));
    rstd_negativePE_Index_long = rstd_negativePE_Index_long(~cellfun('isempty', rstd_negativePE_Index_long));
    rstd_positivePE_long = rstd_positivePE_long(~cellfun('isempty', rstd_positivePE_long));
    rstd_negativePE_long = rstd_negativePE_long(~cellfun('isempty', rstd_negativePE_long));
    RSTD_metric_PE_neural_long = RSTD_metric_PE_neural_long(~cellfun('isempty', RSTD_metric_PE_neural_long));
    RSTD_metric_PE_neural_NORM_long = RSTD_metric_PE_neural_NORM_long(~cellfun('isempty', RSTD_metric_PE_neural_NORM_long));

    % Find indices where RSTD_metric_neural_long is 0
    zero_VE_indices = (RSTD_metric_VE_neural_long == 0);
    zero_VE_indices_NORM = (RSTD_metric_VE_neural_NORM_long == 0);
    zero_PE_indices = (RSTD_metric_PE_neural_long == 0);
    zero_PE_indices_NORM = (RSTD_metric_PE_neural_NORM_long == 0);

    % Extract corresponding values from the other two variables
    rstd_VE_zeros = [rstd_positiveVE_long(zero_PE_indices), rstd_negativePE_long(zero_PE_indices)];
    rstd_VE_zeros_NORM = [rstd_positivePE_long(zero_PE_indices), rstd_negativePE_long(zero_PE_indices)];
    rstd_PE_zeros = [rstd_positivePE_long(zero_PE_indices), rstd_negativePE_long(zero_PE_indices)];
    rstd_PE_zeros_NORM = [rstd_positivePE_long(zero_PE_indices), rstd_negativePE_long(zero_PE_indices)];

    %% ~~~~~~~~~~~~~~~~~~~~~~~~~ DATA ANALYSIS ~~~~~~~~~~~~~~~~~~~~~~~ %%

    % (1) Quantify the significant contacts
    % TODO:
    %  plot histograms of signficant contacts' R-squared values for each TD model.

    % Getting data from models, RSTD R2 and pValues
    for k = 1:nPts
        for j = 1:length(neuralData_All{:,k}.TDdataGradients.neuralFit)
            % RSTD MODEL INFO
            rstdCue_RsqAdj(k,j) = neuralData_All{k}.TDdataGradients.neuralFit(j).rstd_CueModel.Rsquared.Adjusted;
            rstdCue_pValue(k,j) = neuralData_All{k}.TDdataGradients.neuralFit(j).rstd_CueModel.Coefficients(2,6);
            rstdOutcome_RsqAdj(k,j) = neuralData_All{k}.TDdataGradients.neuralFit(j).rstd_OutcomeModel.Rsquared.Adjusted;
            rstdOutcome_pValue(k,j) = neuralData_All{k}.TDdataGradients.neuralFit(j).rstd_OutcomeModel.Coefficients(2,6);

        end
    end

    % Convert the pValue datasets to a double array (total contacts 5167)
    rstdCue_pValue = double(rstdCue_pValue);
    rstdOutcome_pValue = double(rstdOutcome_pValue);

    % Getting significant values per patient (norm by # of electrodes?)
    rstdCue_length(rstdCue_pValue == 0) = NaN;
    lengthElectrodes_PerPatient = sum(~isnan(rstdCue_length), 2);
    rstdCue_pValue_sigPerPatient = sum(rstdCue_pValue < 0.05 & rstdCue_pValue > 0, 2);  % Sum across columns (electrodes)
    rstdOutcome_pValue_sigPerPatient = sum(rstdOutcome_pValue < 0.05 & rstdOutcome_pValue > 0, 2);  % Sum across columns (electrodes)

    % Treat 0s as invalid/missing electrode entries
    validMask_VE = rstdCue_pValue > 0;
    validMask_PE = rstdOutcome_pValue > 0;
    sigMask_VE = (rstdCue_pValue < 0.05) & validMask_VE;
    sigMask_PE = (rstdOutcome_pValue < 0.05) & validMask_PE;
    sigPE_PerPatient = sum(sigMask_PE, 2);  % 71x1 vector
    sigVE_PerPatient = sum(sigMask_VE, 2);  % 71x1 vector

    % Proportion of significant p-values per patient
    sigVE_ProportionPerPatient = sigVE_PerPatient ./ lengthElectrodes_PerPatient;
    sigPE_ProportionPerPatient = sigPE_PerPatient ./ lengthElectrodes_PerPatient;

    % plot rstdMetric by # of significant contacts
    lm = fitlm(sigVE_ProportionPerPatient,RSTDmetric);
    h = lm.plot;
    h(1).Marker = '.';
    h(1).MarkerEdgeColor = 'k';
    h(1).MarkerSize = 15;
    xlabel('# of sig RSTD_VE contacts')
    ylabel('RSTD metric')
    subtitle(sprintf('p = %2f', lm.ModelFitVsNullModel.Pvalue))
    legend off
    axis tight square

    lm = fitlm(sigPE_ProportionPerPatient,RSTDmetric);
    h = lm.plot;
    h(1).Marker = '.';
    h(1).MarkerEdgeColor = 'k';
    h(1).MarkerSize = 15;
    xlabel('prop of sig RSTD_PE contacts')
    ylabel('RSTD metric')
    subtitle(sprintf('p = %2f', lm.ModelFitVsNullModel.Pvalue))
    legend off
    axis tight square

    % Getting pValues that are significant for each model
    % need to add greater than zeros because right now it is a full matrix of the most contacts by patient. which not all pts have.
    sig_rstdCue_Logical = rstdCue_pValue < 0.05 & rstdCue_pValue > 0; % sum(sum(sig_rstdCue_Logical)) % 1790
    sig_rstdOutcome_Logical = rstdOutcome_pValue < 0.05 & rstdOutcome_pValue > 0; % sum(sum(sig_rstdOutcome_Logical)) % 1676
   
    % plot these sums...
    % Define the data (from the sum of logical conditions)
    data = [sum(sum(sig_rstdCue_Logical)), sum(sum(sig_rstdOutcome_Logical))];
    labels = {'rstd VE', 'rstd PE'};
    figure(123);
    bar(data, 'FaceColor', [0.678, 0.847, 0.902]);
    title('Significant All Contacts for RSTD Models');
    xlabel('Models');
    ylabel('Sig Contacts');
    set(gca, 'XTickLabel', labels);
    xtickangle(45);

    % Add total sums to the bars
    for i = 1:length(data)
        % Display the total sum on top of each bar
        text(i, data(i) + 0.05, num2str(data(i)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    end

    %  3) plot significant examples with R-squared.
    % plotting RSTD histogram
    figure(1234)
    subplot(2,2,1);
    histogram(rstdCue_RsqAdj(~sig_rstdCue_Logical), 'BinMethod', 'auto', 'FaceColor', 'k');
    hold on
    histogram(rstdCue_RsqAdj(sig_rstdCue_Logical), 'BinMethod', 'auto', 'FaceColor', 'b');
    title('RSTD: sig cue R^2');
    xlabel('rstdSurprise R^2 Adj');
    ylabel('Frequency');
    subplot(2,2,2);
    histogram(rstdOutcome_RsqAdj(~sig_rstdOutcome_Logical), 'BinMethod', 'auto', 'FaceColor', 'k');
    hold on
    histogram(rstdOutcome_RsqAdj(sig_rstdOutcome_Logical), 'BinMethod', 'auto', 'FaceColor', 'r');
    title('RSTD: sig outcome R^2');
    xlabel('rstd Outcome R^2 Adj');
    ylabel('Frequency');

    % take top R2 values from signifcant contacts
    top100_rstdCue = maxk(rstdCue_RsqAdj(sig_rstdCue_Logical), 100);
    top100_rstdOutcome = maxk(rstdOutcome_RsqAdj(sig_rstdOutcome_Logical), 100);
   
    subplot(2,2,3);
    histogram(top100_rstdCue, 'BinMethod', 'auto', 'FaceColor', 'k');
    title('RSTD: sig cue R^2 top 100');
    xlabel('rstd Cue R^2 Adj');
    ylabel('Frequency');
    xlim([0 1])
    subplot(2,2,4);
    histogram(top100_rstdOutcome, 'BinMethod', 'auto', 'FaceColor', 'k');
    title('RSTD: sig outcome R^2 top 100');
    xlabel('rst Outcome R^2 Adj');
    ylabel('Frequency');
    xlim([0 1])
    
    %  4a) incorporate new contact labels
    for k = 1:nPts
        for j = 1:length(neuralData_All{k}.TDdataGradients.neuralFit)
            trodeLabels_All{k,j} = neuralData_All{k}.TDdataGradients.neuralFit(j).new_trodeLabel;
        end
    end

    % (a) How many contacts in each brain region?
    % number of contacts
    % (A) Amygdala
    amygdala_Contacts = strcmp(trodeLabels_All, 'Amygdala');
    sum(sum(amygdala_Contacts)) % = 221
    % (B) Cingulate
    cingulate_Contacts = strcmp(trodeLabels_All, 'Cingulate');
    sum(sum(cingulate_Contacts)) % = 122
    % (C) Hippocampus
    hippocampus_Contacts = strcmp(trodeLabels_All, 'Hippocampus');
    sum(sum(hippocampus_Contacts)) % = 616
    % (D) Inf.Frontal
    InfFrontal_Contacts = strcmp(trodeLabels_All, 'InfFrontal');
    sum(sum(InfFrontal_Contacts)) % = 188
    % (E) Insula
    insula_Contacts = strcmp(trodeLabels_All, 'Insula');
    sum(sum(insula_Contacts)) % = 180
    % (F) Med.Frontal
    MedFrontal_Contacts = strcmp(trodeLabels_All, 'MedFrontal');
    sum(sum(MedFrontal_Contacts)) % = 318
    % (G) N.Accumbens
    NAccumbens_Contacts = strcmp(trodeLabels_All, 'N.Accumbens');
    sum(sum(NAccumbens_Contacts)) % = 5
    % (H) Orb.Frontal
    OrbFrontal_Contacts = strcmp(trodeLabels_All, 'OrbFrontal');
    sum(sum(OrbFrontal_Contacts)) % = 314
    % (I) Striatum
    striatum_Contacts = strcmp(trodeLabels_All, 'Striatum');
    sum(sum(striatum_Contacts)) % = 165
    % (J) Sup.Frontal
    SupFrontal_Contacts = strcmp(trodeLabels_All, 'SupFrontal');
    sum(sum(SupFrontal_Contacts)) % = 149
    % (K) Thalamus
    thalamus_Contacts = strcmp(trodeLabels_All, 'Thalamus');
    sum(sum(thalamus_Contacts)) % = 123
    % (L) White Matter
    whiteMatter_Contacts = strcmp(trodeLabels_All, 'WhiteMatter');
    sum(sum(whiteMatter_Contacts)) % = 940
    % (K) Unknown
    unknown_Contacts = strcmp(trodeLabels_All, 'Unknown');
    sum(sum(unknown_Contacts)) % = 64

    % how many contacts total?
    sum_allContacts = sum(sum(amygdala_Contacts)) +  sum(sum(cingulate_Contacts)) + sum(sum(hippocampus_Contacts)) + sum(sum(InfFrontal_Contacts)) + sum(sum(insula_Contacts))...
        + sum(sum(MedFrontal_Contacts)) + sum(sum(NAccumbens_Contacts)) + sum(sum(OrbFrontal_Contacts)) + sum(sum(striatum_Contacts)) + sum(sum(SupFrontal_Contacts))...
        + sum(sum(thalamus_Contacts)) + sum(sum(whiteMatter_Contacts)) + sum(sum(unknown_Contacts)); % 3405 (only the 11 ROIs)

    sum_allContacts_woWM_Unk = sum(sum(amygdala_Contacts)) +  sum(sum(cingulate_Contacts)) + sum(sum(hippocampus_Contacts)) + sum(sum(InfFrontal_Contacts)) + sum(sum(insula_Contacts))...
        + sum(sum(MedFrontal_Contacts)) + sum(sum(NAccumbens_Contacts)) + sum(sum(OrbFrontal_Contacts)) + sum(sum(striatum_Contacts)) + sum(sum(SupFrontal_Contacts))...
        + sum(sum(thalamus_Contacts)); %2401 (wo WM or Unks)

    % Getting Dataframes for each Region by Model
    % Amygdala Encoding
    amygdala_byModel_data = [sum(sig_rstdCue_Logical(amygdala_Contacts))/sum(sum(amygdala_Contacts))*100, sum(sig_rstOutcome_Logical(amygdala_Contacts))/sum(sum(amygdala_Contacts))*100];
    % Cingulate Encoding
    cingulate_byModel_data = [sum(sig_rstdCue_Logical(cingulate_Contacts))/sum(sum(cingulate_Contacts))*100, sum(sig_rstOutcome_Logical(cingulate_Contacts))/sum(sum(cingulate_Contacts))*100];
    % Hippocampus Encoding
    hippocampus_byModel_data = [sum(sig_rstdCue_Logical(hippocampus_Contacts))/sum(sum(hippocampus_Contacts))*100, sum(sig_rstOutcome_Logical(hippocampus_Contacts))/sum(sum(hippocampus_Contacts))*100];
    % InfFrontal_Contacts Encoding
    InfFrontal_byModel_data = [sum(sig_rstdCue_Logical(InfFrontal_Contacts))/sum(sum(InfFrontal_Contacts))*100, sum(sig_rstOutcome_Logical(InfFrontal_Contacts))/sum(sum(InfFrontal_Contacts))*100];
    % insula_Contacts Encoding
    insula_byModel_data = [sum(sig_rstdCue_Logical(insula_Contacts))/sum(sum(insula_Contacts))*100, sum(sig_rstOutcome_Logical(insula_Contacts))/sum(sum(insula_Contacts))*100];
    % MedFrontal_Contacts Encoding
    MedFrontal_byModel_data = [sum(sig_rstdCue_Logical(MedFrontal_Contacts))/sum(sum(MedFrontal_Contacts))*100, sum(sig_rstOutcome_Logical(MedFrontal_Contacts))/sum(sum(MedFrontal_Contacts))*100];
    % NAccumbens_Contacts Encoding
    NAccumbens_byModel_data = [sum(sig_rstdCue_Logical(NAccumbens_Contacts))/sum(sum(NAccumbens_Contacts))*100, sum(sig_rstOutcome_Logical(NAccumbens_Contacts))/sum(sum(NAccumbens_Contacts))*100];
    % OrbFrontal_Contacts Encoding
    OrbFrontal_byModel_data = [sum(sig_rstdCue_Logical(OrbFrontal_Contacts))/sum(sum(OrbFrontal_Contacts))*100, sum(sig_rstOutcome_Logical(OrbFrontal_Contacts))/sum(sum(OrbFrontal_Contacts))*100];
    % striatum_Contacts Encoding
    striatum_byModel_data = [sum(sig_rstdCue_Logical(striatum_Contacts))/sum(sum(striatum_Contacts))*100, sum(sig_rstOutcome_Logical(striatum_Contacts))/sum(sum(striatum_Contacts))*100];
    % SupFrontal_Contacts Encoding
    SupFrontal_byModel_data = [sum(sig_rstdCue_Logical(SupFrontal_Contacts))/sum(sum(SupFrontal_Contacts))*100, sum(sig_rstOutcome_Logical(SupFrontal_Contacts))/sum(sum(SupFrontal_Contacts))*100];
    % thalamus_Contacts Encoding
    thalamus_byModel_data = [sum(sig_rstdCue_Logical(thalamus_Contacts))/sum(sum(thalamus_Contacts))*100, sum(sig_rstOutcome_Logical(thalamus_Contacts))/sum(sum(thalamus_Contacts))*100];

    % Figure of encoding of every Brain Region by Each Model
    labels = {'rstd Cue', 'rstd Outcome'};
    figure(12345);
    subplot(3,4,1) % amygdala
    bar(amygdala_byModel_data, 'FaceColor', [0.678, 0.847, 0.902]);
    ylabel('Sig Amygdala Contacts (%)');
    set(gca, 'XTickLabel', labels);
    ylim([0 50])
    xtickangle(45);
    subplot(3,4,2) % cingulate
    bar(cingulate_byModel_data, 'FaceColor', [0.678, 0.847, 0.902]);
    ylabel('Sig Cingulate Contacts (%)');
    set(gca, 'XTickLabel', labels);
    ylim([0 50])
    xtickangle(45);
    subplot(3,4,3) % hippocampus
    bar(hippocampus_byModel_data, 'FaceColor', [0.678, 0.847, 0.902]);
    ylabel('Sig hippocampus Contacts (%)');
    set(gca, 'XTickLabel', labels);
    ylim([0 50])
    xtickangle(45);
    subplot(3,4,4) % InfFrontal
    bar(InfFrontal_byModel_data, 'FaceColor', [0.678, 0.847, 0.902]);
    ylabel('Sig InfFrontal Contacts (%)');
    set(gca, 'XTickLabel', labels);
    ylim([0 50])
    xtickangle(45);
    subplot(3,4,5) % insula
    bar(insula_byModel_data, 'FaceColor', [0.678, 0.847, 0.902]);
    ylabel('Sig insula Contacts (%)');
    set(gca, 'XTickLabel', labels);
    ylim([0 50])
    xtickangle(45);
    subplot(3,4,6) % MedFrontal
    bar(MedFrontal_byModel_data, 'FaceColor', [0.678, 0.847, 0.902]);
    ylabel('Sig MedFrontal Contacts (%)');
    set(gca, 'XTickLabel', labels);
    ylim([0 50])
    xtickangle(45);
    subplot(3,4,7) %  NAccumbens
    bar(NAccumbens_byModel_data, 'FaceColor', [0.678, 0.847, 0.902]);
    ylabel('Sig  NAccumbens Contacts');
    set(gca, 'XTickLabel', labels);
    ylim([0 50])
    xtickangle(45);
    subplot(3,4,8) % OrbFrontal
    bar(OrbFrontal_byModel_data, 'FaceColor', [0.678, 0.847, 0.902]);
    ylabel('Sig OrbFrontal Contacts (%)');
    set(gca, 'XTickLabel', labels);
    ylim([0 50])
    xtickangle(45);
    subplot(3,4,9) % striatum
    bar(striatum_byModel_data, 'FaceColor', [0.678, 0.847, 0.902]);
    ylabel('Sig striatum Contacts (%)');
    set(gca, 'XTickLabel', labels);
    ylim([0 50])
    xtickangle(45);
    subplot(3,4,10) % SupFrontal
    bar(SupFrontal_byModel_data, 'FaceColor', [0.678, 0.847, 0.902]);
    ylabel('Sig SupFrontal Contacts (%)');
    set(gca, 'XTickLabel', labels);
    ylim([0 50])
    xtickangle(45);
    subplot(3,4,11) % Thalamus
    bar(thalamus_byModel_data, 'FaceColor', [0.678, 0.847, 0.902]);
    ylabel('Sig Thalamus Contacts (%)');
    set(gca, 'XTickLabel', labels);
    ylim([0 50])
    xtickangle(45);

    % Getting Dataframes for each Model by Region
    % rstdSurprise Encoding
    rstdCue_byRegion_data = [sum(sig_rstdCue_Logical(amygdala_Contacts))/sum(sum(amygdala_Contacts))*100, sum(sig_rstdCue_Logical(cingulate_Contacts))/sum(sum(cingulate_Contacts))*100,...
        sum(sig_rstdCue_Logical(hippocampus_Contacts))/sum(sum(hippocampus_Contacts))*100, sum(sig_rstdCue_Logical(InfFrontal_Contacts))/sum(sum(InfFrontal_Contacts))*100,...
        sum(sig_rstdCue_Logical(insula_Contacts))/sum(sum(insula_Contacts))*100, sum(sig_rstdCue_Logical(MedFrontal_Contacts))/sum(sum(MedFrontal_Contacts))*100,...
        sum(sig_rstdCue_Logical(NAccumbens_Contacts))/sum(sum(NAccumbens_Contacts))*100, sum(sig_rstdCue_Logical(OrbFrontal_Contacts))/sum(sum(OrbFrontal_Contacts))*100,...
        sum(sig_rstdCue_Logical(striatum_Contacts))/sum(sum(striatum_Contacts))*100, sum(sig_rstdCue_Logical(thalamus_Contacts))/sum(sum(thalamus_Contacts))*100];
    %rstdExpectation Encoding
    rstdOutcome_byRegion_data = [sum(sig_rstdOutcome_Logical(amygdala_Contacts))/sum(sum(amygdala_Contacts))*100, sum(sig_rstdOutcome_Logical(cingulate_Contacts))/sum(sum(cingulate_Contacts))*100,...
        sum(sig_rstdOutcome_Logical(hippocampus_Contacts))/sum(sum(hippocampus_Contacts))*100, sum(sig_rstdOutcome_Logical(InfFrontal_Contacts))/sum(sum(InfFrontal_Contacts))*100,...
        sum(sig_rstdOutcome_Logical(insula_Contacts))/sum(sum(insula_Contacts))*100, sum(sig_rstdOutcome_Logical(MedFrontal_Contacts))/sum(sum(MedFrontal_Contacts))*100,...
        sum(sig_rstdOutcome_Logical(NAccumbens_Contacts))/sum(sum(NAccumbens_Contacts))*100, sum(sig_rstdOutcome_Logical(OrbFrontal_Contacts))/sum(sum(OrbFrontal_Contacts))*100,...
        sum(sig_rstdOutcome_Logical(striatum_Contacts))/sum(sum(striatum_Contacts))*100, sum(sig_rstdOutcome_Logical(thalamus_Contacts))/sum(sum(thalamus_Contacts))*100];
   
    % Figure of encoding of every Brain Region by Each Model
    labels = {'amygdala', 'cingulate','hippocampus', 'infFrontal','insula','medFrontal', 'nAccumbens', 'orbFrontal', 'striatum', 'thalamus'};
    figure(123456);
    subplot(1,2,1) % rstd_Surprise
    bar(rstdCue_byRegion_data, 'FaceColor', [0.678, 0.847, 0.902]);
    ylabel('sig rstd cue contacts (%)');
    set(gca, 'XTickLabel', labels);
    ylim([0 max(rstdCue_byRegion_data)+10])
    xtickangle(45);
    subplot(1,2,2) % rstd_Expectation
    bar(rstdOutcome_byRegion_data, 'FaceColor', [0.678, 0.847, 0.902]);
    ylabel('sig rstd outcome contacts (%)');
    set(gca, 'XTickLabel', labels);
    ylim([0 max(rstdOutcome_byRegion_data)+10])
    xtickangle(45);
    
%% ~~~~~~~~~~~~~~ Examining Encoding Strength of Different Brain Regions ~~~~~~~~~~~~~~~~~~~~~~ %%

% (1) Flatten trodeLabels_All
trodeLabels_All_Flat = [];
for i = 1:size(trodeLabels_All, 1)
    rowLabels = trodeLabels_All(i, :);
    trodeLabels_All_Flat = [trodeLabels_All_Flat; rowLabels(:)];
end
trodeLabels_All_Flat = trodeLabels_All_Flat(~cellfun('isempty', trodeLabels_All_Flat));

% (2) Load Electrode Projections
final_ElecXYZProj = load('\\155.100.91.44\d\Data\Rhiannon\BART_RLDM_outputs\BrainRegions\final_ElecXYZProj');
ElecXYZProj_All = final_ElecXYZProj.final_ElecXYZProj;

% (3) Get RSTD Metric Neural to Project onto Contacts (GOT TO HERE).
amygdala_VE_rstdMetric = RSTD_metric_VE_neural_long(strcmp(trodeLabels_All_Flat, "Amygdala"));
cingulate_VE_rstdMetric = RSTD_metric_VE_neural_long(strcmp(trodeLabels_All_Flat, "Cingulate"));
hippocampus_VE_rstdMetric = RSTD_metric_VE_neural_long(strcmp(trodeLabels_All_Flat, "Hippocampus"));
infFrontal_VE_rstdMetric = RSTD_metric_VE_neural_long(strcmp(trodeLabels_All_Flat, "InfFrontal"));
insula_VE_rstdMetric = RSTD_metric_VE_neural_long(strcmp(trodeLabels_All_Flat, "Insula"));
medFrontal_VE_rstdMetric = RSTD_metric_VE_neural_long(strcmp(trodeLabels_All_Flat, "MedFrontal"));
nAccumbens_VE_rstdMetric = RSTD_metric_VE_neural_long(strcmp(trodeLabels_All_Flat, "N.Accumbens"));
orbFrontal_VE_rstdMetric = RSTD_metric_VE_neural_long(strcmp(trodeLabels_All_Flat, "OrbFrontal"));
striatum_VE_rstdMetric = RSTD_metric_VE_neural_long(strcmp(trodeLabels_All_Flat, "Striatum"));
supFrontal_VE_rstdMetric = RSTD_metric_VE_neural_long(strcmp(trodeLabels_All_Flat, "SupFrontal"));
thalamus_VE_rstdMetric = RSTD_metric_VE_neural_long(strcmp(trodeLabels_All_Flat, "Thalamus"));
motor_VE_rstdMetric = RSTD_metric_VE_neural_long(strcmp(trodeLabels_All_Flat, "Motor"));
unknown_VE_rstdMetric = RSTD_metric_VE_neural_long(strcmp(trodeLabels_All_Flat, "Unknown"));

amygdala_VE_rstdMetric = cell2mat(amygdala_VE_rstdMetric);
cingulate_VE_rstdMetric = cell2mat(cingulate_VE_rstdMetric);
hippocampus_VE_rstdMetric = cell2mat(hippocampus_VE_rstdMetric);
infFrontal_VE_rstdMetric = cell2mat(infFrontal_VE_rstdMetric);
insula_VE_rstdMetric = cell2mat(insula_VE_rstdMetric);
medFrontal_VE_rstdMetric = cell2mat(medFrontal_VE_rstdMetric);
nAccumbens_VE_rstdMetric = cell2mat(nAccumbens_VE_rstdMetric);
orbFrontal_VE_rstdMetric = cell2mat(orbFrontal_VE_rstdMetric);
striatum_VE_rstdMetric = cell2mat(striatum_VE_rstdMetric);
supFrontal_VE_rstdMetric = cell2mat(supFrontal_VE_rstdMetric);
thalamus_VE_rstdMetric = cell2mat(thalamus_VE_rstdMetric);
motor_VE_rstdMetric = cell2mat(motor_VE_rstdMetric);
unknown_VE_rstdMetric = cell2mat(unknown_VE_rstdMetric);


% TO DO: average rstdMetric by region?
% Calculate the average RSTD metric for each region
data = [
    mean(amygdala_VE_rstdMetric(amygdala_VE_rstdMetric ~= 0)), ...
    mean(cingulate_VE_rstdMetric(cingulate_VE_rstdMetric ~= 0)), ...
    mean(hippocampus_VE_rstdMetric(hippocampus_VE_rstdMetric ~= 0)), ...
    mean(infFrontal_VE_rstdMetric(infFrontal_VE_rstdMetric ~= 0)), ...
    mean(insula_VE_rstdMetric(insula_VE_rstdMetric ~= 0)), ...
    mean(medFrontal_VE_rstdMetric(medFrontal_VE_rstdMetric ~= 0)), ...
    mean(nAccumbens_VE_rstdMetric(nAccumbens_VE_rstdMetric ~= 0)), ...
    mean(orbFrontal_VE_rstdMetric(orbFrontal_VE_rstdMetric ~= 0)), ...
    mean(striatum_VE_rstdMetric(striatum_VE_rstdMetric ~= 0)), ...
    mean(supFrontal_VE_rstdMetric(supFrontal_VE_rstdMetric ~= 0)), ...
    mean(thalamus_VE_rstdMetric(thalamus_VE_rstdMetric ~= 0))];

data_posAlpha = [
    mean(amygdala_VE_rstdMetric(amygdala_VE_rstdMetric > 0)), ...
    mean(cingulate_VE_rstdMetric(cingulate_VE_rstdMetric > 0)), ...
    mean(hippocampus_VE_rstdMetric(hippocampus_VE_rstdMetric > 0)), ...
    mean(infFrontal_VE_rstdMetric(infFrontal_VE_rstdMetric > 0)), ...
    mean(insula_VE_rstdMetric(insula_VE_rstdMetric > 0)), ...
    mean(medFrontal_VE_rstdMetric(medFrontal_VE_rstdMetric > 0)), ...
    mean(nAccumbens_VE_rstdMetric(nAccumbens_VE_rstdMetric > 0)), ...
    mean(orbFrontal_VE_rstdMetric(orbFrontal_VE_rstdMetric > 0)), ...
    mean(striatum_VE_rstdMetric(striatum_VE_rstdMetric > 0)), ...
    mean(supFrontal_VE_rstdMetric(supFrontal_VE_rstdMetric > 0)), ...
    mean(thalamus_VE_rstdMetric(thalamus_VE_rstdMetric > 0))];

data_negAlpha = [
    mean(amygdala_VE_rstdMetric(amygdala_VE_rstdMetric < 0)), ...
    mean(cingulate_VE_rstdMetric(cingulate_VE_rstdMetric < 0)), ...
    mean(hippocampus_VE_rstdMetric(hippocampus_VE_rstdMetric < 0)), ...
    mean(infFrontal_VE_rstdMetric(infFrontal_VE_rstdMetric < 0)), ...
    mean(insula_VE_rstdMetric(insula_VE_rstdMetric < 0)), ...
    mean(medFrontal_VE_rstdMetric(medFrontal_VE_rstdMetric < 0)), ...
    mean(nAccumbens_VE_rstdMetric(nAccumbens_VE_rstdMetric < 0)), ...
    mean(orbFrontal_VE_rstdMetric(orbFrontal_VE_rstdMetric < 0)), ...
    mean(striatum_VE_rstdMetric(striatum_VE_rstdMetric < 0)), ...
    mean(supFrontal_VE_rstdMetric(supFrontal_VE_rstdMetric < 0)), ...
    mean(thalamus_VE_rstdMetric(thalamus_VE_rstdMetric < 0))];

% Define region names for the x-axis
regions = {'Amygdala', 'Cingulate', 'Hippocampus', 'Inf. Frontal', 'Insula', ...
           'Med. Frontal', 'NAccumbens', 'Orb. Frontal', 'Striatum', 'Sup. Frontal', 'Thalamus'};
figure;
b = bar(data, 'FaceColor', 'flat');
colormap cool;
colorbar;
caxis([-1 1]);  % Set color limits between -1 and 1
color_map = cool(256); 
for i = 1:length(data)
    normalizedValue = (data(i) + 1) / 2;  % Normalize between -1 and 1
    colorIndex = round(normalizedValue * (size(color_map, 1) - 1)) + 1;
    barColor = color_map(colorIndex, :);
    b.FaceColor = 'flat';
    b.CData(i, :) = barColor;  
end
set(gca, 'XTickLabel', regions);
xtickangle(45); 

title('Average RSTD Metric for Each Region');
xlabel('Regions');
ylabel('Average RSTD Metric');
ylim([-1 1]);


% NEXT FIGURE

% Define metrics
regionMetrics = {
    amygdala_VE_rstdMetric, ...
    cingulate_VE_rstdMetric, ...
    hippocampus_VE_rstdMetric, ...
    infFrontal_VE_rstdMetric, ...
    insula_VE_rstdMetric, ...
    medFrontal_VE_rstdMetric, ...
    nAccumbens_VE_rstdMetric, ...
    orbFrontal_VE_rstdMetric, ...
    striatum_VE_rstdMetric, ...
    supFrontal_VE_rstdMetric, ...
    thalamus_VE_rstdMetric
};

regions = {'Amygdala', 'Cingulate', 'Hippocampus', 'Inf. Frontal', 'Insula', ...
           'Med. Frontal', 'NAccumbens', 'Orb. Frontal', 'Striatum', 'Sup. Frontal', 'Thalamus'};

numRegions = length(data_negAlpha);
regionMeans = zeros(1, numRegions);
regionSEMs = zeros(1, numRegions);
regionNs = zeros(1, numRegions);

for i = 1:numRegions
    vals = data_negAlpha(i);
    validVals = vals(vals ~= 0);
    n = length(validVals);
    regionMeans(i) = mean(validVals);
    regionSEMs(i) = std(validVals) / sqrt(n);
    regionNs(i) = n;
end

% Optionally compute a global weighted mean (across all data)
allValues = [];
for i = 1:numRegions
    allValues = [allValues; data_negAlpha(i)];
end
globalMean = mean(allValues);
globalSEM = std(allValues) / sqrt(length(allValues));

% === PLOTTING ===
figure;
b = bar(regionMeans, 'FaceColor', 'flat');
hold on;
colormap cool;
colorbar;
caxis([-1 1]);
color_map = cool(256);
for i = 1:numRegions
    normVal = (regionMeans(i) + 1) / 2;  % normalize to [0,1]
    colorIdx = round(normVal * (size(color_map,1) - 1)) + 1;
    b.CData(i,:) = color_map(colorIdx, :);
end

% Error bars
x = 1:numRegions;
errorbar(x, regionMeans, regionSEMs, 'k.', 'LineWidth', 1.2);

% Annotate positive means above the bars
for i = 1:numRegions
    posVal = data_posAlpha(i);
    text(x(i), regionMeans(i) + regionSEMs(i) + 0.12, ...
        sprintf('+%.2f', posVal), ...
        'HorizontalAlignment', 'center', 'FontSize', 8, 'FontWeight', 'bold', 'Color', [0 0.5 0]);
end

% Annotate number of electrodes
for i = 1:numRegions
    text(x(i), regionMeans(i) + regionSEMs(i) + 0.05, ...
        sprintf('n=%d', regionNs(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', [0.3 0.3 0.3]);
end

% Labels and axes
set(gca, 'XTick', x, 'XTickLabel', regions);
xtickangle(45);
xlabel('Regions');
ylabel('Average RSTD Metric');
title(sprintf('RSTD Metric per Region (Global Weighted Mean = %.3f)', globalMean));
ylim([-1 1]);
hold off;


% (4) Normalizing RSTD_Neural by RSTD_behavior
% (3) Get RSTD Metric Neural to Project onto Contacts
amygdala_rstdMetric_NORM = RSTD_metric_neural_NORM_long(strcmp(trodeLabels_All_Flat, "Amygdala"));
cingulate_rstdMetric_NORM = RSTD_metric_neural_NORM_long(strcmp(trodeLabels_All_Flat, "Cingulate"));
hippocampus_rstdMetric_NORM = RSTD_metric_neural_NORM_long(strcmp(trodeLabels_All_Flat, "Hippocampus"));
infFrontal_rstdMetric_NORM = RSTD_metric_neural_NORM_long(strcmp(trodeLabels_All_Flat, "InfFrontal"));
insula_rstdMetric_NORM = RSTD_metric_neural_NORM_long(strcmp(trodeLabels_All_Flat, "Insula"));
medFrontal_rstdMetric_NORM = RSTD_metric_neural_NORM_long(strcmp(trodeLabels_All_Flat, "MedFrontal"));
nAccumbens_rstdMetric_NORM = RSTD_metric_neural_NORM_long(strcmp(trodeLabels_All_Flat, "N.Accumbens"));
orbFrontal_rstdMetric_NORM = RSTD_metric_neural_NORM_long(strcmp(trodeLabels_All_Flat, "OrbFrontal"));
striatum_rstdMetric_NORM = RSTD_metric_neural_NORM_long(strcmp(trodeLabels_All_Flat, "Striatum"));
supFrontal_rstdMetric_NORM = RSTD_metric_neural_NORM_long(strcmp(trodeLabels_All_Flat, "SupFrontal"));
thalamus_rstdMetric_NORM = RSTD_metric_neural_NORM_long(strcmp(trodeLabels_All_Flat, "Thalamus"));
motor_rstdMetric_NORM = RSTD_metric_neural_NORM_long(strcmp(trodeLabels_All_Flat, "Motor"));
unknown_rstdMetric_NORM = RSTD_metric_neural_NORM_long(strcmp(trodeLabels_All_Flat, "Unknown"));

amygdala_rstdMetric_NORM = cell2mat(RSTD_metric_neural_NORM_long(strcmp(trodeLabels_All_Flat, "Amygdala")));
cingulate_rstdMetric_NORM = cell2mat(RSTD_metric_neural_NORM_long(strcmp(trodeLabels_All_Flat, "Cingulate")));
hippocampus_rstdMetric_NORM = cell2mat(RSTD_metric_neural_NORM_long(strcmp(trodeLabels_All_Flat, "Hippocampus")));
infFrontal_rstdMetric_NORM = cell2mat(RSTD_metric_neural_NORM_long(strcmp(trodeLabels_All_Flat, "InfFrontal")));
insula_rstdMetric_NORM = cell2mat(RSTD_metric_neural_NORM_long(strcmp(trodeLabels_All_Flat, "Insula")));
medFrontal_rstdMetric_NORM = cell2mat(RSTD_metric_neural_NORM_long(strcmp(trodeLabels_All_Flat, "MedFrontal")));
nAccumbens_rstdMetric_NORM = cell2mat(RSTD_metric_neural_NORM_long(strcmp(trodeLabels_All_Flat, "N.Accumbens")));
orbFrontal_rstdMetric_NORM = cell2mat(RSTD_metric_neural_NORM_long(strcmp(trodeLabels_All_Flat, "OrbFrontal")));
striatum_rstdMetric_NORM = cell2mat(RSTD_metric_neural_NORM_long(strcmp(trodeLabels_All_Flat, "Striatum")));
supFrontal_rstdMetric_NORM = cell2mat(RSTD_metric_neural_NORM_long(strcmp(trodeLabels_All_Flat, "SupFrontal")));
thalamus_rstdMetric_NORM = cell2mat(RSTD_metric_neural_NORM_long(strcmp(trodeLabels_All_Flat, "Thalamus")));

% Calculate the average normalized RSTD metric for each region
data_norm = [
    nanmean(amygdala_rstdMetric_NORM), nanmean(cingulate_rstdMetric_NORM), nanmean(hippocampus_rstdMetric_NORM), ...
    nanmean(infFrontal_rstdMetric_NORM), nanmean(insula_rstdMetric_NORM), nanmean(medFrontal_rstdMetric_NORM), ...
    nanmean(nAccumbens_rstdMetric_NORM), nanmean(orbFrontal_rstdMetric_NORM), nanmean(striatum_rstdMetric_NORM), ...
    nanmean(supFrontal_rstdMetric_NORM), nanmean(thalamus_rstdMetric_NORM)
];

% Define region names for the x-axis
regions = {'Amygdala', 'Cingulate', 'Hippocampus', 'Inf. Frontal', 'Insula', ...
           'Med. Frontal', 'NAccumbens', 'Orb. Frontal', 'Striatum', 'Sup. Frontal', 'Thalamus'};

% Create the bar plot
figure;
b = bar(data_norm, 'FaceColor', 'flat');

% Apply colormap and color limits
colormap cool;
colorbar;
caxis([-1 1]);  % Set color limits between -1 and 1

% Get the colormap matrix (an Nx3 matrix where N is the number of colors)
color_map = cool(256);  % 256 colors from the cool colormap

% Loop through each bar and manually set the color based on its value
for i = 1:length(data_norm)
    % Normalize the RSTD value to map between 0 and 1 for the colormap
    normalizedValue = (data_norm(i) + 1) / 2;  % Normalize between -1 and 1
    
    % Get the corresponding RGB color from the colormap
    colorIndex = round(normalizedValue * (size(color_map, 1) - 1)) + 1;
    barColor = color_map(colorIndex, :);
    
    % Set the color for the i-th bar
    b.FaceColor = 'flat';  % Ensure the FaceColor property is set to 'flat'
    b.CData(i, :) = barColor;  % Assign the color to the i-th bar
end

% Set x-tick labels for regions
set(gca, 'XTickLabel', regions);
xtickangle(45);  % Rotate the x-axis labels for better readability

% Add title and axis labels
title('Average Normalized RSTD Metric for Each Region');
xlabel('Regions');
ylabel('Average Normalized RSTD Metric');

% Set y-axis limits to range from -1 to 1
ylim([-1 1]);

% loading brains
mainPath = '\\155.100.91.44\D\Data\preProcessed\BART_preprocessed\';
load(fullfile(mainPath,ptID,'Imaging','Registered','Surfaces.mat'));

% Creating index to plot single hemispheres
% Step 1: Find indices of vertices in the left hemisphere
left_idx = find(BrainSurfRaw.vertices(:,1) < 0);
right_idx = find(BrainSurfRaw.vertices(:,1) > 0);

% Step 2: Create a mask for faces in both hemis
face_mask_left = all(ismember(BrainSurfRaw.faces, left_idx), 2);
face_mask_right = all(ismember(BrainSurfRaw.faces, right_idx), 2);

% Step 3: Extract faces and remap vertex indices
faces_left = BrainSurfRaw.faces(face_mask_left, :);
faces_right = BrainSurfRaw.faces(face_mask_right, :);

% Step 4: Remap subset of vertices
[uniqueLeft_idx, ~, new_faces_left] = unique(faces_left(:));
[uniqueRight_idx, ~, new_faces_right] = unique(faces_right(:));
vertices_left = BrainSurfRaw.vertices(uniqueLeft_idx, :);
vertices_right = BrainSurfRaw.vertices(uniqueRight_idx, :);
faces_left = reshape(new_faces_left, size(faces_left));
faces_right = reshape(new_faces_right, size(faces_right));

% Encoding for RSTD neural in ROIs
figure(101)
% Cingulate Contacts
subplot(4,4,1)
subtitle('cingulate')
hold on
cingulate_data = ElecXYZProj_All(strcmp(trodeLabels_All_Flat, "Cingulate"), :);
s1 = scatter3(cingulate_data(:,1), cingulate_data(:,2), cingulate_data(:,3), ...
    70, cingulate_VE_rstdMetric, 'filled', 'MarkerEdgeColor', 'k');
colorbar;
colormap cool
view(3);
xlabel('Lateral --- Medial --- Lateral');
ylabel('Posterior --- Anterior');
zlabel('Inferior --- Superior');
% plotting actual glass brain....
patchs = patch('Faces', BrainSurfRaw.faces, 'Vertices', BrainSurfRaw.vertices, 'edgecolor', 'none', 'facecolor', 'flat', 'facealpha', .2);
facecolor = repmat([1 1 1],length(BrainSurfRaw.faces),1);
set(patchs, 'FaceVertexCData', facecolor);
camlight
lighting gouraud
% Amygdala Contacts
subplot(4,4,2)
subtitle('amygdala')
hold on
amygdala_data = ElecXYZProj_All(strcmp(trodeLabels_All_Flat, "Amygdala"), :);
s2 = scatter3(amygdala_data(:,1), amygdala_data(:,2), amygdala_data(:,3), ...
    70, amygdala_VE_rstdMetric, 'filled', 'MarkerEdgeColor', 'k');
colorbar;
colormap cool
view(3);
xlabel('Lateral --- Medial --- Lateral');
ylabel('Posterior --- Anterior');
zlabel('Inferior --- Superior');
% plotting actual glass brain....
patchs = patch('Faces', BrainSurfRaw.faces, 'Vertices', BrainSurfRaw.vertices, 'edgecolor', 'none', 'facecolor', 'flat', 'facealpha', .2);
facecolor = repmat([1 1 1],length(BrainSurfRaw.faces),1);
set(patchs, 'FaceVertexCData', facecolor);
camlight
lighting gouraud
% Hippocampus Contacts
subplot(4,4,3)
subtitle('hippocampus')
hold on
hippocampus_data = ElecXYZProj_All(strcmp(trodeLabels_All_Flat, "Hippocampus"), :);
s3 = scatter3(hippocampus_data(:,1), hippocampus_data(:,2), hippocampus_data(:,3), ...
    70, hippocampus_VE_rstdMetric, 'filled', 'MarkerEdgeColor', 'k');
colorbar;
view(3);
xlabel('Lateral --- Medial --- Lateral');
ylabel('Posterior --- Anterior');
zlabel('Inferior --- Superior');
% plotting actual glass brain....
patchs = patch('Faces', BrainSurfRaw.faces, 'Vertices', BrainSurfRaw.vertices, 'edgecolor', 'none', 'facecolor', 'flat', 'facealpha', .2);
facecolor = repmat([1 1 1],length(BrainSurfRaw.faces),1);
set(patchs, 'FaceVertexCData', facecolor);
camlight
lighting gouraud
% Insula Contacts
subplot(4,4,4)
subtitle('insula')
hold on
insula_data = ElecXYZProj_All(strcmp(trodeLabels_All_Flat, "Insula"), :);
s4 = scatter3(insula_data(:,1), insula_data(:,2), insula_data(:,3), ...
    70, insula_VE_rstdMetric, 'filled', 'MarkerEdgeColor', 'k');
colorbar;
colormap cool;
view(3);
xlabel('Lateral --- Medial --- Lateral');
ylabel('Posterior --- Anterior');
zlabel('Inferior --- Superior');
% plotting actual glass brain....
patchs = patch('Faces', BrainSurfRaw.faces, 'Vertices', BrainSurfRaw.vertices, 'edgecolor', 'none', 'facecolor', 'flat', 'facealpha', .2);
facecolor = repmat([1 1 1],length(BrainSurfRaw.faces),1);
set(patchs, 'FaceVertexCData', facecolor);
camlight
lighting gouraud
% InfFrontal Contacts
subplot(4,4,5)
subtitle('InfFrontal')
hold on
infFrontal_data = ElecXYZProj_All(strcmp(trodeLabels_All_Flat, "InfFrontal"), :);
s5 = scatter3(infFrontal_data(:,1), infFrontal_data(:,2), infFrontal_data(:,3), ...
    70, infFrontal_VE_rstdMetric, 'filled', 'MarkerEdgeColor', 'k');
colorbar;
view(3);
xlabel('Lateral --- Medial --- Lateral');
ylabel('Posterior --- Anterior');
zlabel('Inferior --- Superior');
% plotting actual glass brain....
patchs = patch('Faces', BrainSurfRaw.faces, 'Vertices', BrainSurfRaw.vertices, 'edgecolor', 'none', 'facecolor', 'flat', 'facealpha', .2);
facecolor = repmat([1 1 1],length(BrainSurfRaw.faces),1);
set(patchs, 'FaceVertexCData', facecolor);
camlight
lighting gouraud
% MedFrontal Contacts
subplot(4,4,6)
subtitle('medFrontal')
hold on
medFrontal_data = ElecXYZProj_All(strcmp(trodeLabels_All_Flat, "MedFrontal"), :);
s6 = scatter3(medFrontal_data(:,1), medFrontal_data(:,2), medFrontal_data(:,3), ...
    70, medFrontal_VE_rstdMetric, 'filled', 'MarkerEdgeColor', 'k');
colorbar;
view(3);
xlabel('Lateral --- Medial --- Lateral');
ylabel('Posterior --- Anterior');
zlabel('Inferior --- Superior');
% plotting actual glass brain....
patchs = patch('Faces', BrainSurfRaw.faces, 'Vertices', BrainSurfRaw.vertices, 'edgecolor', 'none', 'facecolor', 'flat', 'facealpha', .2);
facecolor = repmat([1 1 1],length(BrainSurfRaw.faces),1);
set(patchs, 'FaceVertexCData', facecolor);
camlight
lighting gouraud
% NAccumbens Contacts
subplot(4,4,7)
subtitle('NAccumbens')
hold on
nAccumbens_data = ElecXYZProj_All(strcmp(trodeLabels_All_Flat, "N.Accumbens"), :);
s7 = scatter3(nAccumbens_data(:,1), nAccumbens_data(:,2), nAccumbens_data(:,3), ...
    70, nAccumbens_VE_rstdMetric, 'filled', 'MarkerEdgeColor', 'k');
colorbar;
view(3);
xlabel('Lateral --- Medial --- Lateral');
ylabel('Posterior --- Anterior');
zlabel('Inferior --- Superior');
% plotting actual glass brain....
patchs = patch('Faces', BrainSurfRaw.faces, 'Vertices', BrainSurfRaw.vertices, 'edgecolor', 'none', 'facecolor', 'flat', 'facealpha', .2);
facecolor = repmat([1 1 1],length(BrainSurfRaw.faces),1);
set(patchs, 'FaceVertexCData', facecolor);
camlight
lighting gouraud
% OrbFrontal Contacts
subplot(4,4,8)
subtitle('orbFrontal')
hold on
orbFrontal_data = ElecXYZProj_All(strcmp(trodeLabels_All_Flat, "OrbFrontal"), :);
s8 = scatter3(orbFrontal_data(:,1), orbFrontal_data(:,2), orbFrontal_data(:,3), ...
    70, orbFrontal_VE_rstdMetric, 'filled', 'MarkerEdgeColor', 'k');
colorbar;
view(3);
xlabel('Lateral --- Medial --- Lateral');
ylabel('Posterior --- Anterior');
zlabel('Inferior --- Superior');
% plotting actual glass brain....
patchs = patch('Faces', BrainSurfRaw.faces, 'Vertices', BrainSurfRaw.vertices, 'edgecolor', 'none', 'facecolor', 'flat', 'facealpha', .2);
facecolor = repmat([1 1 1],length(BrainSurfRaw.faces),1);
set(patchs, 'FaceVertexCData', facecolor);
camlight
lighting gouraud
% Striatum Contacts
subplot(4,4,9)
subtitle('striatum')
hold on
striatum_data = ElecXYZProj_All(strcmp(trodeLabels_All_Flat, "Striatum"), :);
s9 = scatter3(striatum_data(:,1), striatum_data(:,2), striatum_data(:,3), ...
    70, striatum_VE_rstdMetric, 'filled', 'MarkerEdgeColor', 'k');
colorbar;
view(3);
xlabel('Lateral --- Medial --- Lateral');
ylabel('Posterior --- Anterior');
zlabel('Inferior --- Superior');
% plotting actual glass brain....
patchs = patch('Faces', BrainSurfRaw.faces, 'Vertices', BrainSurfRaw.vertices, 'edgecolor', 'none', 'facecolor', 'flat', 'facealpha', .2);
facecolor = repmat([1 1 1],length(BrainSurfRaw.faces),1);
set(patchs, 'FaceVertexCData', facecolor);
camlight
lighting gouraud
% SupFrontal Contacts
subplot(4,4,10)
subtitle('supFrontal')
hold on
supFrontal_data = ElecXYZProj_All(strcmp(trodeLabels_All_Flat, "SupFrontal"), :);
s10 = scatter3(supFrontal_data(:,1), supFrontal_data(:,2), supFrontal_data(:,3), ...
    70, supFrontal_VE_rstdMetric, 'filled', 'MarkerEdgeColor', 'k');
colorbar;
view(3);
xlabel('Lateral --- Medial --- Lateral');
ylabel('Posterior --- Anterior');
zlabel('Inferior --- Superior');
% plotting actual glass brain....
patchs = patch('Faces', BrainSurfRaw.faces, 'Vertices', BrainSurfRaw.vertices, 'edgecolor', 'none', 'facecolor', 'flat', 'facealpha', .2);
facecolor = repmat([1 1 1],length(BrainSurfRaw.faces),1);
set(patchs, 'FaceVertexCData', facecolor);
camlight
lighting gouraud
% Thalamus Contacts
subplot(4,4,11)
subtitle('thalamus')
hold on
thalamus_data = ElecXYZProj_All(strcmp(trodeLabels_All_Flat, "Thalamus"), :);
s11 = scatter3(thalamus_data(:,1), thalamus_data(:,2), thalamus_data(:,3), ...
    70, thalamus_VE_rstdMetric, 'filled', 'MarkerEdgeColor', 'k');
colorbar;
view(3);
xlabel('Lateral --- Medial --- Lateral');
ylabel('Posterior --- Anterior');
zlabel('Inferior --- Superior');
% plotting actual glass brain....
patchs = patch('Faces', BrainSurfRaw.faces, 'Vertices', BrainSurfRaw.vertices, 'edgecolor', 'none', 'facecolor', 'flat', 'facealpha', .2);
facecolor = repmat([1 1 1],length(BrainSurfRaw.faces),1);
set(patchs, 'FaceVertexCData', facecolor);
camlight
lighting gouraud


%% Examining different anatomical gradients throughout the brain

% (1) create logical values to isolate hemispheres
rightInsula = (insula_data(:,1) > 0);
rightCingulate = (cingulate_data(:,1)> 0); % right ACC: % ML (:,1) > 0; left ACC: % ML (:,1) < 0
rightHippocampus = (hippocampus_data(:,1)> 0);
rightInfFrontal = (infFrontal_data(:,1)> 0);
rightMedFrontal = (medFrontal_data(:,1)> 0);
rightSupFrontal = (supFrontal_data(:,1)> 0);
rightOrbFrontal =(orbFrontal_data(:,1)> 0);
rightStriatum = (striatum_data(:,1)> 0);
rightNAccumbens = (nAccumbens_data(:,1)> 0);
rightAmygdala = (amygdala_data(:,1)> 0);
rightThalamus =(thalamus_data(:,1)> 0);

% (2) Cingulate: Splitting ACC into dorsal ACC and pg/sgACC
right_dACC = (cingulate_data(:,1) > 0) & (cingulate_data(:,2) > 0) & (cingulate_data(:,3) > 15); % (A) rightDACC: % ML (:,1) > 0 & AP (:,2) > 0 & SI (:,3) > 15 
right_psgACC = (cingulate_data(:,1) > 0) & (cingulate_data(:,2) > 0) & (cingulate_data(:,3) < 15); % (B) rightPGACC: % ML (:,1) > 0 & AP (:,2) > 0 & SI (:,3) < 15 
left_dACC = (cingulate_data(:,1) < 0) & (cingulate_data(:,2) > 0) & (cingulate_data(:,3) > 15); % (A) leftDACC: % ML (:,1) < 0 & AP (:,2) > 0 & SI (:,3) > 15 
left_psgACC = (cingulate_data(:,1) < 0) & (cingulate_data(:,2) > 0) & (cingulate_data(:,3) < 15); % (B) leftPGACC: % ML (:,1) < 0 & AP (:,2) > 0 & SI (:,3) < 15 
PCC = (cingulate_data(:,2) < 0); % (B) leftPGACC: % ML (:,1) < 0 & AP (:,2) > 0 & SI (:,3) < 15 

% plotting Cingualte by hemis
figure(9)
subplot(2,1,1)
subtitle('right Cingulate')
hold on
s1 = scatter3(cingulate_data(rightCingulate,1), cingulate_data(rightCingulate,2), cingulate_data(rightCingulate,3), ...
    70, cingulate_VE_rstdMetric(rightCingulate), 'filled', 'MarkerEdgeColor', 'k');
colorbar;
colormap cool
%view(3);
xlabel('Lateral --- Medial --- Lateral');
ylabel('Anterior --- Posterior');
zlabel('Inferior --- Superior');
% plotting actual glass brain....
patchs = patch('Faces', faces_right, 'Vertices', vertices_right, 'edgecolor', 'none', 'facecolor', 'flat', 'facealpha', .2);
facecolor = repmat([1 1 1],length(faces_right),1);
set(patchs, 'FaceVertexCData', facecolor);
camlight 
lighting gouraud
view([-1 0 0]); % for medial view of right hemi -1
subplot(2,1,2)
subtitle('left Cingulate')
hold on
s1 = scatter3(cingulate_data(~rightCingulate,1), cingulate_data(~rightCingulate,2), cingulate_data(~rightCingulate,3), ...
    70, cingulate_VE_rstdMetric(~rightCingulate), 'filled', 'MarkerEdgeColor', 'k');
colorbar;
colormap cool
%view(3);
xlabel('Lateral --- Medial --- Lateral');
ylabel('Posterior --- Anterior');
zlabel('Inferior --- Superior');
% plotting actual glass brain....
patchs = patch('Faces', faces_left, 'Vertices', vertices_left, 'edgecolor', 'none', 'facecolor', 'flat', 'facealpha', .2);
facecolor = repmat([1 1 1],length(faces_left),1);
set(patchs, 'FaceVertexCData', facecolor);
camlight 
lighting gouraud
view([1 0 0]); % for medial view of left hemi 1


%(2) Plot brain figures for half brains and for dACC and psgACC averages
% plotting dACC by hemispheres
figure(7)
subplot(2,1,1)
subtitle('right dACC')
hold on
s1 = scatter3(cingulate_data(right_dACC,1), cingulate_data(right_dACC,2), cingulate_data(right_dACC,3), ...
    70, cingulate_VE_rstdMetric(right_dACC), 'filled', 'MarkerEdgeColor', 'k');
colorbar;
clim([-1 1]);
colormap cool
%view(3);
xlabel('Lateral --- Medial --- Lateral');
ylabel('Posterior --- Anterior');
zlabel('Inferior --- Superior');
% plotting actual glass brain....
patchs = patch('Faces', faces_right, 'Vertices', vertices_right, 'edgecolor', 'none', 'facecolor', 'flat', 'facealpha', .2);
facecolor = repmat([1 1 1],length(faces_right),1);
set(patchs, 'FaceVertexCData', facecolor);
camlight 
lighting gouraud
view([-1 0 0]); % for medial view of right hemi -1
subplot(2,1,2)
subtitle('left dACC')
hold on
s1 = scatter3(cingulate_data(left_dACC,1), cingulate_data(left_dACC,2), cingulate_data(left_dACC,3), ...
    70, cingulate_VE_rstdMetric(left_dACC), 'filled', 'MarkerEdgeColor', 'k');
colorbar;
clim([-1 1]);
colormap cool
%view(3);
xlabel('Lateral --- Medial --- Lateral');
ylabel('Posterior --- Anterior');
zlabel('Inferior --- Superior');
% plotting actual glass brain....
patchs = patch('Faces', faces_left, 'Vertices', vertices_left, 'edgecolor', 'none', 'facecolor', 'flat', 'facealpha', .2);
facecolor = repmat([1 1 1],length(faces_left),1);
set(patchs, 'FaceVertexCData', facecolor);
camlight 
lighting gouraud
view([1 0 0]); % for medial view of left hemi 1

% plotting psgACC by hemispheres
figure(8)
subplot(2,1,1)
subtitle('right psgACC')
hold on
s1 = scatter3(cingulate_data(right_psgACC,1), cingulate_data(right_psgACC,2), cingulate_data(right_psgACC,3), ...
    70, cingulate_VE_rstdMetric(right_psgACC), 'filled', 'MarkerEdgeColor', 'k');
colorbar;
colormap cool
%view(3);
xlabel('Lateral --- Medial --- Lateral');
ylabel('Posterior --- Anterior');
zlabel('Inferior --- Superior');
% plotting actual glass brain....
patchs = patch('Faces', faces_right, 'Vertices', vertices_right, 'edgecolor', 'none', 'facecolor', 'flat', 'facealpha', .2);
facecolor = repmat([1 1 1],length(faces_right),1);
set(patchs, 'FaceVertexCData', facecolor);
camlight 
lighting gouraud
view([-1 0 0]); % for medial view of right hemi -1
subplot(2,1,2)
subtitle('left psgACC')
hold on
s1 = scatter3(cingulate_data(left_psgACC,1), cingulate_data(left_psgACC,2), cingulate_data(left_psgACC,3), ...
    70, cingulate_VE_rstdMetric(left_psgACC), 'filled', 'MarkerEdgeColor', 'k');
colorbar;
colormap cool
%view(3);
xlabel('Lateral --- Medial --- Lateral');
ylabel('Posterior --- Anterior');
zlabel('Inferior --- Superior');
% plotting actual glass brain....
patchs = patch('Faces', faces_left, 'Vertices', vertices_left, 'edgecolor', 'none', 'facecolor', 'flat', 'facealpha', .2);
facecolor = repmat([1 1 1],length(faces_left),1);
set(patchs, 'FaceVertexCData', facecolor);
camlight 
lighting gouraud
view([1 0 0]); % for medial view of left hemi 1
%view([0 1 0]); % for  view front of left hemi 1

% Data
means = [mean(cingulate_VE_rstdMetric(left_psgACC)), mean(cingulate_VE_rstdMetric(right_psgACC)), mean(cingulate_VE_rstdMetric(left_dACC)), mean(cingulate_VE_rstdMetric(right_dACC)), mean(cingulate_VE_rstdMetric(PCC))];
labels = {'Left psgACC', 'Right psgACC', 'Left dACC', 'Right dACC', 'PCC'};
% Colormap and color limits
clim = [-1 1];                      % Color limits
cmap = colormap('cool');           % Get 'cool' colormap
nColors = size(cmap, 1);           % Number of colors in colormap
% Map means to color indices
color_indices = round((means - clim(1)) / (clim(2) - clim(1)) * (nColors - 1)) + 1;
color_indices = max(1, min(nColors, color_indices));  % Ensure indices within valid range
bar_colors = cmap(color_indices, :);                  % Get corresponding colors
% Plot
figure;
hold on;
for i = 1:length(means)
    bar(i, means(i), 'FaceColor', bar_colors(i,:), 'EdgeColor', 'none');
end
% Axes and labels
set(gca, 'XTick', 1:5, 'XTickLabel', labels, 'XTickLabelRotation', 45);
ylabel('Mean rstdMetric');
title('Cingulate rstdMetric');
ylim([-1 1])
box off;
% Colorbar
colormap(cmap);          % Set colormap
caxis(clim);             % Set color axis limits
colorbar;                % Display colorbar


%% 4D regressions of brain axes and RSTD metrics
% currently removed ML from overall models.

% (1) ~~~~~  Insula  ~~~~~~
% All Contacts
insula_T = table(insula_data(:,1), insula_data(:,2), insula_data(:,3), insula_VE_rstdMetric, 'VariableNames', {'ML', 'AP', 'SI', 'insula_rstdMetric'});
insula_right_T = table(insula_data(rightInsula,1), insula_data(rightInsula,2), insula_data(rightInsula,3), insula_VE_rstdMetric(rightInsula),'VariableNames', {'ML', 'AP', 'SI', 'insula_rstdMetric_right'});
insula_left_T = table(insula_data(~rightInsula,1), insula_data(~rightInsula,2), insula_data(~rightInsula,3), insula_VE_rstdMetric(~rightInsula),'VariableNames', {'ML', 'AP', 'SI', 'insula_rstdMetric_left'});

% All Contacts Regressions
lm_insula = fitlm(insula_T, 'insula_rstdMetric ~ AP + SI');
lm_insula_right = fitlm(insula_right_T, 'insula_rstdMetric_right ~ ML + AP + SI');
lm_insula_left = fitlm(insula_left_T, 'insula_rstdMetric_left ~ ML + AP + SI'); % sig SI

 h = lm_insula_left.plot;
        h(1).Marker = '.';
        h(1).MarkerEdgeColor = 'k';
        h(1).MarkerSize = 15;
        xlabel('SI')
        ylabel('RSTD metric')
        subtitle(sprintf('p = %2f', lm_insula_left.ModelFitVsNullModel.Pvalue))
        legend off
        ylim([-1 1])
        yticks(-1:.2:1)
        axis tight square

% plotting Amygdala by hemis
figure(11)
title('left Insula')
hold on
s1 = scatter3(insula_data(~rightInsula,1), insula_data(~rightInsula,2), insula_data(~rightInsula,3), ...
    70, insula_VE_rstdMetric(~rightInsula), 'filled', 'MarkerEdgeColor', 'k');
colorbar;
colormap cool
%view(3);
xlabel('Lateral --- Medial --- Lateral');
ylabel('Posterior --- Anterior');
zlabel('Inferior --- Superior');
% plotting actual glass brain....
patchs = patch('Faces', faces_left, 'Vertices', vertices_left, 'edgecolor', 'none', 'facecolor', 'flat', 'facealpha', .2);
facecolor = repmat([1 1 1],length(faces_left),1);
set(patchs, 'FaceVertexCData', facecolor);
camlight 
lighting gouraud
view([1 0 0]); % for medial view of left hemi 1



% Dropping Zeros from Insula RSTD
insula_rstdMetricZeros = insula_VE_rstdMetric(insula_VE_rstdMetric ~= 0);
insula_rstdMetricZeros_logical = (insula_VE_rstdMetric == 0);
insulaZeros_data = insula_data(~insula_rstdMetricZeros_logical,:);
rightInsulaZeros = (insulaZeros_data(:,1) > 0);

% No Zeros Tables
insula_Zeros_T = table(insulaZeros_data(:,1), insulaZeros_data(:,2), insulaZeros_data(:,3), insula_rstdMetricZeros, 'VariableNames', {'ML_Zeros', 'AP_Zeros', 'SI_Zeros', 'insula_rstdMetric_Zeros'});
insula_Zeros_right_T = table(insulaZeros_data(rightInsulaZeros,1), insulaZeros_data(rightInsulaZeros,2), insulaZeros_data(rightInsulaZeros,3), insula_rstdMetricZeros(rightInsulaZeros),'VariableNames', {'ML_Zeros', 'AP_Zeros', 'SI_Zeros', 'insula_rstdMetric_Zeros_right'});
insula_Zeros_left_T = table(insulaZeros_data(~rightInsulaZeros,1), insulaZeros_data(~rightInsulaZeros,2), insulaZeros_data(~rightInsulaZeros,3), insula_rstdMetricZeros(~rightInsulaZeros),'VariableNames', {'ML_Zeros', 'AP_Zeros', 'SI_Zeros', 'insula_rstdMetric_Zeros_left'});

% No Zeros Regression
lm_insulaZeros = fitlm(insula_Zeros_T, 'insula_rstdMetric_Zeros ~ AP_Zeros + SI_Zeros');
lm_insulaZeros_right = fitlm(insula_Zeros_right_T, 'insula_rstdMetric_Zeros_right ~ ML_Zeros + AP_Zeros + SI_Zeros');
lm_insulaZeros_left = fitlm(insula_Zeros_left_T, 'insula_rstdMetric_Zeros_left ~ ML_Zeros + AP_Zeros + SI_Zeros'); % sig SI

% (2) ~~~~~~ Cingulate ~~~~~~
cingulate_T = table(cingulate_data(:,1), cingulate_data(:,2), cingulate_data(:,3), cingulate_VE_rstdMetric, 'VariableNames', {'ML', 'AP', 'SI', 'cingulate_rstdMetric'});
cingulate_right_T = table(cingulate_data(rightCingulate,1), cingulate_data(rightCingulate,2), cingulate_data(rightCingulate,3), cingulate_VE_rstdMetric(rightCingulate), 'VariableNames', {'ML', 'AP', 'SI', 'cingulate_rstdMetric_right'});
cingulate_left_T = table(cingulate_data(~rightCingulate,1), cingulate_data(~rightCingulate,2), cingulate_data(~rightCingulate,3), cingulate_VE_rstdMetric(~rightCingulate), 'VariableNames', {'ML', 'AP', 'SI', 'cingulate_rstdMetric_left'});

% All Contacts Regressions
lm_cingulate = fitlm(cingulate_T, 'cingulate_rstdMetric ~ AP + SI'); % sig AP
lm_cingulate_right = fitlm(cingulate_right_T, 'cingulate_rstdMetric_right ~ ML + AP + SI'); % sig AP
lm_cingulate_left = fitlm(cingulate_left_T, 'cingulate_rstdMetric_left ~ ML + AP + SI'); % NS

% Logical indexing to include only rows where column 2 (AP) > 0
filtered_T = cingulate_T(cingulate_T{:, 2} > 0, :);
filtered_T_right = cingulate_right_T(cingulate_right_T{:, 2} > 0, :);
filtered_T_left = cingulate_left_T(cingulate_left_T{:, 2} > 0, :);

% Fit the linear model using the filtered table
lm_cingulate_noPCC = fitlm(filtered_T, 'cingulate_rstdMetric ~ AP + SI');
lm_cingulate_right_noPCC = fitlm(filtered_T_right, 'cingulate_rstdMetric_right ~ ML + AP + SI'); % sig AP
lm_cingulate_left_noPCC = fitlm(filtered_T_left, 'cingulate_rstdMetric_left ~ ML + AP + SI'); % NS

 h = lm_cingulate.plot;
        h(1).Marker = '.';
        h(1).MarkerEdgeColor = 'k';
        h(1).MarkerSize = 15;
        xlabel('AP')
        ylabel('RSTD metric')
        subtitle(sprintf('p = %2f', lm_cingulate.ModelFitVsNullModel.Pvalue))
        legend off
        axis tight square

% Dropping Zeros from Cingulate RSTD
cingulate_rstdMetricZeros = cingulate_VE_rstdMetric(cingulate_VE_rstdMetric ~= 0);
cingulate_rstdMetricZeros_logical = (cingulate_VE_rstdMetric == 0);
cingulateZeros_data = cingulate_data(~cingulate_rstdMetricZeros_logical,:);
rightCingulateZeros = (cingulateZeros_data(:,1) > 0);

% No Zeros Tables
cingulate_Zeros_T = table(cingulateZeros_data(:,1), cingulateZeros_data(:,2), cingulateZeros_data(:,3), cingulate_rstdMetricZeros, 'VariableNames', {'ML_Zeros', 'AP_Zeros', 'SI_Zeros', 'cingulate_rstdMetric_Zeros'});
cingulate_Zeros_right_T = table(cingulateZeros_data(rightCingulateZeros,1), cingulateZeros_data(rightCingulateZeros,2), cingulateZeros_data(rightCingulateZeros,3), cingulate_rstdMetricZeros(rightCingulateZeros),'VariableNames', {'ML_Zeros', 'AP_Zeros', 'SI_Zeros', 'cingulate_rstdMetric_Zeros_right'});
cingulate_Zeros_left_T = table(cingulateZeros_data(~rightCingulateZeros,1), cingulateZeros_data(~rightCingulateZeros,2), cingulateZeros_data(~rightCingulateZeros,3), cingulate_rstdMetricZeros(~rightCingulateZeros),'VariableNames', {'ML_Zeros', 'AP_Zeros', 'SI_Zeros', 'cingulate_rstdMetric_Zeros_left'});

% No Zeros Regression
lm_cingulateZeros = fitlm(cingulate_Zeros_T, 'cingulate_rstdMetric_Zeros ~ AP_Zeros + SI_Zeros');
lm_cingulateZeros_right = fitlm(cingulate_Zeros_right_T, 'cingulate_rstdMetric_Zeros_right ~ ML_Zeros + AP_Zeros + SI_Zeros'); % sig AP + SI + overall.
lm_cingulateZeros_left = fitlm(cingulate_Zeros_left_T, 'cingulate_rstdMetric_Zeros_left ~ ML_Zeros + AP_Zeros + SI_Zeros');

%(2a) regressions for only dACC and psgACC
% (A) ~~~~~~ dACC ~~~~~~
dACC_right_T = table(cingulate_data(right_dACC,1), cingulate_data(right_dACC,2), cingulate_data(right_dACC,3), cingulate_VE_rstdMetric(right_dACC), 'VariableNames', {'ML', 'AP', 'SI', 'dACC_rstdMetric_right'});
dACC_left_T = table(cingulate_data(left_dACC,1), cingulate_data(left_dACC,2), cingulate_data(left_dACC,3), cingulate_VE_rstdMetric(left_dACC), 'VariableNames', {'ML', 'AP', 'SI', 'dACC_rstdMetric_left'});
% dACC Regressions
lm_dACC_right = fitlm(dACC_right_T, 'dACC_rstdMetric_right ~ ML + AP + SI'); % NS
lm_dACC_left = fitlm(dACC_left_T, 'dACC_rstdMetric_left ~ ML + AP + SI'); % NS
% (B) ~~~~~~ psgACC ~~~~~~
psgACC_right_T = table(cingulate_data(right_psgACC,1), cingulate_data(right_psgACC,2), cingulate_data(right_psgACC,3), cingulate_VE_rstdMetric(right_psgACC), 'VariableNames', {'ML', 'AP', 'SI', 'psgACC_rstdMetric_right'});
psgACC_left_T = table(cingulate_data(left_psgACC,1), cingulate_data(left_psgACC,2), cingulate_data(left_psgACC,3), cingulate_VE_rstdMetric(left_psgACC), 'VariableNames', {'ML', 'AP', 'SI', 'psgACC_rstdMetric_left'});
% dACC Regressions
lm_psgACC_right = fitlm(psgACC_right_T, 'psgACC_rstdMetric_right ~ ML + AP + SI'); % 
lm_psgACC_left = fitlm(psgACC_left_T, 'psgACC_rstdMetric_left ~ ML + AP + SI'); % 

% (3) ~~~~~ Hippocampus ~~~~~
hippocampus_T = table(hippocampus_data(:,1), hippocampus_data(:,2), hippocampus_data(:,3), hippocampus_VE_rstdMetric, 'VariableNames', {'ML', 'AP', 'SI', 'hippocampus_rstdMetric'});
hippocampus_right_T = table(hippocampus_data(rightHippocampus,1), hippocampus_data(rightHippocampus,2), hippocampus_data(rightHippocampus,3), hippocampus_VE_rstdMetric(rightHippocampus), 'VariableNames', {'ML', 'AP', 'SI', 'hippocampus_rstdMetric_right'});
hippocampus_left_T = table(hippocampus_data(~rightHippocampus,1), hippocampus_data(~rightHippocampus,2), hippocampus_data(~rightHippocampus,3), hippocampus_VE_rstdMetric(~rightHippocampus), 'VariableNames', {'ML', 'AP', 'SI', 'hippocampus_rstdMetric_left'});

lm_hippocampus = fitlm(hippocampus_T, 'hippocampus_rstdMetric ~ AP + SI'); % sig ML
lm_hippocampus_right = fitlm(hippocampus_right_T, 'hippocampus_rstdMetric_right ~ ML + AP + SI'); % sig ML
lm_hippocampus_left = fitlm(hippocampus_left_T, 'hippocampus_rstdMetric_left ~ ML + AP + SI'); % sig AP + SI

% Dropping Zeros from Hippocampus RSTD
hippocampus_rstdMetricZeros = hippocampus_VE_rstdMetric(hippocampus_VE_rstdMetric ~= 0);
hippocampus_rstdMetricZeros_logical = (hippocampus_VE_rstdMetric == 0);
hippocampusZeros_data = hippocampus_data(~hippocampus_rstdMetricZeros_logical,:);
rightHippocampusZeros = (hippocampusZeros_data(:,1) > 0);

% No Zeros Tables
hippocampus_Zeros_T = table(hippocampusZeros_data(:,1), hippocampusZeros_data(:,2), hippocampusZeros_data(:,3), hippocampus_rstdMetricZeros, 'VariableNames', {'ML_Zeros', 'AP_Zeros', 'SI_Zeros', 'hippocampus_rstdMetric_Zeros'});
hippocampus_Zeros_right_T = table(hippocampusZeros_data(rightHippocampusZeros,1), hippocampusZeros_data(rightHippocampusZeros,2), hippocampusZeros_data(rightHippocampusZeros,3), hippocampus_rstdMetricZeros(rightHippocampusZeros),'VariableNames', {'ML_Zeros', 'AP_Zeros', 'SI_Zeros', 'hippocampus_rstdMetric_Zeros_right'});
hippocampus_Zeros_left_T = table(hippocampusZeros_data(~rightHippocampusZeros,1), hippocampusZeros_data(~rightHippocampusZeros,2), hippocampusZeros_data(~rightHippocampusZeros,3), hippocampus_rstdMetricZeros(~rightHippocampusZeros),'VariableNames', {'ML_Zeros', 'AP_Zeros', 'SI_Zeros', 'hippocampus_rstdMetric_Zeros_left'});

% No Zeros Regression
lm_hippocampusZeros = fitlm(hippocampus_Zeros_T, 'hippocampus_rstdMetric_Zeros ~ AP_Zeros + SI_Zeros');
lm_hippocampusZeros_right = fitlm(hippocampus_Zeros_right_T, 'hippocampus_rstdMetric_Zeros_right ~ ML_Zeros + AP_Zeros + SI_Zeros'); % ML
lm_hippocampusZeros_left = fitlm(hippocampus_Zeros_left_T, 'hippocampus_rstdMetric_Zeros_left ~ ML_Zeros + AP_Zeros + SI_Zeros'); % AP + SI

% (4) ~~~~~ InfFrontal ~~~~~ 
infFrontal_T = table(infFrontal_data(:,1), infFrontal_data(:,2), infFrontal_data(:,3), infFrontal_VE_rstdMetric, 'VariableNames', {'ML', 'AP', 'SI', 'infFrontal_rstdMetric'});
infFrontal_right_T = table(infFrontal_data(rightInfFrontal,1), infFrontal_data(rightInfFrontal,2), infFrontal_data(rightInfFrontal,3), infFrontal_VE_rstdMetric(rightInfFrontal), 'VariableNames', {'ML', 'AP', 'SI', 'infFrontal_rstdMetric_right'});
infFrontal_left_T = table(infFrontal_data(~rightInfFrontal,1), infFrontal_data(~rightInfFrontal,2), infFrontal_data(~rightInfFrontal,3), infFrontal_VE_rstdMetric(~rightInfFrontal), 'VariableNames', {'ML', 'AP', 'SI', 'infFrontal_rstdMetric_left'});

lm_infFrontal = fitlm(infFrontal_T, 'infFrontal_rstdMetric ~ AP + SI'); % sig ML, overall
lm_infFrontal_right = fitlm(infFrontal_right_T, 'infFrontal_rstdMetric_right ~ ML + AP + SI');
lm_infFrontal_left = fitlm(infFrontal_left_T, 'infFrontal_rstdMetric_left ~ ML + AP + SI');

% Dropping Zeros from InfFrontal RSTD
infFrontal_rstdMetricZeros = infFrontal_VE_rstdMetric(infFrontal_VE_rstdMetric ~= 0);
infFrontal_rstdMetricZeros_logical = (infFrontal_VE_rstdMetric == 0);
infFrontalZeros_data = infFrontal_data(~infFrontal_rstdMetricZeros_logical,:);
rightInfFrontalZeros = (infFrontalZeros_data(:,1) > 0);

% No Zeros Tables
infFrontal_Zeros_T = table(infFrontalZeros_data(:,1), infFrontalZeros_data(:,2), infFrontalZeros_data(:,3), infFrontal_rstdMetricZeros, 'VariableNames', {'ML_Zeros', 'AP_Zeros', 'SI_Zeros', 'infFrontal_rstdMetric_Zeros'});
infFrontal_Zeros_right_T = table(infFrontalZeros_data(rightInfFrontalZeros,1), infFrontalZeros_data(rightInfFrontalZeros,2), infFrontalZeros_data(rightInfFrontalZeros,3), infFrontal_rstdMetricZeros(rightInfFrontalZeros),'VariableNames', {'ML_Zeros', 'AP_Zeros', 'SI_Zeros', 'infFrontal_rstdMetric_Zeros_right'});
infFrontal_Zeros_left_T = table(infFrontalZeros_data(~rightInfFrontalZeros,1), infFrontalZeros_data(~rightInfFrontalZeros,2), infFrontalZeros_data(~rightInfFrontalZeros,3), infFrontal_rstdMetricZeros(~rightInfFrontalZeros),'VariableNames', {'ML_Zeros', 'AP_Zeros', 'SI_Zeros', 'infFrontal_rstdMetric_Zeros_left'});

% No Zeros Regression
lm_infFrontalZeros = fitlm(infFrontal_Zeros_T, 'infFrontal_rstdMetric_Zeros ~ AP_Zeros + SI_Zeros');
lm_infFrontalZeros_right = fitlm(infFrontal_Zeros_right_T, 'infFrontal_rstdMetric_Zeros_right ~ ML_Zeros + AP_Zeros + SI_Zeros');
lm_infFrontalZeros_left = fitlm(infFrontal_Zeros_left_T, 'infFrontal_rstdMetric_Zeros_left ~ ML_Zeros + AP_Zeros + SI_Zeros');  

% (5) ~~~~~ MedFrontal ~~~~~
medFrontal_T = table(medFrontal_data(:,1), medFrontal_data(:,2), medFrontal_data(:,3), medFrontal_VE_rstdMetric, 'VariableNames', {'ML', 'AP', 'SI', 'medFrontal_rstdMetric'});
medFrontal_right_T = table(medFrontal_data(rightMedFrontal,1), medFrontal_data(rightMedFrontal,2), medFrontal_data(rightMedFrontal,3), medFrontal_VE_rstdMetric(rightMedFrontal), 'VariableNames', {'ML', 'AP', 'SI', 'medFrontal_rstdMetric_right'});
medFrontal_left_T = table(medFrontal_data(~rightMedFrontal,1), medFrontal_data(~rightMedFrontal,2), medFrontal_data(~rightMedFrontal,3), medFrontal_VE_rstdMetric(~rightMedFrontal), 'VariableNames', {'ML', 'AP', 'SI', 'medFrontal_rstdMetric_left'});

lm_medFrontal = fitlm(medFrontal_T, 'medFrontal_rstdMetric ~ AP + SI'); % sig SI, overall
lm_medFrontal_right = fitlm(medFrontal_right_T, 'medFrontal_rstdMetric_right ~ ML + AP + SI'); % sig ML, AP overall
lm_medFrontal_left = fitlm(medFrontal_left_T, 'medFrontal_rstdMetric_left ~ ML + AP + SI');

 h = lm_medFrontal_right.plot;
        h(1).Marker = '.';
        h(1).MarkerEdgeColor = 'k';
        h(1).MarkerSize = 15;
        xlabel('SI')
        ylabel('RSTD metric')
        subtitle(sprintf('p = %2f', lm_medFrontal_right.ModelFitVsNullModel.Pvalue))
        legend off
        ylim([-1 1])
        yticks(-1:.2:1)
        axis tight square



% plotting MedFrontal by hemis
figure(10)
subplot(2,1,1)
subtitle('right medFrontal')
hold on
s1 = scatter3(medFrontal_data(rightMedFrontal,1), medFrontal_data(rightMedFrontal,2), medFrontal_data(rightMedFrontal,3), ...
    70, medFrontal_VE_rstdMetric(rightMedFrontal), 'filled', 'MarkerEdgeColor', 'k');
colorbar;
clim([-1 1]);
colormap cool
%view(3);
xlabel('Lateral --- Medial --- Lateral');
ylabel('Anterior --- Posterior');
zlabel('Inferior --- Superior');
% plotting actual glass brain....
patchs = patch('Faces', faces_right, 'Vertices', vertices_right, 'edgecolor', 'none', 'facecolor', 'flat', 'facealpha', .2);
facecolor = repmat([1 1 1],length(faces_right),1);
set(patchs, 'FaceVertexCData', facecolor);
camlight 
lighting gouraud
view(0, 90); % for medial view of right hemi -1
subplot(2,1,2)
subtitle('left medFrontal')
hold on
s1 = scatter3(medFrontal_data(~rightMedFrontal,1), medFrontal_data(~rightMedFrontal,2), medFrontal_data(~rightMedFrontal,3), ...
    70, medFrontal_VE_rstdMetric(~rightMedFrontal), 'filled', 'MarkerEdgeColor', 'k');
colorbar;
clim([-1 1]);
colormap cool
%view(3);
xlabel('Lateral --- Medial --- Lateral');
ylabel('Posterior --- Anterior');
zlabel('Inferior --- Superior');
% plotting actual glass brain....
patchs = patch('Faces', faces_left, 'Vertices', vertices_left, 'edgecolor', 'none', 'facecolor', 'flat', 'facealpha', .2);
facecolor = repmat([1 1 1],length(faces_left),1);
set(patchs, 'FaceVertexCData', facecolor);
camlight 
lighting gouraud
view(0, 90); % for medial view of left hemi 1

% Dropping Zeros from MedFrontal RSTD
medFrontal_rstdMetricZeros = medFrontal_VE_rstdMetric(medFrontal_VE_rstdMetric ~= 0);
medFrontal_rstdMetricZeros_logical = (medFrontal_VE_rstdMetric == 0);
medFrontalZeros_data = medFrontal_data(~medFrontal_rstdMetricZeros_logical,:);
rightMedFrontalZeros = (medFrontalZeros_data(:,1) > 0);

% No Zeros Tables
medFrontal_Zeros_T = table(medFrontalZeros_data(:,1), medFrontalZeros_data(:,2), medFrontalZeros_data(:,3), medFrontal_rstdMetricZeros, 'VariableNames', {'ML_Zeros', 'AP_Zeros', 'SI_Zeros', 'medFrontal_rstdMetric_Zeros'});
medFrontal_Zeros_right_T = table(medFrontalZeros_data(rightMedFrontalZeros,1), medFrontalZeros_data(rightMedFrontalZeros,2), medFrontalZeros_data(rightMedFrontalZeros,3), medFrontal_rstdMetricZeros(rightMedFrontalZeros),'VariableNames', {'ML_Zeros', 'AP_Zeros', 'SI_Zeros', 'medFrontal_rstdMetric_Zeros_right'});
medFrontal_Zeros_left_T = table(medFrontalZeros_data(~rightMedFrontalZeros,1), medFrontalZeros_data(~rightMedFrontalZeros,2), medFrontalZeros_data(~rightMedFrontalZeros,3), medFrontal_rstdMetricZeros(~rightMedFrontalZeros),'VariableNames', {'ML_Zeros', 'AP_Zeros', 'SI_Zeros', 'medFrontal_rstdMetric_Zeros_left'});

% No Zeros Regression
lm_medFrontalZeros = fitlm(medFrontal_Zeros_T, 'medFrontal_rstdMetric_Zeros ~ AP_Zeros + SI_Zeros');
lm_medFrontalZeros_right = fitlm(medFrontal_Zeros_right_T, 'medFrontal_rstdMetric_Zeros_right ~ ML_Zeros + AP_Zeros + SI_Zeros'); % ML
lm_medFrontalZeros_left = fitlm(medFrontal_Zeros_left_T, 'medFrontal_rstdMetric_Zeros_left ~ ML_Zeros + AP_Zeros + SI_Zeros'); % AP

% (6) ~~~~~ SupFrontal ~~~~~
supFrontal_T = table(supFrontal_data(:,1), supFrontal_data(:,2), supFrontal_data(:,3), supFrontal_VE_rstdMetric, 'VariableNames', {'ML', 'AP', 'SI', 'supFrontal_rstdMetric'});
supFrontal_right_T = table(supFrontal_data(rightSupFrontal,1), supFrontal_data(rightSupFrontal,2), supFrontal_data(rightSupFrontal,3), supFrontal_VE_rstdMetric(rightSupFrontal), 'VariableNames', {'ML', 'AP', 'SI', 'supFrontal_rstdMetric_right'});
supFrontal_left_T = table(supFrontal_data(~rightSupFrontal,1), supFrontal_data(~rightSupFrontal,2), supFrontal_data(~rightSupFrontal,3), supFrontal_VE_rstdMetric(~rightSupFrontal), 'VariableNames', {'ML', 'AP', 'SI', 'supFrontal_rstdMetric_left'});

lm_supFrontal = fitlm(supFrontal_T, 'supFrontal_rstdMetric ~ AP + SI');
lm_supFrontal_right = fitlm(supFrontal_right_T, 'supFrontal_rstdMetric_right ~ ML + AP + SI');
lm_supFrontal_left = fitlm(supFrontal_left_T, 'supFrontal_rstdMetric_left ~ ML + AP + SI');  % Check significance

% Dropping Zeros from SupFrontal RSTD
supFrontal_rstdMetricZeros = supFrontal_VE_rstdMetric(supFrontal_VE_rstdMetric ~= 0);
supFrontal_rstdMetricZeros_logical = (supFrontal_VE_rstdMetric == 0);
supFrontalZeros_data = supFrontal_data(~supFrontal_rstdMetricZeros_logical,:);
rightSupFrontalZeros = (supFrontalZeros_data(:,1) > 0);

% No Zeros Tables
supFrontal_Zeros_T = table(supFrontalZeros_data(:,1), supFrontalZeros_data(:,2), supFrontalZeros_data(:,3), supFrontal_rstdMetricZeros, 'VariableNames', {'ML_Zeros', 'AP_Zeros', 'SI_Zeros', 'supFrontal_rstdMetric_Zeros'});
supFrontal_Zeros_right_T = table(supFrontalZeros_data(rightSupFrontalZeros,1), supFrontalZeros_data(rightSupFrontalZeros,2), supFrontalZeros_data(rightSupFrontalZeros,3), supFrontal_rstdMetricZeros(rightSupFrontalZeros),'VariableNames', {'ML_Zeros', 'AP_Zeros', 'SI_Zeros', 'supFrontal_rstdMetric_Zeros_right'});
supFrontal_Zeros_left_T = table(supFrontalZeros_data(~rightSupFrontalZeros,1), supFrontalZeros_data(~rightSupFrontalZeros,2), supFrontalZeros_data(~rightSupFrontalZeros,3), supFrontal_rstdMetricZeros(~rightSupFrontalZeros),'VariableNames', {'ML_Zeros', 'AP_Zeros', 'SI_Zeros', 'supFrontal_rstdMetric_Zeros_left'});

% No Zeros Regression
lm_supFrontalZeros = fitlm(supFrontal_Zeros_T, 'supFrontal_rstdMetric_Zeros ~ AP_Zeros + SI_Zeros');
lm_supFrontalZeros_right = fitlm(supFrontal_Zeros_right_T, 'supFrontal_rstdMetric_Zeros_right ~ ML_Zeros + AP_Zeros + SI_Zeros');
lm_supFrontalZeros_left = fitlm(supFrontal_Zeros_left_T, 'supFrontal_rstdMetric_Zeros_left ~ ML_Zeros + AP_Zeros + SI_Zeros');

% (7) ~~~~~~ OrbFrontal ~~~~~
OrbFrontal_T = table(orbFrontal_data(:,1), orbFrontal_data(:,2), orbFrontal_data(:,3), orbFrontal_VE_rstdMetric, 'VariableNames', {'ML', 'AP', 'SI', 'OrbFrontal_rstdMetric'});
OrbFrontal_right_T = table(orbFrontal_data(rightOrbFrontal,1), orbFrontal_data(rightOrbFrontal,2), orbFrontal_data(rightOrbFrontal,3), orbFrontal_VE_rstdMetric(rightOrbFrontal), 'VariableNames', {'ML', 'AP', 'SI', 'OrbFrontal_rstdMetric_right'});
OrbFrontal_left_T = table(orbFrontal_data(~rightOrbFrontal,1), orbFrontal_data(~rightOrbFrontal,2), orbFrontal_data(~rightOrbFrontal,3), orbFrontal_VE_rstdMetric(~rightOrbFrontal), 'VariableNames', {'ML', 'AP', 'SI', 'OrbFrontal_rstdMetric_left'});

lm_OrbFrontal = fitlm(OrbFrontal_T, 'OrbFrontal_rstdMetric ~ AP + SI');
lm_OrbFrontal_right = fitlm(OrbFrontal_right_T, 'OrbFrontal_rstdMetric_right ~ ML + AP + SI');
lm_OrbFrontal_left = fitlm(OrbFrontal_left_T, 'OrbFrontal_rstdMetric_left ~ ML + AP + SI');

% Dropping Zeros from OrbFrontal RSTD
OrbFrontal_rstdMetricZeros = orbFrontal_VE_rstdMetric(orbFrontal_VE_rstdMetric ~= 0);
OrbFrontal_rstdMetricZeros_logical = (orbFrontal_VE_rstdMetric == 0);
OrbFrontalZeros_data = orbFrontal_data(~OrbFrontal_rstdMetricZeros_logical,:);
rightOrbFrontalZeros = (OrbFrontalZeros_data(:,1) > 0);

% No Zeros Tables
OrbFrontal_Zeros_T = table(OrbFrontalZeros_data(:,1), OrbFrontalZeros_data(:,2), OrbFrontalZeros_data(:,3), OrbFrontal_rstdMetricZeros, 'VariableNames', {'ML_Zeros', 'AP_Zeros', 'SI_Zeros', 'OrbFrontal_rstdMetric_Zeros'});
OrbFrontal_Zeros_right_T = table(OrbFrontalZeros_data(rightOrbFrontalZeros,1), OrbFrontalZeros_data(rightOrbFrontalZeros,2), OrbFrontalZeros_data(rightOrbFrontalZeros,3), OrbFrontal_rstdMetricZeros(rightOrbFrontalZeros),'VariableNames', {'ML_Zeros', 'AP_Zeros', 'SI_Zeros', 'OrbFrontal_rstdMetric_Zeros_right'});
OrbFrontal_Zeros_left_T = table(OrbFrontalZeros_data(~rightOrbFrontalZeros,1), OrbFrontalZeros_data(~rightOrbFrontalZeros,2), OrbFrontalZeros_data(~rightOrbFrontalZeros,3), OrbFrontal_rstdMetricZeros(~rightOrbFrontalZeros),'VariableNames', {'ML_Zeros', 'AP_Zeros', 'SI_Zeros', 'OrbFrontal_rstdMetric_Zeros_left'});

% No Zeros Regression
lm_OrbFrontalZeros = fitlm(OrbFrontal_Zeros_T, 'OrbFrontal_rstdMetric_Zeros ~ AP_Zeros + SI_Zeros');
lm_OrbFrontalZeros_right = fitlm(OrbFrontal_Zeros_right_T, 'OrbFrontal_rstdMetric_Zeros_right ~ ML_Zeros + AP_Zeros + SI_Zeros');
lm_OrbFrontalZeros_left = fitlm(OrbFrontal_Zeros_left_T, 'OrbFrontal_rstdMetric_Zeros_left ~ ML_Zeros + AP_Zeros + SI_Zeros');  

% (8) ~~~~~ Striatum ~~~~~
striatum_T = table(striatum_data(:,1), striatum_data(:,2), striatum_data(:,3), striatum_VE_rstdMetric, 'VariableNames', {'ML', 'AP', 'SI', 'striatum_rstdMetric'});
striatum_right_T = table(striatum_data(rightStriatum,1), striatum_data(rightStriatum,2), striatum_data(rightStriatum,3), striatum_VE_rstdMetric(rightStriatum), 'VariableNames', {'ML', 'AP', 'SI', 'striatum_rstdMetric_right'});
striatum_left_T = table(striatum_data(~rightStriatum,1), striatum_data(~rightStriatum,2), striatum_data(~rightStriatum,3), striatum_VE_rstdMetric(~rightStriatum), 'VariableNames', {'ML', 'AP', 'SI', 'striatum_rstdMetric_left'});

lm_striatum = fitlm(striatum_T, 'striatum_rstdMetric ~ AP + SI');
lm_striatum_right = fitlm(striatum_right_T, 'striatum_rstdMetric_right ~ ML + AP + SI');
lm_striatum_left = fitlm(striatum_left_T, 'striatum_rstdMetric_left ~ ML + AP + SI'); 


% plotting striatum by hemis
figure(11)
subtitle('striatum')
hold on
s1 = scatter3(striatum_data(:,1),striatum_data(:,2), striatum_data(:,3), ...
    70, striatum_VE_rstdMetric, 'filled', 'MarkerEdgeColor', 'k');
colorbar;
clim([-1 1]); 
colormap cool
%view(3);
xlabel('Lateral --- Medial --- Lateral');
ylabel('Anterior --- Posterior');
zlabel('Inferior --- Superior');
% plotting actual glass brain....
patchs = patch('Faces', faces_right, 'Vertices', vertices_right, 'edgecolor', 'none', 'facecolor', 'flat', 'facealpha', .2);
facecolor = repmat([1 1 1],length(faces_right),1);
set(patchs, 'FaceVertexCData', facecolor);
camlight 
lighting gouraud
view([-1 0 0]); % for medial view of right hemi -1


% Dropping Zeros from Striatum RSTD
striatum_rstdMetricZeros = striatum_VE_rstdMetric(striatum_VE_rstdMetric ~= 0);
striatum_rstdMetricZeros_logical = (striatum_VE_rstdMetric == 0);
striatumZeros_data = striatum_data(~striatum_rstdMetricZeros_logical,:);
rightStriatumZeros = (striatumZeros_data(:,1) > 0);

% No Zeros Tables
striatum_Zeros_T = table(striatumZeros_data(:,1), striatumZeros_data(:,2), striatumZeros_data(:,3), striatum_rstdMetricZeros, 'VariableNames', {'ML_Zeros', 'AP_Zeros', 'SI_Zeros', 'striatum_rstdMetric_Zeros'});
striatum_Zeros_right_T = table(striatumZeros_data(rightStriatumZeros,1), striatumZeros_data(rightStriatumZeros,2), striatumZeros_data(rightStriatumZeros,3), striatum_rstdMetricZeros(rightStriatumZeros),'VariableNames', {'ML_Zeros', 'AP_Zeros', 'SI_Zeros', 'striatum_rstdMetric_Zeros_right'});
striatum_Zeros_left_T = table(striatumZeros_data(~rightStriatumZeros,1), striatumZeros_data(~rightStriatumZeros,2), striatumZeros_data(~rightStriatumZeros,3), striatum_rstdMetricZeros(~rightStriatumZeros),'VariableNames', {'ML_Zeros', 'AP_Zeros', 'SI_Zeros', 'striatum_rstdMetric_Zeros_left'});

% No Zeros Regression
lm_striatumZeros = fitlm(striatum_Zeros_T, 'striatum_rstdMetric_Zeros ~ AP_Zeros + SI_Zeros');
lm_striatumZeros_right = fitlm(striatum_Zeros_right_T, 'striatum_rstdMetric_Zeros_right ~ ML_Zeros + AP_Zeros + SI_Zeros');
lm_striatumZeros_left = fitlm(striatum_Zeros_left_T, 'striatum_rstdMetric_Zeros_left ~ ML_Zeros + AP_Zeros + SI_Zeros'); %

% (9) ~~~~~ nAccumbens ~~~~~
nAccumbens_T = table(nAccumbens_data(:,1), nAccumbens_data(:,2), nAccumbens_data(:,3), nAccumbens_VE_rstdMetric, 'VariableNames', {'ML', 'AP', 'SI', 'nAccumbens_rstdMetric'});
nAccumbens_right_T = table(nAccumbens_data(rightNAccumbens,1), nAccumbens_data(rightNAccumbens,2), nAccumbens_data(rightNAccumbens,3), nAccumbens_VE_rstdMetric(rightNAccumbens), 'VariableNames', {'ML', 'AP', 'SI', 'nAccumbens_rstdMetric_right'});
nAccumbens_left_T = table(nAccumbens_data(~rightNAccumbens,1), nAccumbens_data(~rightNAccumbens,2), nAccumbens_data(~rightNAccumbens,3), nAccumbens_VE_rstdMetric(~rightNAccumbens), 'VariableNames', {'ML', 'AP', 'SI', 'nAccumbens_rstdMetric_left'});

lm_nAccumbens = fitlm(nAccumbens_T, 'nAccumbens_rstdMetric ~ AP + SI');
% Can't do right nACC - there is only one contact
% Can't do left nACC - there is only four contacts
lm_nAccumbens_left = fitlm(nAccumbens_left_T, 'nAccumbens_rstdMetric_left ~ ML + AP + SI');

% plotting NACC by hemis
figure(11)
subtitle('nAccumbens')
hold on
s1 = scatter3(nAccumbens_data(:,1),nAccumbens_data(:,2), nAccumbens_data(:,3), ...
    70, nAccumbens_VE_rstdMetric, 'filled', 'MarkerEdgeColor', 'k');
colorbar;
clim([-1 1]); 
colormap cool
%view(3);
xlabel('Lateral --- Medial --- Lateral');
ylabel('Anterior --- Posterior');
zlabel('Inferior --- Superior');
% plotting actual glass brain....
patchs = patch('Faces', faces_right, 'Vertices', vertices_right, 'edgecolor', 'none', 'facecolor', 'flat', 'facealpha', .2);
facecolor = repmat([1 1 1],length(faces_right),1);
set(patchs, 'FaceVertexCData', facecolor);
camlight 
lighting gouraud
view([-1 0 0]); % for medial view of right hemi -1


% Dropping Zeros from NAccumbens RSTD
nAccumbens_rstdMetricZeros = nAccumbens_VE_rstdMetric(nAccumbens_VE_rstdMetric ~= 0);
nAccumbens_rstdMetricZeros_logical = (nAccumbens_VE_rstdMetric == 0);
nAccumbensZeros_data = nAccumbens_data(~nAccumbens_rstdMetricZeros_logical,:);
rightNAccumbensZeros = (nAccumbensZeros_data(:,1) > 0);

% No Zeros Tables
nAccumbens_Zeros_T = table(nAccumbensZeros_data(:,1), nAccumbensZeros_data(:,2), nAccumbensZeros_data(:,3), nAccumbens_rstdMetricZeros, 'VariableNames', {'ML_Zeros', 'AP_Zeros', 'SI_Zeros', 'nAccumbens_rstdMetric_Zeros'});
nAccumbens_Zeros_right_T = table(nAccumbensZeros_data(rightNAccumbensZeros,1), nAccumbensZeros_data(rightNAccumbensZeros,2), nAccumbensZeros_data(rightNAccumbensZeros,3), nAccumbens_rstdMetricZeros(rightNAccumbensZeros),'VariableNames', {'ML_Zeros', 'AP_Zeros', 'SI_Zeros', 'nAccumbens_rstdMetric_Zeros_right'});
nAccumbens_Zeros_left_T = table(nAccumbensZeros_data(~rightNAccumbensZeros,1), nAccumbensZeros_data(~rightNAccumbensZeros,2), nAccumbensZeros_data(~rightNAccumbensZeros,3), nAccumbens_rstdMetricZeros(~rightNAccumbensZeros),'VariableNames', {'ML_Zeros', 'AP_Zeros', 'SI_Zeros', 'nAccumbens_rstdMetric_Zeros_left'});

% No Zeros Regression
lm_nAccumbensZeros = fitlm(nAccumbens_Zeros_T, 'nAccumbens_rstdMetric_Zeros ~ AP_Zeros + SI_Zeros');
lm_nAccumbensZeros_right = fitlm(nAccumbens_Zeros_right_T, 'nAccumbens_rstdMetric_Zeros_right ~ ML_Zeros + AP_Zeros + SI_Zeros');
lm_nAccumbensZeros_left = fitlm(nAccumbens_Zeros_left_T, 'nAccumbens_rstdMetric_Zeros_left ~ ML_Zeros + AP_Zeros + SI_Zeros'); % sig SI

% (10) ~~~~~ Amygdala ~~~~~
amygdala_T = table(amygdala_data(:,1), amygdala_data(:,2), amygdala_data(:,3), amygdala_VE_rstdMetric, 'VariableNames', {'ML', 'AP', 'SI', 'amygdala_rstdMetric'});
amygdala_right_T = table(amygdala_data(rightAmygdala,1), amygdala_data(rightAmygdala,2), amygdala_data(rightAmygdala,3), amygdala_VE_rstdMetric(rightAmygdala), 'VariableNames', {'ML', 'AP', 'SI', 'amygdala_rstdMetric_right'});
amygdala_left_T = table(amygdala_data(~rightAmygdala,1), amygdala_data(~rightAmygdala,2), amygdala_data(~rightAmygdala,3), amygdala_VE_rstdMetric(~rightAmygdala), 'VariableNames', {'ML', 'AP', 'SI', 'amygdala_rstdMetric_left'});

lm_amygdala = fitlm(amygdala_T, 'amygdala_rstdMetric ~ AP + SI'); % sig AP, overall
lm_amygdala_right = fitlm(amygdala_right_T, 'amygdala_rstdMetric_right ~ ML + AP + SI');
lm_amygdala_left = fitlm(amygdala_left_T, 'amygdala_rstdMetric_left ~ ML + AP + SI'); % sig AP, overall

% plotting Amygdala by hemis
figure(11)
subplot(2,1,1)
subtitle('right Amygdala')
hold on
s1 = scatter3(amygdala_data(rightAmygdala,1), amygdala_data(rightAmygdala,2), amygdala_data(rightAmygdala,3), ...
    70, amygdala_VE_rstdMetric(rightAmygdala), 'filled', 'MarkerEdgeColor', 'k');
colorbar;
colormap cool
%view(3);
xlabel('Lateral --- Medial --- Lateral');
ylabel('Anterior --- Posterior');
zlabel('Inferior --- Superior');
% plotting actual glass brain....
patchs = patch('Faces', faces_right, 'Vertices', vertices_right, 'edgecolor', 'none', 'facecolor', 'flat', 'facealpha', .2);
facecolor = repmat([1 1 1],length(faces_right),1);
set(patchs, 'FaceVertexCData', facecolor);
camlight 
lighting gouraud
view([-1 0 0]); % for medial view of right hemi -1
subplot(2,1,2)
subtitle('left Amygdala')
hold on
s1 = scatter3(amygdala_data(~rightAmygdala,1), amygdala_data(~rightAmygdala,2), amygdala_data(~rightAmygdala,3), ...
    70, amygdala_VE_rstdMetric(~rightAmygdala), 'filled', 'MarkerEdgeColor', 'k');
colorbar;
colormap cool
%view(3);
xlabel('Lateral --- Medial --- Lateral');
ylabel('Posterior --- Anterior');
zlabel('Inferior --- Superior');
% plotting actual glass brain....
patchs = patch('Faces', faces_left, 'Vertices', vertices_left, 'edgecolor', 'none', 'facecolor', 'flat', 'facealpha', .2);
facecolor = repmat([1 1 1],length(faces_left),1);
set(patchs, 'FaceVertexCData', facecolor);
camlight 
lighting gouraud
view([1 0 0]); % for medial view of left hemi 1

% Dropping Zeros from Amygdala RSTD
amygdala_rstdMetricZeros = amygdala_VE_rstdMetric(amygdala_VE_rstdMetric ~= 0);
amygdala_rstdMetricZeros_logical = (amygdala_VE_rstdMetric == 0);
amygdalaZeros_data = amygdala_data(~amygdala_rstdMetricZeros_logical,:);
rightAmygdalaZeros = (amygdalaZeros_data(:,1) > 0);

% No Zeros Tables
amygdala_Zeros_T = table(amygdalaZeros_data(:,1), amygdalaZeros_data(:,2), amygdalaZeros_data(:,3), amygdala_rstdMetricZeros, 'VariableNames', {'ML_Zeros', 'AP_Zeros', 'SI_Zeros', 'amygdala_rstdMetric_Zeros'});
amygdala_Zeros_right_T = table(amygdalaZeros_data(rightAmygdalaZeros,1), amygdalaZeros_data(rightAmygdalaZeros,2), amygdalaZeros_data(rightAmygdalaZeros,3), amygdala_rstdMetricZeros(rightAmygdalaZeros),'VariableNames', {'ML_Zeros', 'AP_Zeros', 'SI_Zeros', 'amygdala_rstdMetric_Zeros_right'});
amygdala_Zeros_left_T = table(amygdalaZeros_data(~rightAmygdalaZeros,1), amygdalaZeros_data(~rightAmygdalaZeros,2), amygdalaZeros_data(~rightAmygdalaZeros,3), amygdala_rstdMetricZeros(~rightAmygdalaZeros),'VariableNames', {'ML_Zeros', 'AP_Zeros', 'SI_Zeros', 'amygdala_rstdMetric_Zeros_left'});

% No Zeros Regression
lm_amygdalaZeros = fitlm(amygdala_Zeros_T, 'amygdala_rstdMetric_Zeros ~ AP_Zeros + SI_Zeros'); % AP
lm_amygdalaZeros_right = fitlm(amygdala_Zeros_right_T, 'amygdala_rstdMetric_Zeros_right ~ ML_Zeros + AP_Zeros + SI_Zeros');
lm_amygdalaZeros_left = fitlm(amygdala_Zeros_left_T, 'amygdala_rstdMetric_Zeros_left ~ ML_Zeros + AP_Zeros + SI_Zeros'); % AP + overall

% (11) ~~~~~~ Thalamus ~~~~~~
thalamus_T = table(thalamus_data(:,1), thalamus_data(:,2), thalamus_data(:,3), thalamus_VE_rstdMetric, 'VariableNames', {'ML', 'AP', 'SI', 'thalamus_rstdMetric'});
thalamus_right_T = table(thalamus_data(rightThalamus,1), thalamus_data(rightThalamus,2), thalamus_data(rightThalamus,3), thalamus_VE_rstdMetric(rightThalamus), 'VariableNames', {'ML', 'AP', 'SI', 'thalamus_rstdMetric_right'});
thalamus_left_T = table(thalamus_data(~rightThalamus,1), thalamus_data(~rightThalamus,2), thalamus_data(~rightThalamus,3), thalamus_VE_rstdMetric(~rightThalamus), 'VariableNames', {'ML', 'AP', 'SI', 'thalamus_rstdMetric_left'});

lm_thalamus = fitlm(thalamus_T, 'thalamus_rstdMetric ~ AP + SI'); % AP
lm_thalamus_right = fitlm(thalamus_right_T, 'thalamus_rstdMetric_right ~ ML + AP + SI');
lm_thalamus_left = fitlm(thalamus_left_T, 'thalamus_rstdMetric_left ~ ML + AP + SI'); % sig AP + SI, overall

% Dropping Zeros from Thalamus RSTD
thalamus_rstdMetricZeros = thalamus_VE_rstdMetric(thalamus_VE_rstdMetric ~= 0);
thalamus_rstdMetricZeros_logical = (thalamus_VE_rstdMetric == 0);
thalamusZeros_data = thalamus_data(~thalamus_rstdMetricZeros_logical,:);
rightThalamusZeros = (thalamusZeros_data(:,1) > 0);

% No Zeros Tables
thalamus_Zeros_T = table(thalamusZeros_data(:,1), thalamusZeros_data(:,2), thalamusZeros_data(:,3), thalamus_rstdMetricZeros, 'VariableNames', {'ML_Zeros', 'AP_Zeros', 'SI_Zeros', 'thalamus_rstdMetric_Zeros'});
thalamus_Zeros_right_T = table(thalamusZeros_data(rightThalamusZeros,1), thalamusZeros_data(rightThalamusZeros,2), thalamusZeros_data(rightThalamusZeros,3), thalamus_rstdMetricZeros(rightThalamusZeros),'VariableNames', {'ML_Zeros', 'AP_Zeros', 'SI_Zeros', 'thalamus_rstdMetric_Zeros_right'});
thalamus_Zeros_left_T = table(thalamusZeros_data(~rightThalamusZeros,1), thalamusZeros_data(~rightThalamusZeros,2), thalamusZeros_data(~rightThalamusZeros,3), thalamus_rstdMetricZeros(~rightThalamusZeros),'VariableNames', {'ML_Zeros', 'AP_Zeros', 'SI_Zeros', 'thalamus_rstdMetric_Zeros_left'});

% No Zeros Regression
lm_thalamusZeros = fitlm(thalamus_Zeros_T, 'thalamus_rstdMetric_Zeros ~ AP_Zeros + SI_Zeros');
lm_thalamusZeros_right = fitlm(thalamus_Zeros_right_T, 'thalamus_rstdMetric_Zeros_right ~ ML_Zeros + AP_Zeros + SI_Zeros');
lm_thalamusZeros_left = fitlm(thalamus_Zeros_left_T, 'thalamus_rstdMetric_Zeros_left ~ ML_Zeros + AP_Zeros + SI_Zeros'); % AP, SI, overall


% plotting Thalamus by hemis
figure(10)
subplot(2,1,1)
subtitle('right Thalamus')
hold on
s1 = scatter3(thalamus_data(rightThalamus,1), thalamus_data(rightThalamus,2), thalamus_data(rightThalamus,3), ...
    70, thalamus_VE_rstdMetric(rightThalamus), 'filled', 'MarkerEdgeColor', 'k');
colorbar;
colormap cool
%view(3);
xlabel('Lateral --- Medial --- Lateral');
ylabel('Anterior --- Posterior');
zlabel('Inferior --- Superior');
% plotting actual glass brain....
patchs = patch('Faces', faces_right, 'Vertices', vertices_right, 'edgecolor', 'none', 'facecolor', 'flat', 'facealpha', .2);
facecolor = repmat([1 1 1],length(faces_right),1);
set(patchs, 'FaceVertexCData', facecolor);
camlight 
lighting gouraud
view([-1 0 0]); % for medial view of right hemi -1
subplot(2,1,2)
subtitle('left Thalamus')
hold on
s1 = scatter3(thalamus_data(~rightThalamus,1), thalamus_data(~rightThalamus,2), thalamus_data(~rightThalamus,3), ...
    70, thalamus_VE_rstdMetric(~rightThalamus), 'filled', 'MarkerEdgeColor', 'k');
colorbar;
colormap cool
%view(3);
xlabel('Lateral --- Medial --- Lateral');
ylabel('Posterior --- Anterior');
zlabel('Inferior --- Superior');
% plotting actual glass brain....
patchs = patch('Faces', faces_left, 'Vertices', vertices_left, 'edgecolor', 'none', 'facecolor', 'flat', 'facealpha', .2);
facecolor = repmat([1 1 1],length(faces_left),1);
set(patchs, 'FaceVertexCData', facecolor);
camlight 
lighting gouraud
view([1 0 0]); % for medial view of left hemi 1












% plotting brains of significant nonsig contacts for each model

% RSTD model
figure(105)
subplot(2,3,[1:2,4:5])
subtitle('rstdSuprise Encoding')
hold on
s5 = scatter3(ElecXYZProj_All(sig_,1), ElecXYZProj_All(:,2), ElecXYZProj_All(:,3), ...
    70, amygdala_VE_rstdMetric, 'filled', 'MarkerEdgeColor', 'k');
colorbar;
colormap cool
view(3);
xlabel('Lateral --- Medial --- Lateral');
ylabel('Anterior --- Posterior');
zlabel('Inferior --- Superior');
% plotting actual glass brain....
patchs = patch('Faces', BrainSurfRaw.faces, 'Vertices', BrainSurfRaw.vertices, 'edgecolor', 'none', 'facecolor', 'flat', 'facealpha', .2);
facecolor = repmat([1 1 1],length(BrainSurfRaw.faces),1);
set(patchs, 'FaceVertexCData', facecolor);
camlight
lighting gouraud









































% load coarse labels created from BART_BrainRegions
    % (a) with hemipsheres
    regions_wHemi_CoarseStruct = load('\\155.100.91.44\d\Data\Rhiannon\BART_RLDM_outputs\BrainRegions\regions_wHemi_CoarseStruct');
    % (b) without hemipsheres
    regions_woHemi_CoarseStruct = load('\\155.100.91.44\d\Data\Rhiannon\BART_RLDM_outputs\BrainRegions\regions_woHemi_CoarseStruct');

    % Getting trodeLabels.
    sum(arrayfun(@(x) numel(x.neuralFit), TDdata)) % check size of TDdata = 5167

    % trodeLabels with Hemisphere
    trodeLabelsTD_wHemi = {};
    trodeLabelsTD_wHemi = regions_wHemi_CoarseStruct.regions_wHemi_CoarseStruct;
    size(trodeLabelsTD_wHemi) % check size of TrodeLabels = 5167

    % trodeLabels with Hemisphere
    trodeLabelsTD_woHemi = {};
    trodeLabelsTD_woHemi = regions_woHemi_CoarseStruct.regions_woHemi_CoarseStruct;
    size(trodeLabelsTD_woHemi) % check size of TrodeLabels = 5167

    % 5) test for signficance within each brain area
    % Loading in impulsivity Data
    riskaversionKLD_yellows = [bhvStruct.impulsivityKLD_yellows]; % apZVal_Yellow
    riskaversionMetric = -riskaversionKLD_yellows;
   
    % initializing VOIs
    impByTrode = [];
    sig_RSTD_PE_Models = [];
    sig_RSTD_VE_Models = [];
    sig_Risk_PE_Models = [];
    sig_Risk_VE_Models = [];
    sig_Reward_PE_Models = [];
    sig_Reward_VE_Models = [];
    sigModelsPOP = [];
    alp = 0.05;

    % MODEL SIGNIFICANCE ~ [rstdPE rstdVE rewVE rewPE risVE risPE];
    for p = 1:size(TDdata,2)
        for chz = 1:length(TDdata(p).neuralFit)
            % RSTD
            sig_RSTD_PE_Models = cat(1,sig_RSTD_PE_Models,TDdata(p).neuralFit(chz).rstdSurpriseModelPOPCTRL.Coefficients{2,6}<alp);
            sig_RSTD_VE_Models = cat(1,sig_RSTD_VE_Models,TDdata(p).neuralFit(chz).rstdExpectationModelPOPCTRL.Coefficients{2,6}<alp);
            % Risk
            sig_Risk_PE_Models = cat(1,sig_Risk_PE_Models,TDdata(p).neuralFit(chz).riskSurpriseModelPOPCTRL.Coefficients{2,6}<alp);
            sig_Risk_VE_Models = cat(1,sig_Risk_VE_Models,TDdata(p).neuralFit(chz).riskExpectationModelPOPCTRL.Coefficients{2,6}<alp);
            % Reward
            sig_Reward_PE_Models = cat(1,sig_Reward_PE_Models,TDdata(p).neuralFit(chz).rewardSurpriseModelPOPCTRL.Coefficients{2,6}<alp);
            sig_Reward_VE_Models = cat(1,sig_Reward_VE_Models,TDdata(p).neuralFit(chz).rewardExpectationModelPOPCTRL.Coefficients{2,6}<alp);
        end
    end

    % Combine into significant model matrix
    sigContacts_allPts = cat(2, ...
        impByTrode,...
        sig_RSTD_PE_Models, ...
        sig_RSTD_VE_Models, ...
        sig_Risk_PE_Models, ...
        sig_Risk_VE_Models, ...
        sig_Reward_PE_Models, ...
        sig_Reward_VE_Models);

    % statistically test how many contacts encoded each model
    num_significant_contacts = sum(sigContacts_allPts(2:end), 2);


    %  6) Compare model significance across brain areas.

 

    % RSTD_supriseModel
    sig_RSTD_PE_Amygdala = sum(sig_RSTD_PE_Models(amygdala_Contacts))/sum(amygdala_Contacts)*100
    sig_RSTD_PE_Cingulate = sum(sig_RSTD_PE_Models(cingulate_Contacts))/sum(cingulate_Contacts)*100
    sig_RSTD_PE_Hippocampus = sum(sig_RSTD_PE_Models(hippocampus_Contacts))/sum(hippocampus_Contacts)*100
    sig_RSTD_PE_InfFrontal = sum(sig_RSTD_PE_Models(InfFrontal_Contacts))/sum(InfFrontal_Contacts)*100
    sig_RSTD_PE_Insula = sum(sig_RSTD_PE_Models(insula_Contacts))/sum(insula_Contacts)*100
    sig_RSTD_PE_MedFrontal = sum(sig_RSTD_PE_Models(MedFrontal_Contacts))/sum(MedFrontal_Contacts)*100
    sig_RSTD_PE_NAccumbens = sum(sig_RSTD_PE_Models(NAccumbens_Contacts))/sum(NAccumbens_Contacts)*100
    sig_RSTD_PE_OrbFrontal = sum(sig_RSTD_PE_Models(OrbFrontal_Contacts))/sum(OrbFrontal_Contacts)*100
    sig_RSTD_PE_Striatum = sum(sig_RSTD_PE_Models(striatum_Contacts))/sum(striatum_Contacts)*100
    sig_RSTD_PE_SupFrontal = sum(sig_RSTD_PE_Models(SupFrontal_Contacts))/sum(SupFrontal_Contacts)*100
    sig_RSTD_PE_Thalamus = sum(sig_RSTD_PE_Models(thalamus_Contacts))/sum(thalamus_Contacts)*100

    % RSTD_expectationModel
    sig_RSTD_VE_Amygdala = sum(sig_RSTD_VE_Models(amygdala_Contacts))/sum(amygdala_Contacts)*100
    sig_RSTD_VE_Cingulate = sum(sig_RSTD_VE_Models(cingulate_Contacts))/sum(cingulate_Contacts)*100
    sig_RSTD_VE_Hippocampus = sum(sig_RSTD_VE_Models(hippocampus_Contacts))/sum(hippocampus_Contacts)*100
    sig_RSTD_VE_InfFrontal = sum(sig_RSTD_VE_Models(InfFrontal_Contacts))/sum(InfFrontal_Contacts)*100
    sig_RSTD_VE_Insula = sum(sig_RSTD_VE_Models(insula_Contacts))/sum(insula_Contacts)*100
    sig_RSTD_VE_MedFrontal = sum(sig_RSTD_VE_Models(MedFrontal_Contacts))/sum(MedFrontal_Contacts)*100
    sig_RSTD_VE_NAccumbens = sum(sig_RSTD_VE_Models(NAccumbens_Contacts))/sum(NAccumbens_Contacts)*100
    sig_RSTD_VE_OrbFrontal = sum(sig_RSTD_VE_Models(OrbFrontal_Contacts))/sum(OrbFrontal_Contacts)*100
    sig_RSTD_VE_Striatum = sum(sig_RSTD_VE_Models(striatum_Contacts))/sum(striatum_Contacts)*100
    sig_RSTD_VE_SupFrontal = sum(sig_RSTD_VE_Models(SupFrontal_Contacts))/sum(SupFrontal_Contacts)*100
    sig_RSTD_VE_Thalamus = sum(sig_RSTD_VE_Models(thalamus_Contacts))/sum(thalamus_Contacts)*100

    % Risk_supriseModel
    sig_Risk_PE_Amygdala = sum(sig_Risk_PE_Models(amygdala_Contacts))/sum(amygdala_Contacts)*100
    sig_Risk_PE_Cingulate = sum(sig_Risk_PE_Models(cingulate_Contacts))/sum(cingulate_Contacts)*100
    sig_Risk_PE_Hippocampus = sum(sig_Risk_PE_Models(hippocampus_Contacts))/sum(hippocampus_Contacts)*100
    sig_Risk_PE_InfFrontal = sum(sig_Risk_PE_Models(InfFrontal_Contacts))/sum(InfFrontal_Contacts)*100
    sig_Risk_PE_Insula = sum(sig_Risk_PE_Models(insula_Contacts))/sum(insula_Contacts)*100
    sig_Risk_PE_MedFrontal = sum(sig_Risk_PE_Models(MedFrontal_Contacts))/sum(MedFrontal_Contacts)*100
    sig_Risk_PE_NAccumbens = sum(sig_Risk_PE_Models(NAccumbens_Contacts))/sum(NAccumbens_Contacts)*100
    sig_Risk_PE_OrbFrontal = sum(sig_Risk_PE_Models(OrbFrontal_Contacts))/sum(OrbFrontal_Contacts)*100
    sig_Risk_PE_Striatum = sum(sig_Risk_PE_Models(striatum_Contacts))/sum(striatum_Contacts)*100
    sig_Risk_PE_SupFrontal = sum(sig_Risk_PE_Models(SupFrontal_Contacts))/sum(SupFrontal_Contacts)*100
    sig_Risk_PE_Thalamus = sum(sig_Risk_PE_Models(thalamus_Contacts))/sum(thalamus_Contacts)*100

    % Risk_expectationModel
    sig_Risk_VE_Amygdala = sum(sig_Risk_VE_Models(amygdala_Contacts))/sum(amygdala_Contacts)*100
    sig_Risk_VE_Cingulate = sum(sig_Risk_VE_Models(cingulate_Contacts))/sum(cingulate_Contacts)*100
    sig_Risk_VE_Hippocampus = sum(sig_Risk_VE_Models(hippocampus_Contacts))/sum(hippocampus_Contacts)*100
    sig_Risk_VE_InfFrontal = sum(sig_Risk_VE_Models(InfFrontal_Contacts))/sum(InfFrontal_Contacts)*100
    sig_Risk_VE_Insula = sum(sig_Risk_VE_Models(insula_Contacts))/sum(insula_Contacts)*100
    sig_Risk_VE_MedFrontal = sum(sig_Risk_VE_Models(MedFrontal_Contacts))/sum(MedFrontal_Contacts)*100
    sig_Risk_VE_NAccumbens = sum(sig_Risk_VE_Models(NAccumbens_Contacts))/sum(NAccumbens_Contacts)*100
    sig_Risk_VE_OrbFrontal = sum(sig_Risk_VE_Models(OrbFrontal_Contacts))/sum(OrbFrontal_Contacts)*100
    sig_Risk_VE_Striatum = sum(sig_Risk_VE_Models(striatum_Contacts))/sum(striatum_Contacts)*100
    sig_Risk_VE_SupFrontal = sum(sig_Risk_VE_Models(SupFrontal_Contacts))/sum(SupFrontal_Contacts)*100
    sig_Risk_VE_Thalamus = sum(sig_Risk_VE_Models(thalamus_Contacts))/sum(thalamus_Contacts)*100

    % Reward_supriseModel
    sig_Reward_PE_Amygdala = sum(sig_Reward_PE_Models(amygdala_Contacts))/sum(amygdala_Contacts)*100
    sig_Reward_PE_Cingulate = sum(sig_Reward_PE_Models(cingulate_Contacts))/sum(cingulate_Contacts)*100
    sig_Reward_PE_Hippocampus = sum(sig_Reward_PE_Models(hippocampus_Contacts))/sum(hippocampus_Contacts)*100
    sig_Reward_PE_InfFrontal = sum(sig_Reward_PE_Models(InfFrontal_Contacts))/sum(InfFrontal_Contacts)*100
    sig_Reward_PE_Insula = sum(sig_Reward_PE_Models(insula_Contacts))/sum(insula_Contacts)*100
    sig_Reward_PE_MedFrontal = sum(sig_Reward_PE_Models(MedFrontal_Contacts))/sum(MedFrontal_Contacts)*100
    sig_Reward_PE_NAccumbens = sum(sig_Reward_PE_Models(NAccumbens_Contacts))/sum(NAccumbens_Contacts)*100
    sig_Reward_PE_OrbFrontal = sum(sig_Reward_PE_Models(OrbFrontal_Contacts))/sum(OrbFrontal_Contacts)*100
    sig_Reward_PE_Striatum = sum(sig_Reward_PE_Models(striatum_Contacts))/sum(striatum_Contacts)*100
    sig_Reward_PE_SupFrontal = sum(sig_Reward_PE_Models(SupFrontal_Contacts))/sum(SupFrontal_Contacts)*100
    sig_Reward_PE_Thalamus = sum(sig_Reward_PE_Models(thalamus_Contacts))/sum(thalamus_Contacts)*100

    % Reward_expectationModel
    sig_Reward_VE_Amygdala = sum(sig_Reward_VE_Models(amygdala_Contacts))/sum(amygdala_Contacts)*100
    sig_Reward_VE_Cingulate = sum(sig_Reward_VE_Models(cingulate_Contacts))/sum(cingulate_Contacts)*100
    sig_Reward_VE_Hippocampus = sum(sig_Reward_VE_Models(hippocampus_Contacts))/sum(hippocampus_Contacts)*100
    sig_Reward_VE_InfFrontal = sum(sig_Reward_VE_Models(InfFrontal_Contacts))/sum(InfFrontal_Contacts)*100
    sig_Reward_VE_Insula = sum(sig_Reward_VE_Models(insula_Contacts))/sum(insula_Contacts)*100
    sig_Reward_VE_MedFrontal = sum(sig_Reward_VE_Models(MedFrontal_Contacts))/sum(MedFrontal_Contacts)*100
    sig_Reward_VE_NAccumbens = sum(sig_Reward_VE_Models(NAccumbens_Contacts))/sum(NAccumbens_Contacts)*100
    sig_Reward_VE_OrbFrontal = sum(sig_Reward_VE_Models(OrbFrontal_Contacts))/sum(OrbFrontal_Contacts)*100
    sig_Reward_VE_Striatum = sum(sig_Reward_VE_Models(striatum_Contacts))/sum(striatum_Contacts)*100
    sig_Reward_VE_SupFrontal = sum(sig_Reward_VE_Models(SupFrontal_Contacts))/sum(SupFrontal_Contacts)*100
    sig_Reward_VE_Thalamus = sum(sig_Reward_VE_Models(thalamus_Contacts))/sum(thalamus_Contacts)*100

    % Plotting region responses to models
    data = [
        sig_RSTD_PE_Amygdala, sig_RSTD_PE_Cingulate, sig_RSTD_PE_Hippocampus, sig_RSTD_PE_InfFrontal, sig_RSTD_PE_Insula, sig_RSTD_PE_MedFrontal, sig_RSTD_PE_NAccumbens, sig_RSTD_PE_OrbFrontal, sig_RSTD_PE_Striatum, sig_RSTD_PE_SupFrontal, sig_RSTD_PE_Thalamus;
        sig_RSTD_VE_Amygdala, sig_RSTD_VE_Cingulate, sig_RSTD_VE_Hippocampus, sig_RSTD_VE_InfFrontal, sig_RSTD_VE_Insula, sig_RSTD_VE_MedFrontal, sig_RSTD_VE_NAccumbens, sig_RSTD_VE_OrbFrontal, sig_RSTD_VE_Striatum, sig_RSTD_VE_SupFrontal, sig_RSTD_VE_Thalamus;
        sig_Risk_PE_Amygdala, sig_Risk_PE_Cingulate, sig_Risk_PE_Hippocampus, sig_Risk_PE_InfFrontal, sig_Risk_PE_Insula, sig_Risk_PE_MedFrontal, sig_Risk_PE_NAccumbens, sig_Risk_PE_OrbFrontal, sig_Risk_PE_Striatum, sig_Risk_PE_SupFrontal, sig_Risk_PE_Thalamus;
        sig_Risk_VE_Amygdala, sig_Risk_VE_Cingulate, sig_Risk_VE_Hippocampus, sig_Risk_VE_InfFrontal, sig_Risk_VE_Insula, sig_Risk_VE_MedFrontal, sig_Risk_VE_NAccumbens, sig_Risk_VE_OrbFrontal, sig_Risk_VE_Striatum, sig_Risk_VE_SupFrontal, sig_Risk_VE_Thalamus;
        sig_Reward_PE_Amygdala, sig_Reward_PE_Cingulate, sig_Reward_PE_Hippocampus, sig_Reward_PE_InfFrontal, sig_Reward_PE_Insula, sig_Reward_PE_MedFrontal, sig_Reward_PE_NAccumbens, sig_Reward_PE_OrbFrontal, sig_Reward_PE_Striatum, sig_Reward_PE_SupFrontal, sig_Reward_PE_Thalamus;
        sig_Reward_VE_Amygdala, sig_Reward_VE_Cingulate, sig_Reward_VE_Hippocampus, sig_Reward_VE_InfFrontal, sig_Reward_VE_Insula, sig_Reward_VE_MedFrontal, sig_Reward_VE_NAccumbens, sig_Reward_VE_OrbFrontal, sig_Reward_VE_Striatum, sig_Reward_VE_SupFrontal, sig_Reward_VE_Thalamus];

    % Define regions for x-axis labels
    regions = {'Amygdala', 'Cingulate', 'Hippocampus', 'InfFrontal', 'Insula',...
        'MedFrontal', 'NAccumbens', 'OrbFrontal', 'Striatum', 'SupFrontal', 'Thalamus'};
    model_names = {'rstd PE', 'rstd VE', 'risk PE', 'risk VE', 'reward PE', 'reward VE'};
    % Plot each model in its own subplot
    figure(1);
    for i = 1:6
        subplot(2, 3, i);
        bar(data(i, :));
        set(gca, 'XTickLabel', regions, 'XTick', 1:length(regions));
        xlabel('Brain Regions');
        ylabel('Proportion of Sig Contacts (%)');
        ylim([0 25])
        title([model_names{i}]);  % Title for each subplot
        xtickangle(45);  % Rotate x-axis labels for readability
    end

    saveas(1, '\\155.100.91.44\d\Data\Rhiannon\BART_RLDM_outputs\TDlearn\allRegions_ModelEncoding.pdf')

    % regress impulsivity by rstd variable
    lm = fitlm(riskaversionMetric, RSTDmetric) % add to abstract 0.0544...
    riskaversionKLD_yellows = [bhvStruct(:).impulsivityKLD_yellows]
    lm = fitlm(riskaversionKLD_yellows, RSTDmetric)

    % sum of significant contacts that encoded each model
    count_rstdSurprise_Model = sum(sigContacts_allPts(:,2)) % 511
    count_rstdExpectation_Model = sum(sigContacts_allPts(:,3)) % 318
    count_riskSurprise_Model = sum(sigContacts_allPts(:,4)) % 362
    count_riskExpectation_Model = sum(sigContacts_allPts(:,5)) % 388
    count_rewardSurprise_Model = sum(sigContacts_allPts(:,6)) % 529
    count_rewardExpectation_Model = sum(sigContacts_allPts(:,7)) % 316

    allsigContacts_Models = [count_rstdSurprise_Model, count_rstdExpectation_Model, count_riskSurprise_Model,...
        count_riskExpectation_Model, count_rewardSurprise_Model,count_rewardExpectation_Model];

    % Significant Model Encoding by Impulsive Choosers
    impBytrode_Logical=logical(impByTrode);

    % make subsets of models
    sig_MI_RSTD_PE_Models = sig_RSTD_PE_Models(impBytrode_Logical);
    sig_LI_RSTD_PE_Models = sig_RSTD_PE_Models(~impBytrode_Logical);
    sig_MI_RSTD_VE_Models = sig_RSTD_VE_Models(impBytrode_Logical);
    sig_LI_RSTD_VE_Models = sig_RSTD_VE_Models(~impBytrode_Logical);
    sig_MI_Risk_PE_Models = sig_Risk_PE_Models(impBytrode_Logical);
    sig_LI_Risk_PE_Models = sig_Risk_PE_Models(~impBytrode_Logical);
    sig_MI_Risk_VE_Models = sig_Risk_VE_Models(impBytrode_Logical);
    sig_LI_Risk_VE_Models = sig_Risk_VE_Models(~impBytrode_Logical);
    sig_MI_Reward_PE_Models = sig_Reward_PE_Models(impBytrode_Logical);
    sig_LI_Reward_PE_Models = sig_Reward_PE_Models(~impBytrode_Logical);
    sig_MI_Reward_VE_Models = sig_Reward_VE_Models(impBytrode_Logical);
    sig_LI_Reward_VE_Models = sig_Reward_VE_Models(~impBytrode_Logical);

    % RSTD_MI_supriseModel
    sig_MI_RSTD_PE_Amygdala = sum(sig_MI_RSTD_PE_Models(amygdala_Contacts(impBytrode_Logical)))/sum(amygdala_Contacts(impBytrode_Logical))*100
    sig_MI_RSTD_PE_Cingulate = sum(sig_MI_RSTD_PE_Models(cingulate_Contacts(impBytrode_Logical)))/sum(cingulate_Contacts(impBytrode_Logical))*100
    sig_MI_RSTD_PE_Hippocampus = sum(sig_MI_RSTD_PE_Models(hippocampus_Contacts(impBytrode_Logical)))/sum(hippocampus_Contacts(impBytrode_Logical))*100
    sig_MI_RSTD_PE_InfFrontal = sum(sig_MI_RSTD_PE_Models(InfFrontal_Contacts(impBytrode_Logical)))/sum(InfFrontal_Contacts(impBytrode_Logical))*100
    sig_MI_RSTD_PE_Insula = sum(sig_MI_RSTD_PE_Models(insula_Contacts(impBytrode_Logical)))/sum(insula_Contacts(impBytrode_Logical))*100
    sig_MI_RSTD_PE_MedFrontal = sum(sig_MI_RSTD_PE_Models(MedFrontal_Contacts(impBytrode_Logical)))/sum(MedFrontal_Contacts(impBytrode_Logical))*100
    sig_MI_RSTD_PE_NAccumbens = sum(sig_MI_RSTD_PE_Models(NAccumbens_Contacts(impBytrode_Logical)))/sum(NAccumbens_Contacts(impBytrode_Logical))*100
    sig_MI_RSTD_PE_OrbFrontal = sum(sig_MI_RSTD_PE_Models(OrbFrontal_Contacts(impBytrode_Logical)))/sum(OrbFrontal_Contacts(impBytrode_Logical))*100
    sig_MI_RSTD_PE_Striatum = sum(sig_MI_RSTD_PE_Models(striatum_Contacts(impBytrode_Logical)))/sum(striatum_Contacts(impBytrode_Logical))*100
    sig_MI_RSTD_PE_SupFrontal = sum(sig_MI_RSTD_PE_Models(SupFrontal_Contacts(impBytrode_Logical)))/sum(SupFrontal_Contacts(impBytrode_Logical))*100
    sig_MI_RSTD_PE_Thalamus = sum(sig_MI_RSTD_PE_Models(thalamus_Contacts(impBytrode_Logical)))/sum(thalamus_Contacts(impBytrode_Logical))*100

    % RSTD_LI_supriseModel
    sig_LI_RSTD_PE_Amygdala = sum(sig_LI_RSTD_PE_Models(amygdala_Contacts(~impBytrode_Logical)))/sum(amygdala_Contacts(~impBytrode_Logical))*100
    sig_LI_RSTD_PE_Cingulate = sum(sig_LI_RSTD_PE_Models(cingulate_Contacts(~impBytrode_Logical)))/sum(cingulate_Contacts(~impBytrode_Logical))*100
    sig_LI_RSTD_PE_Hippocampus = sum(sig_LI_RSTD_PE_Models(hippocampus_Contacts(~impBytrode_Logical)))/sum(hippocampus_Contacts(~impBytrode_Logical))*100
    sig_LI_RSTD_PE_InfFrontal = sum(sig_LI_RSTD_PE_Models(InfFrontal_Contacts(~impBytrode_Logical)))/sum(InfFrontal_Contacts(~impBytrode_Logical))*100
    sig_LI_RSTD_PE_Insula = sum(sig_LI_RSTD_PE_Models(insula_Contacts(~impBytrode_Logical)))/sum(insula_Contacts(~impBytrode_Logical))*100
    sig_LI_RSTD_PE_MedFrontal = sum(sig_LI_RSTD_PE_Models(MedFrontal_Contacts(~impBytrode_Logical)))/sum(MedFrontal_Contacts(~impBytrode_Logical))*100
    sig_LI_RSTD_PE_NAccumbens = sum(sig_LI_RSTD_PE_Models(NAccumbens_Contacts(~impBytrode_Logical)))/sum(NAccumbens_Contacts(~impBytrode_Logical))*100
    sig_LI_RSTD_PE_OrbFrontal = sum(sig_LI_RSTD_PE_Models(OrbFrontal_Contacts(~impBytrode_Logical)))/sum(OrbFrontal_Contacts(~impBytrode_Logical))*100
    sig_LI_RSTD_PE_Striatum = sum(sig_LI_RSTD_PE_Models(striatum_Contacts(~impBytrode_Logical)))/sum(striatum_Contacts(~impBytrode_Logical))*100
    sig_LI_RSTD_PE_SupFrontal = sum(sig_LI_RSTD_PE_Models(SupFrontal_Contacts(~impBytrode_Logical)))/sum(SupFrontal_Contacts(~impBytrode_Logical))*100
    sig_LI_RSTD_PE_Thalamus = sum(sig_LI_RSTD_PE_Models(thalamus_Contacts(~impBytrode_Logical)))/sum(thalamus_Contacts(~impBytrode_Logical))*100

    % RSTD_MI_expectationModel
    sig_MI_RSTD_VE_Amygdala = sum(sig_MI_RSTD_VE_Models(amygdala_Contacts(impBytrode_Logical)))/sum(amygdala_Contacts(impBytrode_Logical))*100
    sig_MI_RSTD_VE_Cingulate = sum(sig_MI_RSTD_VE_Models(cingulate_Contacts(impBytrode_Logical)))/sum(cingulate_Contacts(impBytrode_Logical))*100
    sig_MI_RSTD_VE_Hippocampus = sum(sig_MI_RSTD_VE_Models(hippocampus_Contacts(impBytrode_Logical)))/sum(hippocampus_Contacts(impBytrode_Logical))*100
    sig_MI_RSTD_VE_InfFrontal = sum(sig_MI_RSTD_VE_Models(InfFrontal_Contacts(impBytrode_Logical)))/sum(InfFrontal_Contacts(impBytrode_Logical))*100
    sig_MI_RSTD_VE_Insula = sum(sig_MI_RSTD_VE_Models(insula_Contacts(impBytrode_Logical)))/sum(insula_Contacts(impBytrode_Logical))*100
    sig_MI_RSTD_VE_MedFrontal = sum(sig_MI_RSTD_VE_Models(MedFrontal_Contacts(impBytrode_Logical)))/sum(MedFrontal_Contacts(impBytrode_Logical))*100
    sig_MI_RSTD_VE_NAccumbens = sum(sig_MI_RSTD_VE_Models(NAccumbens_Contacts(impBytrode_Logical)))/sum(NAccumbens_Contacts(impBytrode_Logical))*100
    sig_MI_RSTD_VE_OrbFrontal = sum(sig_MI_RSTD_VE_Models(OrbFrontal_Contacts(impBytrode_Logical)))/sum(OrbFrontal_Contacts(impBytrode_Logical))*100
    sig_MI_RSTD_VE_Striatum = sum(sig_MI_RSTD_VE_Models(striatum_Contacts(impBytrode_Logical)))/sum(striatum_Contacts(impBytrode_Logical))*100
    sig_MI_RSTD_VE_SupFrontal = sum(sig_MI_RSTD_VE_Models(SupFrontal_Contacts(impBytrode_Logical)))/sum(SupFrontal_Contacts(impBytrode_Logical))*100
    sig_MI_RSTD_VE_Thalamus = sum(sig_MI_RSTD_VE_Models(thalamus_Contacts(impBytrode_Logical)))/sum(thalamus_Contacts(impBytrode_Logical))*100

    % RSTD_LI_expectationModel
    sig_LI_RSTD_VE_Amygdala = sum(sig_LI_RSTD_VE_Models(amygdala_Contacts(~impBytrode_Logical)))/sum(amygdala_Contacts(~impBytrode_Logical))*100
    sig_LI_RSTD_VE_Cingulate = sum(sig_LI_RSTD_VE_Models(cingulate_Contacts(~impBytrode_Logical)))/sum(cingulate_Contacts(~impBytrode_Logical))*100
    sig_LI_RSTD_VE_Hippocampus = sum(sig_LI_RSTD_VE_Models(hippocampus_Contacts(~impBytrode_Logical)))/sum(hippocampus_Contacts(~impBytrode_Logical))*100
    sig_LI_RSTD_VE_InfFrontal = sum(sig_LI_RSTD_VE_Models(InfFrontal_Contacts(~impBytrode_Logical)))/sum(InfFrontal_Contacts(~impBytrode_Logical))*100
    sig_LI_RSTD_VE_Insula = sum(sig_LI_RSTD_VE_Models(insula_Contacts(~impBytrode_Logical)))/sum(insula_Contacts(~impBytrode_Logical))*100
    sig_LI_RSTD_VE_MedFrontal = sum(sig_LI_RSTD_VE_Models(MedFrontal_Contacts(~impBytrode_Logical)))/sum(MedFrontal_Contacts(~impBytrode_Logical))*100
    sig_LI_RSTD_VE_NAccumbens = sum(sig_LI_RSTD_VE_Models(NAccumbens_Contacts(~impBytrode_Logical)))/sum(NAccumbens_Contacts(~impBytrode_Logical))*100
    sig_LI_RSTD_VE_OrbFrontal = sum(sig_LI_RSTD_VE_Models(OrbFrontal_Contacts(~impBytrode_Logical)))/sum(OrbFrontal_Contacts(~impBytrode_Logical))*100
    sig_LI_RSTD_VE_Striatum = sum(sig_LI_RSTD_VE_Models(striatum_Contacts(~impBytrode_Logical)))/sum(striatum_Contacts(~impBytrode_Logical))*100
    sig_LI_RSTD_VE_SupFrontal = sum(sig_LI_RSTD_VE_Models(SupFrontal_Contacts(~impBytrode_Logical)))/sum(SupFrontal_Contacts(~impBytrode_Logical))*100
    sig_LI_RSTD_VE_Thalamus = sum(sig_LI_RSTD_VE_Models(thalamus_Contacts(~impBytrode_Logical)))/sum(thalamus_Contacts(~impBytrode_Logical))*100

    % Risk_MI_supriseModel
    sig_MI_Risk_PE_Amygdala = sum(sig_MI_Risk_PE_Models(amygdala_Contacts(impBytrode_Logical)))/sum(amygdala_Contacts(impBytrode_Logical))*100
    sig_MI_Risk_PE_Cingulate = sum(sig_MI_Risk_PE_Models(cingulate_Contacts(impBytrode_Logical)))/sum(cingulate_Contacts(impBytrode_Logical))*100
    sig_MI_Risk_PE_Hippocampus = sum(sig_MI_Risk_PE_Models(hippocampus_Contacts(impBytrode_Logical)))/sum(hippocampus_Contacts(impBytrode_Logical))*100
    sig_MI_Risk_PE_InfFrontal = sum(sig_MI_Risk_PE_Models(InfFrontal_Contacts(impBytrode_Logical)))/sum(InfFrontal_Contacts(impBytrode_Logical))*100
    sig_MI_Risk_PE_Insula = sum(sig_MI_Risk_PE_Models(insula_Contacts(impBytrode_Logical)))/sum(insula_Contacts(impBytrode_Logical))*100
    sig_MI_Risk_PE_MedFrontal = sum(sig_MI_Risk_PE_Models(MedFrontal_Contacts(impBytrode_Logical)))/sum(MedFrontal_Contacts(impBytrode_Logical))*100
    sig_MI_Risk_PE_NAccumbens = sum(sig_MI_Risk_PE_Models(NAccumbens_Contacts(impBytrode_Logical)))/sum(NAccumbens_Contacts(impBytrode_Logical))*100
    sig_MI_Risk_PE_OrbFrontal = sum(sig_MI_Risk_PE_Models(OrbFrontal_Contacts(impBytrode_Logical)))/sum(OrbFrontal_Contacts(impBytrode_Logical))*100
    sig_MI_Risk_PE_Striatum = sum(sig_MI_Risk_PE_Models(striatum_Contacts(impBytrode_Logical)))/sum(striatum_Contacts(impBytrode_Logical))*100
    sig_MI_Risk_PE_SupFrontal = sum(sig_MI_Risk_PE_Models(SupFrontal_Contacts(impBytrode_Logical)))/sum(SupFrontal_Contacts(impBytrode_Logical))*100
    sig_MI_Risk_PE_Thalamus = sum(sig_MI_Risk_PE_Models(thalamus_Contacts(impBytrode_Logical)))/sum(thalamus_Contacts(impBytrode_Logical))*100

    % Risk_LI_supriseModel
    sig_LI_Risk_PE_Amygdala = sum(sig_LI_Risk_PE_Models(amygdala_Contacts(~impBytrode_Logical)))/sum(amygdala_Contacts(~impBytrode_Logical))*100
    sig_LI_Risk_PE_Cingulate = sum(sig_LI_Risk_PE_Models(cingulate_Contacts(~impBytrode_Logical)))/sum(cingulate_Contacts(~impBytrode_Logical))*100
    sig_LI_Risk_PE_Hippocampus = sum(sig_LI_Risk_PE_Models(hippocampus_Contacts(~impBytrode_Logical)))/sum(hippocampus_Contacts(~impBytrode_Logical))*100
    sig_LI_Risk_PE_InfFrontal = sum(sig_LI_Risk_PE_Models(InfFrontal_Contacts(~impBytrode_Logical)))/sum(InfFrontal_Contacts(~impBytrode_Logical))*100
    sig_LI_Risk_PE_Insula = sum(sig_LI_Risk_PE_Models(insula_Contacts(~impBytrode_Logical)))/sum(insula_Contacts(~impBytrode_Logical))*100
    sig_LI_Risk_PE_MedFrontal = sum(sig_LI_Risk_PE_Models(MedFrontal_Contacts(~impBytrode_Logical)))/sum(MedFrontal_Contacts(~impBytrode_Logical))*100
    sig_LI_Risk_PE_NAccumbens = sum(sig_LI_Risk_PE_Models(NAccumbens_Contacts(~impBytrode_Logical)))/sum(NAccumbens_Contacts(~impBytrode_Logical))*100
    sig_LI_Risk_PE_OrbFrontal = sum(sig_LI_Risk_PE_Models(OrbFrontal_Contacts(~impBytrode_Logical)))/sum(OrbFrontal_Contacts(~impBytrode_Logical))*100
    sig_LI_Risk_PE_Striatum = sum(sig_LI_Risk_PE_Models(striatum_Contacts(~impBytrode_Logical)))/sum(striatum_Contacts(~impBytrode_Logical))*100
    sig_LI_Risk_PE_SupFrontal = sum(sig_LI_Risk_PE_Models(SupFrontal_Contacts(~impBytrode_Logical)))/sum(SupFrontal_Contacts(~impBytrode_Logical))*100
    sig_LI_Risk_PE_Thalamus = sum(sig_LI_Risk_PE_Models(thalamus_Contacts(~impBytrode_Logical)))/sum(thalamus_Contacts(~impBytrode_Logical))*100

    % Risk_MI_expectationModel
    sig_MI_Risk_VE_Amygdala = sum(sig_MI_Risk_VE_Models(amygdala_Contacts(impBytrode_Logical)))/sum(amygdala_Contacts(impBytrode_Logical))*100
    sig_MI_Risk_VE_Cingulate = sum(sig_MI_Risk_VE_Models(cingulate_Contacts(impBytrode_Logical)))/sum(cingulate_Contacts(impBytrode_Logical))*100
    sig_MI_Risk_VE_Hippocampus = sum(sig_MI_Risk_VE_Models(hippocampus_Contacts(impBytrode_Logical)))/sum(hippocampus_Contacts(impBytrode_Logical))*100
    sig_MI_Risk_VE_InfFrontal = sum(sig_MI_Risk_VE_Models(InfFrontal_Contacts(impBytrode_Logical)))/sum(InfFrontal_Contacts(impBytrode_Logical))*100
    sig_MI_Risk_VE_Insula = sum(sig_MI_Risk_VE_Models(insula_Contacts(impBytrode_Logical)))/sum(insula_Contacts(impBytrode_Logical))*100
    sig_MI_Risk_VE_MedFrontal = sum(sig_MI_Risk_VE_Models(MedFrontal_Contacts(impBytrode_Logical)))/sum(MedFrontal_Contacts(impBytrode_Logical))*100
    sig_MI_Risk_VE_NAccumbens = sum(sig_MI_Risk_VE_Models(NAccumbens_Contacts(impBytrode_Logical)))/sum(NAccumbens_Contacts(impBytrode_Logical))*100
    sig_MI_Risk_VE_OrbFrontal = sum(sig_MI_Risk_VE_Models(OrbFrontal_Contacts(impBytrode_Logical)))/sum(OrbFrontal_Contacts(impBytrode_Logical))*100
    sig_MI_Risk_VE_Striatum = sum(sig_MI_Risk_VE_Models(striatum_Contacts(impBytrode_Logical)))/sum(striatum_Contacts(impBytrode_Logical))*100
    sig_MI_Risk_VE_SupFrontal = sum(sig_MI_Risk_VE_Models(SupFrontal_Contacts(impBytrode_Logical)))/sum(SupFrontal_Contacts(impBytrode_Logical))*100
    sig_MI_Risk_VE_Thalamus = sum(sig_MI_Risk_VE_Models(thalamus_Contacts(impBytrode_Logical)))/sum(thalamus_Contacts(impBytrode_Logical))*100

    % Risk_LI_expectationModel
    sig_LI_Risk_VE_Amygdala = sum(sig_LI_Risk_VE_Models(amygdala_Contacts(~impBytrode_Logical)))/sum(amygdala_Contacts(~impBytrode_Logical))*100
    sig_LI_Risk_VE_Cingulate = sum(sig_LI_Risk_VE_Models(cingulate_Contacts(~impBytrode_Logical)))/sum(cingulate_Contacts(impBytrode_Logical))*100
    sig_LI_Risk_VE_Hippocampus = sum(sig_LI_Risk_VE_Models(hippocampus_Contacts(~impBytrode_Logical)))/sum(hippocampus_Contacts(~impBytrode_Logical))*100
    sig_LI_Risk_VE_InfFrontal = sum(sig_LI_Risk_VE_Models(InfFrontal_Contacts(~impBytrode_Logical)))/sum(InfFrontal_Contacts(impBytrode_Logical))*100
    sig_LI_Risk_VE_Insula = sum(sig_LI_Risk_VE_Models(insula_Contacts(~impBytrode_Logical)))/sum(insula_Contacts(~impBytrode_Logical))*100
    sig_LI_Risk_VE_MedFrontal = sum(sig_LI_Risk_VE_Models(MedFrontal_Contacts(~impBytrode_Logical)))/sum(MedFrontal_Contacts(impBytrode_Logical))*100
    sig_LI_Risk_VE_NAccumbens = sum(sig_LI_Risk_VE_Models(NAccumbens_Contacts(~impBytrode_Logical)))/sum(NAccumbens_Contacts(~impBytrode_Logical))*100
    sig_LI_Risk_VE_OrbFrontal = sum(sig_LI_Risk_VE_Models(OrbFrontal_Contacts(~impBytrode_Logical)))/sum(OrbFrontal_Contacts(~impBytrode_Logical))*100
    sig_LI_Risk_VE_Striatum = sum(sig_LI_Risk_VE_Models(striatum_Contacts(~impBytrode_Logical)))/sum(striatum_Contacts(~impBytrode_Logical))*100
    sig_LI_Risk_VE_SupFrontal = sum(sig_LI_Risk_VE_Models(SupFrontal_Contacts(~impBytrode_Logical)))/sum(SupFrontal_Contacts(~impBytrode_Logical))*100
    sig_LI_Risk_VE_Thalamus = sum(sig_LI_Risk_VE_Models(thalamus_Contacts(~impBytrode_Logical)))/sum(thalamus_Contacts(~impBytrode_Logical))*100

    % Reward_MI_supriseModel
    sig_MI_Reward_PE_Amygdala = sum(sig_MI_Reward_PE_Models(amygdala_Contacts(impBytrode_Logical)))/sum(amygdala_Contacts(impBytrode_Logical))*100
    sig_MI_Reward_PE_Cingulate = sum(sig_MI_Reward_PE_Models(cingulate_Contacts(impBytrode_Logical)))/sum(cingulate_Contacts(impBytrode_Logical))*100
    sig_MI_Reward_PE_Hippocampus = sum(sig_MI_Reward_PE_Models(hippocampus_Contacts(impBytrode_Logical)))/sum(hippocampus_Contacts(impBytrode_Logical))*100
    sig_MI_Reward_PE_InfFrontal = sum(sig_MI_Reward_PE_Models(InfFrontal_Contacts(impBytrode_Logical)))/sum(InfFrontal_Contacts(impBytrode_Logical))*100
    sig_MI_Reward_PE_Insula = sum(sig_MI_Reward_PE_Models(insula_Contacts(impBytrode_Logical)))/sum(insula_Contacts(impBytrode_Logical))*100
    sig_MI_Reward_PE_MedFrontal = sum(sig_MI_Reward_PE_Models(MedFrontal_Contacts(impBytrode_Logical)))/sum(MedFrontal_Contacts(impBytrode_Logical))*100
    sig_MI_Reward_PE_NAccumbens = sum(sig_MI_Reward_PE_Models(NAccumbens_Contacts(impBytrode_Logical)))/sum(NAccumbens_Contacts(impBytrode_Logical))*100
    sig_MI_Reward_PE_OrbFrontal = sum(sig_MI_Reward_PE_Models(OrbFrontal_Contacts(impBytrode_Logical)))/sum(OrbFrontal_Contacts(impBytrode_Logical))*100
    sig_MI_Reward_PE_Striatum = sum(sig_MI_Reward_PE_Models(striatum_Contacts(impBytrode_Logical)))/sum(striatum_Contacts(impBytrode_Logical))*100
    sig_MI_Reward_PE_SupFrontal = sum(sig_MI_Reward_PE_Models(SupFrontal_Contacts(impBytrode_Logical)))/sum(SupFrontal_Contacts(impBytrode_Logical))*100
    sig_MI_Reward_PE_Thalamus = sum(sig_MI_Reward_PE_Models(thalamus_Contacts(impBytrode_Logical)))/sum(thalamus_Contacts(impBytrode_Logical))*100

    % Reward_LI_supriseModel
    sig_LI_Reward_PE_Amygdala = sum(sig_LI_Reward_PE_Models(amygdala_Contacts(~impBytrode_Logical)))/sum(amygdala_Contacts(~impBytrode_Logical))*100
    sig_LI_Reward_PE_Cingulate = sum(sig_LI_Reward_PE_Models(cingulate_Contacts(~impBytrode_Logical)))/sum(cingulate_Contacts(~impBytrode_Logical))*100
    sig_LI_Reward_PE_Hippocampus = sum(sig_LI_Reward_PE_Models(hippocampus_Contacts(~impBytrode_Logical)))/sum(hippocampus_Contacts(~impBytrode_Logical))*100
    sig_LI_Reward_PE_InfFrontal = sum(sig_LI_Reward_PE_Models(InfFrontal_Contacts(~impBytrode_Logical)))/sum(InfFrontal_Contacts(~impBytrode_Logical))*100
    sig_LI_Reward_PE_Insula = sum(sig_LI_Reward_PE_Models(insula_Contacts(~impBytrode_Logical)))/sum(insula_Contacts(~impBytrode_Logical))*100
    sig_LI_Reward_PE_MedFrontal = sum(sig_LI_Reward_PE_Models(MedFrontal_Contacts(~impBytrode_Logical)))/sum(MedFrontal_Contacts(~impBytrode_Logical))*100
    sig_LI_Reward_PE_NAccumbens = sum(sig_LI_Reward_PE_Models(NAccumbens_Contacts(~impBytrode_Logical)))/sum(NAccumbens_Contacts(~impBytrode_Logical))*100
    sig_LI_Reward_PE_OrbFrontal = sum(sig_LI_Reward_PE_Models(OrbFrontal_Contacts(~impBytrode_Logical)))/sum(OrbFrontal_Contacts(~impBytrode_Logical))*100
    sig_LI_Reward_PE_Striatum = sum(sig_LI_Reward_PE_Models(striatum_Contacts(~impBytrode_Logical)))/sum(striatum_Contacts(~impBytrode_Logical))*100
    sig_LI_Reward_PE_SupFrontal = sum(sig_LI_Reward_PE_Models(SupFrontal_Contacts(~impBytrode_Logical)))/sum(SupFrontal_Contacts(~impBytrode_Logical))*100
    sig_LI_Reward_PE_Thalamus = sum(sig_LI_Reward_PE_Models(thalamus_Contacts(~impBytrode_Logical)))/sum(thalamus_Contacts(~impBytrode_Logical))*100

    % Reward_expectationModel
    sig_MI_Reward_VE_Amygdala = sum(sig_MI_Reward_VE_Models(amygdala_Contacts(impBytrode_Logical)))/sum(amygdala_Contacts(impBytrode_Logical))*100
    sig_MI_Reward_VE_Cingulate = sum(sig_MI_Reward_VE_Models(cingulate_Contacts(impBytrode_Logical)))/sum(cingulate_Contacts(impBytrode_Logical))*100
    sig_MI_Reward_VE_Hippocampus = sum(sig_MI_Reward_VE_Models(hippocampus_Contacts(impBytrode_Logical)))/sum(hippocampus_Contacts(impBytrode_Logical))*100
    sig_MI_Reward_VE_InfFrontal = sum(sig_MI_Reward_VE_Models(InfFrontal_Contacts(impBytrode_Logical)))/sum(InfFrontal_Contacts(impBytrode_Logical))*100
    sig_MI_Reward_VE_Insula = sum(sig_MI_Reward_VE_Models(insula_Contacts(impBytrode_Logical)))/sum(insula_Contacts(impBytrode_Logical))*100
    sig_MI_Reward_VE_MedFrontal = sum(sig_MI_Reward_VE_Models(MedFrontal_Contacts(impBytrode_Logical)))/sum(MedFrontal_Contacts(impBytrode_Logical))*100
    sig_MI_Reward_VE_NAccumbens = sum(sig_MI_Reward_VE_Models(NAccumbens_Contacts(impBytrode_Logical)))/sum(NAccumbens_Contacts(impBytrode_Logical))*100
    sig_MI_Reward_VE_OrbFrontal = sum(sig_MI_Reward_VE_Models(OrbFrontal_Contacts(impBytrode_Logical)))/sum(OrbFrontal_Contacts(impBytrode_Logical))*100
    sig_MI_Reward_VE_Striatum = sum(sig_MI_Reward_VE_Models(striatum_Contacts(impBytrode_Logical)))/sum(striatum_Contacts(impBytrode_Logical))*100
    sig_MI_Reward_VE_SupFrontal = sum(sig_MI_Reward_VE_Models(SupFrontal_Contacts(impBytrode_Logical)))/sum(SupFrontal_Contacts(impBytrode_Logical))*100
    sig_MI_Reward_VE_Thalamus = sum(sig_MI_Reward_VE_Models(thalamus_Contacts(impBytrode_Logical)))/sum(thalamus_Contacts(impBytrode_Logical))*100

    % Reward_expectationModel
    sig_LI_Reward_VE_Amygdala = sum(sig_LI_Reward_VE_Models(amygdala_Contacts(~impBytrode_Logical)))/sum(amygdala_Contacts(~impBytrode_Logical))*100
    sig_LI_Reward_VE_Cingulate = sum(sig_LI_Reward_VE_Models(cingulate_Contacts(~impBytrode_Logical)))/sum(cingulate_Contacts(~impBytrode_Logical))*100
    sig_LI_Reward_VE_Hippocampus = sum(sig_LI_Reward_VE_Models(hippocampus_Contacts(~impBytrode_Logical)))/sum(hippocampus_Contacts(~impBytrode_Logical))*100
    sig_LI_Reward_VE_InfFrontal = sum(sig_MI_Reward_VE_Models(InfFrontal_Contacts(~impBytrode_Logical)))/sum(InfFrontal_Contacts(~impBytrode_Logical))*100
    sig_LI_Reward_VE_Insula = sum(sig_LI_Reward_VE_Models(insula_Contacts(~impBytrode_Logical)))/sum(insula_Contacts(~impBytrode_Logical))*100
    sig_LI_Reward_VE_MedFrontal = sum(sig_LI_Reward_VE_Models(MedFrontal_Contacts(~impBytrode_Logical)))/sum(MedFrontal_Contacts(~impBytrode_Logical))*100
    sig_LI_Reward_VE_NAccumbens = sum(sig_LI_Reward_VE_Models(NAccumbens_Contacts(~impBytrode_Logical)))/sum(NAccumbens_Contacts(~impBytrode_Logical))*100
    sig_LI_Reward_VE_OrbFrontal = sum(sig_LI_Reward_VE_Models(OrbFrontal_Contacts(~impBytrode_Logical)))/sum(OrbFrontal_Contacts(~impBytrode_Logical))*100
    sig_LI_Reward_VE_Striatum = sum(sig_LI_Reward_VE_Models(striatum_Contacts(~impBytrode_Logical)))/sum(striatum_Contacts(~impBytrode_Logical))*100
    sig_LI_Reward_VE_SupFrontal = sum(sig_LI_Reward_VE_Models(SupFrontal_Contacts(~impBytrode_Logical)))/sum(SupFrontal_Contacts(~impBytrode_Logical))*100
    sig_LI_Reward_VE_Thalamus = sum(sig_LI_Reward_VE_Models(thalamus_Contacts(~impBytrode_Logical)))/sum(thalamus_Contacts(~impBytrode_Logical))*100

    % Plotting region responses to models by impulsivity
    data_MI = [
        sig_MI_RSTD_PE_Amygdala, sig_MI_RSTD_PE_Cingulate, sig_MI_RSTD_PE_Hippocampus, sig_MI_RSTD_PE_InfFrontal, sig_MI_RSTD_PE_Insula, sig_MI_RSTD_PE_MedFrontal, sig_MI_RSTD_PE_NAccumbens, sig_MI_RSTD_PE_OrbFrontal, sig_MI_RSTD_PE_Striatum, sig_MI_RSTD_PE_SupFrontal, sig_MI_RSTD_PE_Thalamus;
        sig_MI_RSTD_VE_Amygdala, sig_MI_RSTD_VE_Cingulate, sig_MI_RSTD_VE_Hippocampus, sig_MI_RSTD_VE_InfFrontal, sig_MI_RSTD_VE_Insula, sig_MI_RSTD_VE_MedFrontal, sig_MI_RSTD_VE_NAccumbens, sig_MI_RSTD_VE_OrbFrontal, sig_MI_RSTD_VE_Striatum, sig_MI_RSTD_VE_SupFrontal, sig_MI_RSTD_VE_Thalamus;
        sig_MI_Risk_PE_Amygdala, sig_MI_Risk_PE_Cingulate, sig_MI_Risk_PE_Hippocampus, sig_MI_Risk_PE_InfFrontal, sig_MI_Risk_PE_Insula, sig_MI_Risk_PE_MedFrontal, sig_MI_Risk_PE_NAccumbens, sig_MI_Risk_PE_OrbFrontal, sig_MI_Risk_PE_Striatum, sig_MI_Risk_PE_SupFrontal, sig_MI_Risk_PE_Thalamus;
        sig_MI_Risk_VE_Amygdala, sig_MI_Risk_VE_Cingulate, sig_MI_Risk_VE_Hippocampus, sig_MI_Risk_VE_InfFrontal, sig_MI_Risk_VE_Insula, sig_MI_Risk_VE_MedFrontal, sig_MI_Risk_VE_NAccumbens, sig_MI_Risk_VE_OrbFrontal, sig_MI_Risk_VE_Striatum, sig_MI_Risk_VE_SupFrontal, sig_MI_Risk_VE_Thalamus;
        sig_MI_Reward_PE_Amygdala, sig_MI_Reward_PE_Cingulate, sig_MI_Reward_PE_Hippocampus, sig_MI_Reward_PE_InfFrontal, sig_MI_Reward_PE_Insula, sig_MI_Reward_PE_MedFrontal, sig_MI_Reward_PE_NAccumbens, sig_MI_Reward_PE_OrbFrontal, sig_MI_Reward_PE_Striatum, sig_MI_Reward_PE_SupFrontal, sig_MI_Reward_PE_Thalamus;
        sig_MI_Reward_VE_Amygdala, sig_MI_Reward_VE_Cingulate, sig_MI_Reward_VE_Hippocampus, sig_MI_Reward_VE_InfFrontal, sig_MI_Reward_VE_Insula, sig_MI_Reward_VE_MedFrontal, sig_MI_Reward_VE_NAccumbens, sig_MI_Reward_VE_OrbFrontal, sig_MI_Reward_VE_Striatum, sig_MI_Reward_VE_SupFrontal, sig_MI_Reward_VE_Thalamus];

    % Define regions for x-axis labels
    regions = {'Amygdala', 'Cingulate', 'Hippocampus', 'InfFrontal', 'Insula',...
        'MedFrontal', 'NAccumbens', 'OrbFrontal', 'Striatum', 'SupFrontal', 'Thalamus'};
    model_names = {'rstd PE', 'rstd VE', 'risk PE', 'risk VE', 'reward PE', 'reward VE'};

    % colors for MI and LI individuals
    sage_green = [119, 171, 86] / 255; % RGB values for sage green ICs
    muted_purple = [79, 49, 170] / 255; % RGB values for muted purple ~ICs

    % Plot each model in its own subplot
    figure(2);
    for i = 1:6
        subplot(2, 3, i);
        b = bar(data_MI(i, :));
        set(gca, 'XTickLabel', regions, 'XTick', 1:length(regions));
        b.FaceColor = "flat";
        b.FaceColor = sage_green;
        xlabel('Brain Regions');
        ylabel('Proportion of MI Sig Contacts (%)');
        ylim([0 35])
        title([model_names{i}]);
        xtickangle(45);
    end

    saveas(2, '\\155.100.91.44\d\Data\Rhiannon\BART_RLDM_outputs\TDlearn\allRegions_ModelEncoding_MI.pdf')

    % Plotting region responses to models by impulsivity
    data_LI = [
        sig_LI_RSTD_PE_Amygdala, sig_LI_RSTD_PE_Cingulate, sig_LI_RSTD_PE_Hippocampus, sig_LI_RSTD_PE_InfFrontal, sig_LI_RSTD_PE_Insula, sig_LI_RSTD_PE_MedFrontal, sig_LI_RSTD_PE_NAccumbens, sig_MI_RSTD_PE_OrbFrontal, sig_LI_RSTD_PE_Striatum, sig_LI_RSTD_PE_SupFrontal, sig_LI_RSTD_PE_Thalamus;
        sig_LI_RSTD_VE_Amygdala, sig_LI_RSTD_VE_Cingulate, sig_LI_RSTD_VE_Hippocampus, sig_LI_RSTD_VE_InfFrontal, sig_LI_RSTD_VE_Insula, sig_LI_RSTD_VE_MedFrontal, sig_LI_RSTD_VE_NAccumbens, sig_MI_RSTD_VE_OrbFrontal, sig_LI_RSTD_VE_Striatum, sig_LI_RSTD_VE_SupFrontal, sig_LI_RSTD_VE_Thalamus;
        sig_LI_Risk_PE_Amygdala, sig_LI_Risk_PE_Cingulate, sig_LI_Risk_PE_Hippocampus, sig_LI_Risk_PE_InfFrontal, sig_LI_Risk_PE_Insula, sig_LI_Risk_PE_MedFrontal, sig_LI_Risk_PE_NAccumbens, sig_MI_Risk_PE_OrbFrontal, sig_LI_Risk_PE_Striatum, sig_LI_Risk_PE_SupFrontal, sig_LI_Risk_PE_Thalamus;
        sig_LI_Risk_VE_Amygdala, sig_LI_Risk_VE_Cingulate, sig_LI_Risk_VE_Hippocampus, sig_LI_Risk_VE_InfFrontal, sig_LI_Risk_VE_Insula, sig_LI_Risk_VE_MedFrontal, sig_LI_Risk_VE_NAccumbens, sig_MI_Risk_VE_OrbFrontal, sig_LI_Risk_VE_Striatum, sig_LI_Risk_VE_SupFrontal, sig_LI_Risk_VE_Thalamus;
        sig_LI_Reward_PE_Amygdala, sig_LI_Reward_PE_Cingulate, sig_LI_Reward_PE_Hippocampus, sig_LI_Reward_PE_InfFrontal, sig_LI_Reward_PE_Insula, sig_LI_Reward_PE_MedFrontal, sig_LI_Reward_PE_NAccumbens, sig_LI_Reward_PE_OrbFrontal, sig_LI_Reward_PE_Striatum, sig_LI_Reward_PE_SupFrontal, sig_LI_Reward_PE_Thalamus;
        sig_LI_Reward_VE_Amygdala, sig_LI_Reward_VE_Cingulate, sig_LI_Reward_VE_Hippocampus, sig_LI_Reward_VE_InfFrontal, sig_LI_Reward_VE_Insula, sig_LI_Reward_VE_MedFrontal, sig_LI_Reward_VE_NAccumbens, sig_LI_Reward_VE_OrbFrontal, sig_LI_Reward_VE_Striatum, sig_LI_Reward_VE_SupFrontal, sig_LI_Reward_VE_Thalamus];

    % Define regions for x-axis labels
    regions = {'Amygdala', 'Cingulate', 'Hippocampus', 'InfFrontal', 'Insula',...
        'MedFrontal', 'NAccumbens', 'OrbFrontal', 'Striatum', 'SupFrontal', 'Thalamus'};
    model_names = {'rstd PE', 'rstd VE', 'risk PE', 'risk VE', 'reward PE', 'reward VE'};

    % Plot each model in its own subplot
    figure(3);
    for i = 1:6
        subplot(2, 3, i);
        b = bar(data_LI(i, :));
        set(gca, 'XTickLabel', regions, 'XTick', 1:length(regions));
        b.FaceColor = "flat";
        b.FaceColor = muted_purple;
        xlabel('Brain Regions');
        ylabel('Proportion of LI Sig Contacts (%)');
        ylim([0 35])
        title([model_names{i}]);
        xtickangle(45);
    end

    saveas(3, '\\155.100.91.44\d\Data\Rhiannon\BART_RLDM_outputs\TDlearn\allRegions_ModelEncoding_LI.pdf')

    %  7) write CCN abstract.
    % finish summary statistics.

    %% statistical tests for proportions
    % now testing the hypothesis that more impulsive subjects have more significant contacts?
    % details about how this is done: sigContacts_all patients is a boolean
    % array whose columns are represent significant contacts for
    % 2: RSTD PE
    % 3: RSTD VE
    % 4: Risk PE
    % 5: Risk VE
    % 6: Reward PE
    % 7: Reward VE

    % setting up data vectors for comparison BETWEEN IMPULSIVITY CATEGORIES
    % rstd PEs
    X_rstdPE = [sum(sigContacts_allPts(impBytrode_Logical,2)) sum(sigContacts_allPts(~impBytrode_Logical,2))];
    X_rstdVE = [sum(sigContacts_allPts(impBytrode_Logical,3)) sum(sigContacts_allPts(~impBytrode_Logical,3))];
    X_rstdBoth = [sum(sigContacts_allPts(impBytrode_Logical,2) & sigContacts_allPts(impBytrode_Logical,3)) sum(sigContacts_allPts(~impBytrode_Logical,2) & sigContacts_allPts(~impBytrode_Logical,3))];

    % risk PEs & VEs
    X_riskPE = [sum(sigContacts_allPts(impBytrode_Logical,4)) sum(sigContacts_allPts(~impBytrode_Logical,4))];
    X_riskVE = [sum(sigContacts_allPts(impBytrode_Logical,5)) sum(sigContacts_allPts(~impBytrode_Logical,5))];
    X_riskBoth = [sum(sigContacts_allPts(impBytrode_Logical,4) & sigContacts_allPts(impBytrode_Logical,5)) sum(sigContacts_allPts(~impBytrode_Logical,4) & sigContacts_allPts(~impBytrode_Logical,5))];

    % reward PEs
    X_rewardPE = [sum(sigContacts_allPts(impBytrode_Logical,6)) sum(sigContacts_allPts(~impBytrode_Logical,6))];
    X_rewardVE = [sum(sigContacts_allPts(impBytrode_Logical,7)) sum(sigContacts_allPts(~impBytrode_Logical,7))];
    X_rewardBoth = [sum(sigContacts_allPts(impBytrode_Logical,6) & sigContacts_allPts(impBytrode_Logical,7)) sum(sigContacts_allPts(~moreImpulsive,6) & sigContacts_allPts(~moreImpulsive,7))];


    %% TO HERE>
    % both reward and risk PEs
    X_pPEs = [sum(sigContacts_allPts(moreImpulsive,1) & sigContacts_allPts(moreImpulsive,3)) sum(sigContacts_allPts(~moreImpulsive,1) & sigContacts_allPts(~moreImpulsive,3))];
    X_nPEs = [sum(sigContacts_allPts(moreImpulsive,2) & sigContacts_allPts(moreImpulsive,4)) sum(sigContacts_allPts(~moreImpulsive,2) & sigContacts_allPts(~moreImpulsive,4))];
    X_allPEs = [sum(sigContacts_allPts(moreImpulsive,1) & sigContacts_allPts(moreImpulsive,2) & sigContacts_allPts(moreImpulsive,3) & sigContacts_allPts(moreImpulsive,4)) sum(sigContacts_allPts(~moreImpulsive,1) & sigContacts_allPts(~moreImpulsive,2) & sigContacts_allPts(~moreImpulsive,3) & sigContacts_allPts(~moreImpulsive,4))];
    % reward PE/risk PE comparison
    X_rewardPE = [sum(sigContacts_allPts(:,1)) sum(sigContacts_allPts(:,2))];
    X_riskPE = [sum(sigContacts_allPts(:,3)) sum(sigContacts_allPts(:,4))];

    % hemisphere-specific tests & brain area specific tests.
    % Ho: there is no difference in proportion of encoding contacts in each
    % hemisphere
    X_hemi_rewardPEs = [sum(sum(sigContacts_allPts(leftHemiContacts_allPts,1:2))) sum(sum(sigContacts_allPts(~leftHemiContacts_allPts,1:2)))];
    X_hemi_riskPEs = [sum(sum(sigContacts_allPts(leftHemiContacts_allPts,3:4))) sum(sum(sigContacts_allPts(~leftHemiContacts_allPts,3:4)))];

    % hemisphere X impulsivity level tests
    X_hemi_rewardPEs_imp = [sum(sum(sigContacts_allPts(leftHemiContacts_allPts & moreImpulsive,1:2))) sum(sum(sigContacts_allPts(~leftHemiContacts_allPts & moreImpulsive,1:2)))];
    X_hemi_riskPEs_imp = [sum(sum(sigContacts_allPts(leftHemiContacts_allPts & moreImpulsive,3:4))) sum(sum(sigContacts_allPts(~leftHemiContacts_allPts & moreImpulsive,3:4)))];
    X_hemi_rewardPEs_notImp = [sum(sum(sigContacts_allPts(leftHemiContacts_allPts & ~moreImpulsive,1:2))) sum(sum(sigContacts_allPts(~leftHemiContacts_allPts & ~moreImpulsive,1:2)))];
    X_hemi_riskPEs_notImp = [sum(sum(sigContacts_allPts(leftHemiContacts_allPts & ~moreImpulsive,3:4))) sum(sum(sigContacts_allPts(~leftHemiContacts_allPts & ~moreImpulsive,3:4)))];

    % total numbers
    N = [sum(impBytrode_Logical) sum(~impBytrode_Logical)];
    correct = false;


    %% hypothesis tests
    %  RSTD variable encoding
    X_totRSTD = [sum(X_rstdPE) sum(X_rstdVE)];
    [h_totRSTD, p_totRSTD, chi2stat_totRSTD, df_totRSTD] = prop_test(X_totRSTD, N, correct)
    %  RSTD PE encoding MI/LI
    [h_rstdPE, p_rstdPE, chi2stat_rstdPE, df_rstdPE] = prop_test(X_rstdPE, N, correct)
    %  RSTD VE encoding MI/LI
    [h_rstdVE, p_rstdVE, chi2stat_rstdVE, df_rstdVE] = prop_test(X_rstdVE, N, correct)

    figure
    text(0,2,'tests for overall encoding of RSTD model (rstdPE vs rstdVE):','fontweight','bold')
    text(0,1.8,sprintf('proportion of contacts significantly encoding rstdPE vs. rstdVE variables: (%.2f, %.2f) percent (X^2(%d) = %.2f, p = %f)',(X_totRSTD(1)./N(1))*100,(X_totRSTD(2)./N(2))*100,df_totRSTD,chi2stat_totRSTD,p_totRSTD));
    text(0,1.6,sprintf('proportion of contacts significantly encoding rstdPE MI vs. rstdPE LI : (%.2f, %.2f) percent (X^2(%d) = %.2f, p = %f)',(X_rstdPE(1)./N(1))*100,(X_rstdPE(2)./N(2))*100,df_rstdPE,chi2stat_rstdPE,p_rstdPE));
    text(0,1.4,sprintf('proportion of contacts significantly encoding rstdVE MI vs. rstdVE LI: (%.2f, %.2f) percent (X^2(%d) = %.2f, p = %f)',(X_rstdVE(1)./N(1))*100,(X_rstdVE(2)./N(2))*100,df_rstdVE,chi2stat_rstdVE,p_rstdVE));

    ylim([-2 2.5])
    axis off

    % overall PE encoding (reward and risk)
    X_tot_PE = [sum(X_rewardPE) sum(X_riskPE)];
    [h_totPE, p_totPE, chi2stat_totPE, df_totPE] = prop_test(X_tot_PE, N, correct)
    % RewardPE encoding
    [h_rewPE, p_rewPE, chi2stat_rewPE, df_rewPE] = prop_test(X_rewardPE, N, correct)
    % RiskPE encoding
    [h_risPE, p_risPE, chi2stat_risPE, df_risPE] = prop_test(X_riskPE, N, correct)

    figure
    text(0,2,'tests for overall encoding of PE model (risk vs reward):','fontweight','bold')
    text(0,1.8,sprintf('proportion of contacts significantly encoding riskPE vs. rewardPE variables: (%.2f, %.2f) percent (X^2(%d) = %.2f, p = %f)',(X_tot_PE(1)./N(1))*100,(X_tot_PE(2)./N(2))*100,df_totPE,chi2stat_totPE,p_totPE));
    text(0,1.6,sprintf('proportion of contacts significantly encoding  rewardPE MI vs. rewardPE LI : (%.2f, %.2f) percent (X^2(%d) = %.2f, p = %f)',(X_rewardPE(1)./N(1))*100,(X_rewardPE(2)./N(2))*100,df_rewPE,chi2stat_rewPE,p_rewPE));
    text(0,1.4,sprintf('proportion of contacts significantly encoding  riskPE MI vs. riskPE LI : (%.2f, %.2f) percent (X^2(%d) = %.2f, p = %f)',(X_riskPE(1)./N(1))*100,(X_riskPE(2)./N(2))*100,df_risPE,chi2stat_risPE,p_risPE));

    ylim([-2 2.5])
    axis off


    % overall VE encoding (reward and risk)
    X_tot_VE = [sum(X_rewardVE) sum(X_riskVE)];
    [h_totVE, p_totVE, chi2stat_totVE, df_totVE] = prop_test(X_tot_VE, N, correct)
    % RewardVE encoding
    [h_rewVE, p_rewVE, chi2stat_rewVE, df_rewVE] = prop_test(X_rewardVE, N, correct)
    % RiskVE encoding
    [h_risVE, p_risVE, chi2stat_risVE, df_risVE] = prop_test(X_riskVE, N, correct)

    figure
    text(0,2,'tests for overall encoding of VE model (risk vs reward):','fontweight','bold')
    text(0,1.8,sprintf('proportion of contacts significantly encoding riskVE vs. rewardVE variables: (%.2f, %.2f) percent (X^2(%d) = %.2f, p = %f)',(X_tot_VE(1)./N(1))*100,(X_tot_VE(2)./N(2))*100,df_totVE,chi2stat_totVE,p_totVE));
    text(0,1.6,sprintf('proportion of contacts significantly encoding  rewardVE MI vs. rewardVE LI : (%.2f, %.2f) percent (X^2(%d) = %.2f, p = %f)',(X_rewardVE(1)./N(1))*100,(X_rewardVE(2)./N(2))*100,df_rewVE,chi2stat_rewVE,p_rewVE));
    text(0,1.4,sprintf('proportion of contacts significantly encoding  riskVE MI vs. riskVE LI : (%.2f, %.2f) percent (X^2(%d) = %.2f, p = %f)',(X_riskVE(1)./N(1))*100,(X_riskVE(2)./N(2))*100,df_risVE,chi2stat_risVE,p_risVE));

    ylim([-2 2.5])
    axis off




    %     % tests for reward
    %     [h_pRPE, p_pRPE, chi2stat_pRPE, df_pRPE] = prop_test(X_pRPE , N, correct)
    %     [h_nRPE, p_nRPE, chi2stat_nRPE, df_nRPE] = prop_test(X_nRPE , N, correct)
    %     [h_both, p_both, chi2stat_both, df_both] = prop_test(X_both , N, correct)
    %
    %     % tests for risk
    %     [h_pRiskPE, p_pRiskPE, chi2stat_pRiskPE, df_pRiskPE] = prop_test(X_pRiskPE , N, correct)
    %     [h_nRiskPE, p_nRiskPE, chi2stat_nRiskPE, df_nRiskPE] = prop_test(X_nRiskPE , N, correct)
    %     [h_riskboth, p_riskboth, chi2stat_riskboth, df_riskboth] = prop_test(X_riskboth , N, correct)
    %
    %     % tests for conjunction of risk and reward PEs (positive, negative, both)
    %     [h_pPEs, p_pPEs, chi2stat_pPEs, df_pPEs] = prop_test(X_pPEs , N, correct)
    %     [h_nPEs, p_nPEs, chi2stat_nPEs, df_nPEs] = prop_test(X_nPEs , N, correct)
    %     [h_allPEs, p_allPEs, chi2stat_allPEs, df_allPEs] = prop_test(X_allPEs , N, correct)
    %
    %     % tests for proportions of all signficant contacts between hemispheres
    %     [h_hemi_rewardPEs, p_hemi_rewardPEs, chi2stat_hemi_rewardPEs, df_hemi_rewardPEs] = prop_test(X_hemi_rewardPEs , N, correct)
    %     [h_hemi_riskPEs, p_hemi_riskPEs, chi2stat_hemi_riskPEs, df_hemi_riskPEs] = prop_test(X_hemi_riskPEs , N, correct)
    %
    %     % Ho: no difference in the proportion of contacts in the the left
    %     % hemisphere in more impulsive patients
    %     [h_hemi_rewardPEs_imp, p_hemi_rewardPEs_imp, chi2stat_hemi_rewardPEs_imp, df_hemi_rewardPEs_imp] = prop_test(X_hemi_rewardPEs_imp , N, correct)
    %     [h_hemi_riskPEs_imp, p_hemi_riskPEs_imp, chi2stat_hemi_riskPEs_imp, df_hemi_riskPEs_imp] = prop_test(X_hemi_riskPEs_imp , N, correct)
    %     [h_hemi_rewardPEs_notImp, p_hemi_rewardPEs_notImp, chi2stat_hemi_rewardPEs_notImp, df_hemi_rewardPEs_notImp] = prop_test(X_hemi_rewardPEs_notImp , N, correct)
    %     [h_hemi_riskPEs_notImp, p_hemi_riskPEs_notImp, chi2stat_hemi_riskPEs_notImp, df_hemi_riskPEs_notImp] = prop_test(X_hemi_riskPEs_notImp , N, correct)

    % Any age or sex differences?
    BARTcsv = dir('\\155.100.91.44\d\Data\preProcessed\BART_preprocessed\BART_BISBAS_BDI_Logicals.csv');
    demographicsTable = table;
    for T = 1:length(BARTcsv)
        TMP = readtable(fullfile(BARTcsv(T).folder,BARTcsv(T).name));
        demographicsTable = cat(1,demographicsTable,TMP);
    end

    % Create Metrics
    age = demographicsTable.Age
    sex = demographicsTable.Sex

    % more or less impulsive? M/F Y/O (regression for age)
    %lm(impulsivityMetric, age) % regress age by IM
    %ranksum(impulsivityMetric, sex)

    % plotting a figrue that lists the results.
    pltResultsSummary = true;
    if pltResultsSummary
        figure % TO DO! UPDATE.
        % edited p-values to have all decimal points. (07/13/23, RC)
        text(0,2,'tests for overall encoding of each type of TD model (rewardPE vs riskPE):','fontweight','bold')
        text(0,1.8,sprintf('proportion of contacts significantly encoding rewardPE vs. riskPE variables: (%.2f, %.2f) percent (X^2(%d) = %.2f, p = %f)',(X_tot_PE(1)./N(1))*100,(X_tot_PE(2)./N(2))*100,df_totPE,chi2stat_totPE,p_totPE));
        text(0,1.6,sprintf('proportion of contacts significantly encoding positiveRPE vs. negativeRPE : (%.2f, %.2f) percent (X^2(%d) = %.2f, p = %f)',(X_rewardPE(1)./N(1))*100,(X_rewardPE(2)./N(2))*100,df_rewPE,chi2stat_rewPE,p_rewPE));
        text(0,1.4,sprintf('proportion of contacts significantly encoding RiskPE vs.  RiskPE: (%.2f, %.2f) percent (X^2(%d) = %.2f, p = %f)',(X_riskPE(1)./N(1))*100,(X_riskPE(2)./N(2))*100,df_risPE,chi2stat_ris,p_risPE));

        text(0,1.2,'tests for REWARD PE model encoding by impulsivity classification:','fontweight','bold')
        text(0,1,sprintf('proportion of contacts significantly encoding positive RPE (more impulsive, less impulsive): (%.2f, %.2f) percent (X^2(%d) = %.2f, p = %f)',(X_pRPE(1)./N(1))*100,(X_pRPE(2)./N(2))*100,df_pRPE,chi2stat_pRPE,p_pRPE));
        text(0,0.8,sprintf('proportion of contacts significantly encoding negative RPE (more impulsive, less impulsive): (%.2f, %.2f) percent (X^2(%d) = %.2f, p = %f)',(X_nRPE(1)./N(1))*100,(X_nRPE(2)./N(2))*100,df_nRPE,chi2stat_nRPE,p_nRPE));
        text(0,0.6,sprintf('proportion of contacts significantly encoding both positive and negative RPE (more impulsive, less impulsive): (%.2f, %.2f) percent (X^2(%d) = %.2f, p = %f)',(X_both(1)./N(1))*100,(X_both(2)./N(2))*100,df_both,chi2stat_both,p_both));

        text(0,0.4,'tests for RISK PE model encoding by impulsivity classification:','fontweight','bold')
        text(0,0.2,sprintf('proportion of contacts significantly encoding positive RISK PE (more impulsive, less impulsive): (%.2f, %.2f) percent (X^2(%d) = %.2f, p = %f)',(X_pRiskPE(1)./N(1))*100,(X_pRiskPE(2)./N(2))*100,df_pRiskPE,chi2stat_pRiskPE,p_pRiskPE));
        text(0,0,sprintf('proportion of contacts significantly encoding negative RISK PE (more impulsive, less impulsive): (%.2f, %.2f) percent (X^2(%d) = %.2f, p = %f)',(X_nRiskPE(1)./N(1))*100,(X_nRiskPE(2)./N(2))*100,df_nRiskPE,chi2stat_nRiskPE,p_nRiskPE));
        text(0,-0.2,sprintf('proportion of contacts significantly encoding both positive and negative RISK PE (more impulsive, less impulsive): (%.2f, %.2f) percent (X^2(%d) = %.2f, p = %f)',(X_riskboth(1)./N(1))*100,(X_riskboth(2)./N(2))*100,df_riskboth,chi2stat_riskboth,p_riskboth));

        text(0,-0.4,sprintf('proportion of contacts significantly encoding both positive RPE and positive RISK PE (more impulsive, less impulsive): (%.2f, %.2f) percent (X^2(%d) = %.2f, p = %f)',(X_pPEs(1)./N(1))*100,(X_pPEs(2)./N(2))*100,df_pPEs,chi2stat_pPEs,p_pPEs));
        text(0,-0.6,sprintf('proportion of contacts significantly encoding both negative RPE and negative RISK PE  (more impulsive, less impulsive): (%.2f, %.2f) percent (X^2(%d) = %.2f, p = %f)',(X_nPEs(1)./N(1))*100,(X_nPEs(2)./N(2))*100,df_nPEs,chi2stat_nPEs,p_nPEs));
        text(0,-0.8,sprintf('proportion of contacts significantly encoding both positive and negative PEs (more impulsive, less impulsive): (%.2f, %.2f) percent (X^2(%d) = %.2f, p = %f)',(X_allPEs(1)./N(1))*100,(X_allPEs(2)./N(2))*100,df_allPEs,chi2stat_allPEs,p_allPEs));

        text(0,-1.0,'tests for hemispheric differences:','fontweight','bold')
        text(0,-1.2,sprintf('proportion of contacts significantly encoding reward PE model variables in the left hemisphere (left hemisphere, right hemisphere): (%.2f, %.2f) percent (X^2(%d) = %.2f, p = %f)',(X_hemi_rewardPEs(1)./N(1))*100,(X_hemi_rewardPEs(2)./N(2))*100,df_hemi_rewardPEs,chi2stat_hemi_rewardPEs,p_hemi_rewardPEs));
        text(0,-1.4,sprintf('proportion of contacts significantly encoding risk PE model variables in the left hemisphere (left hemisphere, right hemisphere): (%.2f, %.2f) percent (X^2(%d) = %.2f, p = %f)',(X_hemi_riskPEs(1)./N(1))*100,(X_hemi_riskPEs(2)./N(2))*100,df_hemi_riskPEs,chi2stat_hemi_riskPEs,p_hemi_riskPEs));

        text(0,-1.6,sprintf('proportion of contacts significantly encoding reward PE model variables in the left hemisphere (left hemisphere, right hemisphere) in more impulsive subjects: (%.2f, %.2f) percent (X^2(%d) = %.2f, p = %f)',(X_hemi_rewardPEs_imp(1)./N(1))*100,(X_hemi_rewardPEs_imp(2)./N(2))*100,df_hemi_rewardPEs_imp,chi2stat_hemi_rewardPEs_imp,p_hemi_rewardPEs_imp));
        text(0,-1.8,sprintf('proportion of contacts significantly encoding risk PE model variables in the left hemisphere (left hemisphere, right hemisphere) in more impulsive subjects: (%.2f, %.2f) percent (X^2(%d) = %.2f, p = %f)',(X_hemi_riskPEs_imp(1)./N(1))*100,(X_hemi_riskPEs_imp(2)./N(2))*100,df_hemi_riskPEs_imp,chi2stat_hemi_riskPEs_imp,p_hemi_riskPEs_imp));
        text(0,-2.0,sprintf('proportion of contacts significantly encoding reward PE model variables in the left hemisphere (left hemisphere, right hemisphere) in less impulsive subjects: (%.2f, %.2f) percent (X^2(%d) = %.2f, p = %f)',(X_hemi_rewardPEs_notImp(1)./N(1))*100,(X_hemi_rewardPEs_notImp(2)./N(2))*100,df_hemi_rewardPEs_notImp,chi2stat_hemi_rewardPEs_notImp,p_hemi_rewardPEs_notImp));
        text(0,-2.2,sprintf('proportion of contacts significantly encoding risk PE model variables in the left hemisphere (left hemisphere, right hemisphere) in less impulsive subjects: (%.2f, %.2f) percent (X^2(%d) = %.2f, p = %f)',(X_hemi_riskPEs_notImp(1)./N(1))*100,(X_hemi_riskPEs_notImp(2)./N(2))*100,df_hemi_riskPEs_notImp,chi2stat_hemi_riskPEs_notImp,p_hemi_riskPEs_notImp));

        ylim([-2 2.5])
        axis off

        saveas(gcf,'\\155.100.91.44\D\Data\Rhiannon\BART_RLDM_outputs\TDlearn\TDlearn_neuralFitProportions_allPtsAsymmetricalPredicitonError_CTRL.pdf')
    end

    %% TODO:: visualization of contact anatomical location. Make sure that's fixed now by checking out the example responses.


    keyboard


end

