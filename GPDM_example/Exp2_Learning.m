%%Experiment2: compare different learning methods: SCG, L-BFGS=MAP,  SCG,L-BFGS=B-GPDM
clc; clear; close all
addpath(genpath('src/')) %include all subfolders
addpath modified/ my_util/ data_sets/
% addpath src/gpdm src/gplvm src/netlab src/util src/MOCAP src/NDLUTIL modified/ src/Kernel_PCA/KernelPCA
format long

global USE_GAMMA_PRIOR  % gamma prior for dynamics, only works with RBF kernel
global GAMMA_ALPHA % defines shape of the gamma prior
global USE_LAWRENCE % fix dynamics HPs, as Lawrence suggested (use with thetad = [0.2 0.01 1e6];) 
global FIX_HP % fix all HPs (Hyperparameters)
global MARGINAL_W % marginalize over W while learning X
global MARGINAL_DW % marginalize over scale in dynamics while learning X
global LEARN_SCALE % use different scales for different output dimensions
global REMOVE_REDUNDANT_SCALE % let W absorb the overall scale of reconstruction
global W_VARIANCE % kappa^2 in the paper, not really the variance though
global M_CONST % M value in Jack's master's thesis, adjust weights between ln(p(theta))(HPs of K_Y) and ln(p(thetad))(HPs of K_X)
global BALANCE % Constant in front of dynamics term, set to D/q for the B-GPDM
global SUBSET_SIZE % Number of data to select for EM, set -1 for all data. 
global USE_OLD_MISSING_DATA

M_CONST = 1;  % M_CONST*ln(p(theta))
REMOVE_REDUNDANT_SCALE = 1;
LEARN_SCALE = 1; 
MARGINAL_W = 0; 
MARGINAL_DW = 0; 
W_VARIANCE = 1e6; 
FIX_HP = 0; 
USE_GAMMA_PRIOR = 0; 
GAMMA_ALPHA = [5 10 2.5]; 
USE_LAWRENCE = 0;
BALANCE = 1;
SUBSET_SIZE = -1; 


opt = foptions;
opt(1) = 0; %SCG： display error values
opt(9) = 0; %SCG：set to 1 to check the user defined gradient function

if MARGINAL_W == 1 % do not optimaize W 
    opt(14) = 100; % number of SCG iterations
    extItr = 1; 
else %optimaize W 
    opt(14) = 10; % update W every 10 iterations of SCG
    extItr = 100; % do extItr*opt(14) iterations in total
end  

% modelType(1) : input of dynamics
%   0 => [x_t, x_{t-1}]
%   1 => [x_t, x_t - x_{t-1}] == [x_t, v_{t-1}]
%   2 => [x_t]
% modelType(2) : output of dynamics 
%   0 => x_{t+1} 
%   1 => x_{t+1} - x_t == (v_t)
% modelType(3) : kernel type
%   0 => RBF kernel with weighted dimensions, use with input 0 or 1
%   1 => RBF kernel 
%   2 => Linear kernel
%   3 => weighted Linear kernel + RBF kernel with weighted dimensions, use with
%   input 0 or 1
%   4 => weighted linear kernel
%   5 => linear + RBF

%%Learn single walker model from lin+rbf kernel.
%% Show original data
% Recording rate: 120 Hz
acclaimPlayFile('07-walk.asf', '07_01.amc', 1/120);

%% Load data sequences
modelType = [2 0 5]; 
[Y ,initY, varY, segments] = loadMocapData({['07_01.amc']}, [1], [2],[260]); %allow multiple sequences(use 'segment' to seperate)
%load test sets
[Y_t , ~, ~, ~] = loadMocapData({['07_01.amc']}, [261], [2],[316]);

missing = [];
N = size(Y, 1); D = size(Y, 2);
q = 3; % dimensionality of latent space

N_t = size(Y_t, 1);
meanY_t = mean(Y_t);
Y_t = Y_t - repmat(meanY_t, N_t, 1); %preprocess test data：substract mean
%% PCA
refY = Y; % raw data Y
meanData = mean(Y);
Y = Y - repmat(meanData, N, 1); %substract mean
[v, u] = pca(Y);
v(find(v<0))=0;
X_pca = Y*u(:, 1:q)*diag(1./sqrt(v(1:q))); %sphereing the data to unit covariance

%% GPDM: MAP Estimation of X, W, HPs -- use SCG
% initialize hyperparameters
theta1 = [1 1 exp(1)]; % 3 hyperparameters of K_Y (rbf): theta1=1/L; theta2=rbf amptitude; theta3=1/noise
thetad1 = [0.9 1 0.1 exp(1)]; % 4 hyperparameters of K_X (linear+rbf): thetad1 = linear amptitude; 
                                                % theta2=rbf 1/L; theta3=rbf amptitude ;theta4=1/noise
w1 = ones(D,1);
% MAP estimation
[X_gpdm1, theta1, thetad1, w1] = gpdmfitFull_m1(X_pca, Y, w1, segments, theta1, thetad1, opt, ... 
     extItr, modelType, missing);

%% GPDM: MAP Estimation of X, W, HPs -- use LBFGS
% initialize hyperparameters
theta2 = [1 1 exp(1)]; % 3 hyperparameters of K_Y (rbf): theta1=1/L; theta2=rbf amptitude; theta3=1/noise
thetad2 = [0.9 1 0.1 exp(1)]; % 4 hyperparameters of K_X (linear+rbf): thetad1 = linear amptitude; 
                                                % theta2=rbf 1/L; theta3=rbf amptitude ;theta4=1/noise
w2 = ones(D,1);
% MAP estimation
[X_gpdm2, theta2, thetad2, w2] = gpdmfitFull_lbfgs(X_pca, Y, w2, segments, theta2, thetad2, opt, ... 
     extItr, modelType, missing);

%% Compare learning rate
load log_lbfgs1.mat
load log_scg1.mat
figure()
scatter(1:size(energyLog_lbfgs,2), energyLog_lbfgs, 'or', 'filled'); hold on
scatter(1:size(energyLog_scg,2), energyLog_scg, 'ob', 'filled');
title("Learning Rate/10 iters");
legend("BFGS","SCG"); grid on

%% B-GPDM
BALANCE = D/q;
% initialize hyperparameters
theta3 = [1 1 exp(1)]; % 3 hyperparameters of K_Y (rbf): theta1=1/L; theta2=rbf amptitude; theta3=1/noise
thetad3 = [0.9 1 0.1 exp(1)]; % 4 hyperparameters of K_X (linear+rbf): thetad1 = linear amptitude; 
                                                % theta2=rbf 1/L; theta3=rbf amptitude ;theta4=1/noise
w3 = ones(D,1);
% MAP estimation
[X_gpdm3, theta3, thetad3, w3] = gpdmfitFull_m1(X_pca, Y, w3, segments, theta3, thetad3, opt, ... 
     extItr, modelType, missing);
 
% initialize hyperparameters
theta4 = [1 1 exp(1)]; % 3 hyperparameters of K_Y (rbf): theta1=1/L; theta2=rbf amptitude; theta3=1/noise
thetad4 = [0.9 1 0.1 exp(1)]; % 4 hyperparameters of K_X (linear+rbf): thetad1 = linear amptitude; 
                                                % theta2=rbf 1/L; theta3=rbf amptitude ;theta4=1/noise
w4 = ones(D,1);
% MAP estimation
[X_gpdm4, theta4, thetad4, w4] = gpdmfitFull_lbfgs(X_pca, Y, w4, segments, theta4, thetad4, opt, ... 
     extItr, modelType, missing);

 %% Compare learning rate(B-GPDM)
load log_lbfgs2.mat
load log_scg2.mat
figure()
scatter(1:size(energyLog_lbfgs,2), energyLog_lbfgs, 'or', 'filled'); hold on
scatter(1:size(energyLog_scg,2), energyLog_scg, 'ob', 'filled');
title("Learning Rate/10 iters(B-GPDM)");
legend("BFGS","SCG"); grid on

%% train GPR model for forward mapping to latent space: Y->X
%%cross validation GPR models
%MAP-SCG
multiGPR1_CV = cell(1,size(X_gpdm1, 2));
for i=1:size(multiGPR1_CV, 2)
    multiGPR1_CV{i} = fitrgp(Y, X_gpdm1(:, i), 'KernelFunction','ardsquaredexponential',...
      'FitMethod','exact','PredictMethod','exact', 'Optimizer', 'lbfgs', 'CrossVal','on'); %turn on cross validation
end
%MAP-BFGS
multiGPR2_CV = cell(1,size(X_gpdm2, 2));
for i=1:size(multiGPR2_CV, 2)
    multiGPR2_CV{i} = fitrgp(Y, X_gpdm2(:, i), 'KernelFunction','ardsquaredexponential',...
      'FitMethod','exact','PredictMethod','exact', 'Optimizer', 'lbfgs', 'CrossVal','on'); %turn on cross validation
end
%BGPDM-BFGS
multiGPR4_CV = cell(1,size(X_gpdm4, 2));
for i=1:size(multiGPR4_CV, 2)
    multiGPR4_CV{i} = fitrgp(Y, X_gpdm4(:, i), 'KernelFunction','ardsquaredexponential',...
      'FitMethod','exact','PredictMethod','exact', 'Optimizer', 'lbfgs', 'CrossVal','on'); %turn on cross validation
end

%%GPR prediction models
%MAP-SCG
multiGPR1 = cell(1,size(X_gpdm1, 2));
for i=1:size(multiGPR1, 2)
    multiGPR1{i} = fitrgp(Y, X_gpdm1(:, i), 'KernelFunction','ardsquaredexponential',...
      'FitMethod','exact','PredictMethod','exact', 'Optimizer', 'lbfgs'); %no cross validation
end
%MAP-BFGS
multiGPR2 = cell(1,size(X_gpdm2, 2));
for i=1:size(multiGPR2, 2)
    multiGPR2{i} = fitrgp(Y, X_gpdm2(:, i), 'KernelFunction','ardsquaredexponential',...
      'FitMethod','exact','PredictMethod','exact', 'Optimizer', 'lbfgs'); %no cross validation
end
%BGPDM-BFGS
multiGPR4 = cell(1,size(X_gpdm4, 2));
for i=1:size(multiGPR4, 2)
    multiGPR4{i} = fitrgp(Y, X_gpdm4(:, i), 'KernelFunction','ardsquaredexponential',...
      'FitMethod','exact','PredictMethod','exact', 'Optimizer', 'lbfgs'); %no cross validation
end

%% test k-fold loss(10 folds)--MSE
%MAP-SCG
kfoldloss_1 = 0;
for i=1:size(X_gpdm1, 2)
    kfoldloss_1 =kfoldloss_1 + kfoldLoss(multiGPR1_CV{i});
end
disp("GPR 10-fold loss MSE(MAP-SCG):"); disp(kfoldloss_1);
%MAP-BFGS
kfoldloss_2 = 0;
for i=1:size(X_gpdm2, 2)
    kfoldloss_2 =kfoldloss_2 + kfoldLoss(multiGPR2_CV{i});
end
disp("GPR 10-fold loss MSE(MAP-BFGS):"); disp(kfoldloss_2);

%BGPDM-BFGS
kfoldloss_4 = 0;
for i=1:size(X_gpdm4, 2)
    kfoldloss_4 =kfoldloss_4 + kfoldLoss(multiGPR4_CV{i});
end
disp("GPR 10-fold loss MSE(BGPDM-BFGS):"); disp(kfoldloss_4);

%% Save B-GPDM results
save Exp2_Learning X_gpdm1 X_gpdm2 X_gpdm3 X_gpdm4 Y w1 w2 w3 w4 ...
    theta1 theta2 theta3 theta4  thetad1 thetad2 thetad3 thetad4 modelType N D q meanData ...
    segments initY varY missing refY Y_t meanY_t N_t multiGPR1 multiGPR2 multiGPR4 multiGPR1_CV multiGPR2_CV multiGPR4_CV

%% GPDM: make predictions from learned B-GPDM model
% load Exp2_Learning
%By SCG
% [K3, invK3] = computeKernel(X_gpdm3, theta3); % K_Y
% [Xin, ~] = priorIO(X_gpdm3, segments, modelType); % construct input&output X matrix wrt. the specified type
% [Kd3, invKd3] = computePriorKernel(Xin, thetad3, modelType(3)); % K_X
% simSteps = 256;
% % starts at end of training sequence;
% simStart = [X_gpdm3(segments(1)+1,:), X_gpdm3(end,:)]; %  inputs 2 points in case using 2nd order model
% [X_pred3, ~] = simulatedynamics(X_gpdm3, segments, thetad3, invKd3, simSteps, simStart, modelType);

%By BFGS
[K4, invK4] = computeKernel(X_gpdm4, theta4); % K_Y
[Xin, ~] = priorIO(X_gpdm4, segments, modelType); % construct input&output X matrix wrt. the specified type
[Kd4, invKd4] = computePriorKernel(Xin, thetad4, modelType(3)); % K_X
simSteps = 256;
% starts at end of training sequence;
simStart = [X_gpdm4(segments(1)+1,:), X_gpdm4(end,:)]; %  inputs 2 points in case using 2nd order model
[X_pred4, ~] = simulatedynamics(X_gpdm4, segments, thetad4, invKd4, simSteps, simStart, modelType);

%% GPR model prediction error and GPDM prediction error(RMSE) (On test set)
%dynamic LV prediction of test set sequence
 [Xin1, ~] = priorIO(X_gpdm1, segments, modelType);
[~, invKd1] = computePriorKernel(Xin1, thetad1, modelType(3)); % K_X MAP-SCG
 [Xin2, ~] = priorIO(X_gpdm2, segments, modelType);
[~, invKd2] = computePriorKernel(Xin2, thetad2, modelType(3)); % K_X MAP-BFGS
 [Xin4, ~] = priorIO(X_gpdm4, segments, modelType);
[~, invKd4] = computePriorKernel(Xin4, thetad4, modelType(3)); % K_X BGPDM-BFGS

simSteps = N_t+1;

simStart1 = [X_gpdm1(end-1,:), X_gpdm1(end,:)]; %  inputs 2 points in case using 2nd order model
simStart2 = [X_gpdm2(end-1,:), X_gpdm2(end,:)]; %  inputs 2 points in case using 2nd order model
simStart4 = [X_gpdm4(end-1,:), X_gpdm4(end,:)]; %  inputs 2 points in case using 2nd order model

[X_pred1_t, ~] = simulatedynamics(X_gpdm1, segments, thetad1, invKd1, simSteps, simStart, modelType);%MAP-SCG
X_pred1_t = X_pred1_t(2:end, :);
[X_pred2_t, ~] = simulatedynamics(X_gpdm2, segments, thetad2, invKd2, simSteps, simStart, modelType);%MAP-BFGS
X_pred2_t = X_pred2_t(2:end, :);
[X_pred4_t, ~] = simulatedynamics(X_gpdm4, segments, thetad4, invKd4, simSteps, simStart, modelType);%BGPDM-BFGS
X_pred4_t = X_pred4_t(2:end, :);
% X_pred4_t = X_pred4(2:N_t+1, :);

%%GPR pred RMSE
X_gpPred1 = zeros(size(Y_t,1), q);%MAP-SCG
X_gpPred2 = zeros(size(Y_t,1), q);%MAP-BFGS
X_gpPred4 = zeros(size(Y_t,1), q);%BGPDM-BFGS
for i=1:q 
    X_gpPred1(: ,i) = predict(multiGPR1{i}, Y_t); %MAP-SCG
    X_gpPred2(: ,i) = predict(multiGPR2{i}, Y_t); %MAP-BFGS
    X_gpPred4(: ,i) = predict(multiGPR4{i}, Y_t); %BGPDM-BFGS
end
RMSE_gpPred1 = rmse(X_gpPred1, X_pred1_t);
RMSE_gpPred2 = rmse(X_gpPred2, X_pred2_t);
RMSE_gpPred4 = rmse(X_gpPred4, X_pred4_t);
disp("RMSE(GPR Mapping-MAP-SCG):"); disp(RMSE_gpPred1);
disp("RMSE(GPR Mapping-MAP-BFGS):"); disp(RMSE_gpPred2);
disp("RMSE(GPR Mapping-BGPDM-BFGS):"); disp(RMSE_gpPred4);

%%plot GPR pred results
%MAP-SCG
figure("Position",[200,100,750,600])
plotseries(gca, X_pred1_t, [1], 'b'); %dynamic LV predction
plotseries(gca, X_gpPred1, [1], 'r'); %test set GP mapping
title(gca,"LV-space GP mapping & Dynamic Prediction result (MAP-SCG)")
legend("direction1","GPDM Dynamics","direction2", "GP Mapping");
%MAP-BFGS
figure("Position",[200,100,750,600])
plotseries(gca, X_pred2_t, [1], 'b'); %dynamic LV predction
plotseries(gca, X_gpPred2, [1], 'r'); %test set GP mapping
title(gca,"LV-space GP mapping & Dynamic Prediction result (MAP-BFGS)")
legend("direction1","GPDM Dynamics","direction2", "GP Mapping");
%BGPDM-BFGS
figure("Position",[200,100,750,600])
plotseries(gca, X_pred4_t, [1], 'b'); %dynamic LV predction
plotseries(gca, X_gpPred4, [1], 'r'); %test set GP mapping
title(gca,"LV-space GP mapping & Dynamic Prediction result (BGPDM-BFGS)")
legend("direction1","GPDM Dynamics","direction2", "GP Mapping");

%%GPDM pred RMSE
%MAP-SCG
[Y_pred1_t, ~] = gplvmRecon(X_pred1_t, X_gpdm1, Y, theta1);
RMSE_Pred1_t = rmse(Y_pred1_t, Y_t);
disp("RMSE(GPDM Predict-MAP-SCG):"); disp(RMSE_Pred1_t);
%MAP-BFGS
[Y_pred2_t, ~] = gplvmRecon(X_pred2_t, X_gpdm2, Y, theta2);
RMSE_Pred2_t = rmse(Y_pred2_t, Y_t);
disp("RMSE(GPDM Predict-MAP-BFGS):"); disp(RMSE_Pred2_t);
%BGPDM
[Y_pred4_t, ~] = gplvmRecon(X_pred4_t, X_gpdm4, Y, theta4);
RMSE_Pred4_t = rmse(Y_pred4_t, Y_t);
disp("RMSE(GPDM Predict-BGPDM-BFGS):"); disp(RMSE_Pred4_t);

%%make GPDM pred animation
remakeAmcAnimation(Y_pred1_t, meanY_t, 1/60, '07-walk.asf', '07_01_MAPscg_predtest.amc' , '07_01_MAPscg_predtest.gif',  'yellow', '-.', true);
remakeAmcAnimation(Y_pred2_t, meanY_t, 1/60, '07-walk.asf', '07_01_MAPbfgs_predtest.amc' , '07_01_MAPbfgs_predtest.gif',  'yellow', '-.', true);
remakeAmcAnimation(Y_pred4_t, meanY_t, 1/60, '07-walk.asf', '07_01_BGPDMbfgs_predtest.amc' , '07_01_BGPDMbfgs_predtest.gif',  'yellow', '-.', true);
%% GPDM: generate samples from learned model
% uncomment if want to generate new samples

%  hmcopt = foptions;      % Default options vector.
% hmcopt(1) = 1;			% Switch on diagnostics.
% hmcopt(7) = 100;	    	% Number of steps in leapfrog trajectory.
% hmcopt(9) = 0; 
% hmcopt(14) = 60;		% Number of Monte Carlo samples returned. 
% hmcopt(15) = 20;		% Number of samples omitted at start of chain.
% hmcopt(18) = 0.01;  	% leapfrog step size
% X3_samples = sampledynamics_m1(X_gpdm3, segments, modelType, thetad3, X_pred3, "Exp2_Samples3", hmcopt);
%  X4_samples = sampledynamics_m1(X_gpdm4, segments, modelType, thetad4, X_pred4, "Exp2_Samples4", hmcopt);

%% Plot B-GPDM results in latent space--by SCG
% load Exp2_samples4
%clf;
% hold on;
%sampled X - GPDM
% figure('Name','Samples')
% for n=1:4:size(X3_samples,2)
% plotseries(gca, X3_samples{n}, [1], 'g'); 
% end
% title(gca, 'Sampled Latent Variables (B-GPDM)');

%learned X-GPDM 
figure('Name','Learned Latent Variables')
plotseries(gca, X_gpdm3, segments, 'b');
title(gca, 'Learned Latent Variables (B-GPDM)');

%mean prediction-GPDM
figure('Name','Mean Prediction')
plotseries(gca, X_pred3, [1], 'r'); 
title(gca, 'Mean Prediction (B-GPDM)');
 
 %% Plot B-GPDM-BFGS results in latent space
% load Exp2_samples4
%clf;
% hold on;
%sampled X - GPDM
figure('Name','Samples')
for n=1:4:size(X4_samples,2)
plotseries(gca, X4_samples{n}, [1], 'g'); 
end
title(gca, 'Sampled Latent Variables (B-GPDM)');

%learned X-GPDM 
figure('Name','Learned Latent Variables')
plotseries(gca, X_gpdm4, segments, 'b');
title(gca, 'Learned Latent Variables (B-GPDM)');

%mean prediction-GPDM
figure('Name','Mean Prediction')
plotseries(gca, X_pred4, [1], 'r'); 
title(gca, 'Mean Prediction (B-GPDM)');

%% variance heat maps B-GPDM-BFGS
%Reconstruction variance Map
figure('Position',[200,100,900,650])
gpdmvisualise_m1(gca, X_gpdm4, Y, segments, invK4, theta4, invKd4, thetad4,  modelType, 0)
%%Dynamic Latent Space Prediction Variance Map
figure('Position',[200,100,900,650])
gpdmvisualise_m1(gca, X_gpdm4, Y, segments, invK4, theta4, invKd4, thetad4,  modelType, 1)

%% Reconstruction error-BGPDM-BFGS
%BGPDM
[Y_gpdm4, ~] = gplvmRecon(X_gpdm4, X_gpdm4, Y, theta4);
RMSE_Bgpdm = rmse(Y_gpdm4, Y);
disp("RMSE(B-GPDM):"); disp(RMSE_Bgpdm);

%% Signal-Noise ratio & Characteristic length scale
%MAP-SCG
SNR_gpdm1 = sqrt(theta1(2)*theta1(3));
disp("SNR(beta)(MAP-SCG):"); disp(SNR_gpdm1);
CLS_gpdm1 = sqrt(1/theta1(1)); 
disp("CLS(beta)(MAP-SCG):"); disp(CLS_gpdm1);

SNR_gpdm1d = sqrt((thetad1(1) + thetad1(3)) * thetad1(4));
disp("SNR(alpha)(MAP-SCG):"); disp(SNR_gpdm1d);
CLS_gpdm1d = sqrt(1/thetad1(2)); 
disp("CLS(alpha)(MAP-SCG):"); disp(CLS_gpdm1d);
%MAP-BFGS
SNR_gpdm2 = sqrt(theta2(2)*theta2(3));
disp("SNR(beta)(MAP-BFGS):"); disp(SNR_gpdm2);
CLS_gpdm2 = sqrt(1/theta2(1)); 
disp("CLS(beta)(MAP-BFGS):"); disp(CLS_gpdm2);

SNR_gpdm2d = sqrt((thetad2(1) + thetad2(3)) * thetad2(4));
disp("SNR(alpha)(MAP-BFGS):"); disp(SNR_gpdm2d);
CLS_gpdm2d = sqrt(1/thetad2(2)); 
disp("CLS(alpha)(MAP-BFGS):"); disp(CLS_gpdm2d);

%BGPDM-BFGS
SNR_gpdm4 = sqrt(theta4(2)*theta4(3));
disp("SNR(beta)(B-GPDM):"); disp(SNR_gpdm4);
CLS_gpdm4 = sqrt(1/theta4(1)); 
disp("CLS(beta)(B-GPDM):"); disp(CLS_gpdm4);

SNR_gpdm4d = sqrt((thetad4(1) + thetad4(3)) * thetad4(4));
disp("SNR(alpha)(B-GPDM):"); disp(SNR_gpdm4d);
CLS_gpdm4d = sqrt(1/thetad4(2)); 
disp("CLS(alpha)(B-GPDM):"); disp(CLS_gpdm4d);

%% generate Animation wrt the predicted latent trajectory-BGPDM-BFGS
[Y_pred4, ~] = gplvmRecon(X_pred4, X_gpdm4, Y, theta4);
remakeAmcAnimation(Y_pred4, meanData, 1/60, '07-walk.asf', '07_01_BGPDMreg.amc' , '07_01_BGPDMreg.gif',  'green', '--', true);