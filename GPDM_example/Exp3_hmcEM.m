%%Experiment3: try 2-stage-MAP(hmc-EM) method to train GPDM model
% Notes: results include training set reconstruction error, some figures related 
% to the LVs trained by 2-stage-MAP(hmc-EM), SNR & CLS, test set prediction error, 
% GPR models(Y->X) for each condition, GPR model cross validation, GPR models test set predction error

clc; clear; close all
addpath(genpath('src/')) %include all subfolders
addpath modified/ my_util/ data_sets/
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
opt(1) = 1;
opt(9) = 0;
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
X = zeros(N, q);
refY = Y; % raw data Y
meanData = mean(Y);
Y = Y - repmat(meanData, N, 1); %substract mean
[v, u] = pca(Y);
v(find(v<0))=0;
X_pca = Y*u(:, 1:q)*diag(1./sqrt(v(1:q))); %sphereing the data to unit covariance

%% MAP Estimation of X, W, HPs
% initialize hyperparameters
% theta = [1 1 exp(1)]; % 3 hyperparameters of K_Y (rbf): theta1=1/L; theta2=rbf amptitude; theta3=1/noise
% thetad = [0.9 1 0.1 exp(1)]; % 4 hyperparameters of K_X (linear+rbf): thetad1 = linear amptitude; 
%                                                 % theta2=rbf 1/L; theta3=rbf amptitude ;theta4=1/noise
% w = ones(D,1);
% % MAP estimation
% [X, theta, thetad, w] = gpdmfitFull(X_pca, Y, w, segments, theta, thetad, opt, ... 
%      extItr, modelType, missing);
% save example_model X Y w theta thetad modelType N D q meanData  ...
% segments initY varY missing refY;

%% 2-stage-MAP(hmc-EM) Estimation of X, W, HPs
% initialize hyperparameters
theta_EM = [1 1 exp(1)]; % 3 hyperparameters of K_Y (rbf): theta1=1/L; theta2=rbf amptitude; theta3=1/noise
thetad_EM = [0.9 1 0.1 exp(1)]; % 4 hyperparameters of K_X (linear+rbf): thetad1 = linear amptitude; 
                                                % theta2=rbf 1/L; theta3=rbf amptitude ;theta4=1/noise
w_EM = ones(D,1);

% hmc-EM estimation
hmcopt = foptions;      % Default options vector.
hmcopt(1) = 0;			% Switch off diagnostics.
hmcopt(7) = 100;	    	% Number of steps in leapfrog trajectory.
hmcopt(9) = 0; 
hmcopt(14) = 60;		% Number of Monte Carlo samples returned. 
hmcopt(15) = 20;		% Number of samples omitted at start of chain.
hmcopt(18) = 0.01;  	% leapfrog step size
[X_EM, theta_EM, thetad_EM, w_EM, Xsamples_EM, ~, Isegments] = gpdmfitEM_m1(X_pca, Y, w_EM,...
    segments, theta_EM, thetad_EM, opt, hmcopt, extItr, modelType, 'hmcEM');

%% (2-stage-MAP(hmc-EM)) train corresbonding GPR model of Y->X 
multiGPR_EM = cell(1,size(X_EM, 2));
multiGPR_EM_CV = cell(1,size(X_EM, 2));
for i=1:size(multiGPR_EM, 2)
    multiGPR_EM{i} = fitrgp(Y, X_EM(:, i), 'KernelFunction','ardsquaredexponential',...
      'FitMethod','exact','PredictMethod','exact', 'Optimizer', 'lbfgs');
  multiGPR_EM_CV{i} = fitrgp(Y, X_EM(:, i), 'KernelFunction','ardsquaredexponential',...
      'FitMethod','exact','PredictMethod','exact', 'Optimizer', 'lbfgs', 'CrossVal','on'); %turn on cross validation
end

%% test k-fold loss(10 folds)--MSE
%MAP-SCG
kfoldloss_1 = 0;
for i=1:size(X_EM, 2)
    kfoldloss_1 =kfoldloss_1 + kfoldLoss(multiGPR_EM_CV{i});
end
disp("GPR 10-fold loss MSE(hmc-EM):"); disp(kfoldloss_1);

%% Save results
save Exp3_hmcEM X_EM Y w_EM theta_EM thetad_EM modelType N D q meanData  ...
segments initY varY missing refY Y_t N_t meanY_t multiGPR_EM multiGPR_EM_CV;

%% (2-stage-MAP(hmc-EM)) make predictions from learned model 
load Exp3_hmcEM
[K, invK] = computeKernel(X_EM, theta_EM); % K_Y
[Xin, Xout] = priorIO(X_EM, segments, modelType); % construct input&output X matrix wrt. the specified type
[Kd, invKd] = computePriorKernel(Xin, thetad_EM, modelType(3)); % K_X
simSteps = 256;
% starts at end of training sequence;
simStart = [X_EM(segments(1)+1,:), X_EM(end,:)]; %  inputs 2 points in case using 2nd order model
[X_pred, XRand_pred] = simulatedynamics(X_EM, segments, thetad_EM, invKd, simSteps, simStart, modelType);

%% generate samples from learned model
% uncomment if want to generate new samples

%  hmcopt = foptions;      % Default options vector.
% hmcopt(1) = 1;			% Switch on diagnostics.
% hmcopt(7) = 100;	    	% Number of steps in leapfrog trajectory.
% hmcopt(9) = 0; 
% hmcopt(14) = 60;		% Number of Monte Carlo samples returned. 
% hmcopt(15) = 20;		% Number of samples omitted at start of chain.
% hmcopt(18) = 0.01;  	% leapfrog step size
% X_samples = sampledynamics_m1(X_EM, segments, modelType, thetad_EM, X_pred, "Exp3_samples", hmcopt);
 %% Plot results
load Exp3_samples
%clf;
% hold on;

figure('Name','Samples')
for n=1:4:size(X_samples,2)
plotseries(gca, X_samples{n}, [1], 'g'); 
end
title(gca, 'Sampled Latent Variables (hmc-EM)');

figure('Name','Learned Latent Variables')
plotseries(gca, X_EM, segments, 'b');
title(gca, 'Learned Latent Variables (hmc-EM)');

figure('Name','Mean Prediction')
plotseries(gca, X_pred, [1], 'r'); 
title(gca, 'Mean Prediction (hmc-EM)'); 

%% variance heat maps
%Reconstruction variance Map
figure('Position',[200,100,900,650])
gpdmvisualise_m1(gca, X_EM, Y, segments, invK, theta_EM, invKd, thetad_EM,  modelType, 0)
%%Dynamic Latent Space Prediction Variance Map
figure('Position',[200,100,900,650])
gpdmvisualise_m1(gca, X_EM, Y, segments, invK, theta_EM, invKd, thetad_EM,  modelType, 1)

%% GPR model prediction error and GPDM prediction error(RMSE) (On test set)
%dynamic LV prediction of test set sequence
 [Xin, ~] = priorIO(X_EM, segments, modelType);
[~, invKd] = computePriorKernel(Xin, thetad_EM, modelType(3)); % K_X EM

simSteps = N_t+1;

simStart = [X_EM(end-1,:), X_EM(end,:)]; %  inputs 2 points in case using 2nd order model
[X_pred_t, ~] = simulatedynamics(X_EM, segments, thetad_EM, invKd, simSteps, simStart, modelType);%EM
X_pred_t = X_pred_t(2:end, :); %start from frame No.261

%%GPR pred RMSE
X_gpPred = zeros(size(Y_t,1), q);%EM
for i=1:q 
    X_gpPred(: ,i) = predict(multiGPR_EM{i}, Y_t); %EM
end
RMSE_gpPred = rmse(X_gpPred, X_pred_t);
disp("RMSE(GPR Mapping-hmcEM):"); disp(RMSE_gpPred);

%%plot GPR pred results
%hmc-EM
figure("Position",[200,100,750,600])
plotseries(gca, X_pred_t, [1], 'b'); %dynamic LV predction
plotseries(gca, X_gpPred, [1], 'r'); %test set GP mapping
title(gca,"LV-space GP mapping & Dynamic Prediction result (hmcEM)")
legend("direction1","GPDM Dynamics","direction2", "GP Mapping");

%%GPDM pred RMSE
%MAP-SCG
[Y_pred_t, ~] = gplvmRecon(X_pred_t, X_EM, Y, theta_EM);
RMSE_Pred_t = rmse(Y_pred_t, Y_t);
disp("RMSE(GPDM Predict-hmcEM):"); disp(RMSE_Pred_t);

%%make GPDM pred animation
remakeAmcAnimation(Y_pred_t, meanY_t, 1/60, '07-walk.asf', '07_01_hmcEM_predtest.amc' , '07_01_hmcEM_predtest.gif',  'yellow', '-.', true);
%% Reconstruction error(training set)
%EM
[Y_EM, ~] = gplvmRecon(X_EM, X_EM, Y, theta_EM);
RMSE_EM = rmse(Y_EM, Y);
disp("RMSE(hmcEM):"); disp(RMSE_EM);

%% Signal-Noise ratio & Characteristic length scale
SNR_EM = sqrt(theta_EM(2)*theta_EM(3));
disp("SNR(beta)(hmc-EM):"); disp(SNR_EM);
CLS_EM = sqrt(1/theta_EM(1)); 
disp("CLS(beta)(hmc-EM):"); disp(CLS_EM);

SNR_dEM = sqrt((thetad_EM(1) + thetad_EM(3)) * thetad_EM(4));
disp("SNR(alpha)(hmc-EM):"); disp(SNR_dEM);
CLS_dEM = sqrt(1/thetad_EM(2)); 
disp("CLS(alpha)(hmc-EM):"); disp(CLS_dEM);

%% generate Animation wrt the predicted latent trajectory
remakeAmcAnimation(Y_gpdm, meanData, 1/60, '07-walk.asf', '07_01_hmcEMreg.amc' , '07_01_hmcEMreg.gif',  'cyan', '--', true);
