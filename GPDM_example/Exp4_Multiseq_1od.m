%%Experiment4: hold out test of 1-order GPDM for different learning methods: MAP, B-GPDM
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
% acclaimPlayFile('07-walk.asf', '07_01.amc', 1/120);

%% Load data sequences
modelType = [2 0 5]; % 1-order GPDM
%load training sets
[Y ,initY, varY, segments] = loadMocapData({'07_01.amc', '07_02.amc', '07_09.amc', '07_10.amc'}, ...
                                    repmat(1,1,4), repmat(2,1,4), repmat(260,1,4)); %allow multiple sequences(use 'segment' to seperate)
%load test sets
[Y_t , ~, ~, segments_t] = loadMocapData({['07_11.amc']}, [1], [2],[260]);

missing = [];
N = size(Y, 1); D = size(Y, 2);
N_t = size(Y_t, 1);
q = 3; % dimensionality of latent space

meanY_t = mean(Y_t);
Y_t = Y_t - repmat(meanY_t, N_t, 1); %preprocess test data：substract mean

%% PCA
refY = Y; % raw data Y
meanData = mean(Y);
Y = Y - repmat(meanData, N, 1); %preprocess training data：substract mean
[v, u] = pca(Y);
v(find(v<0))=0;
X_pca = Y*u(:, 1:q)*diag(1./sqrt(v(1:q))); %sphereing the data to unit covariance

%% MAP Estimation of X, W, HPs -- use BFGS
% initialize hyperparameters
theta2 = [1 1 exp(1)]; % 3 hyperparameters of K_Y (rbf): theta1=1/L; theta2=rbf amptitude; theta3=1/noise
thetad2 = [0.9 1 0.1 exp(1)]; % 4 hyperparameters of K_X (linear+rbf): thetad1 = linear amptitude; 
                                                % theta2=rbf 1/L; theta3=rbf amptitude ;theta4=1/noise
w2 = ones(D,1);
% MAP estimation
[X_gpdm2, theta2, thetad2, w2] = gpdmfitFull_lbfgs(X_pca, Y, w2, segments, theta2, thetad2, opt, ... 
     extItr, modelType, missing);

%% (MAP) train corresbonding GPR model of Y->X 
multiGPR1 = cell(1,size(X_gpdm2, 2));
for i=1:size(multiGPR1, 2)
    multiGPR1{i} = fitrgp(Y, X_gpdm2(:, i), 'KernelFunction','ardsquaredexponential',...
      'FitMethod','exact','PredictMethod','exact', 'Optimizer', 'lbfgs');
end

%% B-GPDM Estimation of X, W, HPs -- use BFGS
BALANCE = D/q;
 
% initialize hyperparameters
theta4 = [1 1 exp(1)]; % 3 hyperparameters of K_Y (rbf): theta1=1/L; theta2=rbf amptitude; theta3=1/noise
thetad4 = [0.9 1 0.1 exp(1)]; % 4 hyperparameters of K_X (linear+rbf): thetad1 = linear amptitude; 
                                                % theta2=rbf 1/L; theta3=rbf amptitude ;theta4=1/noise
w4 = ones(D,1);
% MAP estimation
[X_gpdm4, theta4, thetad4, w4] = gpdmfitFull_lbfgs(X_pca, Y, w4, segments, theta4, thetad4, opt, ... 
     extItr, modelType, missing);

%% (B-GPDM) train corresbonding GPR model of Y->X 
multiGPR2 = cell(1,size(X_gpdm4, 2));
for i=1:size(multiGPR2, 2)
    multiGPR2{i} = fitrgp(Y, X_gpdm4(:, i), 'KernelFunction','ardsquaredexponential',...
      'FitMethod','exact','PredictMethod','exact', 'Optimizer', 'lbfgs');
end

%% Save results
save Exp4_Multiseq_1od X_gpdm2 X_gpdm4 Y Y_t w2 w4 theta2 theta4 thetad2 thetad4 modelType N D q meanData meanY_t  ...
segments initY varY missing refY multiGPR1 multiGPR2;

%% Load Results
% load Exp4_Multiseq_1od

%% MAP: make predictions from learned model
[Xin, ~] = priorIO(X_gpdm2, segments, modelType); % construct input&output X matrix wrt. the specified type
[Kd2, invKd2] = computePriorKernel(Xin, thetad2, modelType(3)); % K_X
simSteps = 256;
% starts at end of training sequence;
simStart = [X_gpdm2(segments(1)+1,:), X_gpdm2(end,:)]; %  inputs 2 points in case using 2nd order model
[X_pred2, ~] = simulatedynamics(X_gpdm2, segments, thetad2, invKd2, simSteps, simStart, modelType);

%% B-GPDM: make predictions from learned model
[Xin, ~] = priorIO(X_gpdm4, segments, modelType); % construct input&output X matrix wrt. the specified type
[Kd4, invKd4] = computePriorKernel(Xin, thetad4, modelType(3)); % K_X
simSteps = 256;
% starts at end of training sequence;
simStart = [X_gpdm4(segments(1)+1, :), X_gpdm4(1,:)]; %  inputs 2 points in case using 2nd order model
[X_pred4, ~] = simulatedynamics(X_gpdm4, segments, thetad4, invKd4, simSteps, simStart, modelType);

%% Plot B-GPDM results in latent space--by SCG
%clf;
% hold on;

%learned MAP
figure('Name','Learned Latent Variables')
plotseries(gca, X_gpdm2, segments, 'b');
title(gca, 'Learned Latent Variables (GPDM-MAP)');

%mean prediction-MAP
figure('Name','Mean Prediction')
plotseries(gca, X_pred2, [1], 'r'); 
title(gca, 'Mean Prediction (GPDM-MAP)');

%learned B-GPDM 
figure('Name','Learned Latent Variables')
plotseries(gca, X_gpdm4, segments, 'b');
title(gca, 'Learned Latent Variables (B-GPDM)');

%mean prediction-BGPDM
figure('Name','Mean Prediction')
plotseries(gca, X_pred4, [1], 'r'); 
title(gca, 'Mean Prediction (B-GPDM)');

%% variance heat maps MAP
[~, invK2] = computeKernel(X_gpdm2, theta2); % K_Y
%Reconstruction variance Map
figure('Position',[200,100,900,650])
gpdmvisualise_m1(gca, X_gpdm2, Y, segments, invK2, theta2, invKd2, thetad2,  modelType, 0)
%%Dynamic Latent Space Prediction Variance Map
figure('Position',[200,100,900,650])
gpdmvisualise_m1(gca, X_gpdm2, Y, segments, invK2, theta2, invKd2, thetad2,  modelType, 1)

%% variance heat maps B-GPDM 
[~, invK4] = computeKernel(X_gpdm4, theta4); % K_Y
%Reconstruction variance Map
figure('Position',[200,100,900,650])
gpdmvisualise_m1(gca, X_gpdm4, Y, segments, invK4, theta4, invKd4, thetad4,  modelType, 0)
%%Dynamic Latent Space Prediction Variance Map
figure('Position',[200,100,900,650])
gpdmvisualise_m1(gca, X_gpdm4, Y, segments, invK4, theta4, invKd4, thetad4,  modelType, 1)

%% Signal-Noise ratio & Characteristic length scale
%MAP
SNR_gpdm2 = sqrt(theta2(2)*theta2(3));
disp("SNR(beta)(MAP):"); disp(SNR_gpdm2);
CLS_gpdm2 = sqrt(1/theta2(1)); 
disp("CLS(beta)(MAP):"); disp(CLS_gpdm2);

SNR_gpdm2d = sqrt((thetad2(1) + thetad2(3)) * thetad2(4));
disp("SNR(alpha)(MAP):"); disp(SNR_gpdm2d);
CLS_gpdm2d = sqrt(1/thetad2(2)); 
disp("CLS(alpha)(MAP):"); disp(CLS_gpdm2d);

%BGPDM
SNR_gpdm4 = sqrt(theta4(2)*theta4(3));
disp("SNR(beta)(B-GPDM):"); disp(SNR_gpdm4);
CLS_gpdm4 = sqrt(1/theta4(1)); 
disp("CLS(beta)(B-GPDM):"); disp(CLS_gpdm4);

SNR_gpdm4d = sqrt((thetad4(1) + thetad4(3)) * thetad4(4));
disp("SNR(alpha)(B-GPDM):"); disp(SNR_gpdm4d);
CLS_gpdm4d = sqrt(1/thetad4(2)); 
disp("CLS(alpha)(B-GPDM):"); disp(CLS_gpdm4d);

%% predict motion of test set
simSteps = size(Y_t, 1);
%MAP
x_t = [predict(multiGPR1{1}, Y_t(1, :)), predict(multiGPR1{2}, Y_t(1, :)), predict(multiGPR1{3}, Y_t(1, :))];
simStart = [x_t, x_t]; %  inputs 2 points in case using 2nd order model
[X_pred2_t, ~] = simulatedynamics(X_gpdm2, segments, thetad2, invKd2, simSteps, simStart, modelType);
[Y_gpdm2, ~] = gplvmRecon(X_pred2_t, X_gpdm2, Y, theta2);
RMSE_MAP = rmse(Y_gpdm2, Y_t);
disp("RMSE(MAP):"); disp(RMSE_MAP);

%BGPDM
simSteps = size(Y_t, 1);
% starts at end of training sequence;
x_t = [predict(multiGPR2{1}, Y_t(1, :)), predict(multiGPR2{2}, Y_t(1, :)), predict(multiGPR2{3}, Y_t(1, :))];
simStart = [x_t, x_t]; %  inputs 2 points in case using 2nd order model
[X_pred4_t, ~] = simulatedynamics(X_gpdm4, segments, thetad4, invKd4, simSteps, simStart, modelType);
[Y_gpdm4, ~] = gplvmRecon(X_pred4_t, X_gpdm4, Y, theta4);
RMSE_Bgpdm = rmse(Y_gpdm4, Y_t);
disp("RMSE(B-GPDM):"); disp(RMSE_Bgpdm);

%% generate Animation and draw the predicted latent trajectory
%%MAP
%LV-Trajectory
figure()
plotseries(gca, X_pred2_t, [1], 'r'); 
title(gca, 'Predicted LV Trajectory (GPDM-MAP)');
%animation
remakeAmcAnimation(Y_gpdm2, meanY_t, 1/60, '07-walk.asf', 'Ex4_pred_07_11_MAP.amc' , 'Ex4_pred_07_11_MAP.gif',  'green', '--', true);

%%BGPDM
figure()
plotseries(gca, X_pred4_t, [1], 'r'); 
title(gca, 'Predicted LV Trajectory (B-GPDM)');
%animation
% remakeAmcAnimation(Y_gpdm4, meanY_t, 1/60, '07-walk.asf', 'Ex4_pred_07_11_BGPDM.amc' , 'Ex4_pred_07_11_BGPDM.gif',  'orange', '--', true);