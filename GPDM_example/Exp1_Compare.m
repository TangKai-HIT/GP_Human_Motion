%%Experiment1: compare PCA, KPCA, GPLVM, GDPM results
% Note: using MAP + SCG to train GPLVM & GPDM. KPCA is not able yet
% results include training set reconstruction error, some figures related 
% to the LVs trained by GPLVM & GPDM, SNR & CLS 

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
missing = [];
N = size(Y, 1); D = size(Y, 2);
q = 3; % dimensionality of latent space

%% PCA
refY = Y; % raw data Y
meanData = mean(Y);
Y = Y - repmat(meanData, N, 1); %substract mean
[v, u] = pca(Y);
v(find(v<0))=0;
X_pca = Y*u(:, 1:q)*diag(1./sqrt(v(1:q))); %sphereing the data to unit covariance

%% PPCA
[var, U, lambda] = ppca(Y'*Y, q);
lambda(find(lambda<0))=0;
X_ppca = Y*U(:, 1:q)*diag(1./sqrt(lambda(1:q))); %sphereing the data to unit covariance

%% KPCA
% rbf_kernel = Kernel('type', 'gaussian', 'gamma', 0.46);
% parameter = struct('numComponents', 3, ...
%                    'kernelFunc', rbf_kernel);
% % build a KPCA object
% kpca = KernelPCA(parameter);
% % train KPCA model
% kpca.train(Y);
% %　mapping data
% mappingData = kpca.score;
% visulize the mapping data
% kplot = KernelPCAVisualization();
% kplot.score(kpca); 

%% GPLVM
% Initialise theta
theta_gplvm(1) = 1;
theta_gplvm(2) = 1;
theta_gplvm(3) = 1;
% options for kernel optimisation
opt_Kernel = foptions;
opt_Kernel(1) = 0;
opt_Kernel(9) = 0;
opt_Kernel(14) = 20;
% Fit the GP latent variable model
[X_gplvm, theta_gplvm, ~, scale] = gplvmfitFull_m1(X_pca, Y, theta_gplvm, size(Y,1), opt_Kernel, 100);

%% GPDM: MAP Estimation of X, W, HPs
% initialize hyperparameters
theta = [1 1 exp(1)]; % 3 hyperparameters of K_Y (rbf): theta1=1/L; theta2=rbf amptitude; theta3=1/noise
thetad = [0.9 1 0.1 exp(1)]; % 4 hyperparameters of K_X (linear+rbf): thetad1 = linear amptitude; 
                                                % theta2=rbf 1/L; theta3=rbf amptitude ;theta4=1/noise
w = ones(D,1);
% MAP estimation
[X_gpdm, theta, thetad, w] = gpdmfitFull_m1(X_pca, Y, w, segments, theta, thetad, opt, ... 
     extItr, modelType, missing);

 %% Save Results 
save Exp1_Compare X_pca X_gplvm theta_gplvm X_gpdm Y w theta thetad modelType N D q meanData  ...
segments initY varY missing refY;

%% GPDM: make predictions from learned model
load Exp1_Compare
[K, invK] = computeKernel(X_gpdm, theta); % K_Y
[Xin, Xout] = priorIO(X_gpdm, segments, modelType); % construct input&output X matrix wrt. the specified type
[Kd, invKd] = computePriorKernel(Xin, thetad, modelType(3)); % K_X
simSteps = 256;
% starts at end of training sequence;
simStart = [X_gpdm(segments(1)+1,:), X_gpdm(end,:)]; %  inputs 2 points in case using 2nd order model
[X_pred, XRand_pred] = simulatedynamics(X_gpdm, segments, thetad, invKd, simSteps, simStart, modelType);

%% GPDM: generate samples from learned model
% uncomment if want to generate new samples

%  hmcopt = foptions;      % Default options vector.
% hmcopt(1) = 1;			% Switch on diagnostics.
% hmcopt(7) = 100;	    	% Number of steps in leapfrog trajectory.
% hmcopt(9) = 0; 
% hmcopt(14) = 60;		% Number of Monte Carlo samples returned. 
% hmcopt(15) = 20;		% Number of samples omitted at start of chain.
% hmcopt(18) = 0.01;  	% leapfrog step size
%  X_samples = sampledynamics('Exp1_Compare', X_pred, 'Exp1_samples', hmcopt);

%% Plot PCA, PPCA, KPCA, GPLVM, GPDM results in latent space
% load Exp1_samples
load samples
%clf;
% hold on;

%learned X-PCA
figure()
plotseries(gca, X_pca, segments, 'b');
title(gca, 'Learned Latent Variables (PCA)');

%learned X-PPCA
figure()
plotseries(gca, X_ppca, segments, 'b');
title(gca, 'Learned Latent Variables (PPCA)');

%learned X-KPCA
% figure()
% plotseries(gca, mappingData, [1], 'b');
% title(gca, 'Learned Latent Variables (KPCA)');

%learned X-GPLVM 
figure()
plotseries(gca, X_gplvm, segments, 'b');
title(gca, 'Learned Latent Variables (GPLVM)');

%sampled X - GPDM
figure('Name','Samples')
for n=1:4:size(X_samples,2)
plotseries(gca, X_samples{n}, [1], 'g'); 
end
title(gca, 'Sampled Latent Variables (GPDM)');

%learned X-GPDM 
figure('Name','Learned Latent Variables')
plotseries(gca, X_gpdm, segments, 'b');
title(gca, 'Learned Latent Variables (GPDM)');

%mean prediction-GPDM
figure('Name','Mean Prediction')
plotseries(gca, X_pred, [1], 'r'); 
title(gca, 'Mean Prediction (GPDM)');

%% variance heat maps-GPDM
%Reconstruction variance Map
figure('Position',[200,100,900,650])
gpdmvisualise_m1(gca, X_gpdm, Y, segments, invK, theta, invKd, thetad,  modelType, 0)
%%Dynamic Latent Space Prediction Variance Map
figure('Position',[200,100,900,650])
gpdmvisualise_m1(gca, X_gpdm, Y, segments, invK, theta, invKd, thetad,  modelType, 1)

%% Reconstruction error (RMS)
%PCA
Y_pca = X_pca*diag(sqrt(v(1:q)))*u(:, 1:q)';
RMSE_pca = rmse(Y_pca, Y);
disp("RMSE(PCA):"); disp(RMSE_pca);
%GPLVM
[Y_gplvm, ~] = gplvmRecon(X_gplvm, X_gplvm, Y, theta_gplvm);
RMSE_gplvm = rmse(Y_gplvm, Y);
disp("RMSE(GPLVM):"); disp(RMSE_gplvm);
%GPDM
[Y_gpdm, ~] = gplvmRecon(X_gpdm, X_gpdm, Y, theta);
RMSE_gpdm = rmse(Y_gpdm, Y);
disp("RMSE(GPDM):"); disp(RMSE_gpdm);

%% Signal-Noise ratio & Characteristic length scale
SNR_gplvm = sqrt(theta_gplvm(2)*theta_gplvm(3)); 
disp("SNR(GPLVM):"); disp(SNR_gplvm);
CLS_gplvm = sqrt(1/theta_gplvm(1)); 
disp("CLS(GPLVM):"); disp(CLS_gplvm);

SNR_gpdm = sqrt(theta(2)*theta(3)); % GPDM K_Y
disp("SNR(beta)(GPDM):"); disp(SNR_gpdm);
CLS_gpdm = sqrt(1/theta(1)); 
disp("CLS(beta)(GPDM):"); disp(CLS_gpdm);

SNR_gpdmd = sqrt((thetad(1) + thetad(3)) * thetad(4)); % GPDM K_X
disp("SNR(alpha)(GPDM):"); disp(SNR_gpdmd);
CLS_gpdmd = sqrt(1/thetad(2)); 
disp("CLS(alpha)(GPDM):"); disp(CLS_gpdmd);