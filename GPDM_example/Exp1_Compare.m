%%Experiment1: compare PCA, KPCA, GPLVM, GDPM results
clc; clear; close all

addpath src/gpdm src/gplvm src/netlab src/util src/MOCAP src/NDLUTIL
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
missing = [];
N = size(Y, 1); D = size(Y, 2);
q = 3; % dimensionality of latent space

%% PCA
X = zeros(N, q);
refY = Y; % raw data Y
meanData = mean(Y);
Y = Y - repmat(meanData, N, 1); %substract mean
[v, u] = pca(Y);
v(find(v<0))=0;
X = Y*u(:, 1:q)*diag(1./sqrt(v(1:q))); %sphereing the data to unit covariance

%% MAP Estimation of X, W, HPs
% initialize hyperparameters
theta = [1 1 exp(1)]; % 3 hyperparameters of K_Y (rbf): theta1=1/L; theta2=rbf amptitude; theta3=1/noise
thetad = [0.9 1 0.1 exp(1)]; % 4 hyperparameters of K_X (linear+rbf): thetad1 = linear amptitude; 
                                                % theta2=rbf 1/L; theta3=rbf amptitude ;theta4=1/noise
w = ones(D,1);
% MAP estimation
[X, theta, thetad, w] = gpdmfitFull(X, Y, w, segments, theta, thetad, opt, ... 
     extItr, modelType, missing);
save example_model X Y w theta thetad modelType N D q meanData  ...
segments initY varY missing refY;

%% make predictions from learned model
load example_model
[K, invK] = computeKernel(X, theta); % K_Y
[Xin, Xout] = priorIO(X, segments, modelType); % construct input&output X matrix wrt. the specified type
[Kd, invKd] = computePriorKernel(Xin, thetad, modelType(3)); % K_X
simSteps = 256;
% starts at end of training sequence;
simStart = [X(segments(1)+1,:), X(end,:)]; %  inputs 2 points in case using 2nd order model
[X_pred, XRand_pred] = simulatedynamics(X, segments, thetad, invKd, simSteps, simStart, modelType);

%% generate samples from learned model
% uncomment if want to generate new samples

%  hmcopt = foptions;      % Default options vector.
% hmcopt(1) = 1;			% Switch on diagnostics.
% hmcopt(7) = 100;	    	% Number of steps in leapfrog trajectory.
% hmcopt(9) = 0; 
% hmcopt(14) = 60;		% Number of Monte Carlo samples returned. 
% hmcopt(15) = 20;		% Number of samples omitted at start of chain.
% hmcopt(18) = 0.01;  	% leapfrog step size
%  X_samples = sampledynamics('example_model', X_pred, 'samples', hmcopt);

%% Plot results
load samples
%clf;
% hold on;

figure('Name','Samples')
for n=1:4:size(X_samples,2)
plotseries(gca, X_samples{n}, [1], 'g'); 
end

figure('Name','Learned Latent Variables')
plotseries(gca, X, segments, 'b');

figure('Name','Mean Prediction')
plotseries(gca, X_pred, [1], 'r'); 

