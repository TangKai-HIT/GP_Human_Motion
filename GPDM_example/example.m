addpath src/gpdm src/gplvm src/netlab src/util

format long

global USE_GAMMA_PRIOR  % gamma prior for dynamics, only works with RBF kernel
global GAMMA_ALPHA % defines shape of the gamma prior
global USE_LAWRENCE % fix dynamics HPs, as Lawrence suggested (use with thetad = [0.2 0.01 1e6];) 
global FIX_HP % fix all HPs
global MARGINAL_W % marginalize over W while learning X
global MARGINAL_DW % marginalize over scale in dynamics while learning X
global LEARN_SCALE % use different scales for different output dimensions
global REMOVE_REDUNDANT_SCALE % let W absorb the overall scale of reconstruction
global W_VARIANCE % kappa^2 in the paper, not really the variance though
global M_CONST % M value in Jack's master's thesis
global BALANCE % Constant in front of dynamics term, set to D/q for the B-GPDM
global SUBSET_SIZE % Number of data to select for EM, set -1 for all data. 
global USE_OLD_MISSING_DATA

M_CONST = 1; 
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
if MARGINAL_W == 1
    opt(14) = 100; % total number of iterations
    extItr = 1; 
else
    opt(14) = 10; % rescaling every 10 iterations
    extItr = 100; % do extItr*opt(14) iterations in total
end  

% modelType(1) : input of dynamics
%   0 => [x_t, x_{t-1}]
%   1 => [x_t, x_t - x_{t-1}]
%   2 => [x_t]
% modelType(2) : output of dynamics 
%   0 => x_{t+1} 
%   1 => x_{t+1} - x_t
% modelType(3) : kernel type
%   0 => RBF kernel with weighted dimensions, use with input 0 or 1
%   1 => RBF kernel 
%   2 => Linear kernel
%   3 => weighted Linear kernel + RBF kernel with weighted dimensions, use with
%   input 0 or 1
%   4 => weighted linear kernel
%   5 => linear + RBF

%% Learn single walker model from lin+rbf kernel.
modelType = [2 0 5]; 
[Y ,initY, varY, segments] = loadMocapData({['07_01.amc']}, [1], [2],[260]);
missing = [];
N = size(Y, 1); D = size(Y, 2);
q = 3; % dimensionality of latent space

%% PCA
X = zeros(N, q);
refY = Y; meanData = mean(Y);
Y = Y - repmat(meanData, N, 1); %substract mean
[v, u] = pca(Y);
v(find(v<0))=0;
X = Y*u(:, 1:q)*diag(1./sqrt(v(1:q))); %sphereing the data to unit covariance

%% initialize hyperparameters
theta = [1 1 exp(1)]; %3 hyperparameters of K_Y
thetad = [0.9 1 0.1 exp(1)]; %4 hyperparameters of K_X
w = ones(D,1);

[X, theta, thetad, w] = gpdmfitFull(X, Y, w, segments, theta, thetad, opt, ... 
     extItr, modelType, missing);
save example_model X Y w theta thetad modelType N D q meanData  ...
segments initY varY missing refY;

%% generate samples from learned model

load example_model
[K, invK] = computeKernel(X, theta);
[Xin, Xout] = priorIO(X, segments, modelType);
[Kd, invKd] = computePriorKernel(Xin, thetad, modelType(3));
simSteps = 256;
% starts at end of training sequence;
simStart = [X(segments(1)+1,:), X(end,:)]; %  inputs 2 points in case using 2nd order model
[X_pred, XRand_pred] = simulatedynamics(X, segments, thetad, invKd, simSteps, simStart, modelType);

% uncomment if want to generate new samples

%  hmcopt = foptions;      % Default options vector.
% hmcopt(1) = 1;			% Switch on diagnostics.
% hmcopt(7) = 100;	    	% Number of steps in leapfrog trajectory.
% hmcopt(9) = 0; 
% hmcopt(14) = 60;		% Number of Monte Carlo samples returned. 
% hmcopt(15) = 20;		% Number of samples omitted at start of chain.
% hmcopt(18) = 0.01;  	% leapfrog step size
%  X_samples = sampledynamics('example_model', X_pred, 'samples', hmcopt);

load samples
clf;
% hold on;

for n=1:4:size(X_samples,2)
plotseries(X_samples{n}, [1], 'g'); 
end

plotseries(X, segments, 'b');

plotseries(X_pred, [1], 'r'); 

