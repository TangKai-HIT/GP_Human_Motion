function [Y_mean, Y_var] = gplvmRecon(X_pre, X_learned, Y, theta)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
[~, invK] = computeKernel(X_learned, theta); % K_Y
[Y_mean, Y_var] = manifoldOutputs(X_pre, X_learned, Y, theta, invK);
end

