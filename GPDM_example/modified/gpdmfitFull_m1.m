function [X, theta, thetap, weights] = gpdmfitFull_m1(X, Y, weights,...
    segments, theta, thetap, options, extIters, modelType, missing)
%%gpdmfitFull_m1: simplified version of the original file gpdmfitFull

PRINT_INFO = 0; %print info every external loop
PLOT_L = 1; %plot negative log-likelihood every 10 SCG loop
global FIX_HP
global LEARN_SCALE
global MARGINAL_DW
global W_VARIANCE
global USE_OLD_MISSING_DATA

%% Specify basic parameters
N = size(Y,1);
D = size(Y,2);
q = size(X,2);

nmissing = setdiff(1:N, missing); %去掉缺失帧后的索引

ndp = 0;
if (modelType(3) == 0)
	ndp = 4;
elseif (modelType(3) == 1)
	ndp = 3;
elseif (modelType(3) == 2)
	ndp = 2;
elseif (modelType(3) == 3)
    ndp = 6;
elseif (modelType(3) == 4)
    ndp = 3;
elseif (modelType(3) == 5)
    ndp = 4;
end

opt14 = options(14); %SCG iterations


lastWeights = weights; 
lastX = X;  
lastTheta = [log(theta) log(thetap)]; % use ln(theta), ln(thetap) for parameter optimization

energyLog_scg = []; 
changeW = 0;
changeX = 0; 
changeTheta = 0;

%% start MAP estimation loop
if PLOT_L==1
    figure('Name','MAP: negative Log-likelihood');
end

for iters = 1:extIters
    
    if PRINT_INFO==1
        fprintf(2,'Iteration %d\n',iters);
    end
    
    if (FIX_HP ~= 1) && (LEARN_SCALE == 1)
        [~, invK] = computeKernel(X(nmissing,:), theta); % compute rbf kernel K_Y
        if (USE_OLD_MISSING_DATA == 1) 
        else
        if (~isempty(missing))
            kbold = kernel(X(missing,:), X(nmissing,:), theta)';
            A = Y(nmissing,:)'*invK*kbold;
            Y(missing,:) = A';
            nmissing = 1:N;
            [~, invK] = computeKernel(X(nmissing,:), theta);
        end
        end

        for d=1:D % data Y dimensions
            % update w_d (weight of dimension d)
            if (W_VARIANCE == 0) % kappa^2=0
                denom = Y(nmissing,d)'*invK*Y(nmissing,d); 
                if (denom == 0) 
                    weights(d) = 1; 
                else 
                    weights(d) = sqrt(length(nmissing)/denom); 
                end
            else
                denom = Y(nmissing,d)'*invK*Y(nmissing,d) + 1/W_VARIANCE;
                weights(d) = sqrt(length(nmissing)/denom);
                %             weights(d) = 1;
            end

        end
    end
    
    changeW = max(abs(weights - lastWeights)); 
    lastWeights = weights; 

    lntheta = log(theta);
    lnthetap = log(thetap);

    params = [X(:)' lntheta lnthetap]; % expand X, concatenate with HPs(ln(theta), ln(thetap))
    [params, options, flog] = scg('gpdmlikelihood', params, options, 'gpdmgradient',...
        Y, weights, segments, modelType, missing);

    X = reshape(params(1:N*q), N, q); 
    changeX = max(max(abs(X - lastX))); 
    lastX = X; 

    lntheta = params(end-(ndp+2):end-ndp);
    lnthetap = params(end-(ndp-1):end);      

    theta = exp(lntheta);
    thetap = exp(lnthetap);

    changeTheta = max(abs([log(theta) log(thetap)] - lastTheta));
    lastTheta = [log(theta) log(thetap)];
    
    %%print info
    if PRINT_INFO==1
        if (modelType(3) == 0)
            fprintf('Kernel width: %4.8f\n', 1/theta(1))
            fprintf('RBF Process variance: %4.8f\n', theta(2))
            fprintf('Noise variance: %4.8f\n', 1/theta(3))
            fprintf('Prior Kernel width1: %4.8f\n', 1/thetap(1))
            fprintf('Prior Kernel width2: %4.8f\n', 1/thetap(2))
            fprintf('Prior RBF Process variance: %4.8f\n', thetap(3))
            fprintf('Prior Noise variance: %4.8f\n', 1/thetap(4))
        elseif (modelType(3) == 1)
            fprintf('Kernel width: %4.8f\n', 1/theta(1))
            fprintf('RBF Process variance: %4.8f\n', theta(2))
            fprintf('Noise variance: %4.8f\n', 1/theta(3))
            fprintf('signal to noise ratio: %4.8f\n', sqrt(theta(2)*theta(3))); 
            fprintf('characteristic length scale: %4.8f\n', sqrt(1/theta(1))); 
            fprintf('Prior Kernel width: %4.8f\n', 1/thetap(1))
            fprintf('Prior RBF Process variance: %4.8f\n', thetap(2))
            fprintf('Prior Noise variance: %4.8f\n', 1/thetap(3))
            fprintf('Prior signal to noise ratio: %4.8f\n', sqrt(thetap(2)*thetap(3))); 
            fprintf('Prior characteristic length scale: %4.8f\n', sqrt(1/thetap(1))); 
        elseif (modelType(3) == 2)
            fprintf('Kernel width: %4.8f\n', 1/theta(1))
            fprintf('RBF Process variance: %4.8f\n', theta(2))
            fprintf('Noise variance: %4.8f\n', 1/theta(3))
            fprintf('Prior Process variance: %4.8f\n', thetap(1))
            fprintf('Prior Noise variance: %4.8f\n', 1/thetap(2))
        elseif (modelType(3) == 3)
            fprintf('Kernel width: %4.8f\n', 1/theta(1))
            fprintf('RBF Process variance: %4.8f\n', theta(2))
            fprintf('Noise variance: %4.8f\n', 1/theta(3))
            fprintf('Prior Linear Process variance1: %4.8f\n', thetap(1))
            fprintf('Prior Linear Process variance2: %4.8f\n', thetap(2))
            fprintf('Prior Kernel width1: %4.8f\n', 1/thetap(3))
            fprintf('Prior Kernel width2: %4.8f\n', 1/thetap(4))
            fprintf('Prior RBF Process variance: %4.8f\n', thetap(5))
            fprintf('Prior Noise variance: %4.8f\n', 1/thetap(6))
        elseif (modelType(3) == 4)
            fprintf('Kernel width: %4.8f\n', 1/theta(1))
            fprintf('RBF Process variance: %4.8f\n', theta(2))
            fprintf('Noise variance: %4.8f\n', 1/theta(3))
            fprintf('Prior Process variance1: %4.8f\n', thetap(1))
            fprintf('Prior Process variance2: %4.8f\n', thetap(2))
            fprintf('Prior Noise variance: %4.8f\n', 1/thetap(3))
        elseif (modelType(3) == 5)
            fprintf('Kernel width: %4.8f\n', 1/theta(1))
            fprintf('RBF Process variance: %4.8f\n', theta(2))
            fprintf('Noise variance: %4.8f\n', 1/theta(3))
            fprintf('Prior Linear Process variance: %4.8f\n', thetap(1))
            fprintf('Prior Kernel width: %4.8f\n', 1/thetap(2))
            fprintf('Prior RBF Process variance: %4.8f\n', thetap(3))
            fprintf('Prior Noise variance: %4.8f\n', 1/thetap(4))
        end
        fprintf('Delta X: %4.8f\n', changeX); 
        fprintf('Delta W: %4.8f\n', changeW); 
        fprintf('Delta Theta: %4.8f\n', changeTheta); 
        fprintf('-log Posterior: %4.8f\n', options(8));
    end
    
    if PLOT_L==1
        scatter(gca, iters, options(8), 'ob','filled');
        hold on; grid on;
    end
    
    energyLog_scg = [energyLog_scg options(8)];
    
    w = weights;
    thetad = thetap; 
    
    if (iters==extIters) 
        save log_scg energyLog_scg; 
    end

    if (size(energyLog_scg, 2) > 1)
        if ((energyLog_scg(end-1) - energyLog_scg(end)) < options(3) && ...
                changeX < options(2) && changeW < options(2) && ...
                changeTheta < options(2) && size(flog,2) < options(14))
            changeX
            changeW
            changeTheta
            size(flog,2)
            options(14)
            save log_scg energyLog_scg; 
            return;
        end
    end
  
    if(size(flog,2) == options(14) && flog(end) == flog(1))
        options(14) = opt14 + 10;
    else
        options(14) = opt14; 
    end
    
%     plot(weights);
%     pause;
end

if modelType(3) == 1
    if MARGINAL_DW == 1
        [Xin Xout] = priorIO(X, segments, modelType);
        [Kp, invKp] = computePriorKernel(Xin, thetad, modelType(3));
        denom = 0; 
        for d = 1:q
            denom = denom + Xout(:,d)'*invKp*Xout(:,d); 
        end
        denom = denom + 1/W_VARIANCE;
        priorw = size(Xin,1)*q/denom; 
        thetap(2) = thetap(2)/priorw;
        thetap(3) = thetap(3)*priorw; 
    end
end

if (modelType(3) == 1)
        fprintf('Kernel width: %4.8f\n', 1/theta(1))
        fprintf('RBF Process variance: %4.8f\n', theta(2))
        fprintf('Noise variance: %4.8f\n', 1/theta(3))
        fprintf('signal to noise ratio: %4.8f\n', sqrt(theta(2)*theta(3))); 
        fprintf('characteristic length scale: %4.8f\n', sqrt(1/theta(1))); 
        fprintf('Prior Kernel width: %4.8f\n', 1/thetap(1))
        fprintf('Prior RBF Process variance: %4.8f\n', thetap(2))
        fprintf('Prior Noise variance: %4.8f\n', 1/thetap(3))
        fprintf('Prior signal to noise ratio: %4.8f\n', sqrt(thetap(2)*thetap(3))); 
        fprintf('Prior characteristic length scale: %4.8f\n', sqrt(1/thetap(1)));
end


