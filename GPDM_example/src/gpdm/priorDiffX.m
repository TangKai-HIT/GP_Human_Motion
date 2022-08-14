function [dLp_dx] = priorDiffX(dLp_dxin, dLp_dxout, N, q, segments, ...
priorModel)

% priorModel(1) : input of dynamics
if (priorModel(1) == 0) %   0 => [x_t, x_{t-1}]
    qp = size(dLp_dxin, 2);
    dLp_dx = zeros(N, qp); % N*2q
    S = setdiff(1:N,mod(union(segments-1,segments-2),N)); %mod(-1,N)=N-1 
    S(find(S==N)) = [];
    dLp_dx(S,:) = dLp_dxin;
    dLp_dx(2:end,end-q+1:end) = ...
	dLp_dx(2:end,end-q+1:end) + dLp_dx(1:end-1,1:q);
    dLp_dx(:,1:q) = [];
elseif (priorModel(1) == 1) %   1 => [x_t, x_t - x_{t-1}] == [x_t, v_{t-1}]
    qp = size(dLp_dxin, 2);
    dLp_dx = zeros(N, qp);
    S = setdiff(1:N,mod(union(segments-1,segments-2),N)); %mod(-1,N)=N-1
    S(find(S==N)) = [];
    dLp_dx(S,:) = dLp_dxin;
    dLp_dx(2:end,end-q+1:end) = ...
	dLp_dx(1:end-1,end-q+1:end) - dLp_dx(2:end,end-q+1:end);
    dLp_dx(1,end-q+1:end) = -dLp_dx(1,end-q+1:end);
    dLp_dx(2:end,end-q+1:end) = ...
	dLp_dx(2:end,end-q+1:end) + dLp_dx(1:end-1,1:q);
    dLp_dx(:,1:q) = [];
elseif (priorModel(1) == 2) %   2 => [x_t]
    qp = size(dLp_dxin, 2);
    dLp_dx = zeros(N, qp);
    S = setdiff(1:N,mod(segments-1,N));
    S(find(S==N)) = [];
    dLp_dx(S,:) = dLp_dxin;
elseif (priorModel(1) == 3)
    qp = size(dLp_dxin, 2);
    dLp_dx = zeros(N, qp);
    S = setdiff(1:N,mod(segments-1,N));
    S(find(S==N)) = [];
    dLp_dx(S,:) = dLp_dxin;
    dLp_dx(:,end) = [];
end

% priorModel(2) : output of dynamics 
if (priorModel(2) == 0) %   0 => x_{t+1} 
	if (priorModel(1) == 2 || priorModel(1) == 3)
		S = setdiff(1:N,mod(segments,N));
	else
		S = setdiff(1:N,union(segments,segments+1));
	end
    dLp_dx(S,:) = dLp_dx(S,:) + dLp_dxout;
elseif (priorModel(2) == 1) %   1 => x_{t+1} - x_t == (v_t)
    if (priorModel(1) == 2 || priorModel(1) == 3)
		S = setdiff(1:N,mod(segments,N));
	else
		S = setdiff(1:N,union(segments,segments+1));
	end
    dLp_dx(S,:) = dLp_dx(S,:) + dLp_dxout;
    dLp_dx(S-1,:) = dLp_dx(S-1,:) - dLp_dxout;
end
