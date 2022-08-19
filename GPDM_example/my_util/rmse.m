function e = rmse(Yt, Y)
%RMSE compute root mean square error of two matrix
%   Detailed explanation goes here
E_Y = Yt - Y;

e = sqrt(sum(sum(E_Y.^2)) / size(Y,1));

end

