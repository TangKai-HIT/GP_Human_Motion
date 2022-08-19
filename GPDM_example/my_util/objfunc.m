 function [f,g]=objfunc(t, Y, weights, segments, modelType, missing)
        f=gpdmlikelihood(t, Y, weights, segments, modelType, missing);
        g=gpdmgradient(t, Y, weights, segments, modelType, missing);
    end