function makeGIF(M, filename, frameLength)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
    for j=1:size(M,2)
        if j==1
        [I,map]=rgb2ind(M(j).cdata,256);
        imwrite(I,map,filename,'DelayTime', frameLength)
        else
        imwrite(rgb2ind(M(j).cdata,map),map,filename,'WriteMode','append','DelayTime', frameLength)
        end
    end
end

