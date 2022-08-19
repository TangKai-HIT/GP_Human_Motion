function remakeAmcAnimation(recY, meanY, frameLength, fileNameAsf, amcFileName , gifFileName, color, linestyle, usemask)

% REMAKEAMCANIMATION rewrite reconstructed Y into amc file and make
% animation which will be saved in GIF

% FORMAT
% DESC plays motion capture data by reading in a asf and amc files from disk.
% ARG fileNameAsf : the name of the ASF file to read in.
% ARG amcFileName : the name of the AMC file to write in.
% ARG frameLength : the length of the frames.
% ARG gifFileName : the name of the GIF file to write in.

% MOCAP
figure()
N = size(recY, 1);
newY = zeros(N, 62);
if usemask == 1 || size(recY, 2)~=62
    mask = [32:33 34 35:36 44:45 46 47:48 55 62]; %removed: rhand,rfingers,rthumb, lhand,lfingers,lthumb, rtoes, ltoes
    index = setdiff(1:62, mask);
    newY(:, mask(1:2)) = repmat([-28.8562, -18.8005], N, 1); %rhand
    newY(:, mask(3)) = repmat(7.12502, N, 1); %rfingers
    newY(:, mask(4:5)) = repmat([-2.21279, -48.7688], N, 1); %rthumb
     newY(:, mask(6:7)) =repmat([-19.6688 -28.7993], N, 1); %lhand
    newY(:, mask(8)) = repmat(7.12502, N, 1); %lfingers
    newY(:, mask(9:10)) = repmat([6.65951 1.07601], N, 1); %lthumb
    newY(:, mask(11)) = repmat(-20.7668, N, 1); %rtoes
    newY(:, mask(12)) = repmat(-22.0517, N, 1); %ltoes
else
    index = 1:62;
end

newY(:, index) = recY + meanY; %add mean data back
newY(:, 1:3) = cumsum([zeros(1,3); newY(1:end-1, 1:3)], 1); %sum the velocity to position

matrix_to_amc(amcFileName, newY)

skel = acclaimReadSkel(fileNameAsf);
[channels, skel] = acclaimLoadChannels(amcFileName, skel);
M=skelPlayData_m1(skel, channels, frameLength, color, linestyle);

makeGIF(M, gifFileName, frameLength);