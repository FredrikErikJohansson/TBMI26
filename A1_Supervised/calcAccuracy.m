function [ acc ] = calcAccuracy( cM )
% CALCACCURACY Takes a confusion matrix amd calculates the accuracy

% Add your own code here
correct = 0;
for i = 1:length(cM)
    correct = correct + cM(i,i);
end

acc = correct / sum(sum(cM));

end

