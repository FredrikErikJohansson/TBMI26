function [ cM ] = calcConfusionMatrix( LPred, LTrue )
% CALCCONFUSIONMATRIX returns the confusion matrix of the predicted labels

classes  = unique(LTrue);
NClasses = length(classes);

% Add your own code here
cM = zeros(NClasses);

for i = 1:length(LPred)
    cM(LTrue(i,1), LPred(i,1)) = cM(LTrue(i,1), LPred(i,1)) + 1; 
end

end

