function [ LPred ] = kNN(X, k, XTrain, LTrain)
% KNN Your implementation of the kNN algorithm
%    Inputs:
%              X      - Samples to be classified (matrix)
%              k      - Number of neighbors (scalar)
%              XTrain - Training samples (matrix)
%              LTrain - Correct labels of each sample (vector)
%
%    Output:
%              LPred  - Predicted labels for each sample (vector)

classes = unique(LTrain);
NClasses = length(classes);

% Add your own code here
LPred = zeros(length(X),1);
distances = repmat(1000,length(X),1);
matches = zeros(k,1);

for i = 1:length(X)
    for j = 1:length(XTrain)  
        distances(j) = norm(X(i,:)-XTrain(j,:));
    end
    
    %Store k closest points
    [~,closestPoints] = mink(distances(:,1),k);
   
    %Store the prediced classes
    for z = 1:k
        matches(z) = LTrain(closestPoints(z),1);
    end
    
    %Find the the class with most occurrences
    bestMatch = mode(matches);    
    LPred(i) = bestMatch;
end

end

