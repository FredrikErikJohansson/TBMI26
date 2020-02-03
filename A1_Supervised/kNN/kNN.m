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
LPred = zeros(size(X,1),1);
distances = repmat(1000,size(X,1),1);
matches = zeros(k,1);

for i = 1:length(X)
    for j = 1:length(XTrain)  
        distances(j,1) = norm(X(i,:)-XTrain(j,:));
    end
    
    %Store k closest points
    [~,closestPoints] = mink(distances(:,1),k);
   
    %Find the prediced classes
    for z = 1:NClasses
       for w = 1:k
           if(LTrain(closestPoints(w,1),1) == classes(z,1))
               matches(w) = classes(z,1);
           end
       end
    end
    
    %Find the the class with most occurrences
    bestMatch = mode(matches);    
    LPred(i,1) = bestMatch;
end

end

