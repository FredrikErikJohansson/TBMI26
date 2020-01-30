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
closestPoints = ones(k,1);
LPred = zeros(size(X,1),1);
minDistances = repmat(1000,k,1);

for i = 1:length(X)
    for j = 1:length(XTrain)
        for z = 1:k
            %Store the indices of the shortest distances
            if(norm(X(i,:)-XTrain(j,:)) < minDistances(:))
                if(j ~= closestPoints(:)) % TODO: This may not work properly
                    minDistances(k,1) = norm(X(i,:)-XTrain(j,:)); %Override the longest distance
                    minDistances = sort(minDistances); %Sort to keep the longest distance at back
                    closestPoints(k,1) = j;
                end
            end
        end
    end
    
    %Reset distancces
    minDistances = repmat(1000,k,1);
    
    maxCount = 0;
    currentCount = 0;
    bestMatch = 0;
    
    %Classify based on the number of occurrences in the k nearest 
    for z = 1:NClasses
        for w = 1:length(closestPoints)
            if(LTrain(closestPoints(w,1),1) == classes(z,1))
                currentCount = currentCount + 1;
            end
            
            if(currentCount > maxCount)
                maxCount = currentCount;
                bestMatch = classes(z,1);
            end
        end
    end
    
    LPred(i,1) = bestMatch;
end

end

