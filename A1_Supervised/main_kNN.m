 %% This script will help you test out your kNN code

%% Select which data to use:

% 1 = dot cloud 1
% 2 = dot cloud 2
% 3 = dot cloud 3
% 4 = OCR data

dataSetNr = 1; % Change this to load new data 

% X - Data samples
% D - Desired output from classifier for each sample
% L - Labels for each sample
[X, D, L] = loadDataSet( dataSetNr );

% You can plot and study dataset 1 to 3 by running:
% plotCase(X,D)

%% Select a subset of the training samples

numBins = 3;                    % Number of bins you want to devide your data into
numSamplesPerLabelPerBin = 100; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = true;          % true = select samples at random, false = select the first features

[XBins, DBins, LBins] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom);

% Note: XBins, DBins, LBins will be cell arrays, to extract a single bin from them use e.g.
% XBin1 = XBins{1};
%
% Or use the combineBins helper function to combine several bins into one matrix (good for cross validataion)
% XBinComb = combineBins(XBins, [1,2,3]);

% Add your own code to setup data for training and test here
acc = 0;
maximum = 0;
bestK = 0;
for k = 1:9
    for i = 1:numBins
        XTrain = combineBins(XBins, [1:i-1 i+1:numBins]);
        LTrain = combineBins(LBins, [1:i-1 i+1:numBins]);
        XTest  = XBins{i};
        LTest  = LBins{i};

%% Use kNN to classify data
%  Note: you have to modify the kNN() function yourself.

% Set the number of neighbors

        %k = 3;

        % Classify training data
        LPredTrain = kNN(XTrain, k, XTrain, LTrain);
        % Classify test data
        LPredTest  = kNN(XTest , k, XTrain, LTrain);

        %% Calculate The Confusion Matrix and the Accuracy
        %  Note: you have to modify the calcConfusionMatrix() and calcAccuracy()
        %  functions yourself.

        % The confucionMatrix
        cM = calcConfusionMatrix(LPredTest, LTest);

        % The accuracy
        
        acc = acc + calcAccuracy(cM);
    end
    if((acc/numBins) > maximum)
        bestK = k;
        maximum = (acc/numBins);
    end
    acc = 0;
end
bestK
maximum

%% Plot classifications
%  Note: You should not have to modify this code

if dataSetNr < 4
    plotResultDots(XTrain, LTrain, LPredTrain, XTest, LTest, LPredTest, 'kNN', [], bestK);
else
    plotResultsOCR(XTest, LTest, LPredTest)
end
