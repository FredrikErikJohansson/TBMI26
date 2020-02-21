%% Hyper-parameters

% Number of randomized Haar-features
nbrHaarFeatures = 100;
% Number of training images, will be evenly split between faces and
% non-faces. (Should be even.)
nbrTrainImages = 2000;
% Number of weak classifiers
nbrWeakClassifiers = 9; % Pick something that can be sqrt

%% Load face and non-face data and plot a few examples
load faces;
load nonfaces;
faces = double(faces(:,:,randperm(size(faces,3))));
nonfaces = double(nonfaces(:,:,randperm(size(nonfaces,3))));

figure(1);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(faces(:,:,10*k));
    axis image;
    axis off;
end

figure(2);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(nonfaces(:,:,10*k));
    axis image;
    axis off;
end

%% Generate Haar feature masks
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);

figure(3);
colormap gray;
for k = 1:25
    subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2]);
    axis image;
    axis off;
end

%% Create image sets (do not modify!)

% Create a training data set with examples from both classes.
% Non-faces = class label y=-1, faces = class label y=1
trainImages = cat(3,faces(:,:,1:nbrTrainImages/2),nonfaces(:,:,1:nbrTrainImages/2));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainImages/2), -ones(1,nbrTrainImages/2)];

% Create a test data set, using the rest of the faces and non-faces.
testImages  = cat(3,faces(:,:,(nbrTrainImages/2+1):end),...
                    nonfaces(:,:,(nbrTrainImages/2+1):end));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
yTest = [ones(1,size(faces,3)-nbrTrainImages/2), -ones(1,size(nonfaces,3)-nbrTrainImages/2)];

% Variable for the number of test-data.
nbrTestImages = length(yTest);

%% Implement the AdaBoost training here
%  Use your implementation of WeakClassifier and WeakClassifierError
n = size(xTrain, 2);
d = ones(1, n) / n;
polarity = 1;
alphas = zeros(nbrWeakClassifiers,1);
indices = zeros(nbrWeakClassifiers, 1);
thresholds = zeros(nbrWeakClassifiers, 1);
polarities = zeros(nbrWeakClassifiers, 1);
errors = zeros(nbrWeakClassifiers, 1);
%weights = zeros(nbrWeakClassifiers, 1);

for i=1:nbrWeakClassifiers
    (i/nbrWeakClassifiers)*100
    minE = 0.5;
    for j=1:nbrHaarFeatures
        threshold = xTrain(j,:); % Test all thresholds
        for t = threshold
            C = WeakClassifier(t, polarity, xTrain(j,:));
            E = WeakClassifierError(C, d, yTrain);
            if(E > 0.5)
                polarity = -polarity;
                E = 1 - E;
            end
            
            if(E < minE)
                minE = E;
                alphas(i) = 0.5*log((1-E)/E);
                polarities(i) = polarity;
                thresholds(i) = t;
                indices(i) = j;
                errors(i) = E;
            end
        end
    end
    
    d = (d.*exp(-alphas(i)*yTrain.*(C*polarities(i))));
    d(d > 0.5) = 0.5; % If alpha becomes inf
    d = d./sum(d);
end

%% Evaluate your strong classifier here
%  Evaluate on both the training data and test data, but only the test
%  accuracy can be used as a performance metric since the training accuracy
%  is biased.

result = zeros(nbrWeakClassifiers, nbrTestImages);
for i=1:nbrWeakClassifiers
    result(i,:) = alphas(i) * WeakClassifier(thresholds(i,:), polarities(i,:), xTest(indices(i),:));
    %errors(i,:) = WeakClassifierError(result(i,:), d, yTest);
end

Classifications = sign(sum(result,1));
Accuracy = sum(Classifications(1,:) == yTest(1,:))/nbrTestImages


%% Plot the error of the strong classifier as a function of the number of weak classifiers.
%  Note: you can find this error without re-training with a different
%  number of weak classifiers.

errors = zeros(nbrWeakClassifiers,1);
for i=1:nbrWeakClassifiers
    errors(i) = sum(sign(sum(result(1:i,:),1)) ~= yTest)/nbrWeakClassifiers;
end

figure(4);
plot(errors);


%% Plot some of the misclassified faces and non-faces
%  Use the subplot command to make nice figures with multiple images.



%% Plot your choosen Haar-features
%  Use the subplot command to make nice figures with multiple images.

figure(5);
colormap gray;
for k = 1:nbrWeakClassifiers
    s = sqrt(nbrWeakClassifiers);
    subplot(s,s,k),imagesc(haarFeatureMasks(:,:,(indices(k))),[-1 2]);
    axis image;
    axis off;
end

