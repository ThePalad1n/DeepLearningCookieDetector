folder = 'C:\Users\ThePalad1n\Documents\MATLAB\cookies';
imds = imageDatastore(folder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');
imds.ReadFcn = @(filename)imresize(imread(filename), [227, 227]);


[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 'randomized');

trainLabelCount = countEachLabel(imdsTrain);
disp(['Training set: ', num2str(trainLabelCount.Count')]);

validationLabelCount = countEachLabel(imdsValidation);
disp(['Validation set: ', num2str(validationLabelCount.Count')]);



imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-10,10], ...
    'RandXReflection', true, ...
    'RandYReflection', true, ...
    'RandScale', [0.8, 1.2]);

augimdsTrain = augmentedImageDatastore([227 227 3], imdsTrain, 'DataAugmentation', imageAugmenter);


% Load AlexNet
net = alexnet;

% Modify the layers
layers = net.Layers;
layers(end-2) = fullyConnectedLayer(2, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20); % Change 'fc8' to have 2 outputs
layers(end) = classificationLayer; % Change the final classification layer


options = trainingOptions('sgdm', ...
    'MiniBatchSize', 10, ...
    'MaxEpochs', 2, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', imdsValidation, ...
    'ValidationFrequency', 2, ...
    'Verbose', true, ...
    'Plots', 'training-progress');


netTransfer = trainNetwork(augimdsTrain, layers, options);


[YPred, scores] = classify(netTransfer, imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = mean(YPred == YValidation);
confMat = confusionmat(YValidation, YPred);
confMat = bsxfun(@rdivide, confMat, sum(confMat, 2)); % Normalize the confusion matrix
disp(confMat);
disp(['Validation Accuracy: ', num2str(accuracy)]);

save('cookieNetwork.mat', 'netTransfer');

