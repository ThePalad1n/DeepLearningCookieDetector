% Load the network
loadedNet = load('cookieNetwork.mat');
netTransfer = loadedNet.netTransfer;

% Access the webcam (make sure you have the webcam support package installed)
camera = webcam; % Connect to the camera

% Set up a figure to display the images
f = figure;

% Loop to get images from the webcam and classify them
while ishandle(f)
    % Get an image from the webcam
    picture = camera.snapshot;
    
    % Preprocess the image to match the input size of the network
    pictureResized = imresize(picture, [227, 227]);
    
    % Classify the image using the trained network
    [label, score] = classify(netTransfer, pictureResized);
    
    % Convert the label to string for comparison
    labelStr = string(label);
    
    % Display the image with the classification label and score
    image(picture); % Display the original picture
    if labelStr == "Bad_cookies"
        title(sprintf('Bad Cookies, Score: %.2f', score(label == "Bad_cookies")));
    else
        title(sprintf('Good Cookies, Score: %.2f', score(label == "Good_Cookies")));
    end
    
    % Pause briefly to update the figure window
    drawnow;
end

% Clean up
clear('camera');
