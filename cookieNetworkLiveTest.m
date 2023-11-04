loadedNet = load('cookieNetwork.mat');
netTransfer = loadedNet.netTransfer;

camera = webcam; % Connect to the camera
f = figure;

while ishandle(f)

    picture = camera.snapshot;
    pictureResized = imresize(picture, [227, 227]);
    
    [label, score] = classify(netTransfer, pictureResized);
    labelStr = string(label);
    
    image(picture); % Display the original picture
    if labelStr == "Bad_cookies"
        title(sprintf('Bad Cookies, Score: %.2f', score(label == "Bad_cookies")));
    else
        title(sprintf('Good Cookies, Score: %.2f', score(label == "Good_Cookies")));
    end
    
    drawnow;
end

clear('camera');
