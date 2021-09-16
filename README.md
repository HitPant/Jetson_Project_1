# Description:

The usecase is built on jetson Nano(4GB dev kit)<br>
The usecase inference was executed and tested on jetson Nano.<br>

This usecase is for generating alert: <br>
    1. when loitering is detected in a given area.<br>
    2. if count of person increases beyond allowed number.<br>


**SSD Model** is used for detecting people.<br>

## **Project Structure:**<br>

### loitering_dwell_detection <br>
    --- 1. loiter_dwell_detection.py<br>
    --- 2. config.json <br>
    --- 3. model<br>
        ------ MobileNetSSD_deploy.caffemodel<br>
        ------ MobileNetSSD_deploy.prototxt.txt<br>
    --- 4. tracker<br>
        ------ centroidtracker.py<br>
    --- 5. person_count_alert:<br>
        ------ **stores images with time stamp when person exceed beyond threshold value**<br>

## **Parameters** and **values** for generating alert are defined in **config.json**<br>
### config.json containes the following parameters:<br>
    1. person_duration (Dwell time allowed per person)<br>
    2. personCountExceed (maximum number of persons allowed in a frame at a given time)<br>
    3. vid_source (video source[videofile, rtsp, camera])<br>

Note: To use Opencv with CUDA and use opencv dnn module install Opencv from source.

## Steps to run the the script:
1. Define the following in the config.json file:<br>
    1. video source<br>
    2. person count allowed<br>
    3. dwell time allowed<br>
    <br>
2. Open the terminal in the location where loiter_dwell_detection.py is present.<br>

3. run the following command:
> $ python3 loiter_dwell_detection.py
