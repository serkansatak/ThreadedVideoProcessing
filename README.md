# ThreadedVideoProcessing

Code in this repository is an object oriented approach using official OpenCV [Example](https://github.com/opencv/opencv/blob/master/samples/python/video_threaded.py)

This is an example code for general purpose feel free to contribute!

## Requirements

These are the basic requirements for this project. I didn't tried with previous versions. It may work with them or needs little adjustments who knows.

```
Python >= 3.9 
OpenCV >= 4.7
```

## Usage

You can directly import the VideoProcessor object and assign an operator as if you are assigning a property.
If you don't know what a property is just follow the pattern.

```
# Import Class and OpenCV
import cv2
from VideoThreaded import VideoProcessor

# Create VideoProcessor object
processor = VideoProcessor(${INPUT_FILE_PATH}, ${OUTPUT_FILE_PATH})

# Define a function to use as operator
# Function should take "frame" and "t" as first two arguments rest is up to your operation
def dummy_operation(frame, t, blurRate):
    return cv2.medianBlur(frame, blurRate), t

# While assigning the operator, one should follow the pattern below.
# Arguments except frame and t should be sent as keyword arguments.
processor.operator = {'func': operation, 'kwargs': {'blurRate': 19}}
```

## Authors

* Serkan Şatak - serkansatak1@gmail.com

