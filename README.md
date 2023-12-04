# Background
This project is a fork of the original [yolov5 repository](https://github.com/ultralytics/yolov5/). The purpose of this repository is to create to a cleaner and more modular version of the detector for inference. Please view file `yolov5_detector.py`


# Requirements
    pip3 install -r requirements.txt

# Usage
In your code, import the detector class

    from yolov5_detector import Yolov5

1) The  `__init__`  takes the path to the Yolov5 pytorch weights that are to be loaded and the device (cuda or cpu)

2) The `warmup` function warms up the model

3) The `preprocess` function performs the relevant preprocessing on the image.

4) The `predict` function takes a torch.tensor as input and performs inference on it.

5) The `postprocess` function performs post processing ( NMS ) on the output. It takes a confidence threshold as input along with the tensor from inference.

# TO DO
1) Rescale bounding boxes to the original size of the image.