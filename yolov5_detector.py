import torch
import cv2
import numpy as np
import os
from models.common import DetectMultiBackend
from utils.general import non_max_suppression

class Yolov5: 
    def __init__(self, model_path, device): 
        """loads a model from model_path and convert it to cpu or gpu based on the device. 
        This function needs two attributes: self.model: the model loaded from model_path self.device: 
        the device to run the model on 
        Args: model_path (str): the path to the model weight file device (str): 'cpu' or 'cuda' """ 

        self.model = DetectMultiBackend(model_path)  # Load the model
        self.model = self.model.to(device)           # Move to CUDA if available
        self.model.eval()                            # change to inference mode
        self.device = device  
        
        
    def warmup(self): 
        """warm up the model using a dummy image with all zeros """ 
      
        dummy_image = torch.zeros((1, 3, 640, 640), device=self.device)
        self.model(dummy_image)
        print("ok")
        

    def preprocess(self, image_path): 
        """load the image from image_path and preprocess it to the format that the model accepts. return the processed image. Args: image_path (str): the path to a image """ 

        # Input Size : Change to 1280 for P6 models
        sz = 640
        
        # Read image
        img = cv2.imread(image_path)

        # Change BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to appropriate input size
        img = cv2.resize(img, (sz, sz))   

        # Convert to tensor and move to GPU if available
        img = torch.from_numpy(img).to(self.device) 
        
        # Converting image to correct format with values between 0 and 1
        img = img.half() if self.model.fp16 else img.float()  
        img /= 255
        
        # YOLO expected a batch as input, making image a solo batch
        if len(img.shape) == 3:
          img = img.unsqueeze(0)
        
        # Switch channel positions
        img = torch.permute(img, (0, 3, 1, 2))
        
        return img
                    

        
    def predict(self, image): 
        """run the model on a processed image and return the predictions Args: image (torch.Tensor): the processed image """ 
        # Give input to the model
        output = self.model(image)[0]
        return output


    def postprocess(self, predictions, confidence_threshold=0.25): 
        """filter the predictions based on the confidence threshold and return the filtered results 
        Args: predictions: the return value of self.predict confidence_threshold (float): the confidence threshold to filter the predictions """ 
        # Non maximum suppression
        outputs = non_max_suppression(predictions,conf_thres=confidence_threshold)[0]

        return outputs
        
# Example usage: 
if __name__ == "__main__": 
    model_path = 'yolov5l.pt' # Update with the correct path to your model 
    image_path = 'bus.jpg' # Update with the path to your image 
    detector = Yolov5(model_path, device='cuda' if torch.cuda.is_available() else 'cpu') 
    detector.warmup
    preprocessed_image = detector.preprocess(image_path) 
    predictions = detector.predict(preprocessed_image) 
    filtered_results, scaled_results = detector.postprocess(predictions) 
    print(filtered_results)
