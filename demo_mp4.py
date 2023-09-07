# Basic module
from datetime import datetime
import cv2 
import torch
from matplotlib import pyplot as plt
import PIL
import numpy as np
import scipy
from scipy import ndimage

# Torch and visulization
from torchvision      import transforms

# Metric, loss .etc
from utils.utils import *
from utils.loss import *
from utils.load_param_data import load_param

# my model
from models.AMFU import *
from models.AMFU_noATN import *
from models.AMFU_noResATN import *

video_dir = '/your/own/path'
cap = cv2.VideoCapture(video_dir)

# Torch transform
input_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

# Model init
model = AMFU()
model = model.cuda()
model.apply(weights_init_xavier)

# Load checkpoint
checkpoint       = torch.load('./AMFU-net/result/NUAA-SIRST_AMFU/AMFU_epoch.pth.tar') # pre-trained
model.load_state_dict(checkpoint['state_dict'])

# For test 
model.eval()

while True:
    # Start time 
    start_time = datetime.now()

    # Read Images from mp4
    ret, frame_origin = cap.read()

    # Image to tensor
    frame = Image.fromarray(frame_origin)
    frame = input_transform(frame)
    frame = frame.cuda()
    frame = torch.unsqueeze(frame, 0)

    # Network inference
    preds = model(frame)
    pred = preds[-1]

    # Reshape from tensor
    predsss = np.array((pred > 0).cpu()).astype('int64') * 255
    predsss = np.uint8(predsss)
    img1 = Image.fromarray(predsss.reshape(256, 256))

    # End time
    end_time = datetime.now()
    elapsed_time = end_time - start_time

    # Video streaming
    predicted = np.array(img1)

    # Center of Blob
    centroid = np.zeros((256,256), np.uint8)
    im = predicted
    im = np.where(im > 250, 1, 0 )
    label_im, num = ndimage.label(im)
    slices = ndimage.find_objects(label_im)
    centroids = scipy.ndimage.center_of_mass(im, label_im, range(1,num+1))

    for circle in range(len(centroids)):
        cx = int(centroids[circle][1])
        cy = int(centroids[circle][0])
        cv2.circle(centroid, (cx, cy), 0, (255,255,255), -1)
    
    if ret:
        cv2.imshow('IR', frame_origin)
        cv2.imshow('Detected', predicted)
        cv2.imshow('Centroid', centroid)
        
        # Save Images
        # cv2.imwrite('/home/jay/catkin_GRSL/AMFU-net/tracking/Centroid/centroid.png', centroid)
        # cv2.imwrite('/home/jay/catkin_GRSL/AMFU-net/tracking/Detected/predicted.png', predicted)
        # cv2.imwrite('/home/jay/catkin_GRSL/AMFU-net/tracking/IR/IR.png', frame_origin)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    else:
        break







        


