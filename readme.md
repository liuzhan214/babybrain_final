# Efficient 3D Dense U-Net with Contour Regression for 6-month Infant Brain MRI Segmentation

## 1. hyperparameter setting 

all the hyperparameters are in config/parse.py, the discription .txt file in data gives the path of training, val, or test dataset.

## 2. train and evaluate

cd to babybrain_final directory, and then run train.py to train the Dense U-Net model, run evaluate.py to inference. In addition, densenet9.pth and densenet9_v3.pth is the pretrained weight, we recommend to use densenet9_v3.pth as pretrained weight.

## 3. test and visualiztion

run test.py to get the submit results and visualization result in ./output.