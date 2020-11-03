# FER
# Facial-Expression-Recognition.Pytorch
A CNN based pytorch implementation on facial expression recognition (FER2013), on google colab achieving 58.149% in FER2013



## Dependencies ##
- Python 2.7
- Pytorch >=0.2.0
- h5py (Preprocessing)
- sklearn
- OpenCV


## FER2013 Dataset ##
- Dataset from https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
Image Properties: 48 x 48 pixels (2304 bytes)
labels: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
The training set consists of 28,709 examples. The public test set consists of 3,589 examples. The private test set consists of another 3,589 examples.

## Model Architecture ##

The entire model consists of 14 layers in total. In addition to layers below lists what techniques are applied to build the model.

1. Convolution with 16 different filters in size of (3x3)
2. Max Pooling by 2
  - ReLU activation function
3. Convolution with 64 different filters in size of (3x3)
4. Max Pooling by 2
  - ReLU activation function
5. Convolution with 128 different filters in size of (3x3)
  - ReLU activation function
6. Convolution with 256 different filters in size of (3x3)
7. Max Pooling by 2
  - ReLU activation function
8. Flattening the 3-D output of the last convolutional operations.
9. Fully Connected Layer with 500 units
10. Fully Connected Layer with 200 units
11. Fully Connected Layer with 7 units

### Preprocessing Fer2013 ###
- Dataset is already preprocessed from csv to h5py 


## Training the model ##
Achieving over 96.798% accuracy using batch size as 100.


## Prediction ##
<img src="./Training loss.png" alt="Drawing"/>
<img src="./Training Acc.png" alt="Drawing"/>


