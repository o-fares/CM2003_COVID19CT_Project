# CM2003_COVID19CT_Project
Depository containing our codes and results for the COVID-19 CT classification project

## COVID-CT challenge
The challenge is a classification task of covid patients using CT scans. 

## Data Presentation:

The dataset is composed of COVID and NonCOVID images. Some images come from article and some directly from CT. The images extracted from the article have a poorer resolution but radiologists have confirmed the utility of these images. 
The image files are in /Project/CT_COVID and /Project/CT_NonCOVID

Besides, the splits are given by the author of the challenge in .txt files present in /Project/Data_split/COVID and /Project/Data_split/NonCOVID
 - trainCT_COVID, testCT_COVID, valCT_COVID,
 - trainCT_NonCOVID, testCT_NonCOVID, valCT_NonCOVID
There are 425 images in the train set and 203 in the validation set. 
 
 ### How do we load the data ?
 
  - Recover the txt files to make lists of path to load images
  - Load the images with cv2 and normalize them

## Models and results :
The article proposed to use different models, Densenet169 or Resnet50. We decided to implement both to see the difference of performance between both models. We try also smaller models with less layers or with a smaller number of filters in the Convolution layers. 
The results show that all the models learn pretty well with the train set but the results with the validation set are less conclusive. It seems like the models don't see any benefits in the validation set. Maybe it learns too specific features during the training and then struggle to generalize for the validation set and the test set. 

The different steps we have followed to make the results better are the following :
 - Use data augmentation on the training set because there are not enough images in the dataset.
 - Use smaller models such as Vgg16 or Alexnet model with 5 layers that we have used before in the labs for classification tasks.
 - Use standardization instead of normalization of gray scales.
 - Switch the training set with the test set or the validation set to see if there is a good validation on the training set.
 - Visualize the activation map to see what the models learn on the images.
 
 
