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
 
 ### How do we load the data ?
 
  - Recover the txt files to make lists of path to load images
  - Load the images with cv2 and normalize them

