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
The article proposed to use different models, DenseNet169 or ResNet50. We decided to implement both to see the difference of performance between both models. We try also smaller models with less layers or with a smaller number of filters in the Convolution layers. 
The results show that all the models learn pretty well with the train set but the results with the validation set are less conclusive. It seems like the models don't see any benefits in the validation set. Maybe it learns too specific features during the training and then struggle to generalize for the validation set and the test set. 


<p float="left">
 
 <img width="220" alt="loss_densenet" src="https://user-images.githubusercontent.com/65956573/97592450-f6ed4880-1a00-11eb-9968-dd5ceb6d95aa.PNG"/>
 <img width="214" alt="accuracy_densenet" src="https://user-images.githubusercontent.com/65956573/97592654-2d2ac800-1a01-11eb-88c6-3fd39afb623b.PNG"/>
 <em>DenseNet169 learning curves</em>
</p>


<p float="left">
 <img width="216" alt="loss_resnet" src="https://user-images.githubusercontent.com/65956573/97629269-25354d00-1a2e-11eb-90d6-d6fcbf5fcc80.PNG"/>
 <img width="213" alt="accuracy_resnet" src="https://user-images.githubusercontent.com/65956573/97629250-1f3f6c00-1a2e-11eb-9091-b0517528ed2b.PNG"/>
 <em>ResNet50 learning curves</em>
</p>


The different steps we have followed to make the results better are the following :
 - Use <strong>data augmentation</strong> on the training set because there are not enough images in the dataset.
 - Use <strong>smaller models</strong> such as Vgg16 or Alexnet model with 5 layers that we have used before in the labs for classification tasks.
 - Switch the training set with the test set or the validation set to see if there is a good validation on the training set.
 - Visualize the <strong>activation maps</strong> to see what the models learn on the images.
 - Use <strong>standardization</strong> instead of normalization of gray scales.
 
 ### 1. Data augmentation 
 
![data aug 1](https://user-images.githubusercontent.com/26654114/97636804-f0c78e00-1a39-11eb-9f6e-8b8bafca8ceb.png) 
![data aug 2](https://user-images.githubusercontent.com/26654114/97636803-f0c78e00-1a39-11eb-9fe9-43b13e907ce0.png)
 <em> Data augmentation on the DenseNet model </em>
 
 ### 2. Smaller models: VGG16 and AlexNet 
 
 ![vgg1](https://user-images.githubusercontent.com/26654114/97636797-ef966100-1a39-11eb-9227-02021db29220.png)
 ![vgg2](https://user-images.githubusercontent.com/26654114/97636796-ef966100-1a39-11eb-9aef-707348208b8a.png)
  <em> VGG16 learning curves </em>

![conv1](https://user-images.githubusercontent.com/26654114/97640159-76e6d300-1a40-11eb-875c-0256856e5ce5.png)
![conv2](https://user-images.githubusercontent.com/26654114/97640156-75b5a600-1a40-11eb-90e1-fb16be651024.png)
<em> AlexNet learning curves </em>

 ### 3. Swtich the training and the test sets 
 
![switch1](https://user-images.githubusercontent.com/26654114/97643107-1c04aa00-1a47-11eb-8676-cb1931550b96.png)
![switch2](https://user-images.githubusercontent.com/26654114/97643104-1ad37d00-1a47-11eb-9924-9b56c89c3f93.png)

 ### 4. Activation maps 
 
 <p float="left">
  <img width="213" alt="vizu12" src="https://user-images.githubusercontent.com/26654114/97636802-f02ef780-1a39-11eb-83cd-7e1f83b2ca82.png"/>
 <img width="213" alt="vizu22" src="https://user-images.githubusercontent.com/26654114/97636799-f02ef780-1a39-11eb-8b44-c6addd3d37e4.png"/>
 <img width="216" alt="vizu1" src="https://user-images.githubusercontent.com/26654114/97636793-ef966100-1a39-11eb-8ccd-88238742eb75.png"/>
 <img width="216" alt="vizu21" src="https://user-images.githubusercontent.com/26654114/97636791-ee653400-1a39-11eb-8795-269f7867ab52.png"/>
</p>


 ### 5. Data standardization

 
 
 
 
 
 
