<img src="MEDIGUI_ConvNet_Logo.png" alt="logo" style="display:block; margin:auto;">



# MEDIGUI-ConvNet
MEDIGUI-ConvNet (Medical Imaging Convolutional Neural Network with Graphic User Interface)

Developed by Luca Zammataro, Lunan Foldomics LLC, Copyright (C) 2024

## Description

MEDIGUI-ConvNet is an application that leverages the convenience of interactive widgets in Jupyter to classify MRI and CT SCAN images. It makes it effortless to load datasets, train Convolutional Neural Network (CNN) models, and test these models on images.

The general structure of the application is as follows:

1. Tab for dataset loading: Allows loading a dataset of images in pickle format.
2. Tab for model training: This feature-rich tab equips you with controls to fine-tune model training, such as the number of epochs, batch size, and a variety of CNN architecture options. This flexibility allows you to adapt the training process to your specific needs, giving you a sense of control over your work.
3. Tab for model loading: Allows loading a previously trained CNN model.
4. Tab for model testing: This tab contains a widget for selecting an image from the dataset and displaying the model's prediction. It also allows you to display feature maps from various layers of the CNN.
5. Tab for learning plot visualization: Displays the model's training progress plot over time.

MEDGUI-convnet has undergone rigorous testing in MacOS environments for M1 processors with Jupyter Notebook and in a Google Colab environment. While some functionalities, like the progress bar, may not be fully supported with Jupyter Lab, you can trust in the reliability of the application in these tested environments. 

## Virtual Environment

To run the code, we recommend using a conda virtual environment. You can create a virtual environment named `medgui-convnet` and install the required dependencies by executing the following commands:

```bash
# Create a new virtual environment named medgui-convnet
conda create -n medgui-convnet python=3.9

# Activate the virtual environment
conda activate medgui-convnet  # On Windows
source activate medgui-convnet  # On macOS and Linux

# Install the required packages
pip install numpy==1.23.2
pip install pandas==1.5.3
pip install keras==2.11.0
pip install tensorflow-estimator==2.11.0
pip install tensorflow-macos==2.11.0
pip install tensorflow-metal==0.7.1
pip install matplotlib==3.7.1
pip install scikit-learn==1.2.2
pip install ipywidgets==8.0.4
pip install ipyfilechooser==0.6.0
pip install ipython==8.11.0
pip install tqdm==4.66.1

# Deactivate the virtual environment
conda deactivate  # On Windows
source deactivate  # On macOS and Linux

```
## Run MEDIGUI-ConvNet

The python module medigui_convnet.py contains all the MEDIGUI-ConvNet functions and can be imported in a Jupyter Notebook cell typying:

```
import medigui_convnet
```
the RunGUI function starts automatically. 
You can use the medigui_convnet.ipynb provided if you need it.

## MEDIGUI-ConvNet functions

MEDIGUI-ConvNet functions are directly accessible (you do not need the GUI to use them). Import the module as an object:

```
import medigui_convnet as medigui
```
Upload a dataset with LoadImageArchive(path):
The dataset must be a pickle file of a NumPy array (more information will be available soon about how to construct a dataset)
```
X, Y = medigui.LoadImageArchive('path_to_a_dataset.pickle')
```

Generating training and setting datasets
```
X_train, X_test, Y_train, Y_test = medigui.splitDataset(X=X, Y=Y, test_size=0.2, random_state=42)
```

Check the size of all the arrays
```
# Verify the arrays
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)

```
Defining a model:
You can define the CNN architecture by adjusting filters, the number of neurons, and activation functions.
```
model = defineModel(X=X, Y=Y,
              Conv2D_1_filters=44, Conv2D_1_kernelSize=3,  C2D_1_activation='relu', MP2D_1_filters=2, 
                Conv2D_2_filters=128, Conv2D_2_kernelSize=3,  C2D_2_activation='relu', MP2D_2_filters=2, 
                Conv2D_3_filters=256, Conv2D_3_kernelSize=3,  C2D_3_activation='relu', MP2D_3_filters=2, 
                Conv2D_4_filters=512, Conv2D_4_kernelSize=3,  C2D_4_activation='relu', MP2D_4_filters=2,
                Conv2D_5_filters=512, Conv2D_5_kernelSize=3,  C2D_5_activation='relu', MP2D_5_filters=2, 
                Dense_1_filters=128, Dense_1_activation='relu', l1=0.001, l2=0.001,
                Dense_2_activation='softmax')
```
Training:
You can manipulate training epochs, batch size, and two regularization parameters to fine-tune the training performances.
```
model = medigui.trainCNN(X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test, epochs=30, batch_size=32)
```

## Datasets 
The dataset used (ALZ.training.set) represents the low-resolution version of another dataset provided by Uraninjo https://www.kaggle.com/uraninjo (https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset). All the images have been pre-processed and reduced to 100 X 100 pixels. You can download these datasets in pickle format from our Kaggle page at the following link: https://www.kaggle.com/datasets/lucazammataro/augmented-alzheimer-mri-dataset-100x100-pixel.
The official reference for the dataset is:
A. Yakkundi, Alzheimer’s disease dataset, https://doi.org/10.17632/ch87yswbz4.1, 2023. doi:10.17632/ch87yswbz4.1, mendeley Data, Version 1.

Disclaimer: the software use is intended ONLY for experimental purposes, not for clinical.

## Additional Files
The file "classes.tsv" contains pathological classes associated with the labels. It is pivotal for displaying the results.
Unzip and load the ALZ.training.set.2024-03-15_21-03-51.model.zip file in the application to quickly experience the power of CNN classification.

contact: luca.zammataro@gmail.com

Enjoy MEDIGUI-ConvNet!


