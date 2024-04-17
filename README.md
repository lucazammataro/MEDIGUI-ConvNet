<div style="display:flex; justify-content:center; align-items:center; height:300px;">
    <img src="MEDIGUI_ConvNet_Logo.png" alt="Descrizione dell'immagine">
</div>


# MEDIGUI-ConvNet
MEDIGUI-ConvNet (Medical Imaging Convolutional Neural Network with Graphic User Interface)

## Description

This repository contains the code for MEDIGUI-ConvNet, a Graphic User Interface for Python Jupyter Notebooks, which handles Convolutional Neural Networks for classifying medical images (MRI, CT SCAN, or digital pathology)
MEDGUI-convnet has been tested in MacOS environments for M1 processors with Jupyter Notebook, while its usage with Jupyter Lab appears to be unsupported.

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
Define a model
```
model = medigui.defineModel(X=X, Y=Y, l1=0.01, l2=0.01)
```
Check the size of all the arrays.
```
# Verify the arrays
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)

```
Training
```
model = medigui.trainCNN(X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test, epochs=30, batch_size=32)
```

## Datasets 
The dataset used (ALZ.training.set) represents the low-resolution version of another dataset provided by Uraninjo https://www.kaggle.com/uraninjo (https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset). All the images have been pre-processed and reduced to 100 X 100 pixels. You can download these datasets in pickle format from our Kaggle page at the following link: https://www.kaggle.com/datasets/lucazammataro/augmented-alzheimer-mri-dataset-100x100-pixel.

Disclaimer: the dataset use is intended ONLY for experimental purposes, not for clinical.

## Additional Files
The file "classes.tsv" contains pathological classes associated with the labels. It is pivotal for displaying the results.
Unzip and load the ALZ.training.set.2024-03-15_21-03-51.model.zip file in the application to quickly experience the power of CNN classification.

contact: luca.zammataro@gmail.com

Enjoy MEDIGUI-ConvNet!


