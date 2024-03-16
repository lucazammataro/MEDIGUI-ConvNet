# MEDIGUI-ConvNet
MEDIGUI-ConvNet (Medical Imaging Convolutional Neural Network with Graphic User Interface)

## Description

This repository contains the code for MEDIGUI-ConvNet, a Graphic User Interface for Python Jupyter Notebooks, which handle a Convolutional Neural Networks for classifying medical images (MRI, CT SCAN, or digital pathology)
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
