# Image Tampering Localization Using a Dense Fully Convolutional Network
# Overview
This is the implementation of the method proposed in "Image Tampering Localization Using a Dense Fully Convolutional Network" with tensorflow(1.10.0, gpu version). The aim of this repository is to achieve image tampering localization.
# Network Architecture

# Files structure
## Dense-FCN-for-tampering-localization
-- Models
-- Results
-- testedImages
-- utilis
-- train.py
-- denseFCN.py
-- test.py
# How to train the model
To train the network on your own dataset, do the following:
1. Clone this respository
2. Prepare your dataset with the following steps:
(1) Build the training dataset, the "./data/training_data/tamper" directory is used to store tampering images and the "./data/training_data/masks" directory is used to store ground-truths. 
(2) Build the validation dataset, the "./data/validation_data/tamper" directory is used to store tampering images and the "./data/validation_data/masks" directory is used to store ground-truths. 
3. Open up your terminal and navigate to the cloned repository
4. Type in the following:

