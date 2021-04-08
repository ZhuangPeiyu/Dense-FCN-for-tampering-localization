# Image Tampering Localization Using a Dense Fully Convolutional Network

# Overview
This is the implementation of the method proposed in "Image Tampering Localization Using a Dense Fully Convolutional Network" with tensorflow(1.10.0, gpu version). The aim of this repository is to achieve image tampering localization.
# Network Architecture
![image](https://github.com/ZhuangPeiyu/Dense-FCN-for-tampering-localization/blob/master/networkArchitecture/158b993b1ea5a0b7ee6e460376e3ce2.png)
# Files structure of Dense-FCN-for-tampering-localization
- Models
- Results
- testedImages
- utilis
- train.py
- denseFCN.py
- test.py
- demo.py

# The pre-trained model path
The model trained with Dresden script dataset and fine-tuned with 56 NIST images was uploaded in Dropbox: https://www.dropbox.com/sh/0hkeenrfazob3ci/AAAa6X2hhDnj04LfAR2mSKi9a?dl=0
# How to run
## Test with the trained model

python3 demo --filename test.py

## Train the model from scratch
python3 demo --filename train.py

# Citation
If you use our code please cite: Peiyu Zhuang, Haodong Li, Shunquan Tan, Bin Li, Jiwu Huang, "Image Tampering Localization Using a Dense Fully Convolutional Network," submitted to IEEE Transactions on Information Forensics and Security, 2020.

