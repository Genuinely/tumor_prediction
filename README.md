# Tumor Growth Prediction as Stochastic Video Prediction

Modified from: https://github.com/edenton/svg

# To Run
First, make a new virtual env using requirements.txt.

1. Download tumor dataset and put into a subfolder titled 35_um_data_100x100x48_niis
1. Run  `train.py` with appropriate paramaters which will output model.pth
1. Place the model.pth path as one of the params for `generate.py`

# Dataset
The main data set consists of CT image scans of the leg bones of mice. 
Each data point is a stack of images of dimension 400 x 400$ x 300 where 300 is the number of slices in the region of interest. The data is processed by SITK which registers the CT scan and incorporates spatial information rather than channels. Thus, we can treat each 2D slice as a gray-scale, 1 channel image. There are 251 mice and approximately 100 of the mice have no tumor. Each mouse has 3-6 weeks worth of 3D stacks and the scans are done around once a week. We only consider mice samples with 4 weeks of data points to ensure temporal consistency. In total, we expect around 1000 representative 2D images. The data is loaded dynamically, where the seed represents the data point iderntification. The data set is sourced from the University of Denver.

# Initial Experimental Results
Method | PSNR | SSIM
------------ | ------------- | ------|
SVG | **24.27** | **0.860**
3D-GAN | 23.20 | 0.813

# Architecture Details
The architecture follows the variational auto encoder (VAE) formalism to generate CT scans. The encoder/decoder follow a Deep Convolutional GAN Discriminator/Generater archiecture respectively. An LSTM is used to encode temporal information. 

For more details, see initial report in ``report`` folder.
