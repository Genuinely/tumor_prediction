# tumor_prediction

Modified from: https://github.com/edenton/svg

# To Run
First, make a new virtual env using requirements.txt.

1. Download tumor dataset and put into a subfolder titled 35_um_data_100x100x48_niis
1. Run  `train.py` with appropriate paramaters which will output model.pth
1. Place the model.pth path as one of the params for `generate.py`

# Architecture Details
The architecture is a variational auto encoder to generate CT scans. The encoder/decoder follow a Deep Convolutional GAN Discriminator/Generater archiecture respectively. An LSTM is used to encode temporal information. 
