# tumor_prediction

Modified from: https://github.com/edenton/svg

First, make a new virtual env w/ requirements.txt.

1. download tumor dataset and put into a subfolder titled 35_um_data_100x100x48_niis
1. run  `train.py` with appropriate paramaters which will output model.pth
1. put the model.pth path as one of the params for `generate.py`
