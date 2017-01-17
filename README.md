##Fusion CNN in mxnet
### How to use
1. cd google_model and ./DownloadModel.py
2. python cnn_predict to get the pretrained data from Inception
3. python train_ucf101.py to train the model

###Description
This is a simple method to train the dataset of UCF101 but can get a good result.
We get three pictures from each video in UCF101 then predict them by Inception. 
At last we use the combination of the features of three pictures as the input data.

###Result
This experiment shows that the UCF101 is not only a dataset of actions, but also a dataset of Scenes. So we can use a simple CNN to get a good result.
