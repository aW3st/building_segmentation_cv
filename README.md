# building_segmentation_cv
## Image Segmentation using Pytorch - Developed for DrivenData's Open Cities AI Distaster Relief Competition
Authors: Thomas Kavanagh and Alex Weston

This repo holds code used in our submission Driven Data's Open Cities AI challenge. We use a Resnet backbone combined with an Joint Pyramidal Upsampling module and EncNet Head. 

To reproduce our results, first run scan scenes.py to extract 1024x1024 tiles from each raster provided by Driven Data and upload them to a Google Cloud storage bucket. This may take several days, depending on processing power and internet speeds. 

Then download the training data (either locally or on a remote instance) run train.py to train the model on the downloaded tiles. After the model is done training, run evaluate.py to predict on the test set. 
