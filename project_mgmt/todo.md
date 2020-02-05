# Project Management

NOTE: Let's aim to comment on / strikethrough things we've completed, rather than deleting things entirely. Nice to have everything in a record.

## General Brainstorming

* What if we divided the challenge of predicting on a 1024x1024 image into smaller predictions, on multiple 512x512 images, for example, and then stiching mask / prediction images back into a full size render?

    * The simplest implementation of this might predict on a 2x2 quadrant of 512x512 windows on the original image, where there is no overlap of the windows. But more advanced implementations might predict on several 

* Scanning the scenes for training data extraction... What is the appropriate stride? Our MVP will likely have sliding windows with no overlap. What additional transformations of the training data are appropriate?

* How can we integrate metadata from the training metadata into the model? This might be an advanced modification.

## Priorities
*(markdown for automatically numbered lists is '1.' in front of every list item)*

1. Dev ops, specifically GCP deep learning VM container

1. Save a development training and test set

1. Train an MVP with the UNet-MobileNetV2-Pix2pix model.

1. Consider beginning to scan larger swaths of the training dataset

## Dev Ops, details

The training data too large fit into RAM, and we imagine that training on 1024^2 images will require optimization to run in reasonable compute time. We need to configure VMs both on cloud instances and perhaps in local docker containers for 

We also need to develop an efficient collaborative and versioning workflow to enable best developer practices.

* Configure VM instance on Google Cloud

    * Configure local deep learning VM containers for development on Alex / Thomas' local machines?

        * Also, could potentially run some workflows overnight with some spare machines that Thomas has.

* Where should training data be stored?

    * Does this require SQL-like storage? Are we saving Tensors as binaries? Numpy arrays?