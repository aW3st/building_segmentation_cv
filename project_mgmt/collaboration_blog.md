# Collaboration Log

Record of what's happened, who's done what, and why in this collaboration. At a high level, not small diffs in code. Written in 3rd person, for now.

// double slashes is for dirty notes that need to be converted into blog style.

See the first few paragraphs without slashes for an example of the style.

### ETL / Early steps

// Talk about what brought us to this project / motivation, our early objectives for the collaboration
// Also want to write about the benefits of working remote

Alex investigated how to use STAC files and load them iteratively from the hosted S3 bucket provided by the DrivenData challenge. Geospatial data requires proper consideration of coordinate systems across feature images and target masks, and Alex overcame early hurdles in working with CRS data and associated geometries required to align data. He also wrote some early visualization code to render segmentation mask overlays. He loaded his early ETL into a Colab notebook, which got us off to a running start.

In our first meeting, we began by going over Alex’s early ETL notebook out loud. Thomas hadn’t yet handled the geoPandas or Rasterio libraries, it was helpful for everyone to review. We refined the code as we went.

After becoming famliar enough with manipulating  geospatial data, we switched gears to a strategic overview of deep learning for image segmentation. 

Thomas had reviewed the image segmentation example from Tensorflow, and learned about the U-Net architecture, which we walked through together. The Tensorflow’s image segmentation tutorial notebook we found walked through an implementation the UNet Encoder-Decoder architecture. Not all the layers are trained from scratch, in the notebooks example: a subset of pre-trained encoder output layers are taken from a base mode: in their example, the MobileNetV2 (other pretrained models can be used). The pretrained encoder layers (AKA downsampler or "down stack") are followed sequentially by trainable decoder layers that gradually up-samples the encoder layers into a image mask with the original dimensions as the input image. Tensorflow's own Pix2pix convolutional layer utility package lends high-level apis for constructing these encoder layers by name.

// Thomas turned the Colab code into modules and placed things in a github repo

// Synced repo with a new Google Cloud projects

// Alex researched deployment tech ... mess