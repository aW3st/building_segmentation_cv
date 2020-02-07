# Docker Container Startup, and Sharing

Before starting this, read the tutorial [here](https://cloud.google.com/ai-platform/deep-learning-containers/docs/getting-started-local), and follow instructions up until the Create Container section.

## Creating Our Tensorflow docker container

So, the objective here is to run and contain our docker instance to mainly run Tensorflow models and view results on Tensorboard. Our data ingest operations are less important here. For that and other work, we might as well just develop locally with our favorite IDE.

The reason we would use a container at all is that we can practice running toy models locally on a container, and then when we feel confident with that or we want to run a larger dataset, we can export the container image, clone our repo into that container, and then start running the model on the cloud.

We will use the Google configured tensorflow docker image to run the container. Its a large image, just fair warning. The uri we pass will to docker for a version 2.x deep learning image is `gcr.io/deeplearning-platform-release/tf2-cpu.2-0`. Note that this is the CPU instance for compatibility with our local machines. We may want to change to the GPU instance for the cloud VM. The first time we ask docker to create a container from this hosted image, we will download the image locally. This only happens once.

## Mounting our local repo into the container

Docker containers need explicit access to file storage on the host machine, if that's what you're using. Eventually we will could host the data from google cloud storage. But until then, we can store the data locally (and potentially we could do this on the cloud as well).

The most straightforward way I've found to grant local file access this is to mount a docker container directly into your local git repo. This allows you full write and read access to the repo and any data contained in it.

The container as outlined below will serve a JupyterLab instance on `localhost:8080`. It integrates very neatly with Github and you can push commits direcly from the web IDE interface.

Below is the command I used to initialize the docker container. `tf_container` is the custom name I gave it, and isn't required.

The two-aspect argument following the -v flag (volume), separated by a colon, refers to the host and container mount points respectively. The `/home` folder is a directory within the container itself, and allows the repo files to pop up immediately in the sidebar.

So, the one-liner to get a docker container up and running is the following. The path to your local repo must be an absolute one to directly mount the directory into the container. Otherwise, a proxy volume is created. This way of creating a volume is different: a volume can still be linked to data on your hosted machine, but it will need to be separately bound to your host data. But now that volume can be connected to other containers. Not needed in our case because only one container is mounting to the repo.

```
$ docker run -d -p 8080:8080 --name tf_container -v ~/ABSOLUTE_PATH_TO_REPO/building_segmentation_cv:/home gcr.io/deeplearning-platform-release/tf2-cpu.2-0
```

NOTE: The container has read and write access to the original files on this repo. Git tracks those changes as usual.