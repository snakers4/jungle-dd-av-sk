Jungle video trap challenge solution by AVeysov and SKolbachev
==============================

- Classifying animals using jungle trap videos (200k+, 1TB) with 90%+ accuracy with CNNs;
- Also visit out [blog](https://spark-in.me/tag/group-data-science) and [channel](https://t.me/snakers4) (seriously!);

To replicate our final result from [here](https://www.drivendata.org/competitions/49/deep-learning-camera-trap-animals/leaderboard/) you need more or less to follow these steps:

1. Download the dataset from [here](https://www.drivendata.org/competitions/49/deep-learning-camera-trap-animals/) - (see some hints below);
2. Build the environment via the below Dockerfile;
3. Extract the features and metadata from the videos;
4. Run the final models;


Project Organization
------------

    ├── LICENSE
    ├── README.md           <- The top-level README for developers using this project.
    ├── data
    │   ├── interim         <- Extracted features and metadata
    │   ├── micro           <- The original micro dataset (64x64 videos)
    │   └── raw             <- The original unpacked 1TB raw full size video dataset
    │
    ├── models              <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks           <- Jupyter notebooks (provided just fyi for completeness)
    │
    ├── reports             <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures         <- Generated graphics and figures to be used in reporting
    │
    ├── Dockerfile          <- The Dockerfile to build the environment
    │
    ├── src                 <- Source code for use in this project.
    │   ├── __init__.py     <- Makes src a Python module
    │   │
    │   ├── data            <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features        <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models          <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization   <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── test_environment.py <- A set of small scripts to test the environment

Downloading data
------------

Download and unpack the following datasets to their respective folders from [here](https://www.drivendata.org/competitions/49/deep-learning-camera-trap-animals/data/):
- data/raw - the full 1TB dataset
- data/micro - micro dataset

Download the annotation files and micro dataset using the below python3 script:

```
import collections

file_dict = collections.OrderedDict()
file_dict['data/submission.csv'] = 'https://s3.amazonaws.com/drivendata/data/49/public/submission_format.csv'
file_dict['data/train.csv'] = 'https://s3.amazonaws.com/drivendata/data/49/public/train_labels.csv'
file_dict['data/micro/micro_chimps.tgz'] = 'https://s3.amazonaws.com/drivendata-public-assets/micro_chimps.tgz'

for file,url in file_dict.items():
    url_q = "'" + url + "'"
    ! wget --continue --no-check-certificate --no-proxy -O $file $url_q

```

For the full dataset you will have to be more creative, I recommend using 

Setting up the environment
------------

Key prerequisites:
- Ubuntu 16.04;
- CUDA + CUDNN + Nvidia GPU;
- Docker and nvidia-docker;

Make sure that you are familiar with Docker and building docker images from Dockerfiles and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

Use the provided Dockerfile to build an environment.
Change the **ENTER_YOUR_PASS_HERE** placeholder to the desired root password (is necessary if you want to ssh remotely into the container).

If you are not familiar with Docker, consider following / reading these materials:
- [Docker](https://towardsdatascience.com/how-docker-can-help-you-become-a-more-effective-data-scientist-7fc048ef91d5) for data science;
- A series of posts on our channel with useful links [1](https://t.me/snakers4/1476) [2](https://t.me/snakers4/1479) (if you do not speak Russian - just follow links - they ultimately link to Enlish articles mostly);


Running the environment
------------

1. Build the container;
2. Make sure you have all the necessary ports exposed (i.e. 8888 for jupyter and 6006 for tensorboard);
3. Use the below commands to run the container;
4. Use ```docker exec -it --user root CONTAINER_ID /bin/bash``` to ssh into docker locally;
5. You may also add ssh port pass-though (into EXPOSE statement and into docker run) and ssh into container remotely. Also you may change password auth to ssh-key;

For nvidia-docker use:

```
nvidia-docker run -it -v /PATH/TO/YOUR/FOLDER:/home/keras/notebook -p 8888:8888 -p 6006:6006 --shm-size 8G YOUR_DOCKER_IMAGE_NAME --runtime=nvidia 
```

For nvidia-docker 2 use:

```
docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -it -v /home/av:/home/keras/notebook -p 8888:8888  -p 6006:6006 --shm-size 8G av_image    
```

Testing the environment
------------

1. Run test_environment.py - and make sure it works;
2. The only key major change that occured recently is Pytorch 0.3 being released, but this should not cause any issues - but be cautious;
3. Not all keras + tf versions are friednly with each other;


Extracting features
------------