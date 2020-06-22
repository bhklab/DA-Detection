# DA-Detection

## About
As the field of radiomics grows, medical imaging datasets are becoming increasingly large therefor difficult to manually annotate and clean. In particular, metal objects in a patients body can create bright streak artifacts in computed tomography (CT) images which significantly obscure or alter the image. These artifacts are common in large datasets, and require manual annotation of each image in order to determine their location in the CT image volume. We have developed an automated detection algorithm for identifying the size and position of metal dental artifacts (DAs) in large head and neck cancer CT datasets.


This module consists of sinogram-based detection (SBD) and a convolutional neural network (CNN) which can be used independently as binary DA classifiers for CT volumes, or together as a three-class DA classifier. We have also developed a location detection algorithm, base on repeated thresholding of the images, in order to locate 2D slices containing DAs.

We have also included a thorough analysis of the correlations between hand-picked radiomic features and the size and location of DAs in CT volumes.


## Usage
### Installation
To download all requirements, clone the repository:
```
$ git clone https://github.com/bhklab/DA-Detection.git
```

Then, create a conda environment with the required packages:
```
$ conda env create -f environment.yml
```
