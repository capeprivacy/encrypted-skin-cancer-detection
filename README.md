# Encrypted Skin Cancer Detection 

## Project Overview

The purpose of this project is to demonstrate how simple it is to provide encrypted skin cancer detection with [TF Encrypted](https://github.com/tf-encrypted/tf-encrypted). We released this code in the context of the blog post "A Path to Sub-Second, Encrypted Skin Cancer Detection".

Only three steps are required to provide private predictions:
- Train your deep learning model with the framework of your choice. 
- Export the model to a [protocol buffer file (pb)](https://www.tensorflow.org/guide/extend/model_files#protocol_buffers).
- Load the pb file in TF Encrypted to start serving private predictions. 

## Install

This project requires **Python 3.6** and the following Python libraries installed:

- [TF Encrypted](https://github.com/tf-encrypted/tf-encrypted)
- [TensorFlow](https://github.com/tensorflow/tensorflow) 
- [fastai](https://github.com/fastai/fastai)

You can easily install these libraries with:
```
pip install -r requirements.txt
```

We recommend installing [Anaconda](https://docs.anaconda.com/anaconda/user-guide/getting-started/), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project.

If you experience any issues installing fastai, you can consult the documentation [here](https://docs.fast.ai/).

## Run Private Predictions
If you want, you can run a private prediction right away. We have saved the pb file containing the model weights and the neural network architecture here. We have also saved a pre-processed skin lesion image as a numpy file in this repo: `skin_cancer_image.npy`.

To perform a private prediction locally, you can simply run  this in your terminal:
```
./private_predict --protocol_name securenn \
      --model_name skin_cancer_model.pb \
      --input_file skin_cancer_image.npy \
      --batch_size 1 \
      --iterations 1
```

When running this command line, a TF Encrypted TensoFLow graph gets automatically created based on the pb file. During this process the model weights and input data gets secret shared. 

## Training 

After downloading the data from here, you can train the model by running the notebook `skincancer_vgg16like.ipynb`. The purpose of this model is to classify correctly the malignant skin lesions (Melanoma) images. For this project, we trained the model with the fastai library to quickly experiment with different architectures, data augmentation strategies, cyclical learning rates, etc. But of course this model could have been trained with TensorFLow. 

Once trained, the model gets exported to a pb file with the function `export_to_pb` from the utils file `transform_to_pb_file.py`. This function first transforms the PyTorch model into a TensorFlow graph then exports it to a pb file. 


## Data

For this project, we have gathered 1,321 skin lesions images labeled as Melanona and 2,229 skin lesion images labeled as Nevi or Seborrheic Keratoses. This data were collected from the [2017 ISIC Challenge on Skin Lesion Analysis Towards Melanoma Detection](https://github.com/udacity/dermatologist-ai) and additional data where collected from [ISIC archive](https://isic-archive.com/#images) using the instructions [here](https://github.com/GalAvineri/ISIC-Archive-Downloader).
