# Encrypted Skin Cancer Detection

## Project Overview

The purpose of this project is to demonstrate how simple it is to provide encrypted skin cancer detection with [TF Encrypted](https://github.com/tf-encrypted/tf-encrypted). We released this code in the context of the blog post "A Path to Sub-Second, Encrypted Skin Cancer Detection".

Only three steps are required to provide private predictions:
- Train your deep learning model with the framework of your choice.
- Export the model to a [protocol buffer file (protobuf)](https://www.tensorflow.org/guide/extend/model_files#protocol_buffers).
- Load the protobuf in TF Encrypted to start serving private predictions.

## Install

This project requires **Python 3.6** and the following Python libraries installed:

- [TF Encrypted](https://github.com/tf-encrypted/tf-encrypted)
- [TensorFlow](https://github.com/tensorflow/tensorflow)
- [fastai](https://github.com/fastai/fastai)

You can easily install these libraries with:
```
pip install -r requirements.txt
```

We recommend installing [Anaconda](https://docs.anaconda.com/anaconda/user-guide/getting-started/), a pre-packaged Python distribution that can be used to install and manage all of the necessary libraries and software for this project.

If you experience any issues installing fastai, you can consult the [documentation](https://docs.fast.ai/).

## Run Private Predictions
If you want, you can run a private prediction right away. We supply a protobuf containing the model weights and the neural network architecture in this [GCS storage bucket](https://storage.googleapis.com/tfe-examples-data/skin_cancer/skin_cancer_model.pb). For demonstration purposes, we also provide a pre-processed skin lesion image as a pickled Numpy array: `skin_cancer_image.npy`.

To perform a private prediction locally, you can simply run this command in your terminal:
```
MODEL_URL=https://storage.googleapis.com/tfe-examples-data/skin_cancer/skin_cancer_model.pb
./private_predict --protocol_name securenn \
      --model $MODEL_URL \
      --input-file skin_cancer_image.npy \
      --batch-size 1 \
      --iterations 1
```

Alternatively, you can download the protobuf manually and supply it directly to the CLI:
```
./private_predict --protocol_name securenn \
      --model skin_cancer_model.pb \
      --input-file skin_cancer_image.npy \
      --batch-size 1 \
      --iterations 1
```


When running this command, a TF Encrypted graph is automatically created using the GraphDef inside the supplied protobuf. The original model weights are secret shared during this conversion process. Finally, data provided through the `--input-file` arg is secret shared before the model performs the secure inference.

## Training

After downloading the data from [here](https://storage.googleapis.com/tfe-examples-data/skin_cancer/data.zip), you can train the model by running the notebook `training.ipynb`. The purpose of this model is to classify correctly the malignant skin lesions (Melanoma) images. For this project, we trained the model with the [fastai library](https://github.com/fastai/fastai) to quickly experiment with different architectures, data augmentation strategies, cyclical learning rates, etc. This was mainly due to the availability of the cyclical learning rate finder -- training with TensorFlow, Keras, or a library like [tensor2tensor](https://github.com/tensorflow/tensor2tensor) is usually preferred.

Once trained, the model gets exported to a protobuf file with the function `export_to_pb` from the utility file `transform_to_pb.py`. This function first transforms the PyTorch model into a TensorFlow graph and then exports it to protobuf.


## Data

For this project, we have gathered 1,321 skin lesions images labeled as Melanona and 2,229 skin lesion images labeled as Nevi or Seborrheic Keratoses. This data were collected from the [2017 ISIC Challenge on Skin Lesion Analysis Towards Melanoma Detection](https://github.com/udacity/dermatologist-ai) and additional data where collected from [ISIC archive](https://isic-archive.com/#images) using the instructions [here](https://github.com/GalAvineri/ISIC-Archive-Downloader). You can download the data used for this project from [here](https://storage.googleapis.com/tfe-examples-data/skin_cancer/data.zip).

>Codella N, Gutman D, Celebi ME, Helba B, Marchetti MA, Dusza S, Kalloo A, Liopyris K, Mishra N, Kittler H, Halpern A. "Skin Lesion Analysis Toward Melanoma Detection: A Challenge at the 2017 International Symposium on Biomedical Imaging (ISBI), Hosted by the International Skin Imaging Collaboration (ISIC)". arXiv: 1710.05006 [cs.CV] Available: https://arxiv.org/abs/1710.05006
