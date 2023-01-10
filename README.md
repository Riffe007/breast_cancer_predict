# breast_cancer_predict
Uses a CNN-VAE to predict breast cancer from previous mamograms
# Breast Cancer Classification using CNN-VAE
This repository contains a convolutional neural network with a variational autoencoder (CNN-VAE) model for classifying breast cancer using regular mammogram results. The model is trained on a dataset of mammogram images and labels obtained from the RSNA Breast Cancer Detection competition on Kaggle.

How to use
To use the model, simply run the cnn_vae.py script. The script will load the data, train the model, and make predictions on the test set. It will then evaluate the performance of the model using the accuracy, precision, and recall metrics and print out the values of these metrics.

Model architecture
The CNN-VAE model consists of two main parts: an encoder and a decoder. The encoder consists of a series of convolutional and fully connected layers that map the input image to a latent vector in a lower-dimensional space. The decoder consists of a series of transpose convolutional and fully connected layers that map the latent vector back to the original image space.

The model is trained using the reconstruction loss, which measures the difference between the input image and the reconstructed image, and the KL divergence loss, which measures the difference between the latent distribution and a standard normal distribution.

Model architecture

Evaluation metrics
The performance of the model is evaluated using the following metrics:

Accuracy: The proportion of correct predictions made by the model.
Precision: The proportion of positive predictions that are actually positive.
Recall: The proportion of actual positive cases that are correctly predicted by the model.
Dependencies
The following libraries are required to run the model:

NumPy
Pandas
TensorFlow
Data
The data used to train the model is obtained from the RSNA Breast Cancer Detection competition on Kaggle. It consists of a large number of mammogram images and labels, which are stored in CSV files in the '/kaggle/input/rsna-breast-cancer-detection' folder.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
Hat tip to anyone whose code was used
Inspiration: RSNA Breast Cancer Detection competition on Kaggle
etc
