import numpy as np
import glob
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Define a function to build the CNN-VAE model
def build_cnn_vae(latent_dim=2):
    # Define the encoder architecture
    inputs = Input(shape=(28, 28, 1))
    x = Conv2D(16, 3, strides=2, padding='same', activation='relu')(inputs)
    x = Conv2D(32, 3, strides=2, padding='same', activation='relu')(x)
    x = Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = Flatten()(x)
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)

    # Sample the latent vector
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(z_log
# Load the data
df = pd.read_csv('/kaggle/input/rsna-breast-cancer-detection/train.csv')

# Preprocess the data
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values
X = X.reshape(-1, 28, 28, 1)
X = X / 255.

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tune the hyperparameters
latent_dims = [2, 4, 8, 16, 32]
accuracies = []
precisions = []
recalls = []

for latent_dim in latent_dims:
    # Build the model
    model = build_cnn_vae(latent_dim=latent_dim)

    # Compile the model
    model.compile(optimizer='adam', loss=binary_crossentropy)

    # Fit the model
    model.fit(X_train, X_train, epochs=5)

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred = np.round(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Print the evaluation metrics
    print(f'Latent dimension: {latent_dim}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')

    # Store the evaluation metrics
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)

# Plot the evaluation metrics
import matplotlib.pyplot as plt

plt.plot(latent_dims, accuracies, label='Accuracy')
plt.plot(latent_dims, precisions, label='Precision')
plt.plot(latent_dims, recalls, label='Recall')
plt.xlabel('Latent Dimension')
plt.ylabel('Metric')
plt.legend()
plt.show()

