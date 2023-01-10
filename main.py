import numpy as np
import glob
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.utils import plot_model

# Find all the CSV files in the folder
filenames = glob.glob('/kaggle/input/rsna-breast-cancer-detection/*.csv')

# Load the data from the CSV files and concatenate it into a single dataframe
df = pd.concat([pd.read_csv(f) for f in filenames])

# Split the data into features and target
X = df.drop('diagnosis', axis=1).values
y = df['diagnosis'].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the latent dimension
latent_dim = 2
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
    return z_mean + tf.exp(z_log_var / 2) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])

# Define the decoder architecture
latent_inputs = Input(shape=(latent_dim,))
x = Dense(7*7*64, activation='relu')(latent_inputs)
x = Reshape((7, 7, 64))(x)
x = Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
x = Conv2DTranspose(16, 3, strides=2, padding='same', activation='relu')(x)
outputs = Conv2DTranspose(1, 3, strides=2, padding='same', activation='sigmoid')(x)

# Define the full model
vae = Model(inputs, outputs)

# Define the encoder model (continued)
encoder = Model(inputs, z_mean)

# Define the decoder model
decoder_inputs = Input(shape=(latent_dim,))
_ = decoder(decoder_inputs)
decoder = Model(decoder_inputs, _)

# Define the loss function and optimizer
def kl_loss(z_mean, z_log_var):
    return -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)

loss = binary_crossentropy(K.flatten(inputs), K.flatten(outputs))
loss += kl_loss(z_mean, z_log_var)
vae.add_loss(loss)
vae.compile(optimizer='adam')

# Train the model
vae.fit(X_train, epochs=10, batch_size=32)

# Make predictions on the test set
y_pred = vae.predict(X_test)

# Evaluate the performance of the model
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Print the evaluation metrics
print(f'Accuracy: {acc:.2f}')
print(f'Precision: {prec:.2f}')
print(f'Recall: {recall:.2f}')



