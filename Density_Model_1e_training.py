import joblib
from joblib import Parallel, delayed
from pyscf import gto, scf, dft
import datetime
import time
import psutil
import glob
import numpy as np
import json
import tensorflow as tf
from tqdm import tqdm
import sys
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.stats import kde
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import random
import seaborn as sns
from training_utils import *
from sklearn.preprocessing import MinMaxScaler

from e3nn import o3
from e3nn.o3 import Irreps, FullyConnectedTensorProduct
from e3nn.nn import FullyConnectedNet, Gate
import e3nn.nn as enn
import torch

np.random.seed(0)
random.seed(0)

###############################################

def create_model(): #new best
    model = tf.keras.Sequential()

    # Input Layer with GELU activation
    model.add(tf.keras.layers.Dense(50, activation='gelu', input_shape=(np.shape(train_x)[1],)))

    # Hidden Layers with skip connection and GELU activation
    for _ in range(3):#(3):
        model.add(tf.keras.layers.Dense(250, activation='gelu'))#250, activation='gelu'))

    # Final hidden layer without skip connection
    model.add(tf.keras.layers.Dense(5500, activation='gelu')) #5500

    # Output Layer (you might want to adjust the number of units and activation based on your needs)
    model.add(tf.keras.layers.Dense(134*134)) #NEEDS FLEXIBILITY

    # Learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-5,
        decay_steps=500,
        decay_rate=0.98,
        staircase=True
    )

    # Compile the model with Adam optimizer and L2 loss (mean squared error)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model


def create_model_big(): #new best
    model = tf.keras.Sequential()

    # Input Layer with GELU activation
    model.add(tf.keras.layers.Dense(50, activation='gelu', input_shape=(np.shape(train_x)[1],)))

    # Hidden Layers with skip connection and GELU activation
    for _ in range(3):#(3):
        model.add(tf.keras.layers.Dense(250, activation='gelu'))#250, activation='gelu'))

    # Final hidden layer without skip connection
    model.add(tf.keras.layers.Dense(5500*2, activation='gelu')) #5500

    # Output Layer (you might want to adjust the number of units and activation based on your needs)
    model.add(tf.keras.layers.Dense(44*3)) #NEEDS FLEXIBILITY

    # Learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=500,
        decay_rate=0.98,
        staircase=True
    )

    # Compile the model with Adam optimizer and L2 loss (mean squared error)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model

class TestCallback_old(tf.keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        self.test_mae = []
        self.best_weights = None
        self.best_epoch = 0
        self.best_test_mae = np.Inf

    def on_epoch_end(self, epoch, logs=None, verbose = 1):
        test_x, test_y = self.test_data
        test_predictions = self.model.predict(test_x)
        test_labels = test_y
        # test_predictions = np.reshape(test_predictions, (len(test_x), 1))
        test_mean_absolute_error = np.average(np.abs(test_predictions - test_labels))
        scaled_test_mae = convert_mae_to_original_scale(test_mean_absolute_error, min_deltas, max_deltas)
        self.test_mae.append(scaled_test_mae)

        if scaled_test_mae < self.best_test_mae:
            self.best_test_mae = scaled_test_mae
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch
            if verbose == 1:
                print(f'\nEpoch {epoch + 1}: New best Test MAE achieved: {scaled_test_mae:.4f} kcal/mol')
        if verbose == 1:
            print(f'\nEpoch {epoch + 1}: Scaled Test MAE: {scaled_test_mae:.4f} kcal/mol')


class TestCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_data, min_val, max_val):
        self.test_data = test_data
        self.min_val = min_val
        self.max_val = max_val
        self.test_mae = []
        self.best_weights = None
        self.best_epoch = 0
        self.best_test_mae = np.Inf

    def on_epoch_end(self, epoch, logs=None, verbose=1):
        test_x, test_y = self.test_data
        # print("test_x = ", np.shape(test_x))
        test_predictions = self.model.predict(test_x)
        test_labels = test_y
        # Unnormalize predictions and labels
        unnormalized_predictions = unnormalize_array(test_predictions, self.min_val, self.max_val)
        # Zero out components depending on number of orbitals
        unnormalized_predictions = adjust_predictions(test_x, unnormalized_predictions)
        unnormalized_labels = unnormalize_array(test_y, self.min_val, self.max_val)

        # Calculate MAE on unnormalized data
        #test_mean_absolute_error = np.average(np.abs(unnormalized_predictions - unnormalized_labels))
        #self.test_mae.append(test_mean_absolute_error)

        # Calculate and print average percent error
        unnormalized_predictions_norm = np.abs(unnormalized_predictions)
        unnormalized_labels_norm = np.abs(unnormalized_labels)
        RMAE_unormalized = calculate_rmae(unnormalized_labels, unnormalized_predictions)
        RMAE_normalized = calculate_rmae(test_labels, test_predictions)
        self.test_mae.append(RMAE_unormalized)

        # Check if this is the best MAE
        if RMAE_unormalized < self.best_test_mae:
            self.best_test_mae = RMAE_unormalized #test_mean_absolute_error
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch
            if verbose:
                print(f'\nEpoch {epoch + 1}: New best Test MARD achieved: {RMAE_unormalized:.4f} %')


        #### TESTING
        #outlier_indices = np.where(percent_errors[0] > 100)[0]
        #print("outlier_indices = ", np.shape(outlier_indices))
        #### TESTING
        if verbose:
            #print(f'Average Percent Error = {average_percent_error:.2f}%')
            #print(f'Average Unnormalized Percent Error = {average_unnormalized_percent_error:.2f}%')
            #print(f'RMAE_normalized = {RMAE_normalized:.2f}%')
            print(f'RMAE_unormalized = {RMAE_unormalized:.2f}%')

# Function to reshape the predictions and actuals back to original shape
def reshape_to_original(arr, new_shape=(44, 3)):
    return arr.reshape(-1, *new_shape)


def adjust_predictions(test_x, test_predictions):
    # Copy test_predictions to avoid modifying the original predictions
    adjusted_predictions = np.copy(test_predictions)

    # Iterate through each row in test_x to get the zero count
    for i in range(test_x.shape[0]):
        zero_count = int(test_x[i, -1])  # Get the last value as zero_count
        if zero_count > 0:
            # Set the last `zero_count` elements of the prediction to 0
            adjusted_predictions[i, -zero_count:] = 0
    return adjusted_predictions


############################################### Import Labels and Features
print("Loading labels")
#HF0_Density = np.load("Initial_matrices_10000.npz")["arr_0"] #[:1000]
#HF_Density =  np.load("Density_matrices_10000.npz")["arr_0"] #[:1000]
HF0_Density = np.load(f'Initial_matrices_1e_10000.npz')["arr_0"][:5000]
HF_Density = np.load(f'Density_matrices_1e_10000.npz')["arr_0"][:5000]

deltas = (HF_Density - HF0_Density) # kcal/mol/Angstrom
deltas = deltas.reshape(deltas.shape[0], -1)
print("Targets = ", np.shape(deltas))
print("Done!")

print("Loading features")
X = len(deltas)
zero_counts = []
for row in deltas:
    zero_count = np.sum(row == 0)
    zero_counts.append(zero_count)
#zero_counts_array = np.array(zero_counts).reshape(-1, 1)
#feature_arr = np.hstack((HF0_Density.reshape(X,-1), zero_counts_array))

zero_counts_array = np.array(zero_counts).reshape(-1, 1)
#o3.spherical_harmonics(l=1, x=dist_matrix, normalize=True, normalization='norm')
#feature_arr = o3.spherical_harmonics(l=2, x=torch.tensor(HF0_Density), normalize=True, normalization='norm')
feature_arr = HF0_Density
feature_arr = np.hstack((feature_arr.reshape(X,-1), zero_counts_array))
#feature_arr = HF0_Density.reshape(X,-1)

print("feature_arr = ", np.shape(feature_arr))
print("Done!")


print("HF0_Density = ",np.shape(HF0_Density))
print("HF_Density = ",np.shape(HF_Density))
print("deltas = ", np.shape(deltas))
print("HF_Density range = ", np.round(np.min(HF_Density),3),"->",np.round(np.max(HF_Density),3))
print("HF0_Density range = ", np.round(np.min(HF0_Density),3),"->",np.round(np.max(HF0_Density),3))
print("Done!")
############################################### Split the data

num_samples = len(feature_arr)
min_deltas = np.min(deltas)
max_deltas = np.max(deltas)


print("min deltas = ", min_deltas)
print("max deltas = ",	max_deltas)
print("min abs(deltas) = ", np.min(np.abs(deltas)))
print("max abs(deltas) = ",  np.max(np.abs(deltas)))

deltas_norm = normalize_array(deltas, min_deltas, max_deltas)

# print("HF0_Density = ", HF0_Density[0].reshape(1,-1))
# print("HF_Density = ", HF_Density[0].reshape(1,-1))
# print("deltas = ", deltas[0])
# print("deltas_norm = ", deltas_norm[0])
# print("norm(0) = ", normalize_array(0, min_deltas, max_deltas))

print("deltas_norm = ", np.shape(deltas_norm))



train_x, temp_x, train_y, temp_y = train_test_split(feature_arr, 
						deltas_norm, 
						test_size=0.2, 
						random_state=42)
test_x, val_x, test_y, val_y = train_test_split(temp_x, 
						temp_y,
						test_size=0.5, 
						random_state=42)

## Create an array of indices
#indices = np.arange(len(feature_arr))
## Split the indices
#train_indices, temp_indices = train_test_split(indices, test_size=0.2, random_state=42)
#test_indices, val_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)
#
#train_x = feature_arr[train_indices]
#train_y = deltas_norm[train_indices]
## Do the same for test_x, test_y, val_x, val_y
#test_x = feature_arr[test_indices]
#test_y = deltas_norm[test_indices]
#val_x = feature_arr[val_indices]
#val_y = deltas_norm[val_indices]


print("test_x = ", np.shape(test_x))
print("val_x = ", np.shape(val_x))
print("test_y = ", np.shape(test_y))
print("val_y = ", np.shape(val_y))




###############################################	Make Model

model = create_model()
model.compile(loss='MSE', optimizer='adam')
model.summary()

# Early stopping should probably monitor test loss instead 
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',patience=100,min_delta=.000001,verbose=1,
    restore_best_weights=True)

reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',factor=0.1,patience=50,verbose=1)

############################################### Train Model
#test_callback = TestCallback((test_x, test_y))
test_callback = TestCallback((test_x, test_y), min_deltas, max_deltas)
# Train the model with the TestCallback
history = model.fit(train_x, train_y,
                    epochs=10000,
                    batch_size=2**6,
                    validation_data=(val_x, val_y),
                    callbacks=[early_stopping_callback, reduce_lr_callback, test_callback])

# Set the model's weights to the best weights found during training
if test_callback.best_weights is not None:
    model.set_weights(test_callback.best_weights)

# Evaluate the model with the best weights
test_labels = test_y #np.reshape(test_y, (len(test_y), 1))
test_predictions = model.predict(test_x)

unnormalized_predictions = unnormalize_array(test_predictions, min_deltas, max_deltas)
# Zero out components depending on number of orbitals
unnormalized_predictions = adjust_predictions(test_x, unnormalized_predictions)
unnormalized_labels = unnormalize_array(test_labels, min_deltas, max_deltas)


# Calculate and print average percent error
unnormalized_predictions_norm = np.abs(unnormalized_predictions)
unnormalized_labels_norm = np.abs(unnormalized_labels)
MARD_unormalized = calculate_rmae(unnormalized_labels, unnormalized_predictions)
print("MARD with best weights = " + str(MARD_unormalized) + "%")


### Save model
filename = './saved_data/Density_1e_model_delta_norm_' + str(np.round(MARD_unormalized,3)) + '%_' + str(min_deltas) + '_max_' + str(max_deltas) + '.h5'
model.save(filename)
sys.exit()

##################################### PLOT
#predictions = model.predict(test_x)
# Reshape the first prediction and its actual values back to their original shape (44, 3)
prediction_reshaped = test_predictions[0].reshape(134, 134)
actual_reshaped = test_y[0].reshape(134, 134)  # Assuming test_y is still in the shape (10, 132)

#print("prediction_reshaped = ", prediction_reshaped)
#print("actual_reshaped = ", actual_reshaped)


# Determine the common scale for both heatmaps
vmin = min(np.min(prediction_reshaped), np.min(actual_reshaped))
vmax = max(np.max(prediction_reshaped), np.max(actual_reshaped))

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

# Predicted heatmap
cax_1 = axs[0].imshow(prediction_reshaped, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
axs[0].set_title('Prediction')

# Actual heatmap
cax_2 = axs[1].imshow(actual_reshaped, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
axs[1].set_title('Actual')

# Create a colorbar with a fixed aspect, using the figure (not the axes)
fig.colorbar(cax_2, ax=axs.ravel().tolist(), orientation='vertical')

# Hide x, y ticks for clarity
axs[0].set_xticks([])
axs[0].set_yticks([])
axs[1].set_xticks([])
axs[1].set_yticks([])

# Save the figure
plt.savefig("./saved_data/test_predict_Density_heatmap_comparison.png")
plt.show()
