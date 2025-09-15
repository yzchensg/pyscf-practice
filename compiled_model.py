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
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.stats import kde
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import random
import seaborn as sns
from training_utils import *

np.random.seed(0)
random.seed(0)

## Compilation of the full model
## Delta learning to generate energies from DFT density matrices and Fock matrices


# Where is the Density matrix? and their B3LYP counterparts?

# Where is the Fock matrix? and their B3LYP counterparts?

# Where is the target energy?
targets_path = "./RANDOM_NPZ/target_array_9235.npz" #np.load("target_array_5000.npz", mmap_mode='r')["arr_0"] #[:1000]
feature_scalar_path = "./RANDOM_NPZ/feature_scalar_array_9235.npz" #np.load("feature_scalar_array_5000.npz", mmap_mode='r')["arr_0"] #[:1000]
feature_matrix_path = "./RANDOM_NPZ/feature_matrix_array_9235.npz"

################################################### 
# Define the model architecture
###################################################
def create_model(train_x): #new best
    model = tf.keras.Sequential()

    # Input Layer with GELU activation
    model.add(tf.keras.layers.Dense(50, activation='gelu', input_shape=(np.shape(train_x)[1],)))

    # Hidden Layers with skip connection and GELU activation
    for _ in range(3):#(2):
        model.add(tf.keras.layers.Dense(50, activation='gelu'))

    # Final hidden layer without skip connection
    model.add(tf.keras.layers.Dense(5100, activation='gelu'))

    # Output Layer (you might want to adjust the number of units and activation based on your needs)
    model.add(tf.keras.layers.Dense(1))

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

class TestCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        self.test_mae = []
        self.best_weights = None
        self.best_epoch = 0
        self.best_test_mae = np.inf

    def on_epoch_end(self, epoch, logs=None, verbose = 1):
        test_x, test_y = self.test_data
        test_predictions = self.model.predict(test_x)
        test_labels = np.reshape(test_y, (len(test_y), 1))
        test_predictions = np.reshape(test_predictions, (len(test_x), 1))
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

############################################### Import Labels and Features

def load_labels_and_features(targets_path, feature_scalar_path, feature_matrix_path):
        
    print("Loading labels")
    Targets = np.load(targets_path, mmap_mode='r')["arr_0"] #[:1000] #np.load("target_array_5000.npz", mmap_mode='r')["arr_0"] #[:1000]
    print("Done!")

    print("Loading features")
    feature_scalar_arr = np.load(feature_scalar_path, mmap_mode='r')["arr_0"]#[:1000]
    feature_arr = np.load(feature_matrix_path, mmap_mode='r')["arr_0"]#[:1000]

    print("feature_scalar_arr = ", np.shape(feature_scalar_arr))
    D = feature_arr[:, 0, :, : ]
    F = feature_arr[:, 2, :, : ]

    multiplied_matrices_diag = np.array([np.diag(mat)
                                        for mat in D*F]).reshape(len(feature_scalar_arr), -1)
    D_13 = ((np.abs(D)
            )**(1/3)).reshape(len(feature_scalar_arr), -1)
    D_43 = ((np.abs(D)
            )**(4/3)).reshape(len(feature_scalar_arr), -1)
    feature_arr = np.hstack((D_13,
                            D_43,
                            multiplied_matrices_diag))

    F = 0
    D = 0

    # CALCULATING THE DELTAS
    print("Calculating Deltas")
    HF_SCF_arr = H_kcal_mol(feature_scalar_arr[:,0,4] - (feature_scalar_arr[:,1,4] + feature_scalar_arr[:,2,4]))
    B3LYP_SCF_arr = H_kcal_mol(Targets[:,0,0] - (Targets[:,1,0] + Targets[:,2,0]))
    deltas = B3LYP_SCF_arr - HF_SCF_arr




    print("HF_SCF_arr = ",np.shape(HF_SCF_arr))
    print("B3LYP_SCF_arr = ",np.shape(B3LYP_SCF_arr))
    print("delta range = ", np.round(np.min(deltas),3),"->",np.round(np.max(deltas),3), "kcal/mol")
    print("B3LYP_SCF_arr range = ", np.round(np.min(B3LYP_SCF_arr),3),"->",np.round(np.max(B3LYP_SCF_arr),3), "kcal/mol")
    print("HF_SCF_arr range = ", np.round(np.min(HF_SCF_arr),3),"->",np.round(np.max(HF_SCF_arr),3), "kcal/mol")
    print("Done!")

    num_samples = len(feature_arr)
    min_deltas = np.min(deltas)
    max_deltas = np.max(deltas)

    print("min = ", min_deltas)
    print("max = ",	max_deltas)

    deltas_norm = normalize_array(deltas, min_deltas, max_deltas)



    # Splitting the data
    train_x, temp_x, train_y, temp_y = train_test_split(feature_arr,
                                                        deltas_norm,
                                                        test_size=0.2,
                                                        random_state=42)

    test_x, val_x, test_y, val_y = train_test_split(temp_x, temp_y,
                                                    test_size=0.5,
                                                    random_state=42)
    
    return train_x, train_y, val_x, val_y, test_x, test_y, min_deltas, max_deltas

###############################################	Make Model
def train_model(
        train_x, train_y,
        val_x, val_y,
        test_x, test_y,
        min_deltas, max_deltas,
        model_path=r'./' ,
        plot_path = r'./',
):

    model = create_model(train_x)
    model.compile(loss='MSE', optimizer='adam')
    model.summary()

    # Early stopping should probably monitor test loss instead 
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',patience=100,min_delta=.001,verbose=1,
        restore_best_weights=True)

    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',factor=0.1,patience=50,verbose=1)

    ############################################### Train Model
    test_callback = TestCallback((test_x, test_y))
    # Train the model with the TestCallback
    history = model.fit(train_x, train_y,
                        epochs=1000,
                        batch_size=2**6,
                        validation_data=(val_x, val_y),
                        callbacks=[early_stopping_callback, reduce_lr_callback, test_callback])

    # Set the model's weights to the best weights found during training
    if test_callback.best_weights is not None:
        model.set_weights(test_callback.best_weights)

    # Evaluate the model with the best weights
    test_labels = np.reshape(test_y, (len(test_y), 1))
    test_predictions = np.reshape(model.predict(test_x), (len(test_x), 1))
    mean_absolute_test_err = convert_mae_to_original_scale(np.average(np.abs(test_predictions - test_labels)), min_deltas, max_deltas)
    print("MAE with best weights = " + str(mean_absolute_test_err), "kcal/mol") 


    ### Save model
    filename = model_path + 'binding_model_delta_norm_' + str(np.round(mean_absolute_test_err,3)) +'_RINDO_kcal_mol.h5'
    model.save(filename)

    ############################################### Plot And save results
    # Assuming test_labels and test_predictions are already defined as shown in your code.
    # Calculate the errors
    errors = convert_mae_to_original_scale((test_predictions) - (test_labels), min_deltas, max_deltas)
    # Count the number of values with magnitude greater than 1
    print("Number of errors greater than 1 kcal/mol = ",np.sum(np.abs(errors) > 1))
    print("Percent of errors greater than 1 kcal/mol = ",np.round(np.sum(np.abs(errors) > 1)/len(errors)*100,2), "%")

    # Plotting the energies vs. errors?
    plt.figure(figsize=(12, 6))
    plt.scatter(test_predictions, errors, alpha=0.5)
    # Highlight the region within 1 kcal/mol from the red line
    plt.fill_betweenx(y=[-1, 1], x1=min(test_predictions), x2=max(test_predictions), color='green', alpha=0.2)
    plt.title('Predicted Energies vs. Error')
    plt.xlabel('Predicted Energies (kcal/mol)')
    plt.ylabel('Error (kcal/mol)')
    plt.axhline(0, color='red', linewidth=1)  # Adds a horizontal line at error=0 for reference
    plt.ylim(-5,5)
    plt.grid(True)
    # Save the plot
    plt.savefig(plot_path + "/predicted_energies_vs_error.png")
    plt.close()  # Close the plot to free up memory


############################################### Plot with r^2
def evaluate_r_squared(model, 
                       test_x, test_y, 
                        train_x, train_y,
                       min_deltas, max_deltas,
                       plot_path = r'./'):
    # Evaluate the model
    test_labels = np.reshape(test_y, (len(test_y), 1))

    # Use the predict method on the testing data
    test_predictions = np.reshape(model.predict(test_x), (len(test_x), 1))
    train_predictions = np.reshape(model.predict(train_x), (len(train_x), 1))


    # Fit the testing data
    test_slope, test_intercept, test_r_value, test_p_value, test_std_err = stats.linregress(np.array(test_predictions).flatten(),np.array(test_labels).flatten())
    test_fit_line = lambda x2 : test_slope*x2 + test_intercept

    # Calculate the R^2 value and store it as a string
    r_squared = test_r_value ** 2
    r_squared_str = f'$R^2 = {r_squared:.3f}$'  # Format to 2 decimal places
    print("r_squared = ", r_squared)

    # Sample data: x is linearly spaced numbers between 0 and 10, y is x with added noise
    x = convert_mae_to_original_scale(test_y, min_deltas, max_deltas)
    x2 =convert_mae_to_original_scale(train_y, min_deltas, max_deltas)
    y = convert_mae_to_original_scale(test_predictions.flatten(), min_deltas, max_deltas)
    y2 = convert_mae_to_original_scale(train_predictions, min_deltas, max_deltas)
    # Setting up Seaborn style
    sns.set(style="whitegrid", rc={"grid.linestyle": "--"})

    # Initialize the figure
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, color='lightblue', edgecolors='black', s=40, label=f'Testing set ({r_squared_str})', alpha = 0.5)
    # Adding the ideal line for reference
    plt.plot([np.min(x), np.max(x)], [np.min(x), np.max(x)], '--', color='r', linewidth=1.5)
    # Highlight the region within 1 kcal/mol from the red line
    plt.fill_between(np.linspace(np.min(x), np.max(x), 100), np.linspace(np.min(x), np.max(x), 100) - 1, np.linspace(np.min(x), np.max(x), 100) + 1, color='green', alpha=0.3)
    # Creating a custom circle marker for the "Testing data" legend entry
    testing_marker = Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.Blues(0.8), markersize=10, label='Testing data')
    # Adding the combined legend
    plt.legend(fontsize=20, loc='best')
    plt.tight_layout()
    # Save the plot
    plt.savefig(plot_path + "saved_data/r_squared_plot.png")
    plt.close()  # Close the plot to free up memory


if __name__ == "__main__":
    project_dir = Path('/mnt/e/ML_DFT/binding_E_database')
    targets_path = project_dir / "RANDOM_NPZ/target_array_9235.npz" #np.load("target_array_5000.npz", mmap_mode='r')["arr_0"] #[:1000]
    feature_scalar_path = project_dir /  "RANDOM_NPZ/feature_scalar_array_9235.npz" #np.load("feature_scalar_array_5000.npz", mmap_mode='r')["arr_0"] #[:1000]
    feature_matrix_path = project_dir / "RANDOM_NPZ/feature_matrix_array_9235.npz"
    train_x, train_y, val_x, val_y, test_x, test_y, min_deltas, max_deltas = load_labels_and_features(targets_path, feature_scalar_path, feature_matrix_path)
    print("train_x = ", np.shape(train_x))
    print("train_y = ", np.shape(train_y))
    print("val_x = ", np.shape(val_x))
    print("val_y = ", np.shape(val_y))
    print("test_x = ", np.shape(test_x))
    print("test_y = ", np.shape(test_y))

    train_model(
        train_x, train_y,
        val_x, val_y,
        test_x, test_y,
        min_deltas, max_deltas,
        model_path=r'./saved_data/' ,
        plot_path = r'./saved_data/',
    )