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
import scipy.linalg

############################################## ESSIANTIAL FUNCTIONS
def H_kJ_mol(E):
    """Converts Hartree to kJ/mol"""
    return E*27.211*96.485 #kJ/mol

def H_kcal_mol(E):
    """Converts Hartree to kcal/mol"""
    return E/(0.00159362) #kcal/mol

def kJ_H_mol(E):
    """Converts kJ/mol to Hartree"""
    return E/(27.211*96.485) #kJ/mol

def kcal_H_mol(E):
    """Converts kcal/mol to Hartree"""
    return E*(0.00159362) #kcal/mol

def normalize_array(arr, min_val, max_val):
    # Find the minimum and maximum values in the array
    #min_val = np.min(arr)
    #max_val = np.max(arr)

    # Normalize the array to the range -1 to 1
    normalized = 2 * ((arr - min_val) / (max_val - min_val)) - 1

    return normalized

def unnormalize_array(normalized_arr, min_val, max_val):
    # Unnormalize the array from the range -1 to 1 back to its original range
    original = ((normalized_arr + 1) / 2) * (max_val - min_val) + min_val
    return original

def normalize_array_OLD(arr, min_val, max_val):
    return arr #/(max_val)

def unnormalize_array_OLD(normalized_arr, min_val, max_val):
    return normalized_arr #*max_val

import numpy as np

def calculate_rmae(actual, predictions):
    """
    Calculate the Relative Mean Absolute Error (RMAE).
    Parameters:
    - actual: numpy array of actual values.
    - predictions: numpy array of predicted values, same shape as actual.
    Returns:
    - rmae: Relative Mean Absolute Error.
    """
    # Calculate MAE
    mae = np.mean(np.abs(predictions - actual))
    # Calculate the average absolute value of the actual values
    avg_abs_actual = np.mean(np.abs(actual))
    # Calculate RMAE
    rmae = mae / avg_abs_actual
    return rmae*100


def convert_mae_to_original_scale(normalized_mae, min_val, max_val):
    """
    Convert the Mean Absolute Error (MAE) from a normalized scale back to the original scale.

    Parameters:
    normalized_mae (float): The MAE obtained on the normalized scale.
    min_val (float): The minimum value of the original data before normalization.
    max_val (float): The maximum value of the original data before normalization.

    Returns:
    float: The MAE in the original scale.
    """
    # The range of the original data
    original_range = max_val - min_val

    # Convert the normalized MAE to the original scale
    original_scale_mae = normalized_mae * original_range / 2

    return original_scale_mae


def manual_train_test_val_split(features, targets, test_size=0.2, val_size=0.1):
    """
    Manually split the dataset into training, testing, and validation sets.

    Parameters:
    features (numpy.ndarray): The feature array.
    targets (numpy.ndarray): The target array.
    test_size (float): The proportion of the dataset to include in the test split.
    val_size (float): The proportion of the dataset to include in the validation split.

    Returns:
    train_x (numpy.ndarray): The training features.
    test_x (numpy.ndarray): The test features.
    val_x (numpy.ndarray): The validation features.
    train_y (numpy.ndarray): The training targets.
    test_y (numpy.ndarray): The test targets.
    val_y (numpy.ndarray): The validation targets.
    """

    # Calculate split indices
    total_samples = features.shape[0]
    test_split_index = int(total_samples * (1 - test_size - val_size))
    val_split_index = int(total_samples * (1 - val_size))

    # Shuffle the data
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    features_shuffled = features[indices]
    targets_shuffled = targets[indices]

    # Split the data
    train_x, train_y = features_shuffled[:test_split_index], targets_shuffled[:test_split_index]
    test_x, test_y = features_shuffled[test_split_index:val_split_index], targets_shuffled[test_split_index:val_split_index]
    val_x, val_y = features_shuffled[val_split_index:], targets_shuffled[val_split_index:]

    return train_x, test_x, val_x, train_y, test_y, val_y


def init_guess_by_1e(xyz_file):
    # Read the coordinates from the .xyz file
    with open(xyz_file, "r") as f:
        all_lines = f.readlines()
        # Skip the first two lines
        lines = all_lines[2:]

    # Create the Mole object
    mol = gto.M(atom=lines)
    mol.basis = 'sto-3g'
    mol.build()

    # Compute one-electron integrals
    int1e_kin = mol.intor_symmetric('int1e_kin')
    int1e_nuc = mol.intor_symmetric('ECPscalar')  # Use ECPscalar as an alternative to int1e_nuc
    h1e = int1e_kin + int1e_nuc

    # Compute overlap integrals
    s1e = mol.intor_symmetric('int1e_ovlp')

    # Diagonalize the core Hamiltonian
    mo_energy, mo_coeff = scipy.linalg.eigh(h1e, s1e)

    # Occupy the lowest energy orbitals
    mo_occ = np.zeros_like(mo_energy)
    nocc = mol.nelectron // 2
    mo_occ[:nocc] = 2

    # Construct the density matrix
    dm = np.dot(mo_coeff * mo_occ, mo_coeff.T.conj())

    return dm
