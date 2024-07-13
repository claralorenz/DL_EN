import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from scikeras.wrappers import KerasRegressor
import numpy as np
import random
from sklearn.base import BaseEstimator, TransformerMixin
from keras import optimizers
import os
import pickle
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, MaxPooling1D, LSTM
from sklearn.model_selection import PredefinedSplit
import pickle
from sklearn.model_selection import ParameterGrid
import time
from keras.losses import MeanSquaredError, MeanAbsoluteError
from keras.layers import Input
from keras.models import Model
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_squared_error
import matplotlib.pyplot as plt
import textwrap

def set_wrapped_title(ax, title, max_line_length=40, **kwargs):
    wrapped_title = "\n".join(textwrap.wrap(title, max_line_length))
    ax.set_title(wrapped_title, **kwargs)

def mean_bias_error(y_true, y_pred):
    """
    Calculate the Mean Bias Error (MBE) between true and predicted values.
    
    Parameters:
    y_true : array-like of shape (n_samples,)
        The true values.
    y_pred : array-like of shape (n_samples,)
        The predicted values.
    
    Returns:
    float
        The mean bias error.
    """
    # Convert inputs to NumPy arrays to ensure compatibility
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate the bias errors
    bias_errors = y_pred - y_true
    
    # Calculate the mean of bias errors
    mean_bias_error = np.mean(bias_errors)
    
    return mean_bias_error


def build_model(dense_layers, dropout_rate, learning_rate, units1, units2, n_feat):
    '''
    Build the MLP model with grid search parameters
    '''
    inputs = Input(shape=(n_feat,))
    x = inputs
    
    if dense_layers == 1:
        x = Dense(units=units1, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
    elif dense_layers == 2:
        x = Dense(units=units2, activation='relu')(x)
        x = Dropout(dropout_rate)(x)

    outputs = Dense(units=1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    adam = optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(optimizer=adam, loss='mean_squared_error')  
    
    return model


# # # MLP model 

# Define the base directory
base_dir = '/rwthfs/rz/cluster/home/rwth1434/DL_EN'
datastore_folder_target_name = 'MLP'
destination_folder = os.path.join(base_dir, datastore_folder_target_name)

# Define the directory and name of the folder of the results
folder_of_results = 'Results'
folder_of_results_f = os.path.join(datastore_folder_target_name, folder_of_results)
folder_of_results_directory = destination_folder
folder_of_results = os.path.join(folder_of_results_directory, folder_of_results)

# Create the folder for the results
os.makedirs(folder_of_results, exist_ok=True)

# Define the file names
training_file_name = 'lists_data_MLP.pkl'

target_tr_file_name = 'y_tr.csv'
target_val_file_name = 'y_val.csv'

# Create file paths
training_path = os.path.join(destination_folder, training_file_name)
target_path_tr = os.path.join(destination_folder, target_tr_file_name)
target_path_val = os.path.join(destination_folder, target_val_file_name)

# Load data

with open(training_path, 'rb') as f:
    # Use pickle to load the data dictionary from the file
    loaded_data = pickle.load(f)

X_tr = loaded_data['X_tr_n']
X_val = loaded_data['X_val_n']


y_tr = pd.read_csv(target_path_tr)
y_tr = y_tr.drop(columns='Unnamed: 0')

y_val = pd.read_csv(target_path_val)
y_val = y_val.drop(columns='Unnamed: 0')



# Scale target data
y_tr = np.array(y_tr).reshape(-1, 1)
y_val = np.array(y_val).reshape(-1, 1)
#y_te = np.array(y_te).reshape(-1, 1)

ScalerY = MinMaxScaler()
y_tr_n = ScalerY.fit_transform(y_tr)
y_val_n = ScalerY.transform(y_val)

# Adjust format of input data
X_tr= np.array(X_tr)
X_val=np.array(X_val)


n_feat = X_tr.shape[1]


model = KerasRegressor(model=build_model, optimizer='adam', batch_size=660, units1=100, units2=80, dense_layers=1, dropout_rate=0.0, shuffle=False, random_state=1901, verbose=0, learning_rate=0.0001, epochs=200)

param_grid = {
   'dense_layers': [1, 2],  
    'units1': [200, 500, 1000], 
    'units2': [100, 200, 500],
    'dropout_rate': [0.0, 0.2, 0.35],
}

results = []
#####
for params in ParameterGrid(param_grid):
    model = build_model(params['dense_layers'], params['dropout_rate'],  0.0001, params['units1'], params['units2'],  n_feat)
    #(dense_layers, dropout_rate, learning_rate, units1, units2, n_feat):

    start_time = time.time()
    history = model.fit(X_tr, y_tr_n, epochs=200, batch_size=660, verbose=0)  # Train with validation data
    end_time = time.time()

    # Evaluate on validation set
    y_pred = model.predict(X_val)
    y_pred_original_scale = ScalerY.inverse_transform(y_pred.reshape(-1, 1))
    mae = mean_absolute_error(y_val, y_pred_original_scale)
    rmse = root_mean_squared_error(y_val_n, y_pred)
    rmse_orig = root_mean_squared_error(y_val, y_pred_original_scale)
    mbe = mean_bias_error(y_val, y_pred_original_scale)

    # Extract training history
    training_loss = history.history['loss']

    # Plot learning curve
    epochs = range(1, len(training_loss) + 1)
    plt.figure(dpi=600)
    plt.plot(epochs, training_loss, label='Training MSE')
    set_wrapped_title(plt.gca(), f'Training loss (MSE): {params}')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.tight_layout()
    

    # Save the plot
    plot_filename = f'learning_curve_{params}.png'  # Define a unique filename based on params
    plot_path = os.path.join(folder_of_results, plot_filename)
    plt.savefig(plot_path)
    plt.close()  # Close the plot to free up memory

    # Plot Prediction vs Target values
    plt.figure(dpi=600)
    plt.plot(y_pred_original_scale, label = 'Prediction'),
    plt.plot(y_val, label = 'Target')
    plt.legend()
    set_wrapped_title(plt.gca(), f'Prediction and target comparison for {params}', fontsize=14)
    plt.ylabel('Energy consumption one hour ahead (wH)')
    plt.tight_layout()

    # Hide the x-axis
    plt.gca().axes.get_xaxis().set_visible(False)
    plot_filename2 = f'pred_target{params}.png' 
    plot_path2 = os.path.join(folder_of_results, plot_filename2)
    plt.savefig(plot_path2)
    plt.close()

    # Store results
    result = {
        'params': params,
        'mae': mae,
        'rmse': rmse,
        'rmse_original_scale': rmse_orig,
        'mbe': mbe, 
        'training_time': end_time - start_time,
        'learning_curve_plot': plot_path,
        'pred_target': plot_path2,
        'y_pred_original_scale': y_pred_original_scale
    }
    results.append(result)
    # Store results
    result = {
        'params': params,
        'mae': mae,
        'rmse': rmse,
        'training_time': end_time - start_time,
        'y_pred_original_scale': y_pred_original_scale
    }
    results.append(result)

results_filename = 'Results_MLP_1.pkl'
results_path = os.path.join(folder_of_results, results_filename)
# Open a file in binary write mode
with open(results_path, 'wb') as file:
     pickle.dump(results, file)

