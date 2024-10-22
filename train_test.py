# Import necessary libraries
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib  # For saving and loading model checkpoints
from sklearn.neural_network import MLPRegressor

# Set the maximum number of threads
MAX_THREADS = 6  # 设置线程池最大数量为6



# Genetic algorithm library
try:
    from geneticalgorithm import geneticalgorithm as ga  # For genetic algorithm
except ImportError:
    print("Genetic algorithm library 'geneticalgorithm' not found. Please install it using 'pip install git+https://github.com/rmsolgi/geneticalgorithm.git'.")
    ga = None
import os
import multiprocessing

# Load the training dataset
data_file = "test.csv"
df = pd.read_csv(data_file)

# Preparing input and output data for training
X_train = df.iloc[:, :-1]  # First 13 columns are input features
y_train = df.iloc[:, -1]   # The 14th column is the output target

# Scaling the features for training
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)

# Saving the scaler for future use
scaler_filename = "scaler_new_data.save"
joblib.dump(scaler, scaler_filename)

# Training Script

def train_model(hidden_layer_sizes=(50, 50), max_iter=2000, model_filename="mlp_model_new_data_checkpoint.pkl"):
    # Building the MLPRegressor (BP network)
    mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation='relu', solver='adam', max_iter=max_iter, random_state=42, n_iter_no_change=10, verbose=True, )
    mlp.fit(X_train_scaled, y_train)

    # Save the model checkpoint
    joblib.dump(mlp, model_filename)
    print(f"Model saved to {model_filename}")

    # Calculate RMSE for training set
    y_pred_train = mlp.predict(X_train_scaled)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    print(f"Training RMSE: {rmse_train}")

# Train the model with different number of neurons in hidden layers
if __name__ == "__main__":
    # Set CPU affinity to use a specific number of cores (e.g., 6 cores)
    try:
        p = multiprocessing.current_process()
        p.cpu_affinity([0, 1, 2, 3, 4, 5])  # Set to use 6 CPUs (cores 0 to 5)
    except AttributeError:
        print("CPU affinity could not be set on this system. Please adjust manually if needed.")
    
    train_model(hidden_layer_sizes=(5, 5, 5), model_filename="mlp_model_new_data_5_5_5.pkl")
    train_model(hidden_layer_sizes=(10, 10, 10), model_filename="mlp_model_new_data_10_10_10.pkl")
    train_model(hidden_layer_sizes=(100, 100, 100), model_filename="mlp_model_new_data_100_100_100.pkl")

# Testing Script

def test_model(test_data_file="test.csv", model_filename="mlp_model_new_data_checkpoint.pkl", scaler_filename="scaler_new_data.save"):
    # Load the testing dataset
    df_test = pd.read_csv(test_data_file)
    X_test = df_test.iloc[:, :-1]  # First 13 columns are input features
    y_test = df_test.iloc[:, -1]   # The 14th column is the output target

    # Load the scaler and model
    if os.path.exists(model_filename) and os.path.exists(scaler_filename):
        scaler = joblib.load(scaler_filename)
        mlp = joblib.load(model_filename)
        

        # Rescale the features for testing
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

        # Predict using the loaded model
        y_pred_test = mlp.predict(X_test_scaled)

        # Calculate RMSE
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        print(f"Testing RMSE for model {model_filename}: {rmse_test}")

        # Display the predictions
        results_df = pd.DataFrame({"Experimental Value": y_test, "Predicted Value": y_pred_test})
        print(results_df)
    else:
        print("Model or scaler file not found.")

# Test the trained models
if __name__ == "__main__":
    test_model(model_filename="mlp_model_new_data_5_5_5.pkl")
    test_model(model_filename="mlp_model_new_data_10_10_10.pkl")
    test_model(model_filename="mlp_model_new_data_100_100_100.pkl")

# Genetic Algorithm to Find Optimal Inputs
def optimize_inputs(model_filename="mlp_model_new_data_100_100_100.pkl", scaler_filename="scaler_new_data.save"):
    if ga is None:
        print("Cannot run genetic algorithm optimization. Library 'geneticalgorithm' is not available.")
        return
    # Load the model and scaler
    if os.path.exists(model_filename) and os.path.exists(scaler_filename):
        scaler = joblib.load(scaler_filename)
        mlp = joblib.load(model_filename)

        # Define the objective function for optimization
        def objective_function(X):
            X = np.array(X).reshape(1, -1)
            X_scaled = pd.DataFrame(scaler.transform(X), columns=X_train.columns)
            prediction = mlp.predict(X_scaled)
            return -prediction[0]  # Negative because we want to maximize the output

        # Define the bounds for each input feature
        varbound = np.array([[X_train[col].min(), X_train[col].max()] for col in X_train.columns])

        # Set up the genetic algorithm model
        algorithm_param = {'max_num_iteration': 500, 'population_size': 100, 'mutation_probability': 0.1, 'elit_ratio': 0.01,
                           'crossover_probability': 0.5, 'parents_portion': 0.3, 'crossover_type': 'uniform', 'max_iteration_without_improv': None}

        model = ga(function=objective_function, dimension=X_train.shape[1], variable_type='real', variable_boundaries=varbound, algorithm_parameters=algorithm_param, convergence_curve=False, progress_bar=True)

        # Run the genetic algorithm
        model.run()
        best_solution = model.output_dict['variable']
        best_value = -model.output_dict['function']  # Since we minimized the negative output

        print(f"Optimal Inputs: {best_solution}")
        print(f"Maximum Predicted Output (Enzyme Activity): {best_value}")

    else:
        print("Model or scaler file not found.")

# Run the genetic algorithm to optimize inputs
if __name__ == "__main__":
    optimize_inputs(model_filename="mlp_model_new_data_100_100_100.pkl")
