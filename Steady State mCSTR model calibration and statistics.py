import numpy as np
import math
from scipy.optimize import fsolve, minimize
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import t, chi2

# Function to import the data
def data_importer(folder_path, file_name, sheet):
    """
    Function used to import the experimental data from Excel.
    """
    file_path = os.path.join(folder_path, file_name)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file was not found: {file_path}")
    
    raw_data = pd.read_excel(file_path, sheet_name=sheet)
    
    u = raw_data.iloc[:, 0:3].to_numpy()  # Input
    y = raw_data.iloc[:, 3:5].to_numpy()  # Output
    exp_data = raw_data.iloc[:, 0:5].to_numpy()  # All data
    
    return exp_data

# Define the kinetic model for the CSTR
def cstr_steady_state(u, theta, temp, initQ, initC):
    """
    Defines the steady-state model of the CSTR for a second-order reaction.
    """
    k1 = math.exp(-theta[0] - (theta[1] * 10000.0 / R) * ((1 / (temp + 273.15)) - (1 / (Tref + 273.15))))
    r = k1 * u[0] ** 2
    
    # Steady-state mass balances
    eq1 = (initQ / 60000) * (initC - u[0]) - r * Vr  # A balance
    eq2 = (initQ / 60000) * (0 - u[1]) + r * Vr      # B balance
    
    return [eq1, eq2]

# Function to solve for concentrations given theta, temperature, flowrate, and inlet concentration
def solve_cstr(theta, temp, initQ, initC):
    """
    Solves the steady-state concentrations of A and B in the CSTR using fsolve function.
    """
    initial_guess = [initC, 0.0]
    solution = fsolve(cstr_steady_state, initial_guess, args=(theta, temp, initQ, initC))
    
    return solution

# Objective function to minimize (sum of squared residuals)
def residuals(theta, experimental_data):
    """
    Computes the sum of squared residuals between the model predictions and the experimental data.
    """
    total_residual = 0
    for data in experimental_data:
        temp, initQ, initC, cA_exp, cB_exp = data  # Unpack experimental data

        cA_model, cB_model = solve_cstr(theta, temp, initQ, initC)
        
        residual_A = cA_model - cA_exp
        residual_B = cB_model - cB_exp
        
        total_residual += residual_A**2 + residual_B**2
    
    return total_residual

# Function to evaluate the information matrix
def information_matrix(theta, N_exp, experimental_data, sigma):
    """
    Evaluate the amount of information obtainable from the experiments.
    """
    epsilon = 0.0001
    pert_params = np.zeros((len(theta) + 1, len(theta)))

    for i in range(len(theta) + 1):
        pert_params[i] = theta.copy()

    for i in range(len(theta)):
        pert_params[i][i] = theta[i] * (1 + epsilon)

    y_hat = np.zeros((len(theta) + 1, N_exp, 2))
    for i in range(len(theta) + 1):
        for j, data in enumerate(experimental_data):
            temp, initQ, initC, _, _ = data
            y_hat[i, j] = solve_cstr(pert_params[i], temp, initQ, initC)
    
    sensitivity_matrix = np.zeros((N_exp, len(theta), 2))
    for i in range(N_exp):
        for j in range(len(theta)):
            for k in range(2):
                sensitivity_matrix[i, j, k] = (y_hat[j, i, k] - y_hat[-1, i, k]) / (theta[j] * epsilon)
    
    FIM_matrix = np.zeros((len(theta), len(theta)))
    dynFIM_matrix = np.zeros((N_exp, len(theta), len(theta)))
    for i in range(N_exp):
        for k in range(2):
            FIM_matrix += (1 / (sigma[k] ** 2)) * np.outer(sensitivity_matrix[i][:, k], sensitivity_matrix[i][:, k])
            dynFIM_matrix[i] += (1 / (sigma[k] ** 2)) * np.outer(sensitivity_matrix[i][:, k], sensitivity_matrix[i][:, k])

    return FIM_matrix, sensitivity_matrix, dynFIM_matrix

# Function to evaluate the RFI for each parameter
def calculate_relative_fisher_information_per_param(dynamic_information, total_FIM):
    """
    This function evaluate the Relative Fisher Information (RFI) for each parameter on all experiments.
    """
    N_exp = dynamic_information.shape[0]
    num_params = dynamic_information.shape[1]
    
    rfi_matrix = np.zeros((N_exp, num_params))
    
    # Evaluation of the total norm for each parameter (sum of norms over all experiments)
    total_norms = np.array([np.linalg.norm(total_FIM[:, j]) for j in range(num_params)])

    # Evaluation of the norm for each parameter in each experiment
    for j in range(num_params):
        param_norms = np.array([np.linalg.norm(dynamic_information[i][:, j]) for i in range(N_exp)])
        
        rfi_matrix[:, j] = param_norms / total_norms[j]
    
    return rfi_matrix

# Function to evaluate the trace of the Fisher information matrix and trace 
def calculate_relative_fisher_information_trace(dynamic_information, total_FIM):
    """
    Calcola la Relative Fisher Information (RFI) per ciascun parametro su tutti gli esperimenti.
    """
    N_exp = dynamic_information.shape[0]
    num_params = dynamic_information.shape[1]
    
    rfi_matrix = np.zeros((N_exp, num_params))
    
    # Evaluation of the total norm for each parameter (sum of norms over all experiments)
    total_trace = np.array(np.trace(total_FIM))

    # Evaluation of the norm for each parameter in each experiment
    dynamic_trace = np.array([np.trace(dynamic_information[i]) for i in range(N_exp)])
        
    rfi_matrix = dynamic_trace / total_trace
    
    return rfi_matrix


# Function to plot the RFI for each parameter
def plot_relative_fisher_information_per_param(rfi_matrix):
    """
    Crea un grafico della Relative Fisher Information (RFI) per ciascun parametro.
    """
    N_exp = rfi_matrix.shape[0]
    num_params = rfi_matrix.shape[1]

    for j in range(num_params):
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, N_exp + 1), rfi_matrix[:, j], label=f'RFI KP{j+1}')
        plt.xlabel('Experiment')
        plt.ylabel(f'Relative Fisher Information (RFI) [KP{j+1}]')
        plt.title(f'RFI for KP{j+1} for each experiment')
        plt.xticks(np.arange(0, N_exp + 1, step=10))
        #plt.grid(True)
        plt.legend()
        plt.show()


# Function to plot the RFI for each parameter
def plot_relative_fisher_information_trace(rfi_matrix):
    """
    Crea un grafico della Relative Fisher Information (RFI) per ciascun parametro.
    """
    N_exp = rfi_matrix.shape[0]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, N_exp + 1), rfi_matrix, label=f'RFI')
    plt.xlabel('Experiment')
    plt.ylabel(f'Relative Fisher Information (RFI)')
    plt.title(f'RFI for each experiment')
    plt.xticks(np.arange(0, N_exp + 1, step=10))
    #plt.grid(True)
    plt.legend()
    plt.show()


# Function to plot diagonal elements of the sensitivity matrix
def plot_sensitivity_diagonal(sensitivity_matrix):
    """
    Plot used to visualize the diagonal elements of the sensitivity matrix
    """
    N_exp = sensitivity_matrix.shape[0]  # Number of experiments
    num_params = sensitivity_matrix.shape[1]  # Number of parameters

    # Loop for every parameter
    for j in range(num_params):
        plt.figure(figsize=(10, 6))
        
        # Diagonal elements for the parameter j
        diagonal_sensitivities = [sensitivity_matrix[i, j, j] for i in range(N_exp)]
        
        # Plots of the variation of the diagonal elements
        plt.plot(range(1, N_exp + 1), diagonal_sensitivities, linestyle='-', label=f'Sensitivity KP{j+1}')
        
        # Impostazioni del grafico
        plt.xlabel('Experiments')
        plt.ylabel(f'Sensitivity of the parameter KP{j+1}')
        #plt.title(f'Sensitivity variation of the diagonal of the parameter {j+1}')
        plt.xticks(np.arange(0, N_exp + 1, step=10))  # Labels every 10 experiments
        #plt.grid(True)
        plt.legend()
        plt.show()


def t_test(significance, dof, parameters, variance):
    """
    This function is necessary to perform the t test on the identified parameters
    """
    alpha = 1 - significance
    t_ref = t.ppf((1-alpha), dof)
    conf_int = (t.ppf(1 - (alpha/2), dof)) * np.sqrt(variance)
    t_values = np.array(np.abs(parameters) / conf_int)

    return t_ref, t_values

# Constants
R = 8.314  # Universal gas constant (J/(mol*K))
Tref = 30  # Reference temperature (°C)
Vr = 2.3559 * 1e-3  # Reactor volume (L)
N_exp = 171


# Import the data
folder_path = "C:/Users/Andrea/Desktop/Project/DynamicCSTR/Steady State"
file_name = "SS data.xlsx"
sheet = "Cumulative results"

experimental_data = data_importer(folder_path, file_name, sheet)

# Initial guess for theta
theta_initial = [1.0, 1]

# Perform optimization to minimize the residuals
result = minimize(residuals, theta_initial, args=(experimental_data), method='Nelder-Mead')

# Extract the optimized values of theta
theta_optimal = result.x
print(f"Optimized theta values: theta[0] = {theta_optimal[0]}, theta[1] = {theta_optimal[1]}")

# Calculate concentrations using the optimized theta values
for i, data in enumerate(experimental_data):
    temp, initQ, initC, _, _ = data
    solution = solve_cstr(theta_optimal, temp, initQ, initC)

sigma = [0.017, 0.014]
info_matrix, sensitivity_matrix, dynamic_information = information_matrix(theta_optimal, N_exp, experimental_data, sigma)
print(np.trace(info_matrix))

significance = 0.996
dof = N_exp*6 - 2 # Number of meaurements * Number of states - Number of parameters 

chi2_ref = chi2.ppf(significance, dof)
#print(chi2_ref)
residuals = residuals(theta_optimal, experimental_data)

if residuals > chi2_ref:
    print('\nThe candidate model is falsified for under-fitting.')
    print('The sum of squared residuals: {0:.2f}; is larger than the 95% chi2 value of '
          'reference: {1:.2f};'.format(residuals, chi2_ref))
else:
    print('\nThe candidate model is not falsified by the Goodness-of-fit test.')
    print('The sum of squared residuals: {0:.2f}; below the 95% chi2 values of '
          'reference'.format(residuals))
    
# Perform t-test on the optimized parameters
variance_covariance_matrix = np.linalg.inv(info_matrix)
variance = np.diag(variance_covariance_matrix)

t_ref, t_values = t_test(significance, dof, theta_optimal, variance)

# Calculate the RFI per parameter
relative_fisher_per_param = calculate_relative_fisher_information_per_param(dynamic_information, info_matrix)

# Plot the RFI per parameter
#plot_relative_fisher_information_per_param(relative_fisher_per_param)

# Plot delle sensibilità per tutti i parametri
#plot_sensitivity_diagonal(sensitivity_matrix)

# Profile of the information trace 
relative_trace = calculate_relative_fisher_information_trace(dynamic_information, info_matrix)
#plot_relative_fisher_information_trace(relative_trace)