import numpy as np
import matplotlib.pyplot as plt
import math 
import os
import pandas as pd
from scipy.optimize import fsolve, minimize

# Code used to model the CSTR in steady state conditions
def cstr_steady_state(u, theta, temp, initQ, initC):
    """
    Defines the steady-state model of the CSTR for a second-order reaction.
    """
    Vr = 2.3559 * 1e-3  # Reactor volume (L)

    k1 = math.exp(-theta[0] - (theta[1] * 10000.0 / R) * ((1 / (temp + 273.15)) - (1 / (Tref + 273.15))))
    r = k1 * u[0] ** 2
    
    # Steady-state mass balances
    eq1 = (initQ / 60000) * (initC - u[0]) - r * Vr  # A balance
    eq2 = (initQ / 60000) * (0 - u[1]) + r * Vr      # B balance
    
    return [eq1, eq2]


# Function to solve for concentrations given theta, temperature, flowrate, and inlet concentration
def solve_cstr(theta, temp, initQ, initC):
    """
    Solves the steady-state concentrations of A and B in the CSTR using fsolve.
    """
    initial_guess = [initC, 0.0]
    solution = fsolve(cstr_steady_state, initial_guess, args=(theta, temp, initQ, initC))
    
    return solution


# Function used to import the data
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


# Evaluation of the information 
def information_matrix(theta, N_exp, experimental_data, sigma):
    """
    This function is used to evaluate the amount of information of each steady state experiment
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


theta_optimal = [3.7129760802614644, 2.4424484990150344]  # Optimal parameter values obtained from the model calibration

# Parameters
R = 8.314  # Universal gas constant (J/(mol*K))
Tref = 30  # Reference temperature (°C)
Vr = 2.3559 * 1e-3  # Reactor volume (L)

trace_total = np.trace([[20700.28269832, 633.61402509], [633.61402509, 527.51860947]])

# Now we have to initialize the vector that contains all the values of the RFI
RFIs = []
SM = []  # Sensitivity matrices 

# Experiment 1
folder_path = "C:/Users/Andrea/Desktop/Project/DynamicCSTR/Steady State/15C/2"
file_name = "SS_2_05.xlsx"
sheet = "Sheet1"

experimental_data = data_importer(folder_path, file_name, sheet)

N_exp = np.shape(experimental_data)[0]

# Calculate concentrations using the optimized theta values
for i, data in enumerate(experimental_data):
    temp, initQ, initC, _, _ = data
    solution = solve_cstr(theta_optimal, temp, initQ, initC)

sigma = [0.017, 0.014]
info_matrix, sensitivity_matrix, dynamic_information = information_matrix(theta_optimal, N_exp, experimental_data, sigma)

RFIs +=  [np.trace(np.sum(dynamic_information[0:5], axis=0))/trace_total]  # RFI of the experiment E1a
RFIs +=  [np.trace(np.sum(dynamic_information[5:10], axis=0))/trace_total]  # RFI of the experiment E1b
RFIs +=  [np.trace(np.sum(dynamic_information[10:15], axis=0))/trace_total]  # RFI of the experiment E1c

SM += [np.sum(sensitivity_matrix[0:5], axis=0)]
SM += [np.sum(sensitivity_matrix[5:10], axis=0)]
SM += [np.sum(sensitivity_matrix[10:15], axis=0)]


# Experiment 2
folder_path = "C:/Users/Andrea/Desktop/Project/DynamicCSTR/Steady State/15C/2"
file_name = "SS_2_1.xlsx"
sheet = "Sheet1"

experimental_data = data_importer(folder_path, file_name, sheet)

N_exp = np.shape(experimental_data)[0]

# Calculate concentrations using the optimized theta values
for i, data in enumerate(experimental_data):
    temp, initQ, initC, _, _ = data
    solution = solve_cstr(theta_optimal, temp, initQ, initC)

sigma = [0.017, 0.014]
info_matrix, sensitivity_matrix, dynamic_information = information_matrix(theta_optimal, N_exp, experimental_data, sigma)

RFIs +=  [np.trace(np.sum(dynamic_information[0:5], axis=0))/trace_total]  # RFI of the experiment E2a
RFIs +=  [np.trace(np.sum(dynamic_information[5:10], axis=0))/trace_total]  # RFI of the experiment E2b
RFIs +=  [np.trace(np.sum(dynamic_information[10:15], axis=0))/trace_total]  # RFI of the experiment E2c

SM += [np.sum(sensitivity_matrix[0:5], axis=0)]
SM += [np.sum(sensitivity_matrix[5:10], axis=0)]
SM += [np.sum(sensitivity_matrix[10:15], axis=0)]


# Experiment 3
folder_path = "C:/Users/Andrea/Desktop/Project/DynamicCSTR/Steady State/15C/05"
file_name = "SS_05_05.xlsx"
sheet = "Sheet1"

experimental_data = data_importer(folder_path, file_name, sheet)

N_exp = np.shape(experimental_data)[0]

# Calculate concentrations using the optimized theta values
for i, data in enumerate(experimental_data):
    temp, initQ, initC, _, _ = data
    solution = solve_cstr(theta_optimal, temp, initQ, initC)

sigma = [0.017, 0.014]
info_matrix, sensitivity_matrix, dynamic_information = information_matrix(theta_optimal, N_exp, experimental_data, sigma)


RFIs +=  [np.trace(np.sum(dynamic_information[0:5], axis=0))/trace_total]  # RFI of the experiment E3a
RFIs +=  [np.trace(np.sum(dynamic_information[5:10], axis=0))/trace_total]  # RFI of the experiment E3b
RFIs +=  [np.trace(np.sum(dynamic_information[10:15], axis=0))/trace_total]  # RFI of the experiment E3c

SM += [np.sum(sensitivity_matrix[0:5], axis=0)]
SM += [np.sum(sensitivity_matrix[5:10], axis=0)]
SM += [np.sum(sensitivity_matrix[10:15], axis=0)]


# Experiment 4
folder_path = "C:/Users/Andrea/Desktop/Project/DynamicCSTR/Steady State/15C/05"
file_name = "SS_05_1.xlsx"
sheet = "Sheet1"

experimental_data = data_importer(folder_path, file_name, sheet)

N_exp = np.shape(experimental_data)[0]

# Calculate concentrations using the optimized theta values
for i, data in enumerate(experimental_data):
    temp, initQ, initC, _, _ = data
    solution = solve_cstr(theta_optimal, temp, initQ, initC)

sigma = [0.017, 0.014]
info_matrix, sensitivity_matrix, dynamic_information = information_matrix(theta_optimal, N_exp, experimental_data, sigma)


RFIs +=  [np.trace(np.sum(dynamic_information[0:5], axis=0))/trace_total]  # RFI of the experiment E4a
RFIs +=  [np.trace(np.sum(dynamic_information[5:10], axis=0))/trace_total]  # RFI of the experiment E4b
RFIs +=  [np.trace(np.sum(dynamic_information[10:15], axis=0))/trace_total]  # RFI of the experiment E4c

SM += [np.sum(sensitivity_matrix[0:5], axis=0)]
SM += [np.sum(sensitivity_matrix[5:10], axis=0)]
SM += [np.sum(sensitivity_matrix[10:15], axis=0)]


#####################################################################################################################################
#####################################################################################################################################

# Experiment 5
folder_path = "C:/Users/Andrea/Desktop/Project/DynamicCSTR/Steady State/30C/2"
file_name = "SS_2_05.xlsx"
sheet = "Sheet1"

experimental_data = data_importer(folder_path, file_name, sheet)

N_exp = np.shape(experimental_data)[0]

# Calculate concentrations using the optimized theta values
for i, data in enumerate(experimental_data):
    temp, initQ, initC, _, _ = data
    solution = solve_cstr(theta_optimal, temp, initQ, initC)

sigma = [0.017, 0.014]
info_matrix, sensitivity_matrix, dynamic_information = information_matrix(theta_optimal, N_exp, experimental_data, sigma)

RFIs +=  [np.trace(np.sum(dynamic_information[0:5], axis=0))/trace_total]  # RFI of the experiment E5a
RFIs +=  [np.trace(np.sum(dynamic_information[5:10], axis=0))/trace_total]  # RFI of the experiment E5b
RFIs +=  [np.trace(np.sum(dynamic_information[10:15], axis=0))/trace_total]  # RFI of the experiment E5c

SM += [np.sum(sensitivity_matrix[0:5], axis=0)]
SM += [np.sum(sensitivity_matrix[5:10], axis=0)]
SM += [np.sum(sensitivity_matrix[10:15], axis=0)]


# Experiment 6
folder_path = "C:/Users/Andrea/Desktop/Project/DynamicCSTR/Steady State/30C/2"
file_name = "SS_2_1.xlsx"
sheet = "Sheet1"

experimental_data = data_importer(folder_path, file_name, sheet)

N_exp = np.shape(experimental_data)[0]

# Calculate concentrations using the optimized theta values
for i, data in enumerate(experimental_data):
    temp, initQ, initC, _, _ = data
    solution = solve_cstr(theta_optimal, temp, initQ, initC)

sigma = [0.017, 0.014]
info_matrix, sensitivity_matrix, dynamic_information = information_matrix(theta_optimal, N_exp, experimental_data, sigma)

RFIs +=  [np.trace(np.sum(dynamic_information[0:5], axis=0))/trace_total]  # RFI of the experiment E6a
RFIs +=  [np.trace(np.sum(dynamic_information[5:10], axis=0))/trace_total]  # RFI of the experiment E6b
RFIs +=  [np.trace(np.sum(dynamic_information[10:15], axis=0))/trace_total]  # RFI of the experiment E6c

SM += [np.sum(sensitivity_matrix[0:5], axis=0)]
SM += [np.sum(sensitivity_matrix[5:10], axis=0)]
SM += [np.sum(sensitivity_matrix[10:15], axis=0)]


# Experiment 7
folder_path = "C:/Users/Andrea/Desktop/Project/DynamicCSTR/Steady State/30C/05"
file_name = "SS_05_05.xlsx"
sheet = "Sheet1"

experimental_data = data_importer(folder_path, file_name, sheet)

N_exp = np.shape(experimental_data)[0]

# Calculate concentrations using the optimized theta values
for i, data in enumerate(experimental_data):
    temp, initQ, initC, _, _ = data
    solution = solve_cstr(theta_optimal, temp, initQ, initC)

sigma = [0.017, 0.014]
info_matrix, sensitivity_matrix, dynamic_information = information_matrix(theta_optimal, N_exp, experimental_data, sigma)


RFIs +=  [np.trace(np.sum(dynamic_information[0:5], axis=0))/trace_total]  # RFI of the experiment E7a
RFIs +=  [np.trace(np.sum(dynamic_information[5:10], axis=0))/trace_total]  # RFI of the experiment E7b
RFIs +=  [np.trace(np.sum(dynamic_information[10:15], axis=0))/trace_total]  # RFI of the experiment E7c

SM += [np.sum(sensitivity_matrix[0:5], axis=0)]
SM += [np.sum(sensitivity_matrix[5:10], axis=0)]
SM += [np.sum(sensitivity_matrix[10:15], axis=0)]


# Experiment 8
folder_path = "C:/Users/Andrea/Desktop/Project/DynamicCSTR/Steady State/30C/05"
file_name = "SS_05_1.xlsx"
sheet = "Sheet1"

experimental_data = data_importer(folder_path, file_name, sheet)

N_exp = np.shape(experimental_data)[0]

# Calculate concentrations using the optimized theta values
for i, data in enumerate(experimental_data):
    temp, initQ, initC, _, _ = data
    solution = solve_cstr(theta_optimal, temp, initQ, initC)

sigma = [0.017, 0.014]
info_matrix, sensitivity_matrix, dynamic_information = information_matrix(theta_optimal, N_exp, experimental_data, sigma)


RFIs +=  [np.trace(np.sum(dynamic_information[0:5], axis=0))/trace_total]  # RFI of the experiment E8a
RFIs +=  [np.trace(np.sum(dynamic_information[5:10], axis=0))/trace_total]  # RFI of the experiment E8b
RFIs +=  [np.trace(np.sum(dynamic_information[10:15], axis=0))/trace_total]  # RFI of the experiment E8c

SM += [np.sum(sensitivity_matrix[0:5], axis=0)]
SM += [np.sum(sensitivity_matrix[5:10], axis=0)]
SM += [np.sum(sensitivity_matrix[10:15], axis=0)]


######################################################################################################################################################
######################################################################################################################################################

# Experiment 9
folder_path = "C:/Users/Andrea/Desktop/Project/DynamicCSTR/Steady State/45C/2"
file_name = "SS_2_05.xlsx"
sheet = "Sheet1"

experimental_data = data_importer(folder_path, file_name, sheet)

N_exp = np.shape(experimental_data)[0]

# Calculate concentrations using the optimized theta values
for i, data in enumerate(experimental_data):
    temp, initQ, initC, _, _ = data
    solution = solve_cstr(theta_optimal, temp, initQ, initC)

sigma = [0.017, 0.014]
info_matrix, sensitivity_matrix, dynamic_information = information_matrix(theta_optimal, N_exp, experimental_data, sigma)

RFIs +=  [np.trace(np.sum(dynamic_information[0:5], axis=0))/trace_total]  # RFI of the experiment E9a
RFIs +=  [np.trace(np.sum(dynamic_information[5:10], axis=0))/trace_total]  # RFI of the experiment E9b
RFIs +=  [np.trace(np.sum(dynamic_information[10:15], axis=0))/trace_total]  # RFI of the experiment E9c

SM += [np.sum(sensitivity_matrix[0:5], axis=0)]
SM += [np.sum(sensitivity_matrix[5:10], axis=0)]
SM += [np.sum(sensitivity_matrix[10:15], axis=0)]


# Experiment 10
folder_path = "C:/Users/Andrea/Desktop/Project/DynamicCSTR/Steady State/45C/2"
file_name = "SS_2_1.xlsx"
sheet = "Sheet1"

experimental_data = data_importer(folder_path, file_name, sheet)

N_exp = np.shape(experimental_data)[0]

# Calculate concentrations using the optimized theta values
for i, data in enumerate(experimental_data):
    temp, initQ, initC, _, _ = data
    solution = solve_cstr(theta_optimal, temp, initQ, initC)

sigma = [0.017, 0.014]
info_matrix, sensitivity_matrix, dynamic_information = information_matrix(theta_optimal, N_exp, experimental_data, sigma)

RFIs +=  [np.trace(np.sum(dynamic_information[0:5], axis=0))/trace_total]  # RFI of the experiment E10a
RFIs +=  [np.trace(np.sum(dynamic_information[5:10], axis=0))/trace_total]  # RFI of the experiment E10b
RFIs +=  [np.trace(np.sum(dynamic_information[10:15], axis=0))/trace_total]  # RFI of the experiment E10c

SM += [np.sum(sensitivity_matrix[0:5], axis=0)]
SM += [np.sum(sensitivity_matrix[5:10], axis=0)]
SM += [np.sum(sensitivity_matrix[10:15], axis=0)]


# Experiment 11
folder_path = "C:/Users/Andrea/Desktop/Project/DynamicCSTR/Steady State/45C/05"
file_name = "SS_05_05.xlsx"
sheet = "Sheet1"

experimental_data = data_importer(folder_path, file_name, sheet)

N_exp = np.shape(experimental_data)[0]

# Calculate concentrations using the optimized theta values
for i, data in enumerate(experimental_data):
    temp, initQ, initC, _, _ = data
    solution = solve_cstr(theta_optimal, temp, initQ, initC)

sigma = [0.017, 0.014]
info_matrix, sensitivity_matrix, dynamic_information = information_matrix(theta_optimal, N_exp, experimental_data, sigma)


RFIs +=  [np.trace(np.sum(dynamic_information[0:5], axis=0))/trace_total]  # RFI of the experiment E11a
RFIs +=  [np.trace(np.sum(dynamic_information[5:10], axis=0))/trace_total]  # RFI of the experiment E11b
RFIs +=  [np.trace(np.sum(dynamic_information[10:15], axis=0))/trace_total]  # RFI of the experiment E11c

SM += [np.sum(sensitivity_matrix[0:5], axis=0)]
SM += [np.sum(sensitivity_matrix[5:10], axis=0)]
SM += [np.sum(sensitivity_matrix[10:15], axis=0)]


# Experiment 12
folder_path = "C:/Users/Andrea/Desktop/Project/DynamicCSTR/Steady State/45C/05"
file_name = "SS_05_1.xlsx"
sheet = "Sheet1"

experimental_data = data_importer(folder_path, file_name, sheet)

N_exp = np.shape(experimental_data)[0]

# Calculate concentrations using the optimized theta values
for i, data in enumerate(experimental_data):
    temp, initQ, initC, _, _ = data
    solution = solve_cstr(theta_optimal, temp, initQ, initC)

sigma = [0.017, 0.014]
info_matrix, sensitivity_matrix, dynamic_information = information_matrix(theta_optimal, N_exp, experimental_data, sigma)


RFIs +=  [np.trace(np.sum(dynamic_information[0:5], axis=0))/trace_total]  # RFI of the experiment E12a
RFIs +=  [np.trace(np.sum(dynamic_information[5:10], axis=0))/trace_total]  # RFI of the experiment E12b
RFIs +=  [np.trace(np.sum(dynamic_information[10:15], axis=0))/trace_total]  # RFI of the experiment E12c

print('Total trace:', trace_total)

SM += [np.sum(sensitivity_matrix[0:5], axis=0)]
SM += [np.sum(sensitivity_matrix[5:10], axis=0)]
SM += [np.sum(sensitivity_matrix[10:15], axis=0)]


# Now we have to plot the results
experiment_number = np.arange(1, len(RFIs) + 1)
x_label = ['SS1a', 'SS1b', 'SS1c', 'SS2a', 'SS2b', 'SS2c', 'SS3a', 'SS3b', 'SS3c', 'SS4a', 'SS4b', 'SS4c', 'SS5a', 'SS5b', 'SS5c', 'SS6a', 'SS6b', 'SS6c',
            'SS7a', 'SS7b', 'SS7c', 'SS8a', 'SS8b', 'SS8c', 'SS9a', 'SS9b', 'SS9c', 'SS10a', 'SS10b', 'SS10c', 'SS11a', 'SS11b', 'SS11c', 'SS12a', 'SS12b', 'SS12c']
"""plt.figure(figsize=(10, 6))

plt.bar(experiment_number, RFIs, color='royalblue')
plt.xticks(experiment_number, x_label, rotation=90)  # Mostrare i nomi degli esperimenti ruotati di 90°

# Title and labels
#plt.title('Valori di RFI per Esperimento', fontsize=16)
plt.xlabel('Experiment', fontsize=14)
plt.ylabel('Relative Fisher Information (RFI)', fontsize=14)

plt.tight_layout()
#plt.show()

values_11 = [sm[0, 0] for sm in SM]  # Posizione (1,1)
values_22 = [sm[1, 1] for sm in SM]  # Posizione (2,2)

# 
# Plot (1,1)
plt.figure(figsize=(10, 8))
plt.plot(range(1, len(values_11) + 1), values_11, label=f'Sensitivity KP1')
plt.xlabel("Experiment", fontsize=15)
plt.xticks(experiment_number, x_label, rotation=90, fontsize=11)
plt.ylabel("Sensitivity of the parameter KP1", fontsize=15)
#plt.title("Plot dei valori in SM(1,1)")
#plt.grid(True)
#plt.legend()
#plt.show()

# Plot (2,2)
plt.figure(figsize=(10, 8))
plt.plot(range(1, len(values_22) + 1), values_22, label=f'Sensitivity KP2')
plt.xlabel("Experiment", fontsize=15)
plt.xticks(experiment_number, x_label, rotation=90, fontsize=11)
plt.ylabel("Sensitivity of the parameter KP2", fontsize=15)
#plt.grid(True)
#plt.legend()
#plt.show()"""


# Plot with the averages 

x_label_ss = ['SS1', 'SS2', 'SS3', 'SS4', 'SS5',  'SS6',
            'SS7', 'SS8', 'SS9', 'SS10', 'SS11','SS12']

RFI_ss1 = np.sum(RFIs[0:2])/np.sum(RFIs)
RFI_ss2 = np.sum(RFIs[3:5])/np.sum(RFIs)
RFI_ss3 = np.sum(RFIs[6:8])/np.sum(RFIs)
RFI_ss4 = np.sum(RFIs[9:11])/np.sum(RFIs)
RFI_ss5 = np.sum(RFIs[12:14])/np.sum(RFIs)
RFI_ss6 = np.sum(RFIs[15:17])/np.sum(RFIs)
RFI_ss7 = np.sum(RFIs[18:20])/np.sum(RFIs)
RFI_ss8 = np.sum(RFIs[21:23])/np.sum(RFIs)
RFI_ss9 = np.sum(RFIs[24:26])/np.sum(RFIs)
RFI_ss10 = np.sum(RFIs[27:29])/np.sum(RFIs)
RFI_ss11 = np.sum(RFIs[30:32])/np.sum(RFIs)
RFI_ss12 = np.sum(RFIs[33:35])/np.sum(RFIs)

RFIs_SS = [RFI_ss1, RFI_ss2, RFI_ss3, RFI_ss4, RFI_ss5, RFI_ss6, RFI_ss7, RFI_ss8, RFI_ss9, RFI_ss10, RFI_ss11, RFI_ss12]

exp_num_ss = np.arange(1, len(x_label_ss) + 1)
plt.figure(figsize=(10, 6))

plt.bar(exp_num_ss, RFIs_SS, color='royalblue')
plt.xticks(exp_num_ss, x_label_ss, rotation=90)  # experiment names 

# title and labels
#plt.title('Valori di RFI per Esperimento', fontsize=16)
plt.xlabel('Experiment', fontsize=14)
plt.ylabel('Relative Fisher Information (RFI)', fontsize=14)

plt.tight_layout()
plt.show()