import matplotlib.pyplot as plt
import arviz as az
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.colors as mcolors
import pandas as pd

def weight_mse(found_weights, true_weights):
    found_weights = np.array(found_weights)
    mse = ((found_weights-true_weights)**2).mean()
    return mse

def display_mse_comparison(data):
    """
    Parameters:
        data (dict): Dictionary containing MSE values structured as:
            {
                'weight_mse': {
                    'naive': [..],
                    'greedy': [..],
                    'bayesian': [..]
                },
                'reflectance_mse': {
                    'naive': [..],
                    'greedy': [..],
                    'bayesian': [..]
                }
            }
    """
    weight_df = pd.DataFrame(data['weight_mse'])
    weight_df.index += 1
    reflectance_df = pd.DataFrame(data['reflectance_mse'])
    reflectance_df.index += 1
    
    # Define a function for row-wise coloring using adjusted Reds colormap
    def apply_reds_colormap(row):
        norm = mcolors.Normalize(vmin=row.min(), vmax=row.max())
        cmap = plt.cm.Reds
        
        adjusted_cmap = mcolors.LinearSegmentedColormap.from_list(
            "adjusted_reds",
            cmap(np.linspace(0, 0.8, 256))
        )
        
        styled_row = []
        for val in row:
            color = mcolors.to_hex(adjusted_cmap(norm(val)))
            if val == row.min():
                styled_row.append(f'background-color: {color}; border: 2px solid black;')
            else:
                styled_row.append(f'background-color: {color};')
        return styled_row
    
    weight_styled = weight_df.style.apply(apply_reds_colormap, axis=1).set_caption('Weight MSE')
    reflectance_styled = reflectance_df.style.apply(apply_reds_colormap, axis=1).set_caption('Reflectance MSE')
    
    return weight_styled, reflectance_styled


def posterior_predictive_check(trace, R_meas, ms_wavelengths):
    """
    Compare posterior predictive reflectance curves with the measured reflectance data.
    """
    R_pred_samples = trace.posterior["R_pred"].values

    R_pred_samples = R_pred_samples.reshape(-1, R_pred_samples.shape[-1])

    plt.figure(figsize=(12, 6))
    plt.plot(ms_wavelengths, R_meas, 'o-', label='Measured Reflectance', color='black')
    
    for i in range(min(100, len(R_pred_samples))):
        plt.plot(ms_wavelengths, R_pred_samples[i], alpha=0.1, color='gray')
    
    plt.plot(ms_wavelengths, np.mean(R_pred_samples, axis=0), label='Mean Predicted Reflectance', color='red')
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.title('Posterior Predictive Check: Measured vs Predicted Reflectance')
    plt.legend()
    plt.show()
    
def posterior_diagnostics(trace):
    """
    Visual diagnostic of posterior distributions for the pigment weights
    """
    az.plot_posterior(trace, var_names=['w'])
    plt.show()

def km_reflectance(absorption, scattering):
    F_R = absorption / (scattering + 1e-8)
    a_quad = 1.0
    b_quad = -2.0 * (1.0 + F_R)
    c_quad = 1.0
    
    discriminant = b_quad**2 - 4 * a_quad * c_quad
    sqrt_discriminant = np.sqrt(np.maximum(discriminant, 0.0))
    R = (-b_quad - sqrt_discriminant) / (2 * a_quad)
    return np.clip(R, 0.0, 1.0)

def greedy_predictive_check(w_greedy, R_meas, library_absorption_full, library_scattering_full, library_wavelengths, ms_wavelengths):
    """
    Compare greedy reflectance predictions with the measured reflectance data.
    """

    def interp_to_ms_wavelengths(curves_2d: np.ndarray):
        return np.array([
            np.interp(ms_wavelengths, library_wavelengths, curves_2d[i])
            for i in range(curves_2d.shape[0])
        ])

    K_lib_ms = interp_to_ms_wavelengths(library_absorption_full)
    S_lib_ms = interp_to_ms_wavelengths(library_scattering_full)
    
    K_mix = np.sum(w_greedy[:, None] * K_lib_ms, axis=0)
    S_mix = np.sum(w_greedy[:, None] * S_lib_ms, axis=0)
    R_pred = km_reflectance(K_mix, S_mix)

    plt.figure(figsize=(12, 6))
    plt.plot(ms_wavelengths, R_meas, 'o-', label='Measured Reflectance', color='black')
    plt.plot(ms_wavelengths, R_pred, label='Greedy Predicted Reflectance', color='blue')
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.title('Greedy Predictive Check: Measured vs Predicted Reflectance')
    plt.legend()
    plt.show()
    
def naive_predictive_check(w_naive, R_meas, library_absorption_full, library_scattering_full, library_wavelengths, ms_wavelengths):
    """
    Compare naive reflectance predictions with the measured reflectance data.
    """

    def interp_to_ms_wavelengths(curves_2d: np.ndarray):
        return np.array([
            np.interp(ms_wavelengths, library_wavelengths, curves_2d[i])
            for i in range(curves_2d.shape[0])
        ])

    K_lib_ms = interp_to_ms_wavelengths(library_absorption_full)
    S_lib_ms = interp_to_ms_wavelengths(library_scattering_full)

    K_mix = np.sum(w_naive[:, None] * K_lib_ms, axis=0)
    S_mix = np.sum(w_naive[:, None] * S_lib_ms, axis=0)
    R_pred = km_reflectance(K_mix, S_mix)

    plt.figure(figsize=(12, 6))
    plt.plot(ms_wavelengths, R_meas, 'o-', label='Measured Reflectance', color='black')
    plt.plot(ms_wavelengths, R_pred, label='Naive Predicted Reflectance', color='green')
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.title('Naive Predictive Check: Measured vs Predicted Reflectance')
    plt.legend()
    plt.show()
    
def compare_all_predictions(
    trace, w_greedy, w_naive,
    R_meas, library_absorption_full, library_scattering_full,
    library_wavelengths, ms_wavelengths
):
    """
    Compare Bayesian, Greedy, and Naive predictions in a single plot with three subplots.
    Include Mean Squared Error (MSE) for each method.
    The Bayesian subplot includes sampled predictions.
    """

    def compute_reflectance(w):
        K_lib_ms = np.array([
            np.interp(ms_wavelengths, library_wavelengths, library_absorption_full[i])
            for i in range(library_absorption_full.shape[0])
        ])
        S_lib_ms = np.array([
            np.interp(ms_wavelengths, library_wavelengths, library_scattering_full[i])
            for i in range(library_scattering_full.shape[0])
        ])
        K_mix = np.sum(w[:, None] * K_lib_ms, axis=0)
        S_mix = np.sum(w[:, None] * S_lib_ms, axis=0)
        return km_reflectance(K_mix, S_mix)

    # Bayesian Predictions: Samples and Mean
    R_pred_samples = trace.posterior["R_pred"].values
    R_pred_samples = R_pred_samples.reshape(-1, R_pred_samples.shape[-1])
    R_pred_bayesian = np.mean(R_pred_samples, axis=0)

    # Greedy and Naive Predictions
    R_pred_greedy = compute_reflectance(w_greedy)
    R_pred_naive = compute_reflectance(w_naive)

    # Calculate Mean Squared Errors
    mse_bayesian = mean_squared_error(R_meas, R_pred_bayesian)
    mse_greedy = mean_squared_error(R_meas, R_pred_greedy)
    mse_naive = mean_squared_error(R_meas, R_pred_naive)

    fig, axs = plt.subplots(3, 1, figsize=(12, 12))
    
    # Naive Prediction
    axs[0].plot(ms_wavelengths, R_meas, 'o-', label='Measured Reflectance', color='black')
    axs[0].plot(ms_wavelengths, R_pred_naive, label='Naive Prediction', color='green')
    axs[0].set_title(f'Naive Prediction (Reflectance MSE: {mse_naive:.4f})')
    
     # Greedy Prediction
    axs[1].plot(ms_wavelengths, R_meas, 'o-', label='Measured Reflectance', color='black')
    axs[1].plot(ms_wavelengths, R_pred_greedy, label='Greedy Prediction', color='blue')
    axs[1].set_title(f'Greedy Prediction (Reflectance MSE: {mse_greedy:.4f})')

    # Bayesian Prediction with Samples
    axs[2].plot(ms_wavelengths, R_meas, 'o-', label='Measured Reflectance', color='black')
    for i in range(min(100, len(R_pred_samples))):
        axs[2].plot(ms_wavelengths, R_pred_samples[i], alpha=0.1, color='gray')
    axs[2].plot(ms_wavelengths, R_pred_bayesian, label='Mean Bayesian Prediction', color='red')
    axs[2].set_title(f'Bayesian Prediction (Reflectance MSE: {mse_bayesian:.4f})')

    for ax in axs:
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Reflectance')
        ax.legend()

    plt.tight_layout()
    plt.show()
    
    return mse_naive, mse_greedy, mse_bayesian