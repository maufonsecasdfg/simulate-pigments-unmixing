import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def compute_reflectance_from_absorption_and_scattering(absorption, scattering):

    F_R = absorption / (scattering + 1e-8)  # Avoid division by zero
    
    # Solve the quadratic equation: R^2 - 2(1 + F_R)R + 1 = 0
    a_quad = 1.0
    b_quad = -2.0 * (1.0 + F_R)
    c_quad = 1.0
    
    discriminant = b_quad**2 - 4 * a_quad * c_quad
    discriminant = np.maximum(discriminant, 0)  # Ensure non-negative
    
    sqrt_discriminant = np.sqrt(discriminant)
    
    R = (-b_quad - sqrt_discriminant) / (2 * a_quad)
    
    # Ensure R is within [0,1]
    R = np.clip(R, 0.0, 1.0)
    
    return R

def generate_pigment_coefficients_physical(
    num_pigments=3,
    wavelength_start=400,
    wavelength_end=700,
    max_peaks_per_pigment=3,
    peak_strength_range=(2, 5),
    peak_width_range=(10, 30),
    correlation_strength=0.5,
    K_min=0.1,
    pre_peak_smoothing_sigma=7,
    post_peak_smoothing_sigma=2,
    K_baseline=2.0,
    S_baseline=2.0,
    K_variation=1.,
    S_variation=3.,
    seed=42
):
    """
    Generate physically consistent absorption (K), scattering (S), and reflectance (R)
    using Kubelka-Munk theory, maintaining a negative correlation between K and S.

    Args:
        num_pigments (int): Number of pigments.
        num_wavelengths (int): Number of wavelength points.
        wavelength_start (float): Start wavelength in nm.
        wavelength_end (float): End wavelength in nm.
        max_peaks_per_pigment (int): Maximum number of peaks each pigment can have.
        peak_strength_range (tuple): Min and max strength multipliers for peaks (S or K).
        peak_width_range (tuple): Min and max widths (standard deviation) for Gaussian peaks.
        correlation_strength (float): Scaling factor for negative correlation between K and S.
        K_min (float): Minimum absorption coefficient.
        pre_peak_smoothing_sigma (float): Sigma for Gaussian smoothing before adding peaks.
        post_peak_smoothing_sigma (float): Sigma for Gaussian smoothing after peak addition.
        K_baseline (float): Baseline value for absorption.
        S_baseline (float): Baseline value for scattering.
        K_variation (float): Variation range around K_baseline.
        S_variation (float): Variation range around S_baseline.
        seed (int): Random seed for reproducibility.

    Returns:
        wavelengths (np.ndarray): Wavelengths in nm, shape (num_wavelengths,)
        reflectance (np.ndarray): Reflectance curves, shape (num_pigments, num_wavelengths)
        absorption (np.ndarray): Absorption coefficients, shape (num_pigments, num_wavelengths)
        scattering (np.ndarray): Scattering coefficients, shape (num_pigments, num_wavelengths)
    """
    rng = np.random.default_rng(seed)
    
    # Define wavelength axis
    wavelengths = np.arange(wavelength_start, wavelength_end + 1, 1)
    num_wavelengths = len(wavelengths)
    
    #  Initialize and Smooth Absorption with Baseline
    absorption = rng.uniform(K_baseline - K_variation, K_baseline + K_variation, size=(num_pigments, num_wavelengths))
    for k in range(num_pigments):
        absorption[k] = gaussian_filter1d(absorption[k], sigma=pre_peak_smoothing_sigma)
    
    # Initialize and Smooth Scattering with Baseline
    scattering = rng.uniform(S_baseline - S_variation, S_baseline + S_variation, size=(num_pigments, num_wavelengths))
    for k in range(num_pigments):
        scattering[k] = gaussian_filter1d(scattering[k], sigma=pre_peak_smoothing_sigma)
    
    # Assign Unique Peaks to Absorption and Scattering with Negative Correlation
    for k in range(num_pigments):
        num_peaks = rng.integers(1, max_peaks_per_pigment + 1)  # At least 1 peak
        existing_centers = []
        
        for peak_num in range(num_peaks):
            # Prevent overlapping peaks by ensuring a minimum distance between centers
            min_distance = 30  # in nm
            attempts = 0
            max_attempts = 10
            while attempts < max_attempts:
                center = rng.uniform(wavelength_start + 20, wavelength_end - 20)
                if all(abs(center - ec) > min_distance for ec in existing_centers):
                    existing_centers.append(center)
                    break
                attempts += 1
            else:
                # Could not place a new peak without overlapping; skip this peak
                continue
            
            width = rng.uniform(*peak_width_range)
            
            # Create Gaussian peak
            peak = np.exp(-0.5 * ((wavelengths - center) / width) ** 2)
            
            # Decide randomly whether to add peak to scattering or to absorption
            add_to_S = rng.random() < 0.5 
            
            # Determine peak strength within the specified range
            peak_strength_min, peak_strength_max = peak_strength_range
            peak_strength = rng.uniform(peak_strength_min, peak_strength_max)
            
            if add_to_S:
                scattering[k] += peak_strength * peak
                # Subtract from Absorption to enforce negative correlation
                K_reduction = correlation_strength * peak_strength * peak
                absorption[k] -= K_reduction
                # Ensure Absorption does not fall below K_min
                absorption[k] = np.maximum(absorption[k], K_min)
            else:
                absorption[k] += peak_strength * peak
                # Subtract from Scattering to enforce negative correlation
                S_reduction = correlation_strength * peak_strength * peak
                scattering[k] -= S_reduction
                # Ensure Scattering does not fall below 0
                scattering[k] = np.maximum(scattering[k], 0.0)
        
        # Smooth Scattering and Absorption After Peak Addition to Ensure Smooth Transitions
        scattering[k] = gaussian_filter1d(scattering[k], sigma=post_peak_smoothing_sigma)
        absorption[k] = gaussian_filter1d(absorption[k], sigma=post_peak_smoothing_sigma)
    
    # Ensure Scattering and Absorption remain positive
    scattering = np.maximum(scattering, 0.0)
    absorption = np.maximum(absorption, K_min)
    
    R = compute_reflectance_from_absorption_and_scattering(absorption, scattering)
    
    return wavelengths, R, absorption, scattering