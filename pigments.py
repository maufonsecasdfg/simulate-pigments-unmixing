import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from visualization import reflectance_to_rgb

def compute_reflectance_from_absorption_and_scattering(
    absorption, 
    scattering,
    non_ideal_factor=0.0, 
    seed=42):
    """
    Compute reflectance from absorption and scattering using 
    the standard Kubelka–Munk formula, but allow for a controlled deviation.

    Args:
        absorption (np.ndarray): Array of absorption coefficients
        scattering (np.ndarray): Array of scattering coefficients
        non_ideal_factor (float): 0.0 = Offset added to (K/S). 
                                        0.0 = ideal KM.
                                        >0   => increases effective absorption ratio
                                        <0   => decreases effective absorption ratio
        seed (int): Random seed for reproducible perturbations.

    Returns:
        R (np.ndarray): Reflectance in [0, 1], same shape as absorption/scattering.
    """

    F_R = absorption / (scattering + 1e-8)  # Avoid division by zero
    
    F_R += non_ideal_factor # Perturb the ratio to simulate deviations from KM theory
    
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

def generate_pigments(
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
        non_ideal_factor=0.0,
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
        non_ideal_factor (float): Controls how far from ideal KM theory the reflectance can deviate.
                                  0.0 = perfectly ideal KM, larger = more deviation.
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
    
    R = compute_reflectance_from_absorption_and_scattering(absorption, scattering, non_ideal_factor)
    
    pigment_colors = []
    for k in range(R.shape[0]):
        rgb = reflectance_to_rgb(wavelengths, R[k])
        pigment_colors.append(rgb)
    
    return wavelengths, R, absorption, scattering, pigment_colors


def generate_mixtures(
        wavelengths, 
        absorption, 
        scattering,
        mixture_indexes, 
        weights,
        mixing_non_ideal_factor=0.0,
        reflectance_non_ideal_factor=0.0,
        seed= 42
    ):
    """
    absorption: shape (num_pigments, num_wavelengths)
    scattering: shape (num_pigments, num_wavelengths)
    mixture_indexes: shape (num_mixtures, pigments_per_mixture)
    weights: shape (num_mixtures, pigments_per_mixture)
    mixing_non_ideal_factor (float): Std of the noise added to weights. 0 = no noise
    reflectance_non_ideal_factor (float): Perturbation strength for reflectance calculation
    seed (int): Random seed for reproducibility
    """
    
    # Basic shape checks
    if absorption.shape != scattering.shape:
        raise ValueError("Absorption and scattering arrays must have the same shape!")
    
    if mixture_indexes.shape != weights.shape:
        raise ValueError("mixture_indexes and weights must have the same shape!")
    
    num_mixtures, pigments_per_mixture = mixture_indexes.shape
    
    rng = np.random.default_rng(seed)
    
    if mixing_non_ideal_factor > 0.0:
        # Add noise to weights
        noise = rng.normal(loc=0.0, scale=mixing_non_ideal_factor, size=weights.shape)
        noisy_weights = weights + noise

        noisy_weights = np.clip(noisy_weights, 0, None)

        row_sums = noisy_weights.sum(axis=1, keepdims=True) + 1e-12
        noisy_weights /= row_sums

        weights_used = noisy_weights
    else:
        weights_used = weights
    
    #Select the pigments using the mixture index array
    selected_absorption = absorption[mixture_indexes.ravel()]
    selected_absorption = selected_absorption.reshape(num_mixtures, pigments_per_mixture, -1)
    
    selected_scattering = scattering[mixture_indexes.ravel()]
    selected_scattering = selected_scattering.reshape(num_mixtures, pigments_per_mixture, -1)
    
    # Multiply by weights
    weighted_absorption = selected_absorption * weights_used[..., None]
    weighted_scattering = selected_scattering * weights_used[..., None]
    
    # Sum across the pigment dimension (axis=1) to get shape (num_mixtures, num_wavelengths)
    mixture_absorption = weighted_absorption.sum(axis=1)
    mixture_scattering = weighted_scattering.sum(axis=1)
        
    # Compute reflectance
    mixture_reflectance = compute_reflectance_from_absorption_and_scattering(
        mixture_absorption, 
        mixture_scattering,
        reflectance_non_ideal_factor
    )
    
    mixture_colors = []
    for k in range(mixture_reflectance.shape[0]):
        rgb = reflectance_to_rgb(wavelengths, mixture_reflectance[k])
        mixture_colors.append(rgb)
    
    return mixture_reflectance, mixture_absorption, mixture_scattering, mixture_colors