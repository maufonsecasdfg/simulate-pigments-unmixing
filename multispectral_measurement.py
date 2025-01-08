import numpy as np

def multispectral_experiment(
    wavelengths,
    measured_wavelengths,
    reflectance,
    measurement_noise = 0.01,
    seed=40
):
    """
    Simulate a multispectral experiment with measument noise. Simplified to specific waveleghts instead of wavelegth bands.

    Args:
        wavelengths (np.ndarray): Wavelengths in nm, shape (num_wavelengths,)
        measured_wavelengths (np.ndarray): Wavelengths where reflectance is measured, shape (num_measured_wavelengths,)
        reflectance (np.ndarray): Reflectance curves, shape (num_pigments, num_wavelengths)
        measurement_noise (float): Multiplicative factor for Gaussian(0,1) noise added to measurement

    Returns:
        multispectral_measurement (np.ndarray): Simulated multispectral measurement, shape (num_pigments, num_measured_wavelegths)
    """

    if reflectance.shape[1] != len(wavelengths):
        raise ValueError("Reflectance curve must have the same number of wavelengths as `wavelengths`.")
    
    # Interpolate reflectance to measured wavelengths
    interpolated_reflectance = np.array([
        np.interp(measured_wavelengths, wavelengths, pigment_reflectance)
        for pigment_reflectance in reflectance
    ])
    
    rng = np.random.default_rng(seed)
    
    # Add Gaussian noise to simulate measurement errors
    noise = rng.normal(loc=0.0, scale=measurement_noise, size=interpolated_reflectance.shape)
    multispectral_measurement = np.clip(interpolated_reflectance + noise, 0, 1)
    
    return multispectral_measurement
    