import colour
import numpy as np
import matplotlib.pyplot as plt

def reflectance_to_rgb(wavelengths, reflectance, illuminant='D65', observer='CIE 1964 10 Degree Standard Observer'):
    """
    Convert reflectance spectrum to RGB color using CIE color matching functions.
    
    Args:
        wavelengths (np.ndarray): Wavelengths in nm, shape (num_wavelengths,)
        reflectance (np.ndarray): Reflectance values, shape (num_wavelengths,)
        illuminant (str): Name of the standard illuminant (default: 'D65').
        observer (str): Name of the color matching functions (default: 'CIE 1964 10 Degree Standard Observer').
    
    Returns:
        rgb (tuple): RGB values scaled between 0 and 1.
    """
    # Get the CIE color matching functions
    cmfs = colour.MSDS_CMFS.get(observer)
    if cmfs is None:
        raise ValueError(f"Color matching functions for observer '{observer}' not found.")
    
    # Get the illuminant spectral power distribution
    sd_illuminant = colour.SDS_ILLUMINANTS.get(illuminant)
    if sd_illuminant is None:
        raise ValueError(f"Illuminant '{illuminant}' not found.")

    # Define the spectral shape based on wavelengths
    shape = colour.SpectralShape(
        start=wavelengths[0],
        end=wavelengths[-1],
        interval=wavelengths[1] - wavelengths[0]
    )

    # Interpolate the CMFs and illuminant to match the spectral shape
    cmfs = cmfs.copy().align(shape)
    sd_illuminant = sd_illuminant.copy().align(shape)
    
    # Create spectral distributions for reflectance
    sd_reflectance = colour.SpectralDistribution(reflectance, wavelengths)
    
    # Calculate the XYZ tristimulus values
    X, Y, Z = colour.sd_to_XYZ(sd_reflectance, cmfs, sd_illuminant)
    
    XYZ = np.array([X, Y, Z])
    
    # Normalize by the illuminant's Y to get relative XYZ
    X_illuminant, Y_illuminant, Z_illuminant = colour.sd_to_XYZ(sd_illuminant, cmfs)
    XYZ /= Y_illuminant

    # Convert XYZ to sRGB
    RGB = colour.XYZ_to_sRGB(XYZ*100)  # XYZ_to_sRGB expects XYZ scaled to [0, 100]
    
    # Clip RGB values to [0, 1] range
    RGB = np.clip(RGB, 0, 1)
    
    return tuple(RGB)

def plot_pigment_colors(wavelengths, R, K, S, pigments_colors):
    """
    Plot the reflectance, absorption, and scattering curves along with a color patch.
    
    Args:
        wavelengths (np.ndarray): Wavelengths in nm, shape (num_wavelengths,)
        R (np.ndarray): Reflectance curves, shape (num_pigments, num_wavelengths)
        K (np.ndarray): Absorption coefficients, shape (num_pigments, num_wavelengths)
        S (np.ndarray): Scattering coefficients, shape (num_pigments, num_wavelengths)
        pigments_colors (list): List of RGB tuples for each pigment.
    """
    num_pigments = R.shape[0]
    fig, axes = plt.subplots(num_pigments, 4, figsize=(20, 5 * num_pigments))
    
    if num_pigments == 1:
        axes = np.expand_dims(axes, axis=0)  # Ensure axes is 2D
    
    for k in range(num_pigments):
        # Plot Reflectance
        ax = axes[k, 0]
        ax.plot(wavelengths, R[k], color='red')
        ax.set_title(f'Pigment {k+1} - Reflectance')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Reflectance (R)')
        ax.grid(True)
        
        # Plot Absorption
        ax = axes[k, 1]
        ax.plot(wavelengths, K[k], color='blue')
        ax.set_title(f'Pigment {k+1} - Absorption (K)')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Absorption (K) cm⁻¹')
        ax.grid(True)
        
        # Plot Scattering
        ax = axes[k, 2]
        ax.plot(wavelengths, S[k], color='orange')
        ax.set_title(f'Pigment {k+1} - Scattering (S)')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Scattering (S) cm⁻¹')
        ax.grid(True)
        
        # Plot Color Patch
        ax = axes[k, 3]
        ax.axis('off')  # Hide axes
        ax.imshow([[pigments_colors[k]]], extent=[0, 1, 0, 1])
        ax.set_title(f'Pigment {k+1} - Perceived Color')
    
    plt.tight_layout()
    plt.show()