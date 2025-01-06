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

def plot_pigment_colors(wavelengths, R, K, S, pigment_colors):
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
        ax.imshow([[pigment_colors[k]]], extent=[0, 1, 0, 1])
        ax.set_title(f'Pigment {k+1} - Perceived Color')
    
    plt.tight_layout()
    plt.show()
    
import matplotlib.pyplot as plt
import numpy as np

def plot_mixture_colors(mixture_indexes, mixture_colors, pigment_colors, weights):
    """
    Plots a grid showing the constituent pigments (with their weights) and the
    resulting mixture color for each mixture. 
    
    Args:
        mixture_indexes (np.ndarray): Array of shape (num_mixtures, pigments_per_mixture)
            containing the indices of the pigments that make up each mixture.
        mixture_colors (list or np.ndarray): List (length num_mixtures) of RGB tuples
            for the final mixture colors.
        pigment_colors (list or np.ndarray): List (length num_pigments) of RGB tuples
            for the individual pigment colors.
        weights (np.ndarray): Array of shape (num_mixtures, pigments_per_mixture) containing
            the weights of each pigment within each mixture.
    """
    num_mixtures = mixture_indexes.shape[0]
    pigments_per_mixture = mixture_indexes.shape[1]
    
    # Create a grid with pigments + 2 columns (1 for "=" and 1 for the mixture)
    fig, axes = plt.subplots(
        nrows=num_mixtures, 
        ncols=pigments_per_mixture + 2, 
        figsize=(3 * (pigments_per_mixture + 2), 3 * num_mixtures)
    )
    
    # Ensure axes is always 2D
    if num_mixtures == 1:
        axes = np.array([axes])
    
    for mix_idx in range(num_mixtures):
        for pig_idx in range(pigments_per_mixture):
            ax = axes[mix_idx, pig_idx]
            ax.axis('off')
            
            weight = weights[mix_idx, pig_idx]
            p_idx = mixture_indexes[mix_idx, pig_idx]
            
            if weight == 0 or not (0 <= p_idx < len(pigment_colors)):
                # Invalid index or zero weight -> White square
                ax.imshow(np.ones((1, 1, 3)), extent=[0, 1, 0, 1])
            else:
                pigment_color = pigment_colors[p_idx]
                ax.imshow([[pigment_color]], extent=[0, 1, 0, 1])
                ax.text(
                    0.5, 0.5,
                    f"w={weight:.2f}",
                    ha='center', va='center',
                    color='white',
                    bbox=dict(facecolor='black', alpha=0.5),
                    fontsize=9
                )
                ax.set_title(f"Pigment {p_idx + 1}", fontsize=9)
        
        # "=" Sign Column
        ax_equals = axes[mix_idx, pigments_per_mixture]
        ax_equals.axis('off')
        ax_equals.text(
            0.5, 0.5,
            "=",
            ha='center', va='center',
            fontsize=16,
            fontweight='bold'
        )
        
        # Mixture color
        ax_mix = axes[mix_idx, -1]
        ax_mix.axis('off')
        mixture_color = mixture_colors[mix_idx]
        ax_mix.imshow([[mixture_color]], extent=[0, 1, 0, 1])
        ax_mix.set_title(f"Mixture {mix_idx + 1}", fontsize=10)
    
    plt.subplots_adjust(hspace=1.5, wspace=0.6)
    plt.tight_layout()
    plt.show()
