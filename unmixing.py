import numpy as np
import pymc as pm
from scipy.optimize import minimize, nnls

def pymc_km_reflectance(
    absorption, 
    scattering,
):
    F_R = absorption / (scattering + 1e-8)
    
    a_quad = 1.0
    b_quad = -2.0 * (1.0 + F_R)
    c_quad = 1.0

    discriminant = b_quad**2 - 4 * a_quad * c_quad
    discriminant = pm.math.maximum(discriminant, 0.0) 
    
    sqrt_discriminant = pm.math.sqrt(discriminant)

    R = (-b_quad - sqrt_discriminant) / (2 * a_quad)

    R = pm.math.clip(R, 0.0, 1.0)
    
    return R

def km_reflectance(absorption, scattering):
    F_R = absorption / (scattering + 1e-8)
    a_quad = 1.0
    b_quad = -2.0 * (1.0 + F_R)
    c_quad = 1.0
    
    discriminant = b_quad**2 - 4 * a_quad * c_quad
    sqrt_discriminant = np.sqrt(np.maximum(discriminant, 0.0))
    R = (-b_quad - sqrt_discriminant) / (2 * a_quad)
    return np.clip(R, 0.0, 1.0)

def bayesian_unmixing(
    R_meas: np.ndarray,
    library_absorption_full: np.ndarray,
    library_scattering_full: np.ndarray,
    library_wavelengths: np.ndarray,
    ms_wavelengths: np.ndarray,
    sigma_noise: float = 0.01,
    draws: int = 2000,
    tune: int = 1500,
    chains: int = 4,
    target_accept: float = 0.95
):
    """
    Perform Bayesian unmixing of reflectance data using KM theory with flexible priors,
    and a robust StudentT likelihood. 

    Parameters
    ----------
    R_meas : np.ndarray
        Measured reflectances at N_ms wavelengths (shape: (N_ms,))
    library_absorption_full : np.ndarray
        Absorption curves (K) for each of M library pigments (shape: (M, N_wvl))
    library_scattering_full : np.ndarray
        Scattering curves (S) for each of M library pigments (shape: (M, N_wvl))
    library_wavelengths : np.ndarray
        Wavelengths corresponding to the library (shape: (N_wvl,))
    ms_wavelengths : np.ndarray
        Wavelengths at which R_meas is provided (shape: (N_ms,))
    sigma_noise : float
        Noise prior scale for the reflectance observations
    draws : int
        Number of posterior samples
    tune : int
        Number of NUTS tuning steps
    chains : int
        Number of MCMC chains
    target_accept : float
        Target acceptance rate for the NUTS sampler

    Returns
    -------
    trace : InferenceData
        PyMC inference data with posterior samples
    model : pm.Model
        The PyMC model object
    """

    M, N_wvl = library_absorption_full.shape
    N_ms = len(ms_wavelengths)

    def interp_to_ms_wavelengths(curves_2d: np.ndarray):
        return np.array([
            np.interp(ms_wavelengths, library_wavelengths, curves_2d[i])
            for i in range(M)
        ])

    K_lib_ms = interp_to_ms_wavelengths(library_absorption_full)
    S_lib_ms = interp_to_ms_wavelengths(library_scattering_full)

    with pm.Model() as model:
        raw_w = pm.Normal("raw_w", mu=0.0, sigma=1.0, shape=M)
        w = pm.Deterministic("w", pm.math.exp(raw_w) / pm.math.sum(pm.math.exp(raw_w)))

        K_mix = pm.math.sum(w[:, None] * K_lib_ms, axis=0)
        S_mix = pm.math.sum(w[:, None] * S_lib_ms, axis=0)

        R_naked = pymc_km_reflectance(K_mix, S_mix)
        R_pred = pm.Deterministic("R_pred", R_naked)

        # Hard constraint potential -> Avoid R_pred outside [0, 1]
        invalid_count = pm.math.sum(pm.math.lt(R_pred, 0)) + pm.math.sum(pm.math.gt(R_pred, 1))
        is_invalid = pm.math.gt(invalid_count, 0)
        pm.Potential("valid_reflectance", pm.math.switch(is_invalid, -np.inf, 0.0))

        # Likelihood
        nu = pm.Exponential("nu", lam=1/10)
        sigma = pm.HalfNormal("sigma", sigma=sigma_noise)
        pm.StudentT("obs", nu=nu, mu=R_pred, sigma=sigma, observed=R_meas)

        initvals = {
            "raw_w": np.zeros(M),  
            "nu_log__": np.log(30), 
            "sigma_log__": np.log(sigma_noise)
        }

        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            max_treedepth=20,
            return_inferencedata=True,
            init="adapt_diag",
            initvals=initvals,
            seed=42
        )

    return trace, model


def greedy_unmixing(
    R_meas: np.ndarray,
    library_absorption_full: np.ndarray,
    library_scattering_full: np.ndarray,
    library_wavelengths: np.ndarray,
    ms_wavelengths: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-7
):
    """
    Greedy unmixing using Kubelka-Munk theory with partial weights.
    """
    
    def project_to_simplex(w):
        """
        Projects a vector w onto the simplex { x >= 0, sum(x) = 1 }.
        """
        if np.all(w >= 0) and np.sum(w) == 1.0:
            return w
        
        u = np.sort(w)[::-1]
        cssv = np.cumsum(u)

        rho = np.where(u * np.arange(1, len(w)+1) > (cssv - 1))[0][-1]
        theta = (cssv[rho] - 1.0)/(rho+1.0)
        
        w_proj = np.maximum(w - theta, 0.0)
        w_proj /= np.sum(w_proj)
        return w_proj


    def interp_to_ms_wavelengths(curves_2d):
        return np.array([
            np.interp(ms_wavelengths, library_wavelengths, curves_2d[i])
            for i in range(curves_2d.shape[0])
        ])

    K_lib_ms = interp_to_ms_wavelengths(library_absorption_full)
    S_lib_ms = interp_to_ms_wavelengths(library_scattering_full)

    M = K_lib_ms.shape[0]
    w = np.ones(M) / M
    K_current = np.sum(w[:, None] * K_lib_ms, axis=0)
    S_current = np.sum(w[:, None] * S_lib_ms, axis=0)
    R_current = km_reflectance(K_current, S_current)
    prev_error = np.mean((R_meas - R_current)**2)
    
    for iteration in range(max_iter):
        best_error = prev_error
        best_w = w

        step_size = 0.05
        for i in range(M):
            for j in range(M):
                if i == j:
                    continue
                trial_w = w.copy()
                
                delta = min(step_size, trial_w[j]) 
                trial_w[i] += delta
                trial_w[j] -= delta
                
                trial_w = project_to_simplex(trial_w)
                
                K_trial = np.sum(trial_w[:, None] * K_lib_ms, axis=0)
                S_trial = np.sum(trial_w[:, None] * S_lib_ms, axis=0)
                R_trial = km_reflectance(K_trial, S_trial)
                error = np.mean((R_meas - R_trial)**2)
                
                if error < best_error:
                    best_error = error
                    best_w = trial_w

        if (prev_error - best_error) < tol:
            break
        
        w = best_w
        K_current = np.sum(w[:, None] * K_lib_ms, axis=0)
        S_current = np.sum(w[:, None] * S_lib_ms, axis=0)
        R_current = km_reflectance(K_current, S_current)
        prev_error = best_error
    
    return w, R_current


def naive_unmixing(
    R_meas: np.ndarray,
    library_absorption_full: np.ndarray,
    library_scattering_full: np.ndarray,
    library_wavelengths: np.ndarray,
    ms_wavelengths: np.ndarray
):
    """
    Naive unmixing of reflectance data
    """

    def interp_to_ms_wavelengths(curves_2d: np.ndarray):
        return np.array([
            np.interp(ms_wavelengths, library_wavelengths, curves_2d[i])
            for i in range(curves_2d.shape[0])
        ])

    K_lib_ms = interp_to_ms_wavelengths(library_absorption_full)
    S_lib_ms = interp_to_ms_wavelengths(library_scattering_full)
    
    M = K_lib_ms.shape[0]
    A = np.zeros((len(ms_wavelengths), M))
    
    for i in range(M):
        R = km_reflectance(K_lib_ms[i], S_lib_ms[i])
        A[:, i] = R
    
    w, _ = nnls(A, R_meas)
    
    w /= np.sum(w)
    
    return w