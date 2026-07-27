"""
Module for using neural field theory to simulate neural activity and BOLD signals on cortical 
surfaces.
"""

from __future__ import annotations
from importlib.util import find_spec
from typing import Literal, TYPE_CHECKING, cast
from warnings import warn
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import splu
from neuromodes.eigen import EigenData
from neuromodes.basis import decompose

if TYPE_CHECKING:
    from typing import Literal
    from scipy.sparse import csc_matrix
    from neuromodes.eigen import _CheckKind
    _PDEKind = Literal["fourier", "ode", "fem"]

def sim_nft_waves(
    emodes: NDArray[np.floating] | None,
    evals: NDArray[np.floating] | None,
    nt: int | None = None,
    ext_input: NDArray[np.floating] | None = None,
    dt: float = 1e-4,
    r: float = 17.4,
    gamma: float = 116.0,
    method: _PDEKind = "fourier",
    mass: csc_matrix | None = None,
    stiffness: csc_matrix | None = None, # only used for FEM
    speed_limits: tuple[float, float] | None = (0, 150),
    hetero: NDArray[np.floating] | None = None,
    checks: _CheckKind = True,
    seed: int | None = None,
    cache_input: bool = False,
    n_jobs: int = 1, # only used for FEM
    verbose: int = 0 # only used for FEM
) -> NDArray[np.floating]:
    """
    Simulate neural activity using a Neural Field Theory wave model [1]_ [2]_ [3]_.

    Parameters
    ----------
    emodes : array-like
        The eigenmodes array of shape ``(n_verts, n_modes)``, where ``n_verts`` is the number of
        vertices and ``n_modes`` is the number of eigenmodes.
    evals : array-like
        The eigenvalues array of shape ``(n_modes,)``.
    nt : int, optional
        Number of time points to simulate under white noise input. Note that either ``nt`` or
        ``ext_input`` must be provided. Default is ``None``.
    ext_input : array-like, optional
        External input array of shape ``(n_verts, n_timepoints)``. If ``None``, white noise input is
        generated to simulate ``nt`` time points. Default is ``None``.
    dt : float, optional
        Time step for simulation in seconds. Default is ``1e-4``.
    r : float, optional
        Spatial length scale of wave propagation in millimeters. Default is ``17.4``.
    gamma : float, optional
        Damping rate of wave propagation in seconds^(-1). Default is ``116.0``.
    method : str, optional
        Method for solving the wave PDEs. Either ``'fourier'``, ``'ode'``, or ``fem``. If ``fem``
        (no modal approximation), ``mass`` and ``stiffness`` must be provided while ``emodes`` and
        ``evals`` can be ``None``. Default is ``'fourier'``.
    mass : array-like, optional
        The mass matrix of shape ``(n_verts, n_verts)`` used for decomposition of ``ext_input`` if
        provided and ``method`` is not ``'fem'``. Default is ``None``.
    speed_limits : tuple, optional
        If any wave speeds are outside this range (in meters per second), a warning is raised. If
        ``None``, no warning is raised. Default is ``(0, 150)``.
    hetero : array-like, optional
        Heterogeneity map of shape ``(n_verts,)``, to be provided when ``emodes`` are heterogeneous.
        This is used only to check wave speeds (see ``speed_limits`` above). If not provided, wave
        speed is assumed to be spatially uniform. Default is ``None``.
    checks : bool, optional
        Whether to validate the input arguments. Default is ``True``.
    seed : int, optional
        Random seed for generating external input. Default is ``None``.
    cache_input : bool, optional
        If ``True`` and ``ext_input`` is ``None``, cache the generated random input to avoid
        recomputation for the same values of ``nt``, ``seed``, and number of rows (vertices) in
        ``emodes`` (see :func:`~neuromodes.io._cache_output` for details). Default is ``False``.
    n_jobs : int, optional
        Number of parallel jobs to run when ``method='fem'``. If not ``1``, ``joblib`` must be
        installed. Default is ``1``.
    verbose : int, optional
        ``joblib`` verbosity level for parallel execution when ``method='fem'`` and ``n_jobs > 1``.
        Default is ``0``.

    Returns
    -------
    np.ndarray
        Simulated neural activity of shape ``(n_verts, nt)``.

    Raises
    ------
    ValueError
        If any of ``r``, ``gamma``, or ``dt`` is not positive (or zero in the case of ``r``).
    ValueError
        If ``nt`` is not ``None`` nor a positive integer.
    ValueError
        If ``speed_limits`` is not ``None`` nor a tuple ``(min_speed, max_speed)``, where ``0 ≤
        min_speed < max_speed``.
    ValueError
        If neither ``nt`` nor ``ext_input`` are provided.
    ValueError
        If ``method`` is not ``'fourier'``, ``'ode'``, nor ``'fem'``.
    ValueError
        If ``method='fem'`` and either of ``mass`` or ``stiffness`` is not provided.
    ValueError
        If ``method`` is ``'fourier'`` or ``'ode'`` and either of ``emodes`` or ``evals`` is not
        provided.
    ValueError
        If ``ext_input`` is provided and contains any NaN values.
    ValueError
        If ``n_jobs`` is not ``1`` and ``joblib`` is not installed.

    Notes
    -----
    Prior works have treated ``r`` as a free parameter to fit empirical data [1]_ [2]_, with the
    default value reflecting an optimal fit to human resting-state functional MRI data [2]_.
    Consider adjusting this parameter, as its optimum can vary across analyses (e.g., different
    surfaces, heterogeneous modes, parcellated timeseries, empirical data, fitting metrics, etc.).

    Since the simulation begins at rest, consider discarding the first timepoints of activity to
    allow the system to reach a steady state.

    While the wave model can be run using non-cortical modes, users should consider whether this is
    theoretically sensible and physiologically plausible.

    References
    ----------
    ..  [1] Barnes, V., et al. (2026). Regional heterogeneity shapes macroscopic wave dynamics of
        the human and non-human primate cortex. bioRxiv. https://doi.org/10.64898/2026.01.22.701178
    ..  [2] Pang, J. C., et al. (2023). Geometric constraints on human brain function. Nature.
        https://doi.org/10.1038/s41586-023-06098-1
    ..  [3] Robinson, P. A., et al. (1997). Propagation and stability of waves of electrical
        activity in the cerebral cortex. Physical Review E. https://doi.org/10.1103/physreve.56.826
    """
    # Format / validate arguments
    if checks is not False:
        ved = EigenData(
            emodes=emodes, evals=evals, mass=mass, stiffness=stiffness, 
            hetero=hetero, data=ext_input, checks=checks
            )
        emodes, evals, mass, stiffness, ext_input, hetero = \
            ved.emodes, ved.evals, ved.mass, ved.stiffness, ved.data, ved.hetero
        
    if emodes is not None: 
        n_modes = emodes.shape[1]
    elif stiffness is not None:
        n_verts = stiffness.shape[0]
    elif method == 'fem':
        raise ValueError(f"mass and stiffness matrices must be provided for {method} method.")
    else: 
        raise ValueError(f"emodes must be provided for {method} method.")
        
    r = float(r)
    gamma = float(gamma)
    if r < 0:
        raise ValueError("Parameter r must be non-negative.")
    if gamma <= 0:
        raise ValueError("Parameter gamma must be positive.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if nt is not None and (not isinstance(nt, int) or nt <= 0):
        raise ValueError("nt must be None or a positive integer.")
    if speed_limits is not None:
        if (not isinstance(speed_limits, tuple) or not len(speed_limits) == 2
            or speed_limits[0] < 0 or speed_limits[0] >= speed_limits[1]):
            raise ValueError("speed_limits must be a tuple of (min_speed, max_speed), where "
                             "0 ≤ min_speed < max_speed.")
        speed = calc_wave_speed(r, gamma, hetero=hetero)
        min_speed, max_speed = np.min(speed), np.max(speed)
        if min_speed < speed_limits[0] or max_speed > speed_limits[1]:
            calc_str = min_speed if min_speed == max_speed else f"{min_speed:.1f}-{max_speed:.1f}"
            warn("The combination of r, gamma, and hetero leads to wave speeds "
                 f"outside the range of {speed_limits[0]}-{speed_limits[1]} m/s (calculated "
                 f"{calc_str} m/s). Consider changing these parameters to ensure physiologically "
                 "plausible wave speeds, or adjust speed_limits.")
    if method not in ['fourier', 'ode', 'fem']:
        raise ValueError(f"Invalid PDE method '{method}'; must be 'fourier', 'ode', or 'fem'.")
    if n_jobs != 1:
        if method != 'fem':
            warn("n_jobs is ignored when method is not 'fem'.")
        elif find_spec("joblib") is None:
            raise ImportError("joblib must be installed to use n_jobs != 1 for parallel execution. "
                              "Neuromodes can be installed with the 'cache' extra to include "
                              "joblib as a dependency (e.g., pip install neuromodes[cache]). ")

    if ext_input is not None:
        if nt is not None:
            warn("nt is ignored when ext_input is provided.")
        if seed is not None:
            warn("seed is ignored when ext_input is provided.")
        if cache_input:
            warn("cache_input is ignored when ext_input is provided.")
        if np.isnan(ext_input).any():
            raise ValueError("ext_input contains NaN values, which are not allowed.")
        nt = ext_input.shape[1]
        # Decompose input to modal space
        if method != 'fem':
            input_coeffs = decompose(ext_input, emodes, mass=mass, checks=False)
    elif nt is not None:
        if cache_input and seed is not None:
            from neuromodes.io import _cache_output
            noise_func = _cache_output(_gen_noise)
        else:
            if cache_input and seed is None:
                warn("cache_input is ignored when seed is None.")
            noise_func = _gen_noise
        if method == 'fem':
            n_verts = stiffness.shape[0]
            ext_input = np.asarray(noise_func(n_verts, nt, seed=seed, sample='vertices', mass=mass))
        else:
            # Generate white noise in modal space: faster, memory efficient, and removes areal bias
            input_coeffs = np.asarray(noise_func(n_modes, nt, seed=seed))
    else: # not the nicest, but it makes pyright the happiest
        raise ValueError("Either nt or ext_input must be provided.")

    # Non-modal FEM implementation
    if method == 'fem':
        if mass is None or stiffness is None:
            raise ValueError("Mass and stiffness matrices must be provided for FEM method.")
        return _model_wave_fem(ext_input, dt=dt, r=r, gamma=gamma, mass=mass, stiffness=stiffness,
                               n_jobs=n_jobs, verbose=verbose)
    
    # Standard modal implementation: decompose input and reconstruct output
    _model_wave = _model_wave_fourier if method == 'fourier' else _model_wave_ode
    activity_coeffs = _model_wave(input_coeffs, dt, r, gamma, evals)

    return emodes @ activity_coeffs

def balloon_model(
    activity: NDArray[np.floating],
    dt: float,
    emodes: NDArray[np.floating],
    method: _PDEKind = "fourier",
    mass: csc_matrix | None = None,
    checks: _CheckKind = True,
    **params
) -> NDArray[np.floating]:
    """
    Transform simulated activity to blood oxygen level-dependent (BOLD) signal using the
    Balloon-Windkessel model [1]_ [2]_.
    
    Parameters
    ----------
    activity : array-like
        Simulated neural activity in vertex space of shape ``(n_verts, n_timepoints)``.
    emodes : array-like
        The eigenmodes array of shape ``(n_verts, n_modes)``, where ``n_verts`` is the number of
        vertices and ``n_modes`` is the number of eigenmodes.
    dt : float, optional
        Time step of simulated activity in seconds.
    method : str, optional
        Method for solving the balloon PDEs. Either ``'fourier'`` or ``'ode'``. Default is
        ``'fourier'``.
    mass : array-like, optional
        The mass matrix of shape (n_verts, n_verts) used for the decomposition when method is
        ``'project'``. Default is ``None``.
    checks : bool, optional
        Whether to perform checks on the input arrays. Default is ``True``.
    **params
        Optional balloon model parameters to override defaults (e.g., ``rho``, ``k1``). See
        :func:`_model_balloon_fourier` or :func:`_model_balloon_ode` for available parameters.

    Returns
    -------
    ndarray
        Simulated BOLD signal in vertex space of shape ``(n_verts, n_timepoints)``.

    Raises
    ------
    ValueError
        If ``dt`` is not positive.
    ValueError
        If ``method`` is not ``'fourier'`` or ``'ode'``.

    References
    ----------
    ..  [1] Buxton, R. B., et al. (1998). Dynamics of blood flow and oxygenation changes during
        brain activation: The balloon model. Magnetic Resonance in Med.
        https://doi.org/10.1002/mrm.1910390602
    ..  [2] Stephan, K. E., et al. (2007). Comparing hemodynamic models with DCM. NeuroImage.
        https://doi.org/10.1016/j.neuroimage.2007.07.040
    """
    # Format / validate arguments
    if checks is not False:
        ved = EigenData(emodes=emodes, mass=mass, data=activity, checks=checks)
        emodes, mass, activity = ved.emodes, ved.mass, ved.data

    if np.isnan(activity).any():
        raise ValueError("activity contains NaN values, which are not allowed.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if method not in ['fourier', 'ode']:
        raise ValueError(f"Invalid PDE method '{method}'; must be 'fourier' or 'ode'.")
    for param_name, param_value in params.items():
        if not isinstance(param_value, (int, float)) or param_value <= 0:
            raise ValueError(f"Parameter '{param_name}' must be a positive number.")

    # Eigendecompose activity to get modal coefficients over time
    activity_coeffs = decompose(activity, emodes, mass=mass, checks=False)

    # Apply model to each mode's activity timeseries
    _model_balloon = _model_balloon_fourier if method == 'fourier' else _model_balloon_ode
    bold_coeffs = _model_balloon(activity_coeffs, dt, **params)

    # Transform timeseries from modal coefficients back to vertex space
    return emodes @ bold_coeffs

def calc_wave_speed(
    r: float,
    gamma: float,
    hetero: NDArray[np.floating] | None = None
) -> float | NDArray[np.floating]:
    """
    Calculate wave speed (m/s) based on the two parameters of the wave model. If a scaled
    heterogeneity map is provided, wave speeds are calculated for each cortical vertex (i.e., each
    entry of ``hetero``).
    
    Parameters
    ----------
    r : float
        Axonal length scale for wave propagation in millimeters.
    gamma : float
        Damping parameter for wave propagation in seconds^-1.
    hetero : array-like, optional
        Scaled heterogeneity map of shape (n_verts,). If ``None``, wave speed is assumed to be
        spatially uniform. To scale a heterogeneity map, use :func:eigen.scale_hetero. Default is
        ``None``.
    
    Returns
    -------
    float or np.ndarray
        Wave speed across the whole cortex in meters per second, or at each vertex if ``hetero`` is
        provided.
    """
    speed = (r / 1000) * gamma # Convert r to meters
    if hetero is not None:
        speed *= np.sqrt(hetero)

    return speed

def _gen_noise(
    n_samples: int,
    nt: int,
    sample: Literal['modes', 'vertices'] = 'modes',
    mass: csc_matrix | None = None,
    seed: int | None = None
) -> NDArray[np.floating]:
    """
    Generate reproducible white noise of shape ``(n_samples, nt)`` for a given ``seed``, derived
    from a standard normal distribution. The output is reproducible across nt (i.e.,
    ``_gen_noise(n_samples, nt, seed) == _gen_noise(n_samples, nt+k, seed)[:, :nt]``).

    Parameters
    ----------
    n_samples : int
        Number of samples (rows) in the output noise array. Depending on ``sample``, each row
        represents either an eigenmode or a vertex.
    nt : int
        Number of time points (columns) in the output noise array.
    sample : {'modes', 'vertices'}, optional
        Whether to generate noise in modal space (``'modes'``) or vertex space. Note that if
        ``sample='vertices'``, the mass matrix must be provided to ensure that the expected
        mass-weighted variance of the noise map is 1. Default is ``'modes'``.
    mass : array-like, optional
        The mass matrix of shape ``(n_verts, n_verts)`` used for normalization when
        ``sample='vertices'``. Note that Default is ``None``.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Gaussian white noise array of shape ``(n_samples, nt)``.
    """
    if sample not in ['modes', 'vertices']:
        raise ValueError(f"Invalid sample argument '{sample}'; must be 'modes' or 'vertices'.")
    if sample == 'vertices' and mass is None:
        raise ValueError("Mass matrix must be provided for normalization when sample='vertices'.")

    # Draw samples from N(0, 1) in column-major order to ensure reproducibility across nt, then
    # transpose to desired shape (n_samples, nt)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((nt, n_samples)).T

    if sample == 'modes':
        return noise
    # In vertex space, we need to ensure that expected mass-weighted variance of the noise map is 1
    elif mass.nnz == mass.shape[0]:
        # Lumped mass is easily sqrt'd and inverted
        return noise / np.sqrt(mass.diagonal())[:, None]
    else:
        # Consistent mass matrix requires solving a system
        from scipy.sparse.csgraph import reverse_cuthill_mckee
        from scipy.sparse.linalg import spsolve_triangular

        # Permute rows/cols of mass matrix
        perm = reverse_cuthill_mckee(mass, symmetric_mode=True)
        inv_perm = np.argsort(perm)
        mass_perm = mass[perm, :][:, perm]

        # Factorize mass = L @ D @ L.T
        lu = splu(mass_perm, permc_spec='NATURAL', diag_pivot_thresh=0,
                  options={'SymmetricMode': True})
        L_T = lu.L.tocsr().T

        # Scale noise by D^(-1/2)
        noise_rescaled = noise / np.sqrt(lu.U.diagonal())[:, None]

        # Solve L.T @ x = noise*D(-1/2) for x
        noise_rescaled = spsolve_triangular(L_T, noise_rescaled, lower=False, overwrite_a=True,
                                            overwrite_b=True)

        # Reverse the permutation
        return noise_rescaled[inv_perm]

def _model_wave_fourier(
    input_coeffs: NDArray[np.floating],
    dt: float,
    r: float,
    gamma: float,
    evals: NDArray[np.floating]
) -> NDArray[np.floating]:
    """
    Simulates the time evolution of wave models for all modes using a frequency-domain approach.
    This function applies a Fourier transform to the input mode coefficients, computes the system's
    frequency response, and then applies an inverse Fourier transform to obtain the time-domain
    response of each mode.

    Parameters
    ----------
    input_coeffs : np.ndarray
        Array of mode coefficients at each time representing the input signals to the model, with
        shape ``(n_modes, nt)``.
    dt : float
        Time step for the simulation in seconds.
    r : float
        Spatial length scale of wave propagation in millimeters.
    gamma : float
        Damping rate of wave propagation in seconds^(-1).
    evals : np.ndarray
        The eigenvalues associated with each mode, with shape ``(n_modes,)``.

    Returns
    -------
    out : ndarray
        The real part of the time-domain response of all modes at the specified time points, with
        shape ``(n_modes, nt)``.
    
    Notes
    -----
    This function uses a frequency-domain method to simulate the damped wave response of a causal
    input. To ensure causality (i.e., the input is zero for t < 0), the input is zero-padded on the
    negative time axis and transformed using ``np.fft.ifft``, which mimics the forward Fourier
    transform of a causal signal. The system's frequency response (transfer function) is then
    applied, and ``np.fft.fft`` is used to return to the time domain. This approach is standard for
    simulating linear time-invariant causal systems and is equivalent to convolution with a Green's
    function.

    The sequence is:
      1. Zero-pad input for t < 0 (causality)
      2. Take ifft to get the frequency-domain representation for this causal signal
      3. Apply the frequency response (transfer function)
      4. Use fft to return to the time domain (with appropriate shifts)
    """
    nt = input_coeffs.shape[1]

    # Pad input with zeros on negative side to ensure causality (system is only driven for t >= 0)
    # This is required for the correct Green's function solution of the damped wave equation.
    input_coeffs_padded = np.pad(input_coeffs, ((0, 0), (nt, 0)), constant_values=0)

    # Frequency-domain representation of the causal signal
    # Faster to use `rfft` here than `fftshift(ifft)` (original implementation)
    input_coeffs_f = np.fft.rfft(input_coeffs_padded, axis=1)

    # Frequencies for full signal
    omega = -2 * np.pi * np.fft.rfftfreq(2*nt, d=dt) # keep consistent with _model_balloon_fourier

    # Compute transfer function and apply it to frequency-domain input
    H = gamma**2 / ((-omega**2 - 2j * omega * gamma) + gamma**2 * (1 + r**2 * evals[:, None]))
    out_fft = H * input_coeffs_f

    # Inverse transform to time domain (irfft is fast)
    out_full = np.fft.irfft(out_fft, n=2*nt, axis=1)

    # Return only the non-negative time part (t >= 0)
    return out_full[:, nt:]

def _model_wave_ode(
    input_coeffs: NDArray[np.floating],
    dt: float,
    r: float,
    gamma: float,
    evals: NDArray[np.floating]
) -> NDArray[np.floating]:
    """
    Solves the damped wave ODE for all eigenmodes.

    Parameters
    ----------
    input_coeffs : np.ndarray
        Input drive to the system with shape ``(n_modes, nt)`` (written as qj in equation below).
    dt : float
        Time step for the simulation in seconds.
    gamma : float
        Damping coefficient seconds^-1.
    r : float
        Spatial length scale in millimeters.
    evals : np.ndarray
        Eigenvalues for each mode with shape ``(n_modes,)`` (written as lambdaj in equation below).

    Returns
    -------
    np.ndarray
        Time evolution of phi_j(t), solution to the wave equation, with shape ``(n_modes, nt)``.
    
    Notes
    -----
    The equation is derived from the damped wave equation:
    d^2 phi_j / dt^2 + 2 * gamma * d phi_j / dt + gamma^2 * (1 + r^2 * lambdaj) * phi_j = gamma^2 * qj 
    
    Rearranging gives us the first-order system
        dx1/dt = x2
        dx2/dt = -2 * gamma * x2 - gamma^2 * (1 + r^2 * lambdaj) * x1 + gamma^2 * qval
    """
    n_modes, nt = input_coeffs.shape
    t = np.linspace(0, dt * (nt - 1), nt)
    
    # Simulate wave equation for each mode
    mode_coeffs = np.empty_like(input_coeffs)
    for j in range(n_modes):
        def wave_odes_j(t_, y):
            """Returns the wave ODEs for mode j."""
            x1, x2 = y

            # Interpolate input coefficient at time t_
            qval = np.interp(t_, t, input_coeffs[j, :])
            if isinstance(qval, np.ndarray):
                qval = qval.item()

            # Set expressions for time derivatives
            dx1dt = x2
            dx2dt = -2 * gamma * x2 - gamma**2 * (1 + r**2 * evals[j]) * x1 + gamma**2 * qval
            return [dx1dt, dx2dt]

        # Call ODE solver
        sol = solve_ivp(
            wave_odes_j,
            t_span=(t[0], t[-1]),
            y0=[0.0, 0.0],  # Initial condition: phi_j(0) = 0, dphi_j/dt(0) = 0
            t_eval=t,
            method='RK45',
            rtol=1e-6,
            atol=1e-9
        )

        mode_coeffs[j, :] = sol.y[0]  # Store phi_j(t)

    return mode_coeffs

def _model_wave_fem(
    ext_input: NDArray[np.floating],
    mass: csc_matrix,
    stiffness: csc_matrix,
    dt: float = 1e-4,
    r: float = 17.4,
    gamma: float = 116.0,
    n_jobs: int = 1,
    verbose: int = 0 # for Parallel only (consider making **Parallel_kwargs)
) -> NDArray[np.floating]:
    """
    Simulates the time evolution of wave models for all vertices using a finite element method (FEM)
    approach. This function applies a Fourier transform to the input, computes the system's
    frequency response, and then applies an inverse Fourier transform to obtain the time-domain
    response of each vertex.

    Parameters
    ----------
    ext_input : np.ndarray
        Array of external input at each vertex over time, with shape ``(n_verts, nt)``.
    mass : scipy.sparse.csc_matrix
        The mass matrix of shape ``(n_verts, n_verts)``.
    stiffness : scipy.sparse.csc_matrix
        The stiffness matrix of shape ``(n_verts, n_verts)``.
    dt : float, optional
        Time step for the simulation in seconds. Default is ``1e-4``.
    r : float, optional
        Spatial length scale of wave propagation in millimeters. Default is ``17.4``.
    gamma : float, optional
        Damping rate of wave propagation in seconds^(-1). Default is ``116.0``.
    n_jobs : int, optional
        Number of parallel jobs to run. If not ``1``, ``joblib`` must be installed. Default is
        ``1``.
    verbose : int, optional
        Verbosity level for parallel execution. Default is ``0``.
    """
    nt = ext_input.shape[1]

    # Mass-weight input so that sparser (larger) vertices are compensated by larger amplitude
    input_w = mass @ ext_input

    # Pad input with zeros on negative side to ensure causality (system is only driven for t >= 0)
    # This is required for the correct Green's function solution of the damped wave equation.
    input_padded = np.pad(input_w, ((0, 0), (nt, 0)), constant_values=0)

    # Apply Fourier transform to get frequency-domain representation of the causal signal.
    input_padded_freqs = np.fft.rfft(input_padded, axis=1)

    # Compute components of NFT operator
    spatial = r**2 * stiffness
    omega = -2 * np.pi * np.fft.rfftfreq(2*nt, dt)
    temporal = -omega**2 / gamma**2 - 2j * omega / gamma + 1

    # Main computation
    # Compute activity at each frequency (TODO: consider memory usage, if fine then vectorise)
    eqns = (
        (spatial + temporal[k] * mass, input_padded_freqs[:, k])
        for k in range(len(temporal))
    )

    # Define helper function for parallelization
    # TODO: consider supporting Cholesky decomp if operator is SPD
    def _solve_fem_freq(operator, input):
        return splu(operator).solve(input)

    phi_freqs = None # stops it being unbound and keeps pyright happy
    if n_jobs > 1 or n_jobs == -1:
        from joblib import Parallel, delayed
        phi_freqs = Parallel(n_jobs=n_jobs, verbose=verbose, prefer="threads")(
            delayed(_solve_fem_freq)(op, inp) for op, inp in eqns
        )
    else:
        phi_freqs = [_solve_fem_freq(op, inp) for op, inp in eqns]
    
    phi_freqs = np.stack(cast(list[NDArray[np.complex128]], phi_freqs), axis=1)

    # Inverse transform to time domain
    phi = np.fft.irfft(phi_freqs, axis=1, n=2*nt)

    # Return only the non-negative time part (t >= 0)
    return phi[:, nt:]

def _model_balloon_fourier(
    activity_coeffs: NDArray[np.floating],
    dt: float,
    kappa: float = 0.65,
    tau: float = 0.98,
    alpha: float = 0.32,
    rho: float = 0.34,
    V_0: float = 0.02,
    w_f: float = 0.56,
    k1: float = 3.72,
    k2: float = 0.527,
    k3: float = 0.48
) -> NDArray[np.floating]:
    """
    Simulates the hemodynamic response of all modes using the balloon model in the frequency domain.
    This function computes the balloon model's frequency response and applies it to the input mode
    coefficients via Fourier transforms, returning the modeled hemodynamic response over time.

    Parameters
    ----------
    activity_coeffs : np.ndarray
        Array of mode coefficients representing the input signals to the model, with shape (n_modes,
        nt).
    dt : float
        Time step in seconds.
    kappa : float, optional
        Signal decay rate in seconds^-1. Default is ``0.65``.
    tau : float, optional
        Hemodynamic transit time in seconds. Default is ``0.98``.
    alpha : float, optional
        Grubb's exponent (unitless). Default is ``0.32``.
    rho : float, optional
        Resting oxygen extraction fraction (unitless). Default is ``0.34``.
    V_0 : float, optional
        Resting blood volume fraction (unitless). Default is ``0.02``.
    w_f : float, optional
        Frequency of blood flow response in radians per second. Default is ``0.56``.
    k1 : float, optional
        First coefficient in BOLD signal equation (unitless). Default is ``3.72``
    k2 : float, optional
        Second coefficient in BOLD signal equation (unitless). Default is ``0.527``.
    k3 : float, optional
        Third coefficient in BOLD signal equation (unitless). Default is ``0.48``.

    Returns
    -------
    np.ndarray
        The real part of the time-domain response of all modes at the specified time points, with
        shape (n_modes, nt).

    Notes
    -----
    This function uses a frequency-domain method to simulate the damped wave response of a causal 
    input. To ensure causality (i.e., the input is zero for t < 0), the input is zero-padded on the 
    negative time axis and transformed using ``np.fft.rfft``, which mimics the forward Fourier
    transform of a causal signal. The system's frequency response (transfer function) is then
    applied, and ``np.fft.irfft`` is used to return to the time domain. This approach is standard
    for simulating linear time-invariant causal systems and is equivalent to convolution with a
    Green's function.

    The sequence is:
      1. Zero-pad input for t < 0 (causality)
      2. Take rfft to get the frequency-domain representation for this causal signal
      3. Apply the frequency response (transfer function)
      4. Use irfft to return to the time domain (with appropriate shifts)
    """
    nt = activity_coeffs.shape[1]

    # Calculate balloon model frequency response (Pang et al. 2016)
    omega = -2 * np.pi * np.fft.rfftfreq(2*nt, d=dt)
    beta = (rho + (1 - rho) * np.log(1 - rho)) / rho
    phi_hat_Fz = 1 / (-(omega + 1j * 0.5 * kappa) ** 2 + w_f ** 2)
    phi_hat_yF = V_0 * (alpha * (k2 + k3) * (1 - 1j * tau * omega) 
                                - (k1 + k2) * (alpha + beta - 1 - 1j * tau * alpha * beta * omega)
                                ) / ((1 - 1j * tau * omega) * (1 - 1j * tau * alpha * omega))
    balloon_freq_response = phi_hat_yF * phi_hat_Fz

    # Zero-pad input at t < 0 for causality
    activity_coeffs_padded = np.pad(activity_coeffs, ((0, 0), (nt, 0)), constant_values=0)

    # Apply Fourier transform (implemented as rfft for speed)
    activity_coeffs_f = np.fft.rfft(activity_coeffs_padded, axis=1)

    # Apply frequency response (broadcast along time axis)
    out_fft = balloon_freq_response[None, :] * activity_coeffs_f

    # Inverse transform back to timeseries (inverse of previous transform)
    out_full = np.fft.irfft(out_fft, n=2*nt, axis=1)

    # Remove zero padding
    return out_full[:, nt:]

def _model_balloon_ode(
    activity_coeffs: NDArray[np.floating],
    dt: float,
    kappa: float = 0.65,
    tau: float = 0.98,
    alpha: float = 0.32,
    rho: float = 0.34,
    V_0: float = 0.02,
    gamma_h: float = 0.41,
    k1: float = 3.72,
    k2: float = 0.527,
    k3: float = 0.48
) -> NDArray[np.floating]:
    """
    Simulates the hemodynamic response of all modes using the balloon model in the time domain (ODE 
    approach). This function numerically integrates the balloon model ODEs for each input mode 
    time course.

    Parameters
    ----------
    activity_coeffs : np.ndarray
        Array of mode coefficients representing the input signals to the model, with shape
        ``(n_modes, nt)``.
    dt : float
        Time step for the simulation in seconds.
    kappa : float, optional
        Signal decay rate in seconds^-1. Default is ``0.65``.
    tau : float, optional
        Hemodynamic transit time in seconds. Default is ``0.98``.
    alpha : float, optional
        Grubb's exponent (unitless). Default is ``0.32``.
    V_0 : float, optional
        Resting blood volume fraction (unitless). Default is ``0.02``.
    gamma_h : float, optional
        Hemodynamic gain (unitless). Default is ``0.41``.
    k1 : float, optional
        First coefficient in BOLD signal equation (unitless). Default is ``3.72``.
    k2 : float, optional
        Second coefficient in BOLD signal equation (unitless). Default is ``0.527``.
    k3 : float, optional
        Third coefficient in BOLD signal equation (unitless). Default is ``0.48``.

    Returns
    -------
    np.ndarray
        The BOLD signal time course for all modes at the specified time points, with shape
        ``(n_modes, nt)``.

    Raises
    ------
    RuntimeError
        If the ODE solver fails.
    """
    n_modes, nt = activity_coeffs.shape
    t = np.linspace(0, dt * (nt - 1), nt)

    # Simulate balloon model for each mode
    bold_coeffs = np.empty_like(activity_coeffs)
    for j in range(n_modes):
        def balloon_odes_j(t_, y):
            """Returns the balloon model ODEs for mode j."""
            z, f, v, q = y

            # Interpolate input coefficient at time t_
            N = np.interp(t_, t, activity_coeffs[j])

            # Set expressions for time derivatives
            dzdt = N - kappa * z - gamma_h * (f - 1)
            dfdt = z
            dvdt = (f - v ** (1 / alpha)) / tau
            dqdt = (f * (1 - (1 - rho) ** (1 / f)) / rho - q * v ** (1 / alpha - 1)) / tau
            return [dzdt, dfdt, dvdt, dqdt]

        # Call ODE solver
        sol = solve_ivp(
            balloon_odes_j,
            t_span=(t[0], t[-1]),
            y0=[0.0, 1.0, 1.0, 1.0], # Initial condition for [z, f, v, q]
            t_eval=t,
            method='RK45',
            rtol=1e-6,
            atol=1e-9
        )

        if not sol.success:
            raise RuntimeError("Balloon model ODE solver failed. Try using method='fourier' or "
                               "a smaller timestep (dt) without altering balloon model parameters. "
                               f"scipy.integrate.solve_ivp message: {sol.message}")

        # Apply standard BOLD signal equation
        _, _, v, q = sol.y
        bold_coeffs[j, :] = V_0 * (k1 * (1 - q) + k2 * (1 - q / v) + k3 * (1 - v))

    return bold_coeffs

def calc_nft_fc(
    emodes: NDArray[np.floating],
    evals: NDArray[np.floating],
    r: float
) -> NDArray[np.floating]:
    """
    Calculate the analytical FC for the wave model under white noise input.

    Parameters
    ----------
    emodes : np.ndarray
        Eigenmodes of shape ``(n_verts, n_modes)``.
    evals : np.ndarray
        Eigenvalues corresponding to the modes, with shape ``(n_modes,)``.
    r : float
        Spatial length scale of wave propagation in millimeters.

    Returns
    -------
    np.ndarray
        Analytical FC matrix of shape ``(n_verts, n_verts)``.
    """
    ved = EigenData(emodes=emodes, evals=evals, checks=False)
    emodes, evals = ved.emodes, ved.evals

    # Compute the variance of each mode's activity timeseries by applying Parseval's identity to the
    # NFT operator. This is equivalent to integrating the power spectral density of each mode's
    # response over all frequencies, which yields the variance of the mode's activity timeseries.
    # NOTE: technically the denominator should be multiplied by 2*gamma, but we can ignore this here
    # since correlation normalises this out later (i.e., gamma has no effect on model FC)
    mode_vars = 1.0 / (1 + r**2 * evals)

    # reconstruct (change from modal to vertex basis)
    cov = emodes @ (mode_vars[:, None] * emodes.T)

    # Normalise variance-covariance matrix to get correlation
    stds = np.sqrt(np.diag(cov))
    return cov / stds[:, None] / stds[None, :]