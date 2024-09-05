import numpy as np
from scipy.special import erfc

# Constants
c = 3e8  # Speed of light in m/s

# Free-Space Path Loss (FSPL)
def free_space_path_loss(d, fc):
    """
    Calculate the free-space path loss (FSPL) in dB.
    
    Parameters:
    d  -- distance between transmitter and receiver (meters)
    fc -- carrier frequency (Hz)
    
    Returns:
    FSPL in dB
    """
    return 20 * np.log10((4 * np.pi * d * fc) / c)

# Hata Model for urban areas
def hata_model(fc, ht, hr, d):
    """
    Hata model for urban areas in dB.
    
    Parameters:
    fc -- carrier frequency (MHz)
    ht -- height of the transmitter (meters)
    hr -- height of the receiver (meters)
    d  -- distance between transmitter and receiver (kilometers)
    
    Returns:
    Path loss in dB
    """
    a_hr = (1.1 * np.log10(fc) - 0.7) * hr - (1.56 * np.log10(fc) - 0.8)
    return 69.55 + 26.16 * np.log10(fc) - 13.82 * np.log10(ht) - a_hr + (44.9 - 6.55 * np.log10(ht)) * np.log10(d)

# COST 231 Model (Extension of Hata Model)
def cost231_model(fc, ht, hr, d, Cm=0):
    """
    COST 231 model for urban environments in dB.
    
    Parameters:
    fc  -- carrier frequency (MHz)
    ht  -- height of the transmitter (meters)
    hr  -- height of the receiver (meters)
    d   -- distance between transmitter and receiver (kilometers)
    Cm  -- 0 dB for metropolitan areas, 3 dB for suburban areas (default is 0)
    
    Returns:
    Path loss in dB
    """
    a_hr = (1.1 * np.log10(fc) - 0.7) * hr - (1.56 * np.log10(fc) - 0.8)
    return 46.3 + 33.9 * np.log10(fc) - 13.82 * np.log10(ht) - a_hr + (44.9 - 6.55 * np.log10(ht)) * np.log10(d) + Cm

# Two-Ray Model
def two_ray_model(Pt, d1, d2, R, wavelength):
    """
    Two-ray model to calculate received power.
    
    Parameters:
    Pt        -- transmitted power
    d1, d2    -- distances for line-of-sight and reflected paths
    R         -- reflection coefficient
    wavelength -- wavelength of the signal (meters)
    
    Returns:
    Received power
    """
    return Pt * ((wavelength / (4 * np.pi)) * (1/d1 + R/d2)) ** 2

# Critical Distance in Two-Ray Model
def critical_distance(ht, hr, wavelength):
    """
    Calculate the critical distance in the two-ray model.
    
    Parameters:
    ht -- height of the transmitter (meters)
    hr -- height of the receiver (meters)
    wavelength -- wavelength (meters)
    
    Returns:
    Critical distance in meters
    """
    return (4 * ht * hr) / wavelength

# Delay Spread in Two-Ray Model
def delay_spread(d1, d2):
    """
    Calculate the delay spread between two signals in a two-ray model.
    
    Parameters:
    d1 -- LOS distance (meters)
    d2 -- non-LOS distance (meters)
    
    Returns:
    Delay spread (seconds)
    """
    return abs(d1 - d2) / c

# Outage Probability under Path Loss and Shadowing
def outage_probability(Pmin, Pt, Lpl, sigma_psi):
    """
    Calculate the outage probability under path loss and shadowing.
    
    Parameters:
    Pmin      -- minimum received power (dB)
    Pt        -- transmitted power (dB)
    Lpl       -- path loss at distance d (dB)
    sigma_psi -- standard deviation of shadowing (dB)
    
    Returns:
    Outage probability
    """
    z = (Pmin - (Pt - Lpl)) / sigma_psi
    return 1 - 0.5 * erfc(z / np.sqrt(2))

# Link Budget Equation
def link_budget(Pt, Gt, Gr, L):
    """
    Calculate the received power using the link budget equation.
    
    Parameters:
    Pt -- transmitted power (dBm)
    Gt -- transmitter antenna gain (dB)
    Gr -- receiver antenna gain (dB)
    L  -- total path loss (dB)
    
    Returns:
    Received power (dBm)
    """
    return Pt + Gt + Gr - L

# Doppler Shift
def doppler_shift(v, theta, wavelength):
    """
    Calculate the Doppler shift.
    
    Parameters:
    v          -- velocity of the receiver (m/s)
    theta      -- angle between movement direction and wavefront (radians)
    wavelength -- wavelength of the signal (meters)
    
    Returns:
    Doppler shift (Hz)
    """
    return (v * np.cos(theta)) / wavelength

# Rayleigh Fading PDF
def rayleigh_fading(r, sigma):
    """
    Probability density function for Rayleigh fading.
    
    Parameters:
    r     -- signal amplitude
    sigma -- variance
    
    Returns:
    Probability density
    """
    return (r / sigma**2) * np.exp(-r**2 / (2 * sigma**2))

# Rician Fading PDF
def rician_fading(r, A, sigma):
    """
    Probability density function for Rician fading.
    
    Parameters:
    r     -- signal amplitude
    A     -- peak amplitude of the LOS component
    sigma -- variance
    
    Returns:
    Probability density
    """
    from scipy.special import i0
    return (r / sigma**2) * np.exp(-(r**2 + A**2) / (2 * sigma**2)) * i0((A * r) / sigma**2)

# Cell Coverage Probability
def coverage_probability(gamma, b):
    """
    Calculate the coverage probability.
    
    Parameters:
    gamma -- shadow fading parameter
    b     -- slope parameter
    
    Returns:
    Coverage probability
    """
    return 1 - np.exp(-2 * (gamma**2) / (b**2))

