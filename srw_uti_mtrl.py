#############################################################################
# SRWLib for Python: Metrology utilities
# v 0.01
# Rafael Celestre rafael.celestre@esrf.eu
#############################################################################

import numpy as np
from numpy import fft
from array import *
from scipy.interpolate import interp2d

try:
    from srwlib import *
except:
    from oasys_srw.srwlib import *          # get oasys_srw: https://github.com/oasys-kit/OASYS1-srwpy

    
######################################################################
## Optical Elements
######################################################################       
    
    
def srwl_opt_setup_fractal_surf(_sigma, _exponent, _pix_size, _delta, _atten_len, _apert_h, _apert_v, _xc=0, _yc=0, _e_start=0, _e_fin=0, _qr=0, _seed=None, _C=None, _dist=0):
    """
    Setup Transmission type Optical Element which simulates a rough surface in [m] with a pre-determined PSD.
    
    :param _sigma: standard deviation , i.e. root-mean-square roughness Rq(m)
    :param _exponent: PSD exponent = -2(H+1); Hurst exponent 0<= H <= 1, fractal dimension D = 3-H
    :param _pix_size: pixel size in [m]
    :param _delta: refractive index decrement (can be one number of array vs photon energy)
    :param _atten_len: attenuation length [m] (can be one number of array vs photon energy)
    :param _apert_h: horizontal aperture size [m]
    :param _apert_v: vertical aperture size [m]
    :param _xc: horizontal coordinate of center [m]
    :param _yc: vertical coordinate of center [m]
    :param _e_start: initial photon energy
    :param _e_fin: final photon energy
    :param _qr: roll-off freq. (1/m); qr > (2*pi/Lx or Ly); qr < (pi/PixelWidth) - Nyquist freq.
    :param _seed: seed for random initialisaiton
    :param _C: pre-calculated 2D psd where qx and qy respect the limits imposed by _pix_size, _apert_h and _apert_v
    :param _dist: -1 for phase = 0, 0 for uniform phase rand. distribution, 1 for rand. Gaussian dist.
    :return: transmission (SRWLOptT) type optical element which simulates a rough surface
    """

    _nx = int(_apert_h / _pix_size)
    _ny = int(_apert_h / _pix_size)
    

    _height_prof_data, y, x = fractal_surf(_sigma, _exponent, _pix_size, _ny , ny, _qr=_qr, symmetry=True,
                                           dist=_dist, seed=_seed, psd=False, C=_C)


    pad_y = int(ny*0.15)
    pad_x = int(nx*0.15)

    thcknss = np.pad(_height_prof_data, ((pad_y, pad_y),(pad_x, pad_x)), 'constant', constant_values=0)

    _ny, _nx = thcknss.shape
    xStart = - (_pix_size * (_nx - 1)) / 2.0
    xFin = xStart + _pix_size * (_nx - 1)
    yStart = - (_pix_size * (_ny - 1)) / 2.0
    yFin = yStart + _pix_size * (_ny - 1)

    amplitude_transmission = np.exp(-0.5 * thcknss / _atten_len)
    optical_path_diff = -thcknss * _delta

    arTr = np.empty((2 * _nx * _ny), dtype=np.float)
    arTr[0::2] = np.reshape(amplitude_transmission,(_nx*_ny))
    arTr[1::2] = np.reshape(optical_path_diff,(_nx*_ny))

    return SRWLOptT(_nx, _ny, xFin-xStart, yFin-yStart, _arTr=arTr, _extTr=1, _Fx=1e23, _Fy=1e23, _x=_xc, _y=_yc)


######################################################################
## PSD calculation
######################################################################

def srw_uti_mtrl_psd_avg(profile, axis_x, axis_y, pad=False, mthd=0, _range_r=(0, -1), _delta_r=1,
                         _range_theta=(0, -1), _delta_theta=1, _mask=None):
    '''
    :param profile: height profile as a 2D numpy array in [m] 
    :param axis_x: 1D numpy array in [m] 
    :param axis_y: 1D numpy array in [m] 
    :param pad: (boolean) zero padding to increase the frequency sampling
    :param mthd: (future - add more averaging methods - currently: azimuthally int. only)
    :param _range_r: in cylindrical coords.: start and end of R 
    :param _delta_r: in cylindrical coords.: step for R 
    :param _range_theta: in cylindrical coords.: start and end of theta for average 
    :param _delta_theta: in cylindrical coords.: step for theta 
    :param _mask: boolean mask
    :return: 2D psd and its frequency axes
    '''
    def _f_xy(_theta, _rho):
        x = _rho*np.cos(_theta)
        y = _rho*np.sin(_theta)
        return x, y

    psd_2d, fx, fy = srw_uti_mtrl_psd_2D(profile, axis_x, axis_y, pad=pad)

    xStart = fx[0]
    xFin = fx[-1]
    nx = fx.size

    yStart = fy[0]
    yFin = fy[-1]
    ny = fy.size

    # ********************** generating auxiliary vectors
    x_cen = 0.5*(xFin + xStart)
    y_cen = 0.5*(yFin + yStart)

    _range_r = list(_range_r)
    _range_theta = list(_range_theta)

    if _range_r[1] == -1:
        if xFin - x_cen > yFin - y_cen:
            _range_r[1] = 1 * (yFin - y_cen)
        else:
            _range_r[1] = 1 * (xFin - x_cen)

    if _range_theta[1] == -1:
        _range_theta[1] = 2*np.pi

    nr = int(nx*_delta_r/2)
    ntheta = int((_range_theta[1] - _range_theta[0])*360 * _delta_theta/2/np.pi)

    X = np.linspace(xStart, xFin, nx)
    Y = np.linspace(yStart, yFin, ny)

    R = np.linspace(_range_r[0],_range_r[1], nr)
    THETA = np.linspace(_range_theta[0],_range_theta[1], ntheta)

    psd_avg = np.zeros([nr])
    azimuthal_value = np.zeros([ntheta])

    # ********************** summation
    f = interp2d(X, Y, psd_2d)

    if _mask is not None:
        f_mask = interp2d(X, Y, _mask)

    m = 0
    for rho in R:
        n = 0
        summation = 0
        for angle in THETA:
            x, y = _f_xy(angle, rho)

            if _mask is not None:
                point = f_mask(x, y)
            else:
                point = 1
            if point >= 0.5:
                azimuthal_value[n] = f(x, y)
                summation += 1
            n += 1

        psd_avg[m] = np.sum(azimuthal_value)/summation
        m = m+1

    return psd_avg, R


def srw_uti_mtrl_psd_1D(profile, axis, positive_side=False, pad=False):
    '''
    :param profile: height profile as a 1D numpy array in [m] 
    :param axis: 1D numpy array in [m] 
    :param positive_side: (boolean) crops all negative frequencies
    :param pad: (boolean) zero padding to increase the frequency sampling
    :return: 1D psd and the frequency axis
    '''
    
    PixelWidth = axis[1]-axis[0]
    if pad:
        profile = np.pad(profile, (int(len(profile)/2), int(len(profile)/2)))
        
    m = len(profile)
    freq = fft.fftshift(fft.fftfreq(m, PixelWidth))
    psd = (PixelWidth/m)*np.absolute(fft.fftshift(fft.fft(profile)))**2
    
    if positive_side:
        psd = 2*psd[freq>0]
        freq = freq[freq>0]
    return psd, freq


def srw_uti_mtrl_psd_2D(profile, axis_x, axis_y, pad=False):
    '''
    :param profile: height profile as a 2D numpy array in [m] 
    :param axis_x: 1D numpy array in [m] 
    :param axis_y: 1D numpy array in [m] 
    :param pad: (boolean) zero padding to increase the frequency sampling
    :return: 2D psd and the frequency axis
    '''
        
    dx = axis_x[1]-axis_x[0]
    dy = axis_y[1]-axis_y[0]
    
    if pad:
        pad_y = int(len(axis_y)/2)
        pad_x = int(len(axis_x)/2)
        profile = np.pad(profile, ((pad_y, pad_y),(pad_x, pad_x)), 'constant', constant_values=0)

    m = profile.shape[1]
    n = profile.shape[0]
    psd = (dx*dy/(m*n))*np.absolute(fft.fftshift(fft.fft2(profile)))**2
        
    freqx = fft.fftshift(fft.fftfreq(m, dx))
    freqy = fft.fftshift(fft.fftfreq(n, dy))

    return psd, freqx, freqy

 
######################################################################
## Surface generation
######################################################################


def fractal_surf(sigma, exponent, PixelWidth, m , n, qr=0, symmetry=False, dist=0, seed=None, psd=False, C=None):
    '''
    
    % Fractal topographies with different fractal dimensions.
    
    Adaptation of the MATLAB function 'artificial_surf' (version 1.1.0.0) by Mona Mahboob Kanafi. 
    https://www.mathworks.com/matlabcentral/fileexchange/60817-surface-generator-artificial-randomly-rough-surfaces
    
    parameters (in SI units)
    
    :param sigma: standard deviation , i.e. root-mean-square roughness Rq(m)
    :param exponent: PSD exponent = -2(H+1); Hurst exponent 0<= H <= 1, fractal dimension D = 3-H
    ::param PixelWidth: pixel size in [m]
    :param m: number of pixels in x
    :param n: number of pixels in y
    :param qr: roll-off freq. (1/m); qr > (2*pi/Lx or Ly); qr < (pi/PixelWidth) - Nyquist freq.
    :param symmetry: (bool) apply conjugate symmetry to magnitude and phase - inherited from Matlab
    :param dist: -1 for phase = 0, 0 for uniform phase distribution, 1 for Gaussian dist.
    :param seed: seed for random initialisaiton
    :param psd: (bool) if true, returns Cq and its vectors
    :param C: profile to be applied to Cq. MUST BE C.shape == Cq.shape
    :return: surface profile and axes
    '''

    two_pi = 2*np.pi
    
    # =========================================================================
    Lx = m * PixelWidth # image length in x direction
    Ly = n * PixelWidth # image length in y direction
    A = Lx*Ly

    qx = fft.fftshift(fft.fftfreq(m, PixelWidth)) # image frequency in fx direction
    qy = fft.fftshift(fft.fftfreq(n, PixelWidth)) # image frequency in fy direction
    
    Qx, Qy = np.meshgrid(qx, qy)
    
    # cylindrical coordinates in frequency-space
    rho = np.sqrt(Qy**2 + Qx**2)
    theta = np.arctan2(Qy, Qx)
       
    [y0,x0] = np.where(rho==0) 
    rho[y0, x0] = 1   # avoids division by zero
    
    # 2D matrix of Cq (PSD) values
    if exponent is None:
        Cq = np.ones((n,m))
    else:
        Cq = np.zeros((n,m))
        Cq = rho**exponent
        if qr != 0:
            Cq[np.where(rho < qr)] = qr**exponent
            
    if C is not None:
        C -=np.min(C) 
        Cq = np.multiply(Cq,C)
        
    Cq[y0, x0] = 0

    # =========================================================================
    # applying rms
    Cq *= (sigma/(np.sqrt(np.sum(Cq)/A)))**2
    
    # =========================================================================
    # reversing operation: PSD to fft
    Bq = np.sqrt(Cq/(PixelWidth**2/((n*m))))
    
    # =========================================================================
    # defining a random phase
    np.random.seed(seed)
    if dist == -1:
        phi = np.zeros((n,m))
    elif dist == 0:
        phi = -np.pi + (two_pi)*np.random.uniform(0,1,(n,m))
    elif dist == 1:
        phi = np.pi*np.random.normal(0,1,(n,m))
    
    # =========================================================================
    # apply conjugate symmetry to magnitude and phase - inherited from Matlab
    if symmetry:
        Bq[0,0] = 0
        Bq[0,int(m/2)] = 0
        Bq[int(n/2),int(m/2)] = 0
        Bq[int(n/2),0] = 0
        Bq[1::,1:int(m/2)+1] = np.rot90(Bq[1::,int(m/2)::],2)
        Bq[0,1:int(m/2)+1] = np.flipud(Bq[0,int(m/2)::])
        Bq[int(n/2)::,0] = np.flipud(Bq[1:int(n/2)+1,0]) 
        Bq[int(n/2)::,int(m/2)] = np.flipud(Bq[1:int(n/2)+1,int(m/2)]) 
        if dist != -1:
            phi[0,0] = 0
            phi[0,int(m/2)] = 0
            phi[int(n/2),int(m/2)] = 0
            phi[int(n/2),0] = 0
            phi[1::,1:int(m/2)+1] = -np.rot90(phi[1::,int(m/2)::],2)
            phi[0,1:int(m/2)+1] = -np.flipud(phi[0,int(m/2)::])
            phi[int(n/2)::,0] = -np.flipud(phi[1:int(n/2)+1,0])
            phi[int(n/2)::,int(m/2)] = -np.flipud(phi[1:int(n/2)+1,int(m/2)])

    # =========================================================================
    # Generates topography
    Hm = Bq*np.exp(-1j*phi)

    # phase components.
    z = np.abs(fft.ifftshift(fft.ifft2(Hm))) # generates surface
    x = np.linspace(-m/2, m/2, m)*PixelWidth
    y = np.linspace(-n/2, n/2, n)*PixelWidth

    if psd:
        return z, y, x, Cq, qy, qx
    else:
        return z, y, x
    
"""
- A meridional or tangential (beam direction) ray is a ray that is confined to the plane containing the 
system's optical axis and the object point from which the ray originated

.- A sagittal or transverse ray from an off-axis object point is a ray that propagates in the plane that is 
perpendicular to the meridional plane and contains the principal ray. Sagittal rays intersect the pupil along a line 
that is perpendicular to the meridional plane for the ray's object point and passes through the optical axis. 
If the axis direction is defined to be the z axis, and the meridional plane is the y-z plane, sagittal rays intersect 
the pupil at yp=0. The principal ray is both sagittal and meridional. All other sagittal rays are skew rays.

"""

def single_f_sin_bumps_2d(_size_tang=1., _size_sag=1., _npt=100, _nps=100, _b_amp=1, _b_freq_t=None, _b_per_t=None,
                          _b_freq_s=None, _b_per_s=None, _b_phase=0, _b_offset=0, _file_path=None):
    """
     2D sinusoid profile
    :param _size_tang: meridional (or tangential) width
    :param _size_sag: sagittal (or transverse) width
    :param _npt: tangential number of pixels
    :param _nps: transverse number of pixels
    :param _b_amp: sinusoid amplitude
    :param _b_freq_t: sinusoid tangential frequency [1/m]
    :param _b_per_t: sinusoid tangential period [m] (alternative to _b_freq_t)
    :param _b_freq_s: sinusoid sagittal frequency [1/m]
    :param _b_per_s: sinusoid sagittal period [m] (alternative to _b_freq_s)
    :param _b_phase: initial phase [rad]
    :param _b_offset: dc offset [m]
    :param _file_path: full file path including file name and extension
    :return: profile, s_axis, t_axis
    """

    t_axis = np.linspace(-_size_tang / 2, _size_tang / 2, _npt)
    s_axis = np.linspace(-_size_sag / 2, _size_sag / 2, _nps)
    S, T = np.meshgrid(s_axis, t_axis, indexing='xy')

    if _b_freq_t is None and _b_per_t is None:
        omega_t = 0
    elif _b_freq_t is None:
        omega_t = 2 * np.pi / _b_per_t
    elif _b_per_t is None:
        omega_t = 2 * np.pi * _b_freq_t

    if _b_freq_s is None and _b_per_s is None:
        omega_s = 0
    elif _b_freq_s is None:
        omega_s = 2 * np.pi / _b_per_s
    elif _b_per_t is None:
        omega_s = 2 * np.pi * _b_freq_s

    profile = _b_amp*np.sin(omega_t*T+omega_s*S+_b_phase)+_b_offset

    if _file_path is not None:
        save_profile(profile, s_axis, t_axis, _file_path)

    return profile, s_axis, t_axis


def single_gaussian_bump_2d(_size_tang=1., _size_sag=1., _npt=100, _nps=100, _b_amp=1, _b_pos_t=0, _b_fwhm_t=0.1,
                            _b_pos_s=0, _b_fwhm_s=0.1, _file_path=None):
    """
     2D Gaussian bump profile
    :param _size_tang: meridional (or tangential) width
    :param _size_sag: sagittal (or transverse) width
    :param _npt: tangential number of pixels
    :param _nps: transverse number of pixels
    :param _b_amp: bump amplitude
    :param _b_pos_t: tangential mean [m]
    :param _b_fwhm_t: tangential fwhm [m]
    :param _b_pos_s: sagittal mean [m]
    :param _b_fwhm_s: sagittal mefwhm [m]
    :param _file_path: full file path including file name and extension
    :return: profile, s_axis, t_axis
    """

    t_axis = np.linspace(-_size_tang/2, _size_tang/2, _npt)
    s_axis = np.linspace(-_size_sag/2, _size_sag/2, _nps)
    S, T = np.meshgrid(s_axis, t_axis, indexing='xy')

    profile = _b_amp*np.exp(-0.5*((T-_b_pos_t)/(_b_fwhm_t/2.35482))**2)*np.exp(-0.5*((S-_b_pos_s)/(_b_fwhm_s/2.35482))**2)

    if _file_path is not None:
        save_mirror_profile(profile, s_axis, t_axis, _file_path)

    return profile, s_axis, t_axis

