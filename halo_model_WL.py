import numpy as np

from scipy.integrate import quad, cumulative_trapezoid, trapezoid, simpson
from scipy.interpolate import interp1d,RegularGridInterpolator, InterpolatedUnivariateSpline, lagrange
from scipy.optimize import brentq, root_scalar
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import solve_ivp
from scipy.special import gamma as Gamma, sici

import camb
from camb import get_matter_power_interpolator, model, nonlinear

from fastpt import FASTPT


class units_and_constants:
    def __init__(self):
        self.Mpc      = 1.
        self.kpc      = self.Mpc/1000.
        self.pc       = self.kpc/1000.
        self.cm       = self.pc/3.086e18
        self.km       = 1.e5*self.cm
        self.s        = 1.
        self.yr       = 3.15576e7*self.s
        self.Gyr      = 1.e9*self.yr
        self.Msun     = 1.
        self.gram     = self.Msun/1.988e33
        self.c        = 2.9979e10*self.cm/self.s
        self.G        = 6.6742e-8*self.cm**3/self.gram/self.s**2

class utility:
    def __init__(self):
        pass

    def find_closest_index_value(self,x:float, xs:np.array) -> tuple:
        '''
        Find the index, value pair of the closest values in array 'arr' to value 'x'
        '''
        idx = (np.abs(xs-x)).argmin()
        return idx, xs[idx]

    def derivative_from_samples(self,x:float, xs:np.array, fs:np.array) -> float:
        '''
        Calculates the derivative of the function f(x) which is sampled as fs at values xs
        Approximates the function as quadratic using the samples and Legendre polynomials
        Args:
            x: Point at which to calculate the derivative
            xs: Sample locations
            fs: Value of function at sample locations
        '''
        ix, _ = self.find_closest_index_value(x, xs)
        if ix == 0:
            imin, imax = (0, 1) if x < xs[0] else (0, 2) # Tuple braces seem to be necessarry here
        elif ix == len(xs)-1:
            nx = len(xs)
            imin, imax = (nx-2, nx-1) if x > xs[-1] else (nx-3, nx-1) # Tuple braces seem to be necessarry here
        else:
            imin, imax = ix-1, ix+1
        poly = lagrange(xs[imin:imax+1], fs[imin:imax+1])
        return poly.deriv()(x)

class cosmology(units_and_constants,utility):
    
    def __init__(self):
        units_and_constants.__init__(self)
        utility.__init__(self)
        self.OmegaB        = 0.049
        self.OmegaM        = 0.315
        self.OmegaC        = self.OmegaM-self.OmegaB
        self.OmegaL        = 1.-self.OmegaM
        self.h             = 0.674
        self.H0            = self.h*100*self.km/self.s/self.Mpc 
        self.rhocrit0      = 3*self.H0**2/(8.0*np.pi*self.G)  
        
        self.ns       = 0.965
        self.sigma8   = 0.811

        self.rhom = self.rhocrit0*self.OmegaM

    def Hubble(self, z):
        return self.H0*np.sqrt(self.OmegaM*(1.+z)**3+self.OmegaL)

    def g(self, z):
        return self.OmegaM*(1.+z)**3+self.OmegaL

    def growthD(self, z):
        Omega_Lz = self.OmegaL/(self.OmegaL+self.OmegaM*(1.+z)**3)
        Omega_Mz = 1-Omega_Lz
        phiz     = Omega_Mz**(4./7.)-Omega_Lz+(1.+Omega_Mz/2.0)*(1.+Omega_Lz/70.0)
        phi0     = self.OmegaM**(4./7.)-self.OmegaL+(1.+self.OmegaM/2.0)*(1.+self.OmegaL/70.0)
        return (Omega_Mz/self.OmegaM)*(phi0/phiz)/(1.+z)

    def Om_at_z(self,z):
        return self.OmegaM*(1.+z)**3/self.g(z)

    def comoving_distance(self, z):
        """
        Comoving distance [Mpc] at redshift z, accepts scalar and vector input
        """
        integrand = lambda z: 1 / self.Hubble(z)
        if np.isscalar(z):
            integral, _ = quad(integrand, 0, z)
            return  self.c * integral
        else:
            integrals =  np.array([quad(integrand, 0, x)[0] for x in z])
            return  self.c * integrals
    
    def chi_to_z_interp(self,zmin,zmax):
        """
        Creates interpolator from z in range (zmin,zmax) to chi 

        """
        z_grid = np.linspace(zmin, zmax, 1000)  
        chi_grid = self.comoving_distance(z_grid)
        
        chi_to_z_interp = interp1d(chi_grid, z_grid, kind='cubic', bounds_error=True)
        
        return chi_to_z_interp
    
class halo_model(cosmology):

    def __init__(self):
        super().__init__()
        self.dc0 = (3./20.)*(12.*np.pi)**(2./3.) 
        self.Dv0 = 18.*np.pi**2   

    def lagrangian_radius(self,M):
        return np.cbrt(3*M/(4*np.pi*self.rhom))

    def sigma_R(self, R, Pk, k_grid):
        R_input = R
        R = np.atleast_1d(R)

        k = k_grid[None, :]      # (1, nk)
        Rv = R[:, None]          # (nR, 1)

        x = k * Rv               # (nR, nk)

        W = np.ones_like(x)
        W = 3 * (np.sin(x) - x * np.cos(x)) / x**3

        k2Pk = k_grid**2 * Pk
        integrand = k2Pk[None, :] * W**2

        integral = simpson(integrand, k_grid, axis=1)
        sigma = np.sqrt(integral / (2 * np.pi**2))

        if np.isscalar(R_input):
            return float(sigma[0])
        return sigma

    def sigma8_z(self, Pk, k_grid):
        return self.sigma_R(8.0/self.h, Pk, k_grid)

    def neff_at_R(self, R, Pk, k_grid, eps=1e-3):
        """
        Compute n_eff at a single R using symmetric log-derivative.
        eps = fractional log step size
        """

        R1 = R * np.exp(-eps)
        R2 = R * np.exp(+eps)

        sigma1 = self.sigma_R(R1, Pk, k_grid)
        sigma2 = self.sigma_R(R2, Pk, k_grid)

        ln_sigma2_1 = np.log(sigma1**2)
        ln_sigma2_2 = np.log(sigma2**2)

        dlnsigma2_dlnR = (ln_sigma2_2 - ln_sigma2_1) / (2*eps)

        neff = -dlnsigma2_dlnR - 3.0

        return neff
    
    def sigmaV(self,R:float, Pk:callable, kmin=0., kmax=np.inf, eps=1e-4) -> float:

        def window_func(k, R):
            x = k * R
            W = np.ones_like(x)
            W = 3 * (np.sin(x) - x*np.cos(x)) / x**3
            return W
    
        def sigmaV_integrand(k:np.ndarray, R:float, Pk:callable) -> np.ndarray:
            if R == 0.:
                integrand = Pk(k)
            else:
                integrand = Pk(k)*window_func(k,R)**2
            return integrand

        sigmaV_squared, _ = quad(lambda k: sigmaV_integrand(k, R, Pk), kmin, kmax, epsabs=0., epsrel=eps)
        sigmaV = np.sqrt(sigmaV_squared/(2.*np.pi**2))
        sigmaV /= np.sqrt(3.) # Convert from 3D displacement to 1D displacement
        return sigmaV

    def dc_NakamuraSuto(self,Om_mz:float) -> float:
        '''
        LCDM fitting function for the critical linear collapse density from Nakamura & Suto
        (1997; https://arxiv.org/abs/astro-ph/9612074)
        Cosmology dependence is very weak
        '''
        return self.dc0*(1.+0.012299*np.log10(Om_mz))

    def Dv_BryanNorman(self,z) -> float:
        '''
        LCDM fitting function for virial overdensity from Bryan & Norman
        (1998; https://arxiv.org/abs/astro-ph/9710107)
        Note that here Dv is defined relative to background matter density,
        whereas in paper it is relative to critical density
        For Omega_m = 0.3 LCDM Dv ~ 330.
        '''
        Om_mz = self.Om_at_z(z)
        x = Om_mz-1.
        Dv = self.Dv0+82.*x-39.*x**2
        return Dv/Om_mz
    
    def f_Mead(self,x:float, y:float, p0:float, p1:float, p2:float, p3:float) -> float:
        return p0+p1*(1.-x)+p2*(1.-x)**2+p3*(1.-y)

    def dc_Mead(self,z:float, Om_m:float, f_nu:float, g:float, G:float) -> float:
        '''
        delta_c fitting function from Mead (2017; 1606.05345)
        All input parameters should be evaluated as functions of a/z
        '''
        a = 1/(1+z)
        # See Appendix A of Mead (2017) for naming convention
        p10, p11, p12, p13 = -0.0069, -0.0208, 0.0312, 0.0021
        p20, p21, p22, p23 = 0.0001, -0.0647, -0.0417, 0.0646
        a1, _ = 1, 0
    
        # Linear collapse threshold
        dc_Mead = 1.
        dc_Mead = dc_Mead+self.f_Mead(g/a, G/a, p10, p11, p12, p13)*np.log10(Om_m)**a1
        dc_Mead = dc_Mead+self.f_Mead(g/a, G/a, p20, p21, p22, p23)
        dc_Mead = dc_Mead*self.dc0*(1.-0.041*f_nu)
        return dc_Mead

    def Dv_Mead(self,a:float, Om_m:float, g:float, G:float) -> float:
        '''
        Delta_v fitting function from Mead (2017; 1606.05345)
        All input parameters should be evaluated as functions of a/z
        '''

        # See Appendix A of Mead (2017) for naming convention
        p30, p31, p32, p33 = -0.79, -10.17, 2.51, 6.51
        p40, p41, p42, p43 = -1.89, 0.38, 18.8, -15.87
        a3, a4 = 1, 2

        # Halo virial overdensity
        Dv_Mead = 1.
        Dv_Mead = Dv_Mead+self.f_Mead(g/a, G/a, p30, p31, p32, p33)*np.log10(Om_m)**a3
        Dv_Mead = Dv_Mead+self.f_Mead(g/a, G/a, p40, p41, p42, p43)*np.log10(Om_m)**a4
        Dv_Mead = Dv_Mead*self.Dv0
        return Dv_Mead

    def Omega_m(self,a:float, CAMB_results) -> float:
        '''
        Evolution of Omgea_m with scale-factor ignoring radiation
        '''
        Om_c = CAMB_results.get_Omega(var='cdm', z=0.)
        Om_b = CAMB_results.get_Omega(var='baryon', z=0.)
        Om_nu = CAMB_results.get_Omega(var='nu', z=0.)
        Om_m = Om_c+Om_b+Om_nu
        return Om_m*a**-3/self.Hubble2(a, CAMB_results)

    def Hubble2(self,a:float, CAMB_results) -> float:
        '''
        Squared Hubble parameter ignoring radiation
        Massive neutrinos are counted as 'matter'
        '''
        Om_c = CAMB_results.get_Omega(var='cdm', z=0.)
        Om_b = CAMB_results.get_Omega(var='baryon', z=0.)
        Om_nu = CAMB_results.get_Omega(var='nu', z=0.)
        Om_m = Om_c+Om_b+Om_nu
        Om_w = 1.-Om_m
        Om = 1.
        H2 = Om_m*a**-3+Om_w*self.X_w(a)+(1.-Om)*a**-2
        return H2

    def w(self,a:float, CAMB_results) -> float:
        '''
        Dark energy equation of state for w0, wa models
        '''
        w0, wa = (-1., 0.)
        return w0+(1.-a)*wa

    def X_w(self,a:float) -> float:
        '''
        Cosmological dark energy density for w0, wa models
        '''
        w0, wa = (-1., 0.)
        return a**(-3.*(1.+w0+wa))*np.exp(-3.*wa*(1.-a))

    def AH(self,a:float, CAMB_results) -> float:
        '''
        Acceleration parameter ignoring radiation
        Massive neutrinos are counted as 'matter'
        '''
        Om_c = CAMB_results.get_Omega(var='cdm', z=0.)
        Om_b = CAMB_results.get_Omega(var='baryon', z=0.)
        Om_nu = CAMB_results.get_Omega(var='nu', z=0.)
        Om_m = Om_c+Om_b+Om_nu
        Om_w = 1.-Om_m
        AH = -0.5*(Om_m*a**-3+(1.+3.*self.w(a, CAMB_results))*Om_w*self.X_w(a))
        return AH

    def get_growth_interpolator(self,CAMB_results) -> callable:
        na = 129 
        a_init = 1e-4
        a = np.linspace(a_init, 1., na)
        f = 1.-self.Omega_m(a_init, CAMB_results) # Early mass density
        d_init = a_init**(1.-3.*f/5.)            # Initial condition (~ a_init; but f factor accounts for EDE-ish)
        v_init = (1.-3.*f/5.)*a_init**(-3.*f/5.) # Initial condition (~ 1; but f factor accounts for EDE-ish)
        y0 = (d_init, v_init)
        def fun(a, y):
            d, v = y[0], y[1]
            dxda = v
            fv = -(2.+self.AH(a, CAMB_results)/self.Hubble2(a, CAMB_results,))*v/a
            fd = 1.5*self.Omega_m(a, CAMB_results)*d/a**2
            dvda = fv+fd
            return dxda, dvda
        g = solve_ivp(fun, (a[0], a[-1]), y0, t_eval=a).y[0]
        g_interp = interp1d(a, g, kind='cubic', assume_sorted=True)
        return g_interp

    def conc200(self,M200,z): 
        """ Correa et al. (2015) """
        alpha_cMz_1 = 1.7543-0.2766*(1.+z)+0.02039*(1.+z)**2
        beta_cMz_1  = 0.2753+0.00351*(1.+z)-0.3038*(1.+z)**0.0269
        gamma_cMz_1 = -0.01537+0.02102*(1.+z)**-0.1475
        c_Mz_1      = np.power(10.,alpha_cMz_1+beta_cMz_1*np.log10(M200/self.Msun) \
                               *(1+gamma_cMz_1*np.log10(M200/self.Msun)**2))
        alpha_cMz_2 = 1.3081-0.1078*(1.+z)+0.00398*(1.+z)**2
        beta_cMz_2  = 0.0223-0.0944*(1.+z)**-0.3907
        c_Mz_2      = pow(10,alpha_cMz_2+beta_cMz_2*np.log10(M200/self.Msun))
        return np.where(z<=4.,c_Mz_1,c_Mz_2)

    def f_sheth_tormen(self,nu, A=0.3222, p=0.3, q=0.707):
        A = np.sqrt(2.*q)/(np.sqrt(np.pi) + Gamma(0.5-p)/2**p)
        return A*(1.+((q*nu**2)**(-p)))*np.exp(-q*nu**2/2.)

    def b_sheth_tormen(self, nu, dc,p=0.3, q=0.707) -> np.ndarray:
        '''
        Halo linear bias b(nu) with nu=delta_c/sigma(M)
        Integral of b(nu)*g(nu) over all nu is unity
        '''
        return 1.+(q*(nu**2)-1.+2.*p/(1.+(q*nu**2)**p))/dc

    def u_nfw(self,k, c, rs):
        """
        Analytical Fourier transform u(k|M_200) of an NFW halo.

        """
        k  = np.atleast_1d(k).astype(float)
        rs = np.atleast_1d(rs).astype(float)
        c  = np.atleast_1d(c).astype(float)

        if rs.shape != c.shape:
            raise ValueError("rs and c must have the same shape")

        # Broadcast k against halo parameters
        x = k[:, None] * rs[None, :]

        denom = np.log(1.0 + c) - c / (1.0 + c)
        denom = denom[None, :]  # broadcast over k

        # sine and cosine integrals
        Si_x,  Ci_x  = sici(x)
        Si_xc, Ci_xc = sici((1.0 + c)[None, :] * x)

        u = (
            np.cos(x) * (Ci_xc - Ci_x)
            + np.sin(x) * (Si_xc - Si_x)
            - np.sin(c[None, :] * x) / ((1.0 + c)[None, :] * x)
        ) / denom

        # Return scalar if everything was scalar
        if u.shape == (1, 1):
            return u[0, 0]
        return u

    def get_nonlinear_radius(self,Rmin:float, Rmax:float, dc:float, Pk_lin_z,k_grid) -> float:
        Rnl_root = lambda R: self.sigma_R(R, Pk_lin_z,k_grid)-dc
        Rnl = root_scalar(Rnl_root, bracket=(Rmin, Rmax)).root
        return Rnl

    def get_effective_index(self,Rnl:float, R:np.ndarray, sigmaR:np.ndarray) -> float:
        neff = -3.-2.*self.derivative_from_samples(np.log(Rnl), np.log(R), np.log(sigmaR))
        return neff

class lensing(cosmology):
    def __init__(self):
        super().__init__()

    def nz_to_nchi_interp(self, nz, z, z_mean=0, shift=0, stretch=1):
        z = (z - z_mean) / stretch + z_mean - shift
        nz = nz / stretch

        norm = np.trapezoid(nz, z)
        nz = nz / norm

        nchi = nz * self.Hubble(z) / self.c 
        chi = self.comoving_distance(z)

        nchi_interp = interp1d(chi, nchi, kind='cubic', bounds_error=False, fill_value=0)
        return nchi_interp

    def lensing_efficiency(self, nchi_interp, shear=0):
        z = self.z
        chi = self.chi

        prefactor = 1.5 * self.H0**2 * self.OmegaM / self.c**2

        f = nchi_interp(chi) / chi
        fchi = nchi_interp(chi)

        int_f = cumulative_trapezoid(f, chi, initial=0)
        int_fchi = cumulative_trapezoid(fchi, chi, initial=0)

        total_f = int_f[-1]
        total_fchi = int_fchi[-1]

        integral_from_i = (total_fchi - int_fchi) - chi * (total_f - int_f)

        q = prefactor * chi * (1 + z) * integral_from_i

        return q * (1 + shear)
    
class power_spectrum(halo_model):

    def __init__(self, zmin=0.01, zmax=2.0, nz=128, kmin=1e-4, kmax=1e3,nk=128,model='halomodel',tweaks = True,file_dir=None,DM_type='CDM',**tweak_args):
        super().__init__()
        if model == 'subhalomodel':
            self.load_subhalo_data(file_dir=file_dir,DM_type=DM_type,tweaks=tweaks,**tweak_args)
        else:
            self.zmin = zmin
            self.zmax = zmax
            self.nz = nz
            self.kmin = kmin 
            self.kmax = kmax 
            self.nk = nk
            self.k = np.logspace(np.log10(kmin), np.log10(kmax), nk) # 1/Mpc
            self.z_ps = np.linspace(zmin, zmax, nz)
            
            print(f'kmin = {kmin:.2e} 1/Mpc, kmax = {kmax:.2e} 1/Mpc')
            print(f'zmin = {zmin:.2f}, zmax = {zmax:.2f}')

            #Plin 
            self.camb_pars, self.camb_results = self.camb_init()
            self.Plin_interp = self.get_Pmm_interp_from_camb(False)
            self.Plin = self.Plin_interp(self.z_ps, self.k)

            # Build Pmm interpolator
            if model == 'halomodel':
                if tweaks:
                    self.Pmm = self.get_P_halomodel_sashimi(bloating=True,two_halo_suppression=True,transition_smoothing=True,concentration_boost=True,two_halo_term='dewiggled',**tweak_args)['total']
                    print(f"Computed Pmm using standard halo model with tweaks")
                else:
                    self.Pmm = self.get_P_halomodel_sashimi(bloating=False,two_halo_suppression=False,transition_smoothing=False,concentration_boost=False,two_halo_term='dewiggled')['total']
                print(f"Computed Pmm using standard halo model without tweaks")

            elif model == 'linear':
                self.Pmm_interp = self.Plin_interp
                self.Pmm = self.Plin
                print(f"Computed Pmm using CAMB linear model")

            elif model == 'HMcode':
                if tweaks:
                    self.Pmm = self.get_P_halomodel(bloating=True,two_halo_suppression=True,transition_smoothing=True,two_halo_term='dewiggled',**tweak_args)['total']
                    print(f"Computed Pmm using HMcode halo model with tweaks")
                else:
                    self.Pmm = self.get_P_halomodel(bloating=False,two_halo_suppression=False,transition_smoothing=False,two_halo_term='dewiggled',B=4.)['total']
                    print(f"Computed Pmm using HMcode halo model without tweaks")

            else:
                self.Pmm_interp = self.get_Pmm_interp_from_camb(model)
                self.Pmm = self.Pmm_interp(self.z_ps, self.k)
                print(f"Computed Pmm using CAMB non-linear model {model}")

    def camb_init(self, set_sigma8=True):  

        pars = camb.CAMBparams(WantCls=False)
        wb, wc = self.OmegaB * self.h**2, self.OmegaC * self.h**2

        # This function sets standard and helium set using BBN consistency
        pars.set_cosmology(ombh2=wb, omch2=wc, H0=100.*self.h, mnu=0.0, omk=0.0)
        pars.set_dark_energy(w=-1, wa=0, dark_energy_model='ppf')
        
        As = 2e-9 
        pars.InitPower.set_params(As=As, ns=self.ns, r=0.)

        z = np.linspace(0, self.zmax, self.nz)
        pars.set_matter_power(redshifts=z, kmax=self.kmax) #maximum k to calculate (where k is just k, not k/h)
        
        results = camb.get_results(pars)
        sigma8_fid = results.get_sigma8_0()  

        if set_sigma8:
            As *= (self.sigma8 / sigma8_fid)**2
            self.As = As
            pars.InitPower.set_params(As=self.As, ns=self.ns, r=0.)
            pars.set_matter_power(redshifts=z, kmax=self.kmax)
            results = camb.get_results(pars)

        return  pars, results

    def get_Pmm_interp_from_camb(self,model):  
        PK = get_matter_power_interpolator(
            params=self.camb_pars,
            nonlinear=model,
            hubble_units=False,
            k_hunit=False,
            kmax=self.kmax,
            zmin=0,
            zmax=self.zmax,
            extrap_kmax = 1e10
            )
        return PK.P  
    
    def get_P_interp_from_file(self, filename, field):

        data = np.load(filename)

        k = data["k"]                   # (Nk,)
        z = data["z"]                   # (Nz,)
        P = data[field]                 # (Nz, Nk)

        # Check if k and z grids cover the required ranges

        if k.min() > self.kmin:
            raise ValueError(
                "k grid in file does not cover the required range: "
                "kmin = {:.3e} > {:.3e}".format(k.min(), self.kmin)
            )

        if k.max() < self.kmax:
            raise ValueError(
                "k grid in file does not cover the required range: "
                "kmax = {:.3e} < {:.3e}".format(k.max(), self.kmax)
            )

        if z.min() > self.z_ps.min():
            raise ValueError(
                "z grid in file does not cover the required range: "
                "zmin = {:.3f} > {:.3f}".format(z.min(), self.z_ps.min())
            )

        if z.max() < self.z_ps.max():
            raise ValueError(
                "z grid in file does not cover the required range: "
                "zmax = {:.3f} < {:.3f}".format(z.max(), self.z_ps.max())
            )

        return RegularGridInterpolator(
            (z, k),
            P,
            bounds_error=False,
            fill_value=None
        )

    def Tk_EH_nowiggle(self,k:np.ndarray, h:float, wm:float, wb:float, T_CMB=2.725) -> np.ndarray:
        '''
        No-wiggle transfer function from astro-ph:9709112
        '''
        # These only needs to be calculated once
        rb = wb/wm     # Baryon ratio
        e = np.exp(1.) # e
        s = 44.5*np.log(9.83/wm)/np.sqrt(1.+10.*wb**0.75)              # Equation (26)
        alpha = 1.-0.328*np.log(431.*wm)*rb+0.38*np.log(22.3*wm)*rb**2 # Equation (31)

        # Functions of k
        Gamma = (wm/h)*(alpha+(1.-alpha)/(1.+(0.43*k*s*h)**4)) # Equation (30)
        q = k*(T_CMB/2.7)**2/Gamma # Equation (28)
        L = np.log(2.*e+1.8*q)     # Equation (29)
        C = 14.2+731./(1.+62.5*q)  # Equation (29)
        Tk_nw = L/(L+C*q**2)       # Equation (29)
        return Tk_nw

    def get_P_dewiggled(self, z_arr,k_arr,sigma_dlnk=0.25):
        """
        Extract BAO wiggle component from linear power spectrum
        using EH no-wiggle template ratio smoothing.
        """

        Plin = self.Plin

        # ---- Check log spacing in k ----
        dlnk = np.log(k_arr[1] / k_arr[0])
        if not np.allclose(np.diff(np.log(k_arr)), dlnk, rtol=1e-5):
            raise ValueError("Dewiggle requires log-spaced k grid")

        sigma = sigma_dlnk / dlnk

        # ---- EH no-wiggle template (1D in k) ----
        Tk_now = self.Tk_EH_nowiggle(k_arr, self.h, self.OmegaM * self.h**2, self.OmegaB * self.h**2)
        Pk_now = (k_arr**self.ns) * Tk_now**2 #* (2*np.pi**2) / k_arr**3

        # ---- Ratio smoothing ----
        ratio = Plin / Pk_now[None, :]
        ratio_smooth = gaussian_filter1d(ratio, sigma, axis=1)

        P_smooth = ratio_smooth * Pk_now[None, :]
        P_wiggle = Plin - P_smooth

        P_dwg = np.zeros_like(Plin)

        # ---- Loop over redshifts for sigma_v ----
        for i, z_i in enumerate(z_arr):

            interp = interp1d(
                k_arr,
                Plin[i, :],
                bounds_error=False,
                fill_value="extrapolate"
            )

            sigmaV = self.sigmaV(z_i, interp)

            P_dwg[i, :] = Plin[i, :] - (1.-np.exp(-(k_arr*sigmaV)**2)) * P_wiggle[i, :]
        
        return P_dwg
    
    def get_sigma8(self):
        sigma8s = np.zeros_like(self.z_ps)
        for i, z in enumerate(self.z_ps):
            sigma8s[i] = self.sigma8_z(self.Pmm[i,:],self.k)
        return sigma8s

    def get_P_halomodel(
        self,
        Mmin=1,
        Mmax=1e16,
        nM=128,          
        bloating=True,
        two_halo_suppression=True,
        transition_smoothing=True,
        two_halo_term = 'dewiggled',
        k_d_amp=0.03841,
        k_d_exp=-1.089,
        f_amp=0.2696,
        f_exp=0.9403,
        n_d=2.853,
        one_halo_supp_amp=0.03786,
        one_halo_supp_exp=-1.013,
        eta_amp=0.1281,
        eta_exp=-0.3644,
        B=5.196,
        alpha_amp=1.875,
        alpha_exp=1.603,
        ):

        M_grid = np.logspace(np.log10(Mmin), np.log10(Mmax), nM)
        z_grid = self.z_ps
        k_grid = self.k

        nz = len(z_grid)
        nk = len(k_grid)

        if two_halo_term == 'dewiggled':
            integrate_two_halo = False
            P_2h = self.get_P_dewiggled(z_grid,k_grid)
        elif two_halo_term == 'linear':
            integrate_two_halo = False
            P_2h = self.Plin
        elif two_halo_term == 'halomodel':
            integrate_two_halo = True
            P_2h = np.zeros((nz,nk))
        else:
            raise KeyError("valid two halo options are: 'dewiggled', 'linear', and 'halomodel'")

        g = self.get_growth_interpolator(self.camb_results)

        P_1h = np.zeros((nz,nk))
        P_total = np.zeros((nz,nk))
        
        R = self.lagrangian_radius(M_grid)
        rhom = self.rhom

        for i, z in enumerate(z_grid):

            Plin = self.Plin[i,:]
            sigma8_z = self.sigma8_z(Plin, k_grid)

            a = 1./(1.+z)
            a_init = 1e-4
            G, _ = quad(lambda a: g(a)/a, a_init, a)
            G += g(a_init)

            Om = self.Om_at_z(z)

            Dv = self.Dv_Mead(a, Om, g(a), G)
            dc = self.dc_NakamuraSuto(Om)

            sigmaM = self.sigma_R(R, Plin, k_grid)
            nu = dc/sigmaM

            u = np.zeros((nM,nk))

            gamma = 0.01
            Rc = (3*(gamma*M_grid)/(4*np.pi*rhom))**(1./3.)
            sigma_c = self.sigma_R(Rc, Plin, k_grid)

            for M_id,M in enumerate(M_grid):
                fac = g(a) * dc/sigma_c[M_id]
                if fac >= g(a):
                    zf = z
                else:
                    af_root = lambda af: g(af) - fac
                    af = root_scalar(af_root, bracket=(1e-3, 1.)).root
                    zf  = -1+1/af

                cvir = B*(1+zf)/(1+z)
                rvir = R[M_id]/np.cbrt(Dv)
                rs_vir = rvir/cvir

                if bloating:
                    eta = eta_amp*sigma8_z**eta_exp
                    nu_eta = nu[M_id]**eta

                    u[M_id,:] = self.u_nfw(k_grid*nu_eta, cvir, rs_vir)[:,0]
                else:
                    u[M_id,:]  = self.u_nfw(k_grid, cvir, rs_vir)[:,0]

            f_nu = self.f_sheth_tormen(nu)

            #deriv = np.zeros(len(R))
            #logR, logsigmaM = np.log(R), np.log(sigmaM)
            #for iR, _logR in enumerate(logR):
            #    deriv[iR] = 2.*self.derivative_from_samples(_logR, logR, logsigmaM)
            #dnu_dlnm = -(nu/6.)*deriv

            integrand = f_nu[:,None] * u**2 * (M_grid[:,None]/rhom)
            P_1h[i,:] = trapezoid(integrand, nu, axis=0)

            #integrand = f_nu[:,None] * (M_grid[:,None]/rhom) * u**2 * dnu_dlnm[:,None]
            #P_1h[i,:] = trapezoid(integrand, np.log(M_grid), axis=0)
                
            k_star = one_halo_supp_amp * sigma8_z**one_halo_supp_exp
            P_1h[i,:] *= (k_grid/k_star)**4/(1+(k_grid/k_star)**4)

            if integrate_two_halo:
                b_nu = self.b_sheth_tormen(nu,dc)
                A = 1.-quad(lambda nu: self.f_sheth_tormen(nu) * self.b_sheth_tormen(nu,dc), nu[0], np.inf)[0]

                integrand = b_nu[:,None] * f_nu[:,None] * u 
                int = trapezoid(integrand, nu, axis=0)
                P_2h[i,:] = Plin * (int + A * u[0,:])**2

            if two_halo_suppression:
                k_d = k_d_amp * sigma8_z**k_d_exp
                f = f_amp * sigma8_z**f_exp
                P_2h[i,:] *= 1 - f*(k_grid/k_d)**n_d/(1+(k_grid/k_d)**n_d)

            P_total[i,:] = P_1h[i,:] + P_2h[i,:]

            if transition_smoothing:
                R_nl = self.get_nonlinear_radius(R.min(),R.max(),dc,Plin,k_grid)
                n_eff = self.get_effective_index(R_nl,R,sigmaM)
                alpha = alpha_amp * alpha_exp**(n_eff)

                P_total[i,:] = (P_2h[i,:]**alpha+P_1h[i,:]**alpha)**(1./alpha) 

        return {'total':P_total,'1h':P_1h,'2h':P_2h}
    
    def get_P_halomodel_sashimi(
        self,
        Mmin=1e-6,
        Mmax=1e16,
        nM=128,          
        bloating=True,
        two_halo_suppression=True,
        transition_smoothing=True,
        concentration_boost=True,
        two_halo_term = 'dewiggled',
        k_d_amp=0.03841,
        k_d_exp=-1.089,
        f_amp=0.2696,
        f_exp=0.9403,
        n_d=2.853,
        one_halo_supp_amp=0.03786,
        one_halo_supp_exp=-1.013,
        eta_amp= 0.1281,#2.38309998e-05,#0.2405,#2.38309998e-05, #was 0.1281 -> fit for shm
        eta_exp= -0.3644,#6.87349482e-02,#6.87349482e-02,# was-0.3644, -> fit for shm
        alpha_amp=1.875,#2.17150518e+00, -> fit for shm
        alpha_exp=1.603,#1.63891522e+00, -> fit for shm
        conc_boost=5.196/4, #-> fit for shm
        subhalos=False,
        DM_type='CDM',
        ):

        if DM_type=='CDM':
            from sashimi_c import halo_model
            hm = halo_model()
        elif DM_type=='SIDM':
            from sashimi_si import halo_model
            hm = halo_model()

        if subhalos:
            nM=self.nM
            M_grid = self.M_grid
        else:
            M_grid = np.logspace(np.log10(Mmin), np.log10(Mmax), nM)

        z_grid = self.z_ps
        k_grid = self.k

        #nz = self.nz
        #nk = self.nk
        nz = len(self.z_ps)  
        nk = len(self.k)    

        if two_halo_term == 'dewiggled':
            integrate_two_halo = False
            P_2h = self.get_P_dewiggled(z_grid,k_grid)
        elif two_halo_term == 'linear':
            integrate_two_halo = False
            P_2h = self.Plin
        elif two_halo_term == 'halomodel':
            integrate_two_halo = True
            P_2h = np.zeros((nz,nk))
        else:
            raise KeyError("valid two halo options are: 'dewiggled', 'linear', and 'halomodel'")
        
        P_1h = np.zeros((nz,nk))
        P_total = np.zeros((nz,nk))
        if subhalos:
            P_1h_sm_sm = np.zeros((nz,nk))
            P_1h_sm_sh = np.zeros((nz,nk))
            P_1h_1sh = np.zeros((nz,nk))
            P_1h_2sh = np.zeros((nz,nk))
            
            P_2h_sm_sm = np.zeros((nz,nk))
            P_2h_sm_sh = np.zeros((nz,nk))
            P_2h_2sh = np.zeros((nz,nk))

        rhom = self.rhom
                    
        #lnM = np.log(M_grid)
        #deriv = np.zeros(len(R))
        #logR = np.log(R)

        for i, z in enumerate(z_grid):
            Plin = self.Plin[i,:]
            #M8 = (4*np.pi/3) * self.rhocrit0 * self.OmegaM * (8/self.h)**3
            #sigma8 = hm.sigmaMz(M8,z)
            sigma8 = self.sigma8_z(Plin, k_grid)
            
            if subhalos: 
                R = self.lagrangian_radius(self.Mvir_grid[:,i])
            else:
                R = self.lagrangian_radius(M_grid)

            dc = hm.deltac_func(z)*hm.growthD(z) #different from HMcode (z-independent)
            #sigmaM = hm.sigmaMz(M_grid,z) #different from HMcode
            sigmaM = self.sigma_R(R, Plin, k_grid)
            nu = dc/sigmaM

            Om = self.Om_at_z(z)
            Dv = hm.Delc(Om-1)/Om

            #logsigmaM = np.log(sigmaM)
            #for iR, _logR in enumerate(logR):
            #    deriv[iR] = 2.*self.derivative_from_samples(_logR, logR, logsigmaM)
            #dnu_dlnm = -(nu/6.)*deriv
            f_nu = self.f_sheth_tormen(nu)
            #dndlnM = rhom/M_grid * f_nu * dnu_dlnm
            
            u = np.zeros((nM,nk))   
            for M_id,M in enumerate(M_grid):
                c200 = hm.conc200(M,z) #different from HMcode
                r200 = np.cbrt(3*M/(4*np.pi*self.rhocrit0*hm.g(z)*200.))*(1+z) #different from HMcode (200c)
                rvir = R[M_id]/np.cbrt(Dv)
                cvir = c200*rvir/r200
                if concentration_boost:
                    cvir *= conc_boost

                rs = rvir/cvir

                if bloating:
                    eta = eta_amp*sigma8**eta_exp
                    nu_eta = nu[M_id]**eta

                    u[M_id,:] = self.u_nfw(k_grid*nu_eta, cvir, rs)[:,0]
                else:
                    u[M_id,:]  = self.u_nfw(k_grid, cvir, rs)[:,0]

            
            if subhalos:
                Msm_grid = self.Msm_grid[:,i]
                usm = np.zeros((nM,nk))
                for M_id,M in enumerate(Msm_grid):
                    c200 = hm.conc200(M,z) #different from HMcode
                    r200 = np.cbrt(3*M/(4*np.pi*self.rhocrit0*hm.g(z)*200.))*(1+z) #different from HMcode (200c)
                    rvir = self.lagrangian_radius(M)/np.cbrt(Dv)
                    cvir = c200*rvir/r200
                    rs = rvir/cvir

                    if concentration_boost:
                        cvir *= conc_boost

                    if bloating:
                        eta = eta_amp*sigma8**eta_exp
                        nu_eta = nu[M_id]**eta

                        usm[M_id,:] = self.u_nfw(k_grid*nu_eta, cvir, rs)[:,0]
                    else:
                        usm[M_id,:]  = self.u_nfw(k_grid, cvir, rs)[:,0]
    
            if subhalos:
                M_grid = self.Mvir_grid[:,i]

                for k_id in range(nk):
                    I = self.I_kmz[k_id, :, i]
                    J = self.J_kmz[k_id, :, i]

                    integrand_ss =  M_grid/rhom * f_nu * (Msm_grid / M_grid)**2  * usm[:,k_id]**2
                    integrand_sc =  M_grid/rhom * f_nu * Msm_grid / M_grid * usm[:,k_id] * u[:,k_id] * I
                    integrand_cc = M_grid/rhom * f_nu * u[:,k_id]**2 * I**2
                    integrand_self_c = M_grid/rhom * f_nu * J
                
                    P_1h_sm_sm[i, k_id] = trapezoid(integrand_ss, nu) 
                    P_1h_sm_sh[i, k_id] = 2 * trapezoid(integrand_sc, nu)
                    P_1h_2sh[i, k_id] = trapezoid(integrand_cc, nu)
                    P_1h_1sh[i, k_id] = trapezoid(integrand_self_c, nu)

                    #integrand_ss =  (M_grid/rhom)**2 * dndlnM * (Msm_grid / M_grid)**2  * usm[:,k_id]**2
                    #integrand_sc =  (M_grid/rhom)**2  * dndlnM * Msm_grid / M_grid * usm[:,k_id] * u[:,k_id] * I
                    #integrand_cc = (M_grid/rhom)**2  * dndlnM * u[:,k_id]**2 * I**2
                    #integrand_self_c = (M_grid/rhom)**2  * dndlnM * J

                    #P_1h_sm_sm[i, k_id] = trapezoid(integrand_ss, lnM) 
                    #P_1h_sm_sh[i, k_id] = 2 * trapezoid(integrand_sc, lnM)
                    #P_1h_2sh[i, k_id] = trapezoid(integrand_cc, lnM)
                    #P_1h_1sh[i, k_id] = trapezoid(integrand_self_c, lnM)

                k_star = one_halo_supp_amp * sigma8**one_halo_supp_exp
                P_1h_sm_sm[i,:] *= (k_grid/k_star)**4/(1+(k_grid/k_star)**4)
                P_1h_sm_sh[i,:] *= (k_grid/k_star)**4/(1+(k_grid/k_star)**4)
                P_1h_2sh[i,:] *= (k_grid/k_star)**4/(1+(k_grid/k_star)**4)
                P_1h_1sh[i,:] *= (k_grid/k_star)**4/(1+(k_grid/k_star)**4)

                P_1h[i,:] = P_1h_sm_sm[i,:]+P_1h_sm_sh[i,:]+P_1h_2sh[i,:]+P_1h_1sh[i,:]

            else:
                integrand = (M_grid[:,None]/rhom) * f_nu[:,None] * u**2
                P_1h[i,:] = trapezoid(integrand, nu, axis=0)
                    
                k_star = one_halo_supp_amp * sigma8**one_halo_supp_exp
                P_1h[i,:] *= (k_grid/k_star)**4/(1+(k_grid/k_star)**4)

            if integrate_two_halo:
                if subhalos:
                    b_nu = self.b_sheth_tormen(nu,dc)

                    rho_sm = rhom*trapezoid((1-self.fsh_list[:,i]) * f_nu * b_nu, nu)
                    rho_sh = rhom*trapezoid(self.fsh_list[:,i] * f_nu * b_nu, nu)

                    A_sm = (rhom - (rho_sm+rho_sh)) / (1 + rho_sh/rho_sm) / rhom
                    A_sh = A_sm * rho_sh / rho_sm

                    for k_id in range(nk):
                        I = self.I_kmz[k_id, :, i]

                        integrand_sm = (1-self.fsh_list[:,i]) * f_nu  * b_nu * usm[:,k_id] #Msm/M = 1-fsh
                        integrand_sh = f_nu * b_nu * u[:,k_id] * I

                        S = trapezoid(integrand_sm, nu) + usm[0,k_id] * A_sm 
                        C = trapezoid(integrand_sh, nu) 
                        if self.fsh_list[0,i] != 0.0: 
                                C += I[0] * u[0,k_id] * A_sh / self.fsh_list[0,i] 

                        P_2h_sm_sm[i, k_id] = Plin[k_id] * S ** 2
                        P_2h_sm_sh[i, k_id] = 2 * Plin[k_id] * S * C
                        P_2h_2sh[i, k_id] = Plin[k_id] * C ** 2
                            
                    P_2h[i,:] = P_2h_sm_sm[i,:]+P_2h_sm_sh[i,:]+P_2h_2sh[i,:]
                
                else:
                    b_nu = self.b_sheth_tormen(nu,dc)
                    A = 1.-quad(lambda nu: self.f_sheth_tormen(nu) * self.b_sheth_tormen(nu,dc), nu[0], np.inf)[0]

                    integrand = b_nu[:,None] * f_nu[:,None] * u 
                    int = trapezoid(integrand, nu, axis=0)
                    P_2h[i,:] = Plin * (int + A * u[0,:])**2

            if two_halo_suppression:
                k_d = k_d_amp * sigma8**k_d_exp
                f = f_amp * sigma8**f_exp
                P_2h[i,:] *= 1 - f*(k_grid/k_d)**n_d/(1+(k_grid/k_d)**n_d)

            P_total[i,:] = P_1h[i,:] + P_2h[i,:]

            if transition_smoothing:
                R_nl = self.get_nonlinear_radius(R.min(),R.max(),dc,Plin,k_grid)
                n_eff = self.get_effective_index(R_nl,R,sigmaM)
                alpha = alpha_amp * alpha_exp**(n_eff)

                P_total[i,:] = (P_2h[i,:]**alpha+P_1h[i,:]**alpha)**(1./alpha) 

        if subhalos:
            if integrate_two_halo:
                return {'total':P_total,
                        '1h':P_1h,'1h_sm_sm':P_1h_sm_sm,'1h_sm_sh':P_1h_sm_sh,'1h_1sh':P_1h_1sh,'1h_2sh':P_1h_2sh,
                        '2h':P_2h,'2h_sm_sm':P_2h_sm_sm,'2h_sm_sh':P_2h_sm_sh,'2h_2sh':P_2h_2sh}
            else:
                return {'total':P_total,
                        '1h':P_1h,'1h_sm_sm':P_1h_sm_sm,'1h_sm_sh':P_1h_sm_sh,'1h_1sh':P_1h_1sh,'1h_2sh':P_1h_2sh,
                        '2h':P_2h}
        else:
            return {'total':P_total,'1h':P_1h,'2h':P_2h}

    def load_subhalo_data(self,file_dir,DM_type='CDM',tweaks=True,**tweak_args):

        if DM_type=='CDM':
            from sashimi_c import halo_model
            hm = halo_model()
        elif DM_type=='SIDM':
            from sashimi_si import halo_model
            hm = halo_model()
        data = np.load(file_dir)

        k = data['k_grid']
        z_ps = data['z_grid']

        nk = len(k)
        nz = len(z_ps)

        kmin = k.min()
        kmax = k.max()
        print(f'kmin = {kmin:.2e} 1/Mpc, kmax = {kmax:.2e} 1/Mpc')

        zmin = z_ps.min()
        zmax = z_ps.max()
        print(f'zmin = {zmin:.2f}, zmax = {zmax:.2f}')

        self.M_grid = data['M_grid']
        self.nM = len(self.M_grid)

        Mmin = self.M_grid.min()
        Mmax = self.M_grid.max()
        print(f'Mmin = {Mmin:.2e} Msun, Mmax = {Mmax:.2e} Msun')

        self.m_mz  = data['m_mz']
        self.I_kmz = data['I_kmz']
        self.J_kmz = data['J_kmz']
 
        m_min = 1e30
        m_max = 0
        for z_id in range(nz):
            for M_id in range(self.nM):
                m_min1 = self.m_mz[M_id][z_id].min()
                m_max1 = self.m_mz[M_id][z_id].max()
                if m_min1 < m_min:
                    m_min = m_min1
                if m_max1 > m_max:
                    m_max = m_max1

        print(f'min subhalo mass = {m_min:.2e} Msun')
        print(f'max subhalo mass = {m_max:.2e} Msun')

        self.Msm_grid = np.zeros((self.nM,nz))
        self.fsh_list = np.zeros((self.nM,nz))
        self.Mvir_grid = np.zeros((self.nM,nz))
        for z_id,z in enumerate(z_ps):
            #self.fsh_list[:,z_id] = np.array(self.m_mz[:,z_id]/hm.Mzi(self.M_grid,z))
            #self.Msm_grid[:,z_id] = self.M_grid*(1-self.fsh_list[:,z_id])

            Mvir = hm.Mvir_from_M200(self.M_grid,z)
            self.fsh_list[:,z_id] = np.array(self.m_mz[:,z_id]/hm.Mvir_from_M200(hm.Mzi(self.M_grid,z),z))
            self.Msm_grid[:,z_id] = Mvir*(1-self.fsh_list[:,z_id])
            self.Mvir_grid[:,z_id] = Mvir

        self.k=k
        self.nk=nk
        self.kmax=kmax
        self.kmin=kmin
        self.z_ps=z_ps
        self.nz=nz
        self.zmax=zmax
        self.zmin=zmin

        #Plin 
        self.camb_pars, self.camb_results =  self.camb_init()
        self.Plin_interp = self.get_Pmm_interp_from_camb(False)
        self.Plin = self.Plin_interp(self.z_ps, self.k)

        if tweaks:
            self.Pmm = self.get_P_halomodel_sashimi(subhalos=True,two_halo_term='dewiggled',DM_type=DM_type,bloating=True,two_halo_suppression=True,transition_smoothing=True,concentration_boost=True,**tweak_args)['total']
            print(f"Computed Pmm using halo model with subhalos and tweaks")
        else:
            self.Pmm = self.get_P_halomodel_sashimi(subhalos=True,two_halo_term='dewiggled',DM_type=DM_type,bloating=False,two_halo_suppression=False,transition_smoothing=False,concentration_boost=False)['total']
            print(f"Computed Pmm using halo model with subhalos and without tweaks")
    
    def get_Pmm(self):
        """
        Matter-matter power spectrum
        """
        return self.Pmm
    
    def get_Pgm(self, galaxy_bias):
        """
        Galaxy-matter power spectrum
        """
        return self.Pmm * galaxy_bias

    def get_Pgg(self, galaxy_bias):
        """
        Galaxy-galaxy power specttrum
        """
        return self.Pmm * galaxy_bias**2
    
    def get_PgI(
        self,
        NLA=False,
        C1=5e-14,
        IA_pars=np.array([0.7, -1.36, -1.7, -2.5, 1.0, 0.62])
        ):

        k = self.k     
        kh = self.k * self.h                 
        z_values = self.z_ps

        rho_crit = self.rhocrit0  # Msun Mpc-3
        C1 *= self.h**-2 #h−2 Msun−1 Mpc3 -> Msun-1 Mpc3
        Om0 = self.OmegaM

        Pmm = self.Pmm

        # ---------- NLA MODEL ----------
        if NLA:
            GI = np.zeros_like(Pmm)
            for i, z in enumerate(z_values):
                A1 = (
                    -IA_pars[0] * C1 * rho_crit * Om0
                    / self.growthD(z)
                    * ((1 + z) / (1 + IA_pars[5]))**IA_pars[2]
                )
                GI[i] = A1 * Pmm[i]

            return GI

        fastpt = FASTPT(kh, low_extrap=-6, high_extrap=3, n_pad=int(0.5*self.nk))

        GI = np.zeros((len(z_values), len(k)))

        for i, z in enumerate(z_values):

            # Power spectrum as array in (Mpc/h)^3 units, on kh grid
            P_mm_h = Pmm[i, :] * self.h**3

            A1 = (
                -IA_pars[0] * C1 * rho_crit * Om0
                / self.growthD(z)
                * ((1 + z) / (1 + IA_pars[5]))**IA_pars[2]
            )
            A2 = (
                5 * IA_pars[1] * C1 * rho_crit * Om0
                / self.growthD(z)**2
                * ((1 + z) / (1 + IA_pars[5]))**IA_pars[3]
            )

            P_deltaE1, P_deltaE2, *_ = fastpt.IA_ta(P_mm_h, C_window=.75)
            P_A, P_B, *_ = fastpt.IA_mix(P_mm_h, C_window=.75)

            GI[i] = (
                A1 * Pmm[i]
                + IA_pars[4] * A1 * (P_deltaE1 + P_deltaE2) / self.h**3
                + A2 * (P_A + P_B) / self.h**3
            )

        return GI

    def make_P_interp(self, P,log=False):
        if log:
            return RegularGridInterpolator((self.z_ps, self.k),np.log(P),bounds_error=False,fill_value=None)
        else:
            return RegularGridInterpolator((self.z_ps, self.k),P,bounds_error=False,fill_value=None)

class angular_power_spectrum(power_spectrum,lensing):

    def __init__(self, zmin=0.01, zmax=2.0, nz=128, kmin=1e-4, kmax=1e3,nk=128,model = 'halomodel',file_dir=None,DM_type='CDM',tweaks=True,**tweak_args):
        super().__init__(zmin=zmin, zmax=zmax, nz=nz, kmin=kmin, kmax=kmax, nk=nk, model=model,file_dir=file_dir,DM_type=DM_type,tweaks=tweaks,**tweak_args)
        if self.zmin == 0:
            self.zmin=zmin
            self.zmax=zmax
            self.nz=nz
            self.z = np.linspace(zmin, zmax, nz)

        # chi-grid [Mpc]
        chi_min = self.comoving_distance(zmin)
        chi_max = self.comoving_distance(zmax)
        self.chi = np.linspace(chi_min, chi_max, nz)

        # chi <-> z mapping
        self.chi_to_z = self.chi_to_z_interp(zmin=zmin, zmax=zmax)
        self.z = self.chi_to_z(self.chi)

    def cls_gI(self,l_bins, zl, zs, nz_lens, nz_source, zl_mean, nz_lens_stretch = 1,  nz_lens_shift = 0, NLA=True,  C1=5e-14,IA_pars = np.array([0.7,-1.36,-1.7,-2.5,1.0,0.62])):
        chi = self.chi
        z = self.z
        PgI = self.get_PgI(NLA=NLA,C1=C1,IA_pars = IA_pars)
        P_interp = self.make_P_interp(PgI)

        nchi_lens_interp = self.nz_to_nchi_interp(nz_lens,zl,z_mean=zl_mean,shift = nz_lens_shift, stretch = nz_lens_stretch)
        nchi_source_interp = self.nz_to_nchi_interp(nz_source,zs)

        cls = []
        for l in l_bins:
            k = (l + 0.5) / chi    
            pts = np.column_stack([z, k])
            PgI_chi = P_interp(pts)
            integrand = nchi_lens_interp(chi) * nchi_source_interp(chi) * PgI_chi / chi**2
            cls.append(simpson(integrand, chi))

        return np.array(cls)

    def cls_muI(self,l_bins,zl, zs, nz_lens, nz_source, zl_mean, magnification_bias, nz_lens_stretch = 1,  nz_lens_shift = 0, NLA=True, C1=5e-14,IA_pars = np.array([0.7,-1.36,-1.7,-2.5,1.0,0.62])):
        chi = self.chi
        z = self.z
        PgI = self.get_PgI(NLA=NLA,C1=C1,IA_pars = IA_pars)
        P_interp = self.make_P_interp(PgI)

        nchi_lens_interp = self.nz_to_nchi_interp(nz_lens,zl,z_mean=zl_mean,shift = nz_lens_shift, stretch = nz_lens_stretch)
        nchi_source_interp = self.nz_to_nchi_interp(nz_source,zs)

        q_lens = self.lensing_efficiency(nchi_interp = nchi_lens_interp,shear=0)

        cls = []
        for l in l_bins:
            k = (l + 0.5) / chi 
            pts = np.column_stack([z, k])
            PgI_chi = P_interp(pts)
            integrand = magnification_bias * q_lens  * nchi_lens_interp(chi) * nchi_source_interp(chi) * PgI_chi / chi**2
            cls.append( simpson(integrand, chi))

        return np.array(cls)

    def cls_muk(self,l_bins, galaxy_bias, zl, zs, nz_lens, nz_source, zl_mean,magnification_bias, nz_lens_stretch = 1,  nz_lens_shift = 0, shear=0):
        chi = self.chi
        z = self.z
        Pgm = self.get_Pgm(galaxy_bias=galaxy_bias) 
        logP_interp = self.make_P_interp(Pgm,log=True)

        nchi_lens_interp = self.nz_to_nchi_interp(nz_lens,zl,zl_mean,shift = nz_lens_shift, stretch = nz_lens_stretch)
        nchi_source_interp = self.nz_to_nchi_interp(nz_source,zs)

        q_source = self.lensing_efficiency(nchi_interp = nchi_source_interp,shear=shear)
        q_lens = self.lensing_efficiency(nchi_interp = nchi_lens_interp,shear=0)

        cls = []
        for l in l_bins:
            k = (l + 0.5) / chi
            pts = np.column_stack([z, k])
            Pgm_chi = np.exp(logP_interp(pts)) 
            integrand = (magnification_bias * q_lens * nchi_lens_interp(chi) * q_source * Pgm_chi / chi**2)
            cls.append(simpson(integrand, chi))

        return np.array(cls)
 
    def cls_gk(self, l_bins, galaxy_bias, zl, zs, nz_lens, nz_source, zl_mean, nz_lens_stretch = 1,  nz_lens_shift = 0, shear=0):
        chi = self.chi
        z = self.z
        Pgm = self.get_Pgm(galaxy_bias=galaxy_bias) 
        logP_interp = self.make_P_interp(Pgm,log=True)

        nchi_lens_interp = self.nz_to_nchi_interp(nz_lens,zl,zl_mean,shift = nz_lens_shift, stretch = nz_lens_stretch)
        nchi_source_interp = self.nz_to_nchi_interp(nz_source,zs)

        q_source = self.lensing_efficiency(nchi_interp = nchi_source_interp,shear=shear)

        cls = []
        for l in l_bins:
            k = (l + 0.5) / chi
            pts = np.column_stack([z, k])
            Pgm_chi = np.exp(logP_interp(pts)) 
            integrand = q_source * nchi_lens_interp(chi) * Pgm_chi / chi**2
            cls.append( simpson(integrand, chi))

        return np.array(cls)

    def cls_gg(self, l_bins, galaxy_bias, zl, nz_lens,  zl_mean, nz_lens_stretch = 1,  nz_lens_shift = 0):
        chi = self.chi
        z = self.z
        Pgg = self.get_Pgg(galaxy_bias = galaxy_bias)
        logP_interp = self.make_P_interp(Pgg,log=True)

        nchi_lens_interp = self.nz_to_nchi_interp(nz_lens,zl,zl_mean,shift = nz_lens_shift, stretch = nz_lens_stretch)

        cls = []
        for l in l_bins:
            k = (l + 0.5) / chi
            pts = np.column_stack([z, k])
            Pgg_chi = np.exp(logP_interp(pts))  
            integrand = nchi_lens_interp(chi)**2 * Pgg_chi / chi**2
            cls.append( simpson(integrand, chi))

        return np.array(cls)
    
    def cls_kk(self, l_bins, zs, nz_source,  shear=0):
        chi = self.chi
        z = self.z
        Pmm = self.get_Pmm()
        logP_interp = self.make_P_interp(Pmm,log=True)

        nchi_source_interp = self.nz_to_nchi_interp(nz_source,zs)

        q_source = self.lensing_efficiency(nchi_interp = nchi_source_interp,shear=shear)

        cls = []
        for l in l_bins:
            k = (l + 0.5) / chi 
            pts = np.column_stack([z, k])
            Pmm_chi = np.exp(logP_interp(pts))
            integrand = q_source **2 * Pmm_chi / chi**2
            cls.append( simpson(integrand, chi))

        return np.array(cls)

    def cls_for_MCMC(self, theta,l_bins, galaxy_bias, zl, zs, nz_lens, nz_source, zl_mean, magnification_bias, nz_lens_stretch,  nz_lens_shift,shear, NLA=True, C1=5e-14,IA_pars = np.array([0.7,-1.36,-1.7,-2.5,1.0,0.62])):
        self.Pmm = self.get_P_halomodel_sashimi(subhalos=True,bloating=True,two_halo_suppression=True,transition_smoothing=True,concentration_boost=False,eta_amp=theta[0],eta_exp=theta[1],alpha_amp=theta[2],alpha_exp=theta[3])['total']

        cls_gI = self.cls_gI(l_bins,zl,zs,nz_lens,nz_source,zl_mean,nz_lens_stretch,nz_lens_shift,NLA,C1,IA_pars)
        cls_muI = self.cls_muI(l_bins,zl,zs,nz_lens,nz_source,zl_mean,magnification_bias,nz_lens_stretch,nz_lens_shift,NLA,C1,IA_pars)
        cls_muk = self.cls_muk(l_bins, galaxy_bias, zl, zs, nz_lens, nz_source, zl_mean,magnification_bias,nz_lens_stretch,nz_lens_shift,shear)
        cls_gk = self.cls_gk(l_bins, galaxy_bias, zl, zs, nz_lens, nz_source, zl_mean,nz_lens_stretch,nz_lens_shift,shear)

        return cls_gI + cls_muI + cls_muk + cls_gk