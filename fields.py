import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.special import jv, hankel1 as hv,jvp, h1vp as hvp, jn_zeros
from scipy.constants import hbar,c as c0, epsilon_0 as ϵ0, mu_0 as μ0
from tqdm.notebook import tqdm
import tikzplotlib as tpl
import itertools 
from scipy.ndimage import gaussian_filter1d

# make plots be visually uniform with LaTeX
plt.rcParams.update(
    {
    "pgf.texsystem":   "pdflatex", # or any other engine you want to use
    "text.usetex":     True,       # use TeX for all texts
    "font.family":     "serif",
    "font.serif":      [],         # empty entries should cause the usage of the document fonts
    "font.sans-serif": [],
    "font.monospace":  [],
    "font.size":       10,         # control font sizes of different elements
    "axes.labelsize":  10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "text.latex.preamble": r"\usepackage{physics}\renewcommand{\vec}{\vb*}\usepackage{siunitx}\renewcommand{\Re}{\real}\renewcommand{\Im}{\imaginary}"
}
)
# fix issue with dash sequences in legends https://github.com/nschloe/tikzplotlib/issues/567
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
Line2D._us_dashSeq    = property(lambda self: self._dash_pattern[1])
Line2D._us_dashOffset = property(lambda self: self._dash_pattern[0])
Legend._ncol = property(lambda self: self._ncols)

def material(ε_in,ε_out,μ_in=1,μ_out=1):
    '''Sets the refractive index of the material inside and outside the cylinder'''
    global n1; n1=np.sqrt(ε_in*μ_in)    # Refractive index of the material inside
    global n2; n2=np.sqrt(ε_out*μ_out)  # Refractive index of the material outside
    global ε1; ε1=ε_in  # Permittivity inside 
    global ε2; ε2=ε_out # Permittivity outside
    global μ1; μ1=μ_in  # Permeability inside
    global μ2; μ2=μ_out # Permeability outside
    global NA; NA=np.sqrt(n1**2-n2**2)  # Numerical aperture sin(θ_A)=NA where θ_A is the acceptance angle

def find_nearest(array, value, ix=False):     # Finds the nearest value in an array
    array = np.asarray(array)       # https://stackoverflow.com/a/2566508
    idx = (np.abs(array - value)).argmin()  
    if ix:  return idx
    else:   return array[idx]   

def vector_field_at(X,Y,Z,F,x=None,y=None,z=0,ρ=None,φ=None):   # Returns the vector field F at the points x,y,z
    if ρ is not None and φ is not None: x,y=ρ*np.cos(φ),ρ*np.sin(φ)
    # returns Fx,Fy,Fz at X=x Y=y Z=z
    ix=find_nearest(X[:,0,0],x,ix=True)
    iy=find_nearest(Y[0,:,0],y,ix=True)
    iz=find_nearest(Z[0,0,:],z,ix=True)
    return F[0][ix,iy,iz],F[1][ix,iy,iz],F[2][ix,iy,iz]

def v_norm(F):                # Returns the norm of a vector field
    return np.max(np.real(np.sqrt(np.conj(F[0])*(F[0])+np.conj(F[1])*(F[1])+np.conj(F[2])*(F[2]))))
def tan_norm(F):            # Returns the norm of a vector field in the transverse plane
    return np.max(np.real(np.sqrt(np.conj(F[0])*(F[0])+np.conj(F[1])*(F[1]))))
def z_norm(F):              # Returns the norm of a vector field in the longitudinal direction
    return np.max(np.real(np.sqrt(np.conj(F[2])*(F[2]))))

def spin2cart(F0,Fp,Fm):
    #returns Fx,Fy,Fz
    return (Fp+Fm)/np.sqrt(2),1j*(Fp-Fm)/np.sqrt(2),F0
def cart2spin(Fx,Fy,Fz):
    #returns F0,Fp,Fm
    return Fz,(Fx-1j*Fy)/np.sqrt(2),(Fx+1j*Fy)/np.sqrt(2)
def cart2cylin(Fx,Fy,Fz,φ):
    #returns Fρ,Fφ,Fz
    return Fx*np.cos(φ)+Fy*np.sin(φ),-Fx*np.sin(φ)+Fy*np.cos(φ),Fz
def cylin2cart(Fρ,Fφ,Fz,φ):
    #returns Fx,Fy,Fz
    return Fρ*np.cos(φ)-Fφ*np.sin(φ),Fρ*np.sin(φ)+Fφ*np.cos(φ),Fz
def spin2cylin(F0,Fp,Fm,φ):
    #returns Fρ,Fφ,Fz
    return cart2cylin(*spin2cart(Fp,Fm,F0),φ)
def cylin2spin(Fρ,Fφ,Fz,φ):
    #returns Fp,Fm,F0
    return cart2spin(*cylin2cart(Fρ,Fφ,Fz,φ))

def water_dispersion(λ0): # relative permittivity of distilled water at 20°C valid in λ0∈(500, 1750)nm [https://refractiveindex.info/?shelf=main&book=H2O&page=Kedenburg]
    ε=(1+0.75831/(1-0.01007/(λ0*1e6)**2)+0.08495/(1-8.91377/(λ0*1e6)**2))
    return ε
def SiN_dispersion(λ0=None): # relative permittivity of SiN at 20°C valid in λ0∈(250, 1700)nm [https://refractiveindex.info/?shelf=main&book=Si3N4&page=Vogt-2.13]
    λ = np.array([0.25, 0.26, 0.27, 0.28, 0.29, 0.3 , 0.31, 0.32, 0.33, 0.34, 0.35,
       0.36, 0.37, 0.38, 0.39, 0.4 , 0.41, 0.42, 0.43, 0.44, 0.45, 0.46,
       0.47, 0.48, 0.49, 0.5 , 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57,
       0.58, 0.59, 0.6 , 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68,
       0.69, 0.7 , 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79,
       0.8 , 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9 ,
       0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.  , 1.01,
       1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1 , 1.11, 1.12,
       1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.2 , 1.21, 1.22, 1.23,
       1.24, 1.25, 1.26, 1.27, 1.28, 1.29, 1.3 , 1.31, 1.32, 1.33, 1.34,
       1.35, 1.36, 1.37, 1.38, 1.39, 1.4 , 1.41, 1.42, 1.43, 1.44, 1.45,
       1.46, 1.47, 1.48, 1.49, 1.5 , 1.51, 1.52, 1.53, 1.54, 1.55, 1.56,
       1.57, 1.58, 1.59, 1.6 , 1.61, 1.62, 1.63, 1.64, 1.65, 1.66, 1.67,
       1.68, 1.69, 1.7 ])
    n = np.array([2.515, 2.51 , 2.497, 2.479, 2.458, 2.437, 2.416, 2.395, 2.376,
       2.358, 2.34 , 2.324, 2.31 , 2.296, 2.283, 2.271, 2.26 , 2.249,
       2.239, 2.23 , 2.222, 2.214, 2.206, 2.199, 2.193, 2.186, 2.181,
       2.175, 2.17 , 2.165, 2.16 , 2.156, 2.152, 2.148, 2.145, 2.141,
       2.138, 2.136, 2.133, 2.131, 2.128, 2.126, 2.124, 2.122, 2.12 ,
       2.118, 2.117, 2.115, 2.114, 2.112, 2.111, 2.109, 2.108, 2.107,
       2.106, 2.105, 2.104, 2.103, 2.102, 2.101, 2.1  , 2.099, 2.098,
       2.097, 2.097, 2.096, 2.095, 2.094, 2.094, 2.093, 2.092, 2.092,
       2.091, 2.091, 2.09 , 2.09 , 2.089, 2.089, 2.088, 2.088, 2.087,
       2.087, 2.086, 2.086, 2.085, 2.085, 2.085, 2.084, 2.084, 2.084,
       2.083, 2.083, 2.083, 2.082, 2.082, 2.082, 2.081, 2.081, 2.081,
       2.08 , 2.08 , 2.08 , 2.08 , 2.079, 2.079, 2.079, 2.079, 2.079,
       2.078, 2.078, 2.078, 2.078, 2.078, 2.077, 2.077, 2.077, 2.077,
       2.077, 2.076, 2.076, 2.076, 2.076, 2.076, 2.076, 2.075, 2.075,
       2.075, 2.075, 2.075, 2.075, 2.075, 2.074, 2.074, 2.074, 2.074,
       2.074, 2.074, 2.074, 2.074, 2.073, 2.073, 2.073, 2.073, 2.073,
       2.073, 2.073])
    k = np.array([3.50e-01, 3.05e-01, 2.62e-01, 2.23e-01, 1.90e-01, 1.62e-01,
       1.38e-01, 1.18e-01, 1.01e-01, 8.71e-02, 7.49e-02, 6.44e-02,
       5.54e-02, 4.76e-02, 4.09e-02, 3.50e-02, 2.99e-02, 2.55e-02,
       2.16e-02, 1.82e-02, 1.52e-02, 1.27e-02, 1.04e-02, 8.45e-03,
       6.76e-03, 5.31e-03, 4.08e-03, 3.04e-03, 2.18e-03, 1.48e-03,
       9.34e-04, 5.22e-04, 2.35e-04, 6.43e-05, 1.06e-06, 0.00e+00,
       0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
       0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
       0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
       0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
       0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
       0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
       0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
       0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
       0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
       0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
       0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
       0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
       0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
       0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
       0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
       0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
       0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
       0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
       0.00e+00, 0.00e+00])
    ε=(n+1j*k)**2
    if λ0 is None: return λ*1e-6,ε
    else:   
        i = find_nearest(λ,λ0*1e6,ix=True)
        return ε[i]

def boundary_conditions(r0_per_λ0,neff,ℓ=1):
    ''''Returns a matrix representing the boundary conditions for Ez, Eφ, Hz and Hφ 
    to be continuous at the interface'''
    k0_r0=2*np.pi*r0_per_λ0 #k0·r0
    kz_r0=neff*k0_r0        #kz·r0
    k1_r0=n1*k0_r0          #k1·r0
    k2_r0=n2*k0_r0          #k2·r0
    κ1_r0=np.emath.sqrt(n1**2-neff**2)*k0_r0    #κ1·r0
    κ2_r0=np.emath.sqrt(n2**2-neff**2)*k0_r0    #κ2·r0
    with np.errstate(divide='ignore', invalid='ignore'): 
        J=jv(ℓ,κ1_r0) 
        Jp=jvp(ℓ,κ1_r0)
        H=hv(ℓ,κ2_r0)
        Hp=hvp(ℓ,κ2_r0)
        a11=np.sqrt(ε2)*J
        a12=0j*r0_per_λ0
        a13=-np.sqrt(ε1)*H
        a14=0j*r0_per_λ0
        a21=np.sqrt(ε2)*(ℓ*kz_r0/((κ1_r0)**2))*J
        a22=1j*np.sqrt(ε2)*(k1_r0/κ1_r0)*Jp
        a23=-np.sqrt(ε1)*(ℓ*kz_r0/((κ2_r0)**2))*H
        a24=-1j*np.sqrt(ε1)*(k2_r0/κ2_r0)*Hp
        a31=0j*r0_per_λ0
        a32=np.sqrt(μ2)*J
        a33=0j*r0_per_λ0
        a34=-np.sqrt(μ1)*H
        a41=-1j*(k1_r0/κ1_r0)*Jp
        a42=np.sqrt(μ2)*(ℓ*kz_r0/((κ1_r0)**2))*J
        a43=1j*np.sqrt(μ1)*(k2_r0/κ2_r0)*Hp
        a44=-np.sqrt(μ1)*(ℓ*kz_r0/((κ2_r0)**2))*H
    output=np.array([[a11,a12,a13,a14],[a21,a22,a23,a24],[a31,a32,a33,a34],[a41,a42,a43,a44]],dtype=complex)
    return output

def logdetA(r0_per_λ0,neff,ℓ=1):
    '''Returns the log of the determinant of the matrix representing the boundary conditions 
    for Ez, Eφ, Hz and Hφ'''
    a=boundary_conditions(r0_per_λ0,neff,ℓ=ℓ)   
    output=np.log10(np.abs(
        a[0,3]*a[1,2]*a[2,1]*a[3,0] - a[0,2]*a[1,3]*a[2,1]*a[3,0] - a[0,3]*a[1,1]*a[2,2]*a[3,0] + 
        a[0,1]*a[1,3]*a[2,2]*a[3,0] + a[0,2]*a[1,1]*a[2,3]*a[3,0] - a[0,1]*a[1,2]*a[2,3]*a[3,0] - 
        a[0,3]*a[1,2]*a[2,0]*a[3,1] + a[0,2]*a[1,3]*a[2,0]*a[3,1] + a[0,3]*a[1,0]*a[2,2]*a[3,1] - 
        a[0,0]*a[1,3]*a[2,2]*a[3,1] - a[0,2]*a[1,0]*a[2,3]*a[3,1] + a[0,0]*a[1,2]*a[2,3]*a[3,1] + 
        a[0,3]*a[1,1]*a[2,0]*a[3,2] - a[0,1]*a[1,3]*a[2,0]*a[3,2] - a[0,3]*a[1,0]*a[2,1]*a[3,2] + 
        a[0,0]*a[1,3]*a[2,1]*a[3,2] + a[0,1]*a[1,0]*a[2,3]*a[3,2] - a[0,0]*a[1,1]*a[2,3]*a[3,2] - 
        a[0,2]*a[1,1]*a[2,0]*a[3,3] + a[0,1]*a[1,2]*a[2,0]*a[3,3] + a[0,2]*a[1,0]*a[2,1]*a[3,3] - 
        a[0,0]*a[1,2]*a[2,1]*a[3,3] - a[0,1]*a[1,0]*a[2,2]*a[3,3] + a[0,0]*a[1,1]*a[2,2]*a[3,3]))
    # return output
    return np.where((neff<n1) & (neff>n2),output,0)

def get_neff(rmin=0,rmax=2,N=500,neff_fig=False,r0_over_λ0=None,ℓ=1,mode=0):  
    '''Returns the effective refractive index as function r0/λ0 for a given ℓ mode and 
    eigenvector V of magnitudes of the fields'''  
    r0_per_λ0=np.linspace(rmin,rmax,N+1,endpoint=False)[1:] # r0/λ0
    neff=np.linspace(n2,n1,N+1,endpoint=False)[1:]          # n_eff
    Ro, Ne = np.meshgrid(r0_per_λ0, neff)                   # r0/λ0, n_eff meshgrid
    logdetM=logdetA(Ro,Ne,ℓ=ℓ)                              # log10|detM|

    # Finds n_eff for which log10|detM|=0
    n=np.zeros_like(r0_per_λ0)  
    for i in range(len(r0_per_λ0)):   
        # plt.plot(neff,-logdetM[:,i])
        # plt.show()  
        try: # Finds the first peak of -log10|detM| and sets n_eff to the corresponding value
            if μ1!=1 and μ2!=1: 
                if mode==0: n[i]=neff[np.max(signal.find_peaks(-logdetM[:,i], width=(None,45),prominence=.05)[0])]
                elif mode==1 and ℓ==1: n[i]=neff[np.max(signal.find_peaks(-logdetM[:,i], width=(None,45),prominence=.05)[0][:-mode])]
                else:       n[i]=neff[np.max(signal.find_peaks(-logdetM[:,i])[0][:-mode])]

                # else:       n[i]=neff[np.max(signal.find_peaks(-logdetM[:,i], width=(None,49),prominence=.009)[0][:-mode])]
            else:
                if mode==0: n[i]=neff[np.max(signal.find_peaks(-logdetM[:,i])[0])]
                else:       n[i]=neff[np.max(signal.find_peaks(-logdetM[:,i])[0][:-mode])]

        except ValueError: n[i]=np.nan #n_eff=0 if no peaks
    r0_per_λ0=r0_per_λ0[~np.isnan(n)]
    n=n[~np.isnan(n)]
    
    # Calculates amplitudes    
    M=np.transpose(boundary_conditions(r0_per_λ0,n,ℓ=ℓ),(2,0,1))
    (w,v)=np.linalg.eig(M)              # w,v are eigenvalues and eigenvectors of M
    argus=np.argmin(np.abs(w),axis=1)   # argus is the index of the eigenvalue closest to zero
    V=v[np.arange(len(n)),:,argus]           # V is the eigenvector corresponding to the eigenvalue closest to zero

    # Plots n_eff
    if neff_fig:
        plt.pcolormesh(2*np.pi*Ro,Ne,-logdetM,cmap='Greys')
        plt.colorbar(label=r'$-\log_{10}|\det M|$')
        plt.xlim(xmax=2*np.pi*rmax,xmin=2*np.pi*rmin) 
        plt.plot(2*np.pi*r0_per_λ0,n,label=f'$n_\\mathrm{{eff}}(k_0r_0)$')
        if r0_over_λ0 is None: 
            pass
        else:
            idx=int(np.argwhere(r0_per_λ0==find_nearest(r0_per_λ0,r0_over_λ0)))
            plt.scatter(2*np.pi*r0_per_λ0[idx],n[idx])
        plt.ylim(n2,n1)
        
        plt.grid(color='#bfbfbf',linestyle='-')#color='black', linestyle='--')
        plt.ylabel(r'$n_\mathrm{eff}=\lambda_0/\lambda_z$')
        plt.xlabel(r'$k_0r_0=2\pi r_0/\lambda_0$')
        plt.legend()
        plt.show()
    if r0_over_λ0 is None: # Returns all values
        return (r0_per_λ0,n,V) 
    else: # Returns only the value closest to r0/λ0
        idx=int(np.argwhere(r0_per_λ0==find_nearest(r0_per_λ0,r0_over_λ0))) 
        return (r0_per_λ0[idx],n[idx],V[idx])
    
# def find_best_r0(k0r0min=0,k0r0max=5,res=20,N=500):
#     ε1_per_ε2=np.linspace(1.1,3,res+1,endpoint=False)[1:]
#     ε1_minus_ε2=1
#     # ε1_per_ε2=2
#     # ε1_minus_ε2=np.linspace(0.1,5,res+1,endpoint=False)[1:]
#     R=np.zeros_like(ε1_per_ε2)
#     for i in tqdm(range(len(ε1_per_ε2))):
#         # ε1= ε1_per_ε2[i]*(ε1_minus_ε2)/(ε1_per_ε2[i]-1)
#         # ε2= (ε1_minus_ε2)/(ε1_per_ε2[i]-1)
#         ε2=1
#         ε1=ε1_per_ε2[i]*ε2
#         material(ε1,ε2)
#         (r0_per_λ0,_,_)=get_neff(rmin=k0r0min/(2*np.pi),rmax=k0r0max/(2*np.pi),N=N,ℓ=0,mode=0)
#         R[i]=2*np.pi*r0_per_λ0[0]*np.sqrt(ε1-ε2)
#     plt.plot(ε1_per_ε2,R)
#     plt.show()
#     return R
    # ε1_per_ε2=np.linspace(1.1,3,res+1,endpoint=False)[1:]
    # ε1_minus_ε2=np.linspace(0,5,res+1,endpoint=False)[1:]
    # (A,B)=np.meshgrid(ε1_per_ε2,ε1_minus_ε2)
    # R=np.zeros_like(A)
    # with tqdm(total=len(ε1_per_ε2)*len(ε1_minus_ε2)) as pbar:
    #     for i in tqdm(range(len(ε1_per_ε2))):
    #         for j in tqdm(range(len(ε1_minus_ε2)),leave=False):
    #             ε1= ε1_per_ε2[i]*(ε1_minus_ε2[j])/(ε1_per_ε2[i]-1)
    #             ε2= (ε1_minus_ε2[j])/(ε1_per_ε2[i]-1)
    #             material(ε1,ε2)
    #             (r0_per_λ0,_,_)=get_neff(rmin=k0r0min/(2*np.pi),rmax=k0r0max/(2*np.pi),N=N,ℓ=0,mode=0)
    #             R[i,j]=2*np.pi*r0_per_λ0[0]*NA
    #             pbar.update(1)
    # plt.pcolor(A,B,R)
    # plt.plot()
    # return A,B,R
    
def plt_modes(r0,λ0,k0r0min,k0r0max,res=500,pdfs=False,rescale=False,l_modes=3,m_modes=3):
    if rescale: 
        plt.xlim(xmax=k0r0max,xmin=k0r0min) 
        ax = plt.gca()
        plt.ylabel(r'Normalised propagation constant $b={(k_z^2-k^2_2)}/{(k_1^2-k^2_2)}$')
        plt.xlabel(r'Normalised frequency $V=r_0\sqrt{k_1^2-k^2_2}$')
        plt.axvline((2*np.pi*NA)*(r0/λ0),linestyle='dashed',color='black', linewidth=1)
        # plt.axvline(jn_zeros(0,1)[0],linestyle='dashed',color='black', linewidth=1)
        plt.ylim(0,1)
        for l in range(0,l_modes):
            linestyle = itertools.cycle(('-', '--', '-.', ':'))
            color = next(ax._get_lines.prop_cycler)['color']
            for m in range(0,m_modes):
                (r0_per_λ0,n,_)=get_neff(rmin=(k0r0min/NA)/(2*np.pi),rmax=(k0r0max/NA)/(2*np.pi),N=res,ℓ=l,mode=m)
                y=gaussian_filter1d((n[n>=0]**2-n2**2)/NA**2,1)
                x=2*np.pi*r0_per_λ0[n>=0]*NA
                plt.plot(x,y,label=f'$(\ell,n)=({l},{m})$',color=color,linestyle=next(linestyle))
    else: 
        plt.xlim(xmax=k0r0max,xmin=k0r0min) 
        ax = plt.gca()
        plt.ylabel(r'Effective refractive index $k_z/k_0$')
        plt.xlabel(r'Radius of the fibre in units of reduced wavelength $k_0r_0$')
        plt.axvline((2*np.pi)*r0/λ0,linestyle='dashed',color='black', linewidth=1)
        plt.ylim(n2,n1)
        for l in range(0,3):
            linestyle = itertools.cycle(('-', '--', '-.', ':'))
            color = next(ax._get_lines.prop_cycler)['color']
            for m in range(0,3):
                (r0_per_λ0,n,_)=get_neff(rmin=k0r0min/(2*np.pi),rmax=k0r0max/(2*np.pi),N=res,ℓ=l,mode=m)
                plt.plot(2*np.pi*r0_per_λ0[n>0],gaussian_filter1d(n[n>0],1),label=f'$(\ell,n)=({l},{m})$',color=color,linestyle=next(linestyle))
    plt.grid(color='#bfbfbf',linestyle='-')#rgb(191, 191, 191)    
    plt.legend()
    if pdfs:
        tpl.save("figures/dispersion.tikz", flavor="latex")
    else: 
        plt.show()
    return None
    
def A_ℓsZ_ℓs(ℓ,s,A1,A2,k1,k2,kz,r0,r,φ,z):
    ''' returns the spin-weighted plane harmonics function for a cylindrical coordinate system'''
    return np.where(r/r0<1,
                    A1*jv(ℓ-s,np.emath.sqrt(k1**2-kz**2)*r)*np.exp(1j*((ℓ-s)*φ+kz*z)),
                    A2*hv(ℓ-s,np.emath.sqrt(k2**2-kz**2)*r)*np.exp(1j*((ℓ-s)*φ+kz*z)))

def get_ε(x,y,r0):
    r=np.sqrt(x**2+y**2)        # radial coordinate
    ϵ1=ϵ0*n1**2; ϵ2=ϵ0*n2**2    # ϵ1,ϵ2
    return np.where(r/r0<1,ϵ1,ϵ2)
def get_μ(x,y,r0):
    r=np.sqrt(x**2+y**2)        # radial coordinate
    μ1=μ0*n1**2; μ2=μ0*n2**2    # μ1,μ2
    return np.where(r/r0<1,μ1,μ2)

def E_nℓs(x,y,z,λ0,r0,n,V,ℓ=1,s=0):
    ''' returns the n,ℓ,s mode of Enℓs in SI units'''
    r=np.sqrt(x**2+y**2)        # radial coordinate
    φ=np.arctan2(y,x)           # azimuthal coordinate
    k1=2*np.pi*n1/λ0            # k1
    k2=2*np.pi*n2/λ0            # k2
    kz=2*np.pi*n/λ0             # kz
    if s==0:
        A1=(np.sqrt(2))*V[0] # A1 s=0
        A2=(np.sqrt(2))*V[2] # A2 s=0
    elif s==-1:
        A1=(k1*V[1]+1j*kz*V[0])/(np.emath.sqrt(k1**2-kz**2))
        A2=(k2*V[3]+1j*kz*V[2])/(np.emath.sqrt(k2**2-kz**2))
    elif s==1:
        A1=(k1*V[1]-1j*kz*V[0])/(np.emath.sqrt(k1**2-kz**2))
        A2=(k2*V[3]-1j*kz*V[2])/(np.emath.sqrt(k2**2-kz**2))
    return A_ℓsZ_ℓs(ℓ,s,A1,A2,k1,k2,kz,r0,r,φ,z)/np.sqrt(get_ε(x,y,r0))

def H_nℓs(x,y,z,λ0,r0,n,V,ℓ=1,s=0):
    ''' returns the n,ℓ,s mode of Hnℓs and μ in SI units'''
    r=np.sqrt(x**2+y**2)        # radial coordinate
    φ=np.arctan2(y,x)           # azimuthal coordinate
    k1=2*np.pi*n1/λ0            # k1
    k2=2*np.pi*n2/λ0            # k2
    kz=2*np.pi*n/λ0             # kz
    if s==0:
        A1=(np.sqrt(2))*V[1] # A1 s=0
        A2=(np.sqrt(2))*V[3] # A2 s=0
    elif s==-1:
        A1=(-k1*V[0]+1j*kz*V[1])/(np.emath.sqrt(k1**2-kz**2))
        A2=(-k2*V[2]+1j*kz*V[3])/(np.emath.sqrt(k2**2-kz**2))
    elif s==1:
        A1=(-k1*V[0]-1j*kz*V[1])/(np.emath.sqrt(k1**2-kz**2))
        A2=(-k2*V[2]-1j*kz*V[3])/(np.emath.sqrt(k2**2-kz**2))
    return A_ℓsZ_ℓs(ℓ,s,A1,A2,k1,k2,kz,r0,r,φ,z)/np.sqrt(get_μ(x,y,r0))

def get_E_nℓ(x,y,z,λ0,r0,ℓ=1,mode=0,n=None,V=None):
    ''' returns the n,ℓ mode of Enℓ in SI units'''
    if n is None or V is None: (_,n,V)=get_neff(r0_over_λ0=r0/λ0,ℓ=ℓ,mode=mode)
    Ep=E_nℓs(x,y,z,λ0,r0,n,V,ℓ=ℓ,s=1)
    Em=E_nℓs(x,y,z,λ0,r0,n,V,ℓ=ℓ,s=-1)
    E0=E_nℓs(x,y,z,λ0,r0,n,V,ℓ=ℓ,s=0)
    return spin2cart(E0,Ep,Em)

def get_H_nℓ(x,y,z,λ0,r0,ℓ=1,mode=0,n=None,V=None):
    ''' returns the n,ℓ mode of Hnℓ and μ in SI units'''
    if n is None or V is None: (_,n,V)=get_neff(r0_over_λ0=r0/λ0,ℓ=ℓ,mode=mode)
    Hp=H_nℓs(x,y,z,λ0,r0,n,V,ℓ=ℓ,s=1)
    Hm=H_nℓs(x,y,z,λ0,r0,n,V,ℓ=ℓ,s=-1)
    H0=H_nℓs(x,y,z,λ0,r0,n,V,ℓ=ℓ,s=0)
    return spin2cart(H0,Hp,Hm)

def get_E(x,y,z,λ0,r0,nr,Vr,nl,Vl,R=1,L=0,ℓ=1,mode=0):
    pE=get_E_nℓ(x,y,z,λ0,r0,ℓ=ℓ,mode=mode,n=nr,V=Vr)
    mE=get_E_nℓ(x,y,z,λ0,r0,ℓ=-ℓ,mode=mode,n=nl,V=Vl)
    return ((R*pE[0]+L*mE[0]),(R*pE[1]+L*mE[1]),(R*pE[2]+L*mE[2]))
def get_H(x,y,z,λ0,r0,nr,Vr,nl,Vl,R=1,L=0,ℓ=1,mode=0):
    pH=get_H_nℓ(x,y,z,λ0,r0,ℓ=ℓ,mode=mode,n=nr,V=Vr)
    mH=get_H_nℓ(x,y,z,λ0,r0,ℓ=-ℓ,mode=mode,n=nl,V=Vl)
    return ((R*pH[0]+L*mH[0]),(R*pH[1]+L*mH[1]),(R*pH[2]+L*mH[2]))

def curl(A,dx,dy,dz):           
    '''returns the curl of A'''
    dAx=np.gradient(A[0],dx,dy,dz) 
    dAy=np.gradient(A[1],dx,dy,dz)
    dAz=np.gradient(A[2],dx,dy,dz)
    return np.array([dAy[2]-dAz[1],dAz[0]-dAx[2],dAx[1]-dAy[0]])

# Poynting vector
def get_Re_Π(E,H):
    '''returns the real Poynting vector in SI units'''
    return np.real(np.cross(E,np.conj(H),axis=0))/2
def get_Im_Π(E,H):
    '''returns the imaginary Poynting vector in SI units'''
    return np.imag(np.cross(E,np.conj(H),axis=0))/2
def calculate_power(E,H,dx,dy):
    '''returns the power in SI units'''
    Re_Π=np.real(np.cross(E,np.conj(H),axis=0))/2
    return np.sum(Re_Π[2]*dx*dy,axis=(0,1))

# Spin densities
def get_ωS0(ϵ,E,μ,H):
    ''' returns ω times total spin density in SI units'''
    return np.imag(ϵ*np.cross(np.conj(E),E,axis=0)+μ*np.cross(np.conj(H),H,axis=0))/4
def get_ωS1(ϵ,E,μ,H):
    ''' returns ω times dual-odd spin density in SI units'''
    return np.imag(ϵ*np.cross(np.conj(E),E,axis=0)-μ*np.cross(np.conj(H),H,axis=0))/4

# Energy densities
def get_W0(ϵ,E,μ,H):
    ''' returns total energy density in SI units'''
    W0=0
    for i in range(3):
        W0+=np.real(ϵ*np.conj(E[i])*E[i]+μ*np.conj(H[i])*H[i])/4
    return W0
def get_W2(ϵ,E,μ,H):
    ''' returns time-odd energy density in SI units'''
    W2=0
    for i in range(3):
        W2-=np.sqrt(ϵ*μ)*np.real(np.conj(E[i])*H[i])/2
    return W2
def get_W3(ϵ,E,μ,H):
    ''' returns parity-odd energy density in SI units'''
    W3=0
    for i in range(3):
        W3+=np.sqrt(ϵ*μ)*np.imag(np.conj(H[i])*E[i])/2
    return W3
def get_W1(ϵ,E,μ,H):
    ''' returns dual-odd energy density in SI units'''
    W1=0
    for i in range(3):
        W1+=np.real(ϵ*np.conj(E[i])*E[i]-μ*np.conj(H[i])*H[i])/4
    return W1

# Orbital/canonical linear momentum densities
def get_ωp0(ϵ,E,μ,H,dx,dy,dz):
    ''' returns linear momentum density in SI units'''
    p=0
    for i in range(3):
        p+=np.imag(ϵ*np.conj(E[i])*np.gradient(E[i],dx,dy,dz)+μ*np.conj(H[i])*np.gradient(H[i],dx,dy,dz))/4
    return p
def get_ωp2(ϵ,E,μ,H,dx,dy,dz):
    ''' returns time-even linear momentum density in SI units'''
    p=0
    for i in range(3):
        p-=np.sqrt(ϵ*μ)*np.imag(np.conj(E[i])*np.gradient(H[i],dx,dy,dz)-H[i]*np.gradient(np.conj(E[i]),dx,dy,dz))/4
    return p
def get_ωp3(ϵ,E,μ,H,dx,dy,dz):
    ''' returns parity-even linear momentum density in SI units'''
    p=0
    for i in range(3):
        p+=np.sqrt(ϵ*μ)*np.real(np.conj(E[i])*np.gradient(H[i],dx,dy,dz)-H[i]*np.gradient(np.conj(E[i]),dx,dy,dz))/4
    return p
def get_ωp1(ϵ,E,μ,H,dx,dy,dz):
    ''' returns dual-odd linear momentum density in SI units'''
    p=0
    for i in range(3):
        p+=np.imag(ϵ*np.conj(E[i])*np.gradient(E[i],dx,dy,dz)-μ*np.conj(H[i])*np.gradient(H[i],dx,dy,dz))/4
    return p

def getW(X,Y,λ0,r0,ℓ=1,s=None,mode=0,n=None,V=None):
    ''' returns the n,ℓ mode of Hnℓ and μ in SI units'''
    if n is None or V is None: (_,n,V)=get_neff(r0_over_λ0=r0/λ0,ℓ=ℓ,mode=mode)
    ''' returns energies in SI units'''
    if s is None:
        We=0;Wm=0;Wc=0+0j
        for s in [-1,0,1]:
            E=E_nℓs(X,Y,0,λ0,r0,n,V,ℓ=ℓ,s=s)
            H=H_nℓs(X,Y,0,λ0,r0,n,V,ℓ=ℓ,s=s)
            sqrt_ε_E=E*np.sqrt(get_ε(X,Y,r0))
            sqrt_μ_H=H*np.sqrt(get_μ(X,Y,r0))
            We+=np.conj(sqrt_ε_E)*sqrt_ε_E/4
            Wm+=np.conj(sqrt_μ_H)*sqrt_μ_H/4
            Wc+=1j*np.conj(sqrt_ε_E)*sqrt_μ_H/2
    else:
        E=E_nℓs(X,Y,0,λ0,r0,n,V,ℓ=ℓ,s=s)
        H=H_nℓs(X,Y,0,λ0,r0,n,V,ℓ=ℓ,s=s)
        sqrt_ε_E=E*np.sqrt(get_ε(X,Y,r0))
        sqrt_μ_H=H*np.sqrt(get_μ(X,Y,r0))
        We=np.conj(sqrt_ε_E)*sqrt_ε_E/4
        Wm=np.conj(sqrt_μ_H)*sqrt_μ_H/4
        Wc=1j*np.conj(sqrt_ε_E)*sqrt_μ_H/2
    return np.real(We+Wm), np.imag(Wc), -np.real(Wc), np.real(We-Wm)

def getk(X,Y,λ0,r0,s,ℓ=1,mode=0,n=None):
    ''' returns wavevetor in cartesian coordinates for an s mode'''
    if n is None: (_,n,_)=get_neff(r0_over_λ0=r0/λ0,ℓ=ℓ,mode=mode)
    r=np.sqrt(X**2+Y**2)        # radial coordinate
    φ=np.arctan2(Y,X)           # azimuthal coordinate
    k1=2*np.pi*n1/λ0            # k1
    k2=2*np.pi*n2/λ0            # k2
    kz=2*np.pi*n/λ0
    κ1=np.emath.sqrt(k1**2-kz**2)
    κ2=np.emath.sqrt(k2**2-kz**2)
    Zℓs=np.where(r/r0<1,
                jv(ℓ-s,np.emath.sqrt(k1**2-kz**2)*r),
                hv(ℓ-s,np.emath.sqrt(k2**2-kz**2)*r))
    Zℓsp=np.where(r/r0<1,
                (jv(ℓ-s-1,np.emath.sqrt(k1**2-kz**2)*r)-jv(ℓ-s+1,np.emath.sqrt(k1**2-kz**2)*r))/2,
                (hv(ℓ-s-1,np.emath.sqrt(k2**2-kz**2)*r)-hv(ℓ-s+1,np.emath.sqrt(k2**2-kz**2)*r))/2)
    kφ=(ℓ-s)/r
    kr=-1j*(np.where(r/r0<1,κ1,κ2))*(Zℓsp/Zℓs)
    # calculate kx and ky
    kx=kr*np.cos(φ)-kφ*np.sin(φ)
    ky=kr*np.sin(φ)+kφ*np.cos(φ)
    k=np.zeros((3,len(X),len(X)),dtype=complex)
    k[0,:,:]=np.where(r>0,kx,0)
    k[1,:,:]=np.where(r>0,ky,0)
    k[2,:,:]=kz
    return k

def pltvector(X,Y,zix,V,ax,r0,λ0,W=None,scale=None,zscale=None,step=1,title='',out=False,mask=None,noticks=False):
    '''Plot the field in the x-y plane at z=zix'''
    if mask is not None: V[0]=mask*V[0]; V[1]=mask*V[1]; 
    if W is None: 
        if zscale is None: zscale=np.max(np.abs(V[2][:,:,zix]))
        pcol=ax.pcolormesh(X[:,:,zix]/λ0,Y[:,:,zix]/λ0,V[2][:,:,zix],cmap='bwr',vmin=-zscale,vmax=zscale,rasterized=True) #longitudinal component
    else: 
        if zscale is None: zscale=np.max(np.abs(W[:,:,zix]))
        pcol=ax.pcolormesh(X[:,:,zix]/λ0,Y[:,:,zix]/λ0,W[:,:,zix],cmap='bwr',vmin=-zscale,vmax=zscale,rasterized=True)
    norm=np.real(np.sqrt(np.conj(V[0][:,:,zix])*(V[0][:,:,zix])+np.conj(V[1][:,:,zix])*(V[1][:,:,zix])))
    if scale is None: scale=10*np.max(norm)
    norm=np.where(norm>0,norm,0)
    '''Plot the field in the x-y plane at z=zix'''
    quiv=ax.quiver(X[::step,::step,zix]/λ0,Y[::step,::step,zix]/λ0,V[0][::step,::step,zix],V[1][::step,::step,zix],scale=scale) #transverse components
    circle=plt.Circle((0, 0), r0/λ0, color='black',fill=False,linestyle='--');ax.add_artist(circle);ax.set_aspect('equal')  #fibre boundary
    ax.set_title(title) 
    if noticks: ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    if out: return pcol,quiv

def get(r0,λ0,R=1,L=0,fld_plot=False,neff_fig=False,zix=1,z=0,zgrid=False,window=None,resolution=250,ℓ=1,mode=0,sgnkz=1):
    if window is None: m=1.5*r0/λ0
    else: m=window #size of plot
    N=resolution   #number of points in each direction
    step=int(N/8)  #step size for quiver plot

    (_,nr,Vr)=get_neff(rmin=0,rmax=2,N=500,r0_over_λ0=r0/λ0,neff_fig=neff_fig,ℓ=ℓ,mode=mode)
    (_,nl,Vl)=get_neff(rmin=0,rmax=2,N=500,r0_over_λ0=r0/λ0,neff_fig=False,ℓ=-ℓ,mode=mode)
    x=λ0*np.linspace(-m,m,int(N))   #x coordinates
    y=λ0*np.linspace(-m,m,int(N))   #y coordinates
    dx=np.diff(x)[0]
    if zgrid: z=λ0*np.linspace(-m/4,m/4,int(N/4))   #z coordinates
    else:     z=np.linspace(λ0*z-dx,λ0*z+dx,int(3))   #z coordinates
    X, Y, Z = np.meshgrid(x, y, z,indexing='ij')  #3D grid of coordinates

    # generate fields 
    ε=get_ε(X,Y,r0)   #permittivity
    μ=get_μ(X,Y,r0)   #permeability
    nr=sgnkz*nr
    nl=sgnkz*nl
    if sgnkz==1: 
        (Ex,Ey,Ez)=get_E(X,Y,Z,λ0,r0,nr,Vr,nl,Vl,R=R,L=L,ℓ=ℓ)   #electric field
        (Hx,Hy,Hz)=get_H(X,Y,Z,λ0,r0,nr,Vr,nl,Vl,R=R,L=L,ℓ=ℓ)   #magnetic field
    else:
        (Ex,Ey,Ez)=get_E(X,Y,Z,λ0,r0,nl,Vl,nr,Vr,R=L,L=R,ℓ=ℓ)   #electric field
        (Hx,Hy,Hz)=get_H(X,Y,Z,λ0,r0,nl,Vl,nr,Vr,R=L,L=R,ℓ=ℓ)   #magnetic field
    E=(Ex,Ey,Ez)        #electric field vector
    H=(Hx,Hy,Hz)        #magnetic field vector
    if fld_plot: plt(r0,λ0,E,H,X,Y,zix=zix,step=step)
    return (ε,E,μ,H,X,Y,Z)


def get_α(λ0,a,εp,μp,κp):
    '''returns the polarisabilites of the particle'''
    V = (4/3)*np.pi*a**3
    k=2*np.pi*np.sqrt(ε2*μ2)/λ0
    k3_per_6pi=k**3/(6*np.pi)
    de0=(εp+2*ε2)*(μp+2*μ2)-κp**2
    αe0= 3*V*((εp-ε2)*(μp+2*μ2)-κp**2)/de0
    αm0= 3*V*((μp-μ2)*(εp+2*ε2)-κp**2)/de0
    αc0= 9*V*κp/de0
    (αe,αm,αc)=radiation_correction(λ0,αe0,αm0,αc0)
    return (αe,αm,αc)

def radiation_correction(λ0,αe0,αm0,αc0):
    k=2*np.pi*np.sqrt(ε2*μ2)/λ0
    k3_per_6pi=k**3/(6*np.pi)
    de = 1-1j*(k3_per_6pi)*(αe0+αm0)+k3_per_6pi**2*(αc0**2-αe0*αm0)
    αe = (αe0-1j*(k3_per_6pi)*(αc0**2-αe0*αm0))/de
    αm = (αm0-1j*(k3_per_6pi)*(αc0**2-αe0*αm0))/de
    αc = (αc0)/de
    return (αe,αm,αc)

def get2D(r0,λ0,window=None,resolution=250):
    if window is None: m=1.5*r0/λ0
    else: m=window #size of plot
    N=resolution   #number of points in each direction

    x=λ0*np.linspace(-m,m,int(N))   #x coordinates
    y=λ0*np.linspace(-m,m,int(N))   #y coordinates
    X, Y= np.meshgrid(x, y, indexing='ij')  #2D grid of coordinates

    return (X,Y)

def plot_EH(r0,λ0,E,H,X,Y,zix=1,step=None,w=10,pdf=False,pdfname="figure",c_bar=True,arrows=8):
    step=int(len(X)/arrows)
    fig, ax = plt.subplots(ncols=4,figsize=(10,3),constrained_layout=True, sharex=True, sharey=True)  #create figure with plots of E/|E| and H/|H| 
    pltvector(X,Y,zix,np.real(E)/v_norm(E),ax[0],r0,λ0,zscale=1,scale=w,step=step,title=r'$\Re(\vec{E})/|\vec{E}|$')
    pltvector(X,Y,zix,np.imag(E)/v_norm(E),ax[1],r0,λ0,zscale=1,scale=w,step=step,title=r'$\Im(\vec{E})/|\vec{E}|$')
    pltvector(X,Y,zix,np.real(H)/v_norm(H),ax[2],r0,λ0,zscale=1,scale=w,step=step,title=r'$\Re(\vec{H})/|\vec{H}|$')
    (pcol,_)=pltvector(X,Y,zix,np.imag(H)/v_norm(H),ax[3],r0,λ0,zscale=1,scale=w,step=step,title=r'$\Im(\vec{H})/|\vec{H}|$',out=True)
    fig.supxlabel(r'$x/\lambda_0$');fig.supylabel(r'$y/\lambda_0$')#,x=0.08)
    if c_bar:
        fig.colorbar(pcol, ticks=[-1,0,1],aspect=11,shrink=.8,pad=0.08)
    if pdf:plt.savefig("figures/EH_"+pdfname+".pdf",format='pdf')
    else:plt.show(fig)

def observables(ϵ,E,μ,H,X,Y,Z,Power=1):
    dx=np.diff(X[:,0,0])[0]
    dy=np.diff(Y[0,:,0])[0]
    dz=np.diff(Z[0,0,:])[0]
    if Power is None: Power=calculate_power(E,H,dx,dy)
    
    #calculate energy densities
    W0=get_W0(ϵ,E,μ,H)   #total energy density
    W2=get_W2(ϵ,E,μ,H)   #imaginary chiral energy density
    W3=get_W3(ϵ,E,μ,H)   #real chiral energy density
    W1=get_W1(ϵ,E,μ,H)   #difference between electric and magnetic energy densities

    #gradients of energy densities
    grad_W0=np.asarray(np.gradient(W0,dx,dy,dz))/Power  #gradient of total energy density
    grad_W2=np.asarray(np.gradient(W2,dx,dy,dz))/Power  #gradient of imaginary chiral energy density
    grad_W3=np.asarray(np.gradient(W3,dx,dy,dz))/Power  #gradient of real chiral energy density
    grad_W1=np.asarray(np.gradient(W1,dx,dy,dz))/Power  #gradient of difference between electric and magnetic energy densities

    #calculate momentum densities
    ωp0=get_ωp0(ϵ,E,μ,H,dx,dy,dz)/Power   #total momentum density
    ωp2=get_ωp2(ϵ,E,μ,H,dx,dy,dz)/Power   #imaginary chiral density
    ωp3=get_ωp3(ϵ,E,μ,H,dx,dy,dz)/Power   #real chiral momentum density    
    ωp1=get_ωp1(ϵ,E,μ,H,dx,dy,dz)/Power   #difference between electric and magnetic momentum densities

    #calculate poynting vector and spin densities
    Re_Π=get_Re_Π(E,H)/Power  #real part of poynting vector
    Im_Π=get_Im_Π(E,H)/Power  #imaginary part of poynting vector
    ωS0=get_ωS0(ϵ,E,μ,H)/Power  #total spin density
    ωS1=get_ωS1(ϵ,E,μ,H)/Power  #difference between electric and magnetic spin densities
    return (W0,W1,W2,W3,grad_W0,grad_W1,grad_W2,grad_W3,ωp0,ωp1,ωp2,ωp3,ωS0,Im_Π,Re_Π,ωS1)

def plot_forces(r0,λ0,ϵ,μ,E,H,X,Y,Z,zix=1,w=4,arrows=None,basis='pauli',pdfs=False,pdfname="figure",inside=True,plot_fields=False,Power=1):
    c=1/np.sqrt(ϵ*μ)            # speed of light in the medium [m/s]
    ω=2*np.pi*c0/λ0             # frequency [Hz]
    k=ω/c                       # wave number [1/m]
    dx=np.diff(X[:,0,0])[0]
    dy=np.diff(Y[0,:,0])[0]
    dr=np.sqrt(dx**2+dy**2)     # radial step size 
    R=np.sqrt(X**2+Y**2)        # radial coordinate
    if inside: a=np.tile(np.reshape(1*(R>(r0+dr))+1*(R<(r0-dr)),(1,np.shape(X)[0],np.shape(Y)[1],np.shape(Z)[2])),(3,1,1,1));b=np.ones_like(a)
    else:      a=np.tile(np.reshape(1*(R>(r0+dr)),(1,np.shape(X)[0],np.shape(Y)[1],np.shape(Z)[2])),(3,1,1,1));b=a
    (W0,W1,W2,W3,grad_W0,grad_W1,grad_W2,grad_W3,ωp0,ωp1,ωp2,ωp3,ωS0,Im_Π,Re_Π,ωS1)=observables(ϵ,E,μ,H,X,Y,Z,Power=Power)
    if arrows is None:step=int(len(X)/8)
    else: step=int(len(X)/arrows)

    if basis=='pauli':
        scale=np.max((v_norm(a*grad_W0),v_norm(a*grad_W3),v_norm(a*grad_W2),v_norm(a*grad_W1),v_norm(a*2*ωp0),v_norm(a*2*ωp3),v_norm(a*2*ωp2),v_norm(a*2*ωp1),v_norm(a*k*Re_Π/c),v_norm(a*k*Im_Π/c),v_norm(a*k*ωS0),v_norm(a*k*ωS1)))
        tscale=scale#np.max((tan_norm(a*grad_W0),tan_norm(a*grad_W3),tan_norm(a*grad_W2),tan_norm(a*grad_W1),tan_norm(a*2*ωp0),tan_norm(a*2*ωp3),tan_norm(a*2*ωp2),tan_norm(a*2*ωp1),tan_norm(a*k*Re_Π/c),tan_norm(a*k*Im_Π/c),tan_norm(a*k*ωS0),tan_norm(a*k*ωS1)))
        zscale=scale#np.max((z_norm(b*grad_W0),z_norm(b*grad_W3),z_norm(b*grad_W2),z_norm(b*grad_W1),z_norm(b*2*ωp0),z_norm(b*2*ωp3),z_norm(b*2*ωp2),z_norm(b*2*ωp1),z_norm(b*k*Re_Π/c),z_norm(b*k*Im_Π/c),z_norm(b*k*ωS0),z_norm(b*k*ωS1)))
        # Wmax=np.max(np.abs((W0,W3,W2,W1))) #maximum value of energy densities
        # grad_titles=(r'$\grad W_0=\grad (W_\mathrm{e}+W_\mathrm{m})$',r'$\grad W_1=\grad \Im(W_c)$',r'$-\grad W_2=\grad \Re(W_c)$',r'$\grad W_3=\grad (W_\mathrm{e}-W_\mathrm{m})$')
        grad_titles=(r'$\grad W_0=\grad (W_\mathrm{e}+W_\mathrm{m})$',r'$\grad W_1=\grad (W_\mathrm{e}-W_\mathrm{m})$',r'$\grad W_2=-\grad \Im W_c $',r'$\grad W_3=\grad \Re W_c $')
        # pres_titles=(r'$2\omega\vec{p}_0=2\omega(\vec{p}_\mathrm{e}+\vec{p}_\mathrm{m})$',r'$2\omega\vec{p}_1=2\omega\Im(\vec{p}_\mathrm{c})$',r'$-2\omega\vec{p}_2=2\omega\Re(\vec{p}_\mathrm{c})$',r'$2\omega\vec{p}_3=2\omega(\vec{p}_\mathrm{e}-\vec{p}_\mathrm{m})$')
        pres_titles=(r'$2\omega\vec{p}_0=2\omega(\vec{p}_\mathrm{e}+\vec{p}_\mathrm{m})$',r'$2\omega\vec{p}_1=2\omega(\vec{p}_\mathrm{e}-\vec{p}_\mathrm{m})$',r'$2\omega\vec{p}_2=-2\omega\Im \vec{p}_\mathrm{c} $',r'$2\omega\vec{p}_3=2\omega\Re \vec{p}_\mathrm{c} $')
        # spin_titles=(r'$-k\omega\vec{S}_2=k\Re(\vec{\varPi})/c$',r'$k\omega\vec{S}_3=k\omega(\vec{S}_\mathrm{e}-\vec{S}_\mathrm{m})$',r'$k\omega\vec{S}_0=k\omega(\vec{S}_\mathrm{e}+\vec{S}_\mathrm{m})$',r'$k\omega\vec{S}_1=k\Im(\vec{\varPi})/c$')
        spin_titles=(r'$k\omega\vec{S}_3=k\Re\vec{\varPi}/c$',r'$k\omega\vec{S}_2=-k\Im\vec{\varPi}/c$',r'$k\omega\vec{S}_1=k\omega(\vec{S}_\mathrm{e}-\vec{S}_\mathrm{m})$',r'$k\omega\vec{S}_0=k\omega(\vec{S}_\mathrm{e}+\vec{S}_\mathrm{m})$')
        grad_forces=(grad_W0,grad_W1,grad_W2,grad_W3)
        # energy_dens=(b[0,:,:,:]*W0,b[0,:,:,:]*W1,b[0,:,:,:]*W2,b[0,:,:,:]*W3)
        pres_forces=(2*ωp0,2*ωp1,2*ωp2,2*ωp3)
        spin_forces=(k*Re_Π/c,k*Im_Π/c,k*ωS1,k*ωS0)
    elif basis=='e-m-c':
        We=(W0+W1)/2
        Wm=(W0-W1)/2
        grad_We=(grad_W0+grad_W1)/2
        grad_Wm=(grad_W0-grad_W1)/2
        ωpe=(ωp0+ωp1)/2
        ωpm=(ωp0-ωp1)/2
        ωSe=(ωS0+ωS1)/2
        ωSm=(ωS0-ωS1)/2
        scale=np.max((v_norm(a*grad_We),v_norm(a*grad_Wm),v_norm(a*grad_W2),v_norm(a*grad_W1),v_norm(a*2*ωpe),v_norm(a*2*ωpm),v_norm(a*2*ωp2),v_norm(a*2*ωp1),v_norm(a*k*Re_Π/c),v_norm(a*k*Im_Π/c),v_norm(a*k*ωSe),v_norm(a*k*ωSm)))
        tscale=scale#np.max((tan_norm(a*grad_We),tan_norm(a*grad_Wm),tan_norm(a*grad_W2),tan_norm(a*grad_W1),tan_norm(a*2*ωpe),tan_norm(a*2*ωpm),tan_norm(a*2*ωp2),tan_norm(a*2*ωp1),tan_norm(a*k*Re_Π/c),tan_norm(a*k*Im_Π/c),tan_norm(a*k*ωSe),tan_norm(a*k*ωSm)))
        zscale=scale#np.max((z_norm(b*grad_We),z_norm(b*grad_Wm),z_norm(b*grad_W2),z_norm(b*grad_W1),z_norm(b*2*ωpe),z_norm(b*2*ωpm),z_norm(b*2*ωp2),z_norm(b*2*ωp1),z_norm(b*k*Re_Π/c),z_norm(b*k*Im_Π/c),z_norm(b*k*ωSe),z_norm(b*k*ωSm)))
        # Wmax=np.max(np.abs((b[0,:,:,:]*We,b[0,:,:,:]*Wm,b[0,:,:,:]*W2,b[0,:,:,:]*W1))) #maximum value of energy densities
        grad_titles=(r'$\grad W_\mathrm{e}$',r'$\grad W_\mathrm{m}$',r'$\grad\Re(W_\mathrm{c})$',r'$\grad \Im(W_\mathrm{c})$')
        pres_titles=(r'$2\omega\vec{p}_\mathrm{e}$',r'$2\omega\vec{p}_\mathrm{m}$',r'$2\omega\Re(\vec{p}_\mathrm{c})$',r'$2\omega\Im(\vec{p}_\mathrm{c})$')
        spin_titles=(r'$k\omega\vec{S}_\mathrm{e}$',r'$k\omega\vec{S}_\mathrm{m}$',r'$k\Re(\vec{\varPi})/c$',r'$k\Im(\vec{\varPi})/c$')
        grad_forces=(grad_We,grad_Wm,grad_W3,-grad_W2)
        # energy_dens=(We,Wm,W2,W1)
        pres_forces=(2*ωpe,2*ωpm,2*ωp3,-2*ωp2)
        spin_forces=(k*ωSe,k*ωSm,k*Re_Π/c,k*Im_Π/c)
    if plot_fields: 
        # plot_EH(r0,λ0,E,H,X,Y,zix=zix,step=step,w=w)
        field_titles=(r'$\Re\vec{E}/|\vec{E}|$',r'$\Im\vec{E}/|\vec{E}|$',r'$\Re\vec{H}/|\vec{H}|$',r'$\Im\vec{H}/|\vec{H}|$')
        field_forces=(np.real(E)/v_norm(E),np.imag(E)/v_norm(E),np.real(H)/v_norm(H),np.imag(H)/v_norm(H))
        plot_titles=(r'Fields',r'Gradient forces',r'Pressure forces',r'Recoil forces')
        forces=(field_forces,grad_forces,pres_forces,spin_forces)
        titles=(field_titles,grad_titles,pres_titles,spin_titles)
        zscales=(1,zscale,zscale,zscale)
        tscales=(10/w,tscale,tscale,tscale)
        fig, ax = plt.subplots(ncols=len(grad_titles),nrows=4,figsize=(10,10),constrained_layout=True, sharex=True, sharey=True)
    else: 
        forces=(grad_forces,pres_forces,spin_forces)
        titles=(grad_titles,pres_titles,spin_titles)
        fig, ax = plt.subplots(ncols=len(grad_titles),nrows=3,figsize=(10,7.9),constrained_layout=True, sharex=True, sharey=True)
        zscales=(zscale,zscale,zscale)
        tscales=(tscale,tscale,tscale)
    
    for i in range(len(titles)):
        for j in range(len(grad_titles)):
            (pcol,quiv)=pltvector(X,Y,zix,b[0,:,:,:]*forces[i][j],ax[i,j],r0,λ0,scale=w*tscales[i],zscale=zscales[i],step=step,title=titles[i][j],out=True,mask=a[0,:,:,:],noticks=True)
            if i==0 and j==0: hdl=pcol
        if plot_fields: 
            ax[i,0].set_ylabel(plot_titles[i])
            ax[i,0].yaxis.get_label().set_fontsize(11)
            ax[i,0].yaxis.set_label_coords(-0.05,0.5)
            # ax[i,len(grad_titles)-1].yaxis.set_label_position("right")
            # else: ax[i,0].set_ylabel(plot_titles[i]+'\n'+r'$\times 10^{-1}$')
        # cbar=fig.colorbar(pcol, ticks=[-zscales[i],0,zscales[i]],aspect=11,shrink=.9,pad=0.08)
        # cbar=fig.colorbar(pcol, ax=ax.ravel().tolist())
        # cbar.ax.set_yticklabels(['max', '0', 'max'])
        # if i!=0: cbar.ax.set_yticklabels(['max', '0', 'max'])
        #Specifying figure coordinates works fine:
        
        # cbar_ax.set_ticks_position('top')
        # cb1.ax.yaxis.set_ticks_position("top")
        
        
        fig.supxlabel(r'$x$-component $\to$',x=0.052 ,y=1.01)
        fig.supylabel(r'$y$-component $\to$',x=-0.03 ,y=.95 )
    fig_coord = [.767,1.011,0.22,0.02]
    cbar_ax = fig.add_axes(fig_coord)

    clevs = [-1,0,1]
    cb1 = plt.colorbar(hdl, cax=cbar_ax, orientation='horizontal', ticks=clevs,location='top')
    cb1.set_label(label=r'$z$-component',size=12, labelpad=10)
    cb1.ax.set_xticklabels(['min','0','max'  ])    
    if pdfs:plt.savefig("figures/"+pdfname+".pdf",format='pdf', bbox_inches = "tight")
    else:plt.show(fig)

def plot_force_per_enantiomer(r0,λ0,α,a,ϵ,μ,E,H,X,Y,Z,zix=1,w=4,arrows=None,pdfs=False,pdfname="figure",basis='enantiomer',Power=None):
    c=1/np.sqrt(ϵ*μ)            # speed of light in the medium [m/s]
    ω=2*np.pi*c0/λ0             # frequency [Hz]
    k=ω/c                       # wave number [1/m]
    R=np.sqrt(X**2+Y**2)        # radial coordinate
    b=np.tile(np.reshape(1*(R>(r0+a)),(1,np.shape(X)[0],np.shape(Y)[1],np.shape(Z)[2])),(3,1,1,1))
    (W0,W1,W2,W3,grad_W0,grad_W1,grad_W2,grad_W3,ωp0,ωp1,ωp2,ωp3,ωS0,Im_Π,Re_Π,ωS1)=observables(ϵ,E,μ,H,X,Y,Z,Power=Power)
    if arrows is None:step=int(len(X)/8)
    else: step=int(len(X)/arrows)
    βe=k**3*np.real(np.conj(α[0])*α[2])/(3*np.pi)
    βm=k**3*np.real(np.conj(α[1])*α[2])/(3*np.pi)
    βr=k**3*np.real(np.conj(α[0])*α[1]+np.conj(α[2])*α[2])/(6*np.pi)
    βi=k**3*np.imag(np.conj(α[0])*α[1])/(6*np.pi)
    grad_We=(grad_W0+grad_W1)/2
    grad_Wm=(grad_W0-grad_W1)/2
    ωpe=(ωp0+ωp1)/2
    ωpm=(ωp0-ωp1)/2
    ωSe=(ωS0+ωS1)/2
    ωSm=(ωS0-ωS1)/2
    Fa=b*(np.real(α[0])*grad_We+np.real(α[1])*grad_Wm+2*np.imag(α[0])*ωpe+2*np.imag(α[1])*ωpm-k*(βr*Re_Π/c+βi*Im_Π/c))
    Fc=b*(np.real(α[2])*grad_W3+2*np.imag(α[2])*ωp3-k*(βe*ωSe+βm*ωSm))
    unit=1e12 # convert to fN/mW
    dx=np.diff(X[:,0,0])[0]
    dy=np.diff(Y[0,:,0])[0]
    dr=np.sqrt(dx**2+dy**2)     # radial step size 
    
    if basis=='enantiomer':
        fig, ax = plt.subplots(ncols=4,figsize=(10,3),constrained_layout=True, sharex=True, sharey=True)
        scale=np.max((unit*v_norm(Fa+Fc),unit*v_norm(Fa-Fc),unit*v_norm(Fa),unit*v_norm(Fc)))
        titles=(r'$\vec{F}_\text{L}=\vec{F}_\text{a}-\vec{F}_\text{c}$',r'$\vec{F}_\text{R}=\vec{F}_\text{a}+\vec{F}_\text{c}$',r'$\vec{F}_\text{a}$',r'$\vec{F}_\text{c}$')
        forces=(Fa-Fc,Fa+Fc,Fa,Fc)
        ρ=(r0+a+dr)
        Fa_right=vector_field_at(X,Y,Z,Fa,z=0,ρ=ρ,φ=0)
        Fc_right=vector_field_at(X,Y,Z,Fc,z=0,ρ=ρ,φ=0)
        Fa_top=vector_field_at(X,Y,Z,Fa,z=0,ρ=ρ,φ=np.pi/2)
        Fc_top=vector_field_at(X,Y,Z,Fc,z=0,ρ=ρ,φ=np.pi/2)
        FR_right=vector_field_at(X,Y,Z,Fa+Fc,z=0,ρ=ρ,φ=0)
        FL_right=vector_field_at(X,Y,Z,Fa-Fc,z=0,ρ=ρ,φ=0)
        FR_top=vector_field_at(X,Y,Z,Fa+Fc,z=0,ρ=ρ,φ=np.pi/2)
        FL_top=vector_field_at(X,Y,Z,Fa-Fc,z=0,ρ=ρ,φ=np.pi/2)
        print("Right:", "\t""Fa⊥ = {:.3g} [fN/mW]".format(unit*((tan_norm(Fa_right)))), "\t","Fc⊥ = {:.3g} [fN/mW]".format(unit*((tan_norm(Fc_right)))), "\t","Fa∥ = {:.3g} [fN/mW]".format(unit*((z_norm(Fa_right)))), "\t","Fc∥ = {:.3g} [fN/mW]".format(unit*((z_norm(Fc_right)))))
        print("Right:", "\t""FL⊥ = {:.3g} [fN/mW]".format(unit*((tan_norm(FL_right)))), "\t","FR⊥ = {:.3g} [fN/mW]".format(unit*((tan_norm(FR_right)))), "\t","FL∥ = {:.3g} [fN/mW]".format(unit*((z_norm(FL_right)))), "\t","FR∥ = {:.3g} [fN/mW]".format(unit*((z_norm(FR_right)))))
        print("Top:", "\t""Fa⊥ = {:.3g} [fN/mW]".format(unit*((tan_norm(Fa_top)))), "\t","Fc⊥ = {:.3g} [fN/mW]".format(unit*((tan_norm(Fc_top)))), "\t","Fa∥ = {:.3g} [fN/mW]".format(unit*((z_norm(Fa_top)))), "\t","Fc∥ = {:.3g} [fN/mW]".format(unit*((z_norm(Fc_top)))))
        print("Top:", "\t""FL⊥ = {:.3g} [fN/mW]".format(unit*((tan_norm(FL_top)))), "\t","FR⊥ = {:.3g} [fN/mW]".format(unit*((tan_norm(FR_top)))), "\t","FL∥ = {:.3g} [fN/mW]".format(unit*((z_norm(FL_top)))), "\t","FR∥ = {:.3g} [fN/mW]".format(unit*((z_norm(FR_top)))))
        # print("Top: ","FR⊥ = {:.3g} [fN/mW]".format(unit*((tan_norm(Fa_top+Fc_top)))), "\t","FL⊥ = {:.3g} [fN/mW]".format(unit*((tan_norm(Fa_top-Fc_top)))))

        # print("F_a = {:.3g} [fN/mW]".format(unit*np.max((v_norm(Fa)))),"\t","F_c = {:.3g} [fN/mW]".format(unit*np.max((v_norm(Fc)))))
        # print("FR_z = {:.3g} [fN/mW]".format(unit*np.max((z_norm(Fa+Fc)))),"\t","FR_⊥ = {:.3g} [fN/mW]".format(unit*np.max((tan_norm(Fa+Fc)))))
        # print("FL_z = {:.3g} [fN/mW]".format(unit*np.max((z_norm(Fa-Fc)))),"\t","FL_⊥ = {:.3g} [fN/mW]".format(unit*np.max((tan_norm(Fa-Fc)))))
        for i in range(4):
            (pcol,quiv)=pltvector(X,Y,zix,unit*forces[i],ax[i],r0,λ0,scale=w*scale,zscale=scale,step=step,title=titles[i],out=True,mask=b[0,:,:,:])
        cbar=fig.colorbar(pcol, ticks=[-scale,0,scale],aspect=11,shrink=.8,pad=0.08)
        cbar.ax.set_yticklabels(['min', '0', 'max'])

    elif basis=='e-m-c':
        grad_titles=(r'$\Re(\alpha_\mathrm{e})\grad W_\mathrm{e}$',r'$\Re(\alpha_\mathrm{m})\grad W_\mathrm{m}$',r'$\Re(\alpha_\mathrm{c})\grad\Re(W_\mathrm{c})$',r'$\Re(\alpha_\mathrm{t})\grad \Im(W_\mathrm{c})$')
        pres_titles=(r'$2\omega\Im(\alpha_\mathrm{e})\vec{p}_\mathrm{e}$',r'$2\omega\Im(\alpha_\mathrm{m})\vec{p}_\mathrm{m}$',r'$2\omega\Im(\alpha_\mathrm{c})\Re(\vec{p}_\mathrm{c})$',r'$2\omega\Im(\alpha_\mathrm{t})\Im(\vec{p}_\mathrm{c})$')
        spin_titles=(r'$-\gamma^\text{e}_\text{rec}\omega\vec{S}_\mathrm{e}$',r'$-\gamma^\text{m}_\text{rec}\omega\vec{S}_\mathrm{m}$',r'$-\sigma_\mathrm{rec}\Re(\vec{\varPi})/c$',r'$-\sigma_\mathrm{im}\Im(\vec{\varPi})/c$')
        grad_forces=(b[0,:,:,:]*np.real(α[0])*b*grad_We,b[0,:,:,:]*np.real(α[1])*b*grad_Wm,b[0,:,:,:]*np.real(α[2])*b*grad_W3,0*grad_W2)
        pres_forces=(b[0,:,:,:]*np.imag(α[0])*b*2*ωpe,b[0,:,:,:]*np.imag(α[1])*2*ωpm,b[0,:,:,:]*np.imag(α[2])*2*ωp3,0*2*ωp2)
        spin_forces=(-b[0,:,:,:]*βe*k*ωSe,-b[0,:,:,:]*βm*k*ωSm,-b[0,:,:,:]*βr*k*Re_Π/c,-b[0,:,:,:]*βi*k*Im_Π/c)
        forces=(grad_forces,pres_forces,spin_forces)
        titles=(grad_titles,pres_titles,spin_titles)
        scale=np.max((v_norm(grad_forces[0]),v_norm(grad_forces[1]),v_norm(grad_forces[2]),v_norm(grad_forces[3]),v_norm(pres_forces[0]),v_norm(pres_forces[1]),v_norm(pres_forces[2]),v_norm(pres_forces[3]),v_norm(spin_forces[0]),v_norm(spin_forces[1]),v_norm(spin_forces[2]),v_norm(spin_forces[3])))
        tscale=scale#np.max((tan_norm(grad_forces[0]),tan_norm(grad_forces[1]),tan_norm(grad_forces[2]),tan_norm(grad_forces[3]),tan_norm(pres_forces[0]),tan_norm(pres_forces[1]),tan_norm(pres_forces[2]),tan_norm(pres_forces[3]),tan_norm(spin_forces[0]),tan_norm(spin_forces[1]),tan_norm(spin_forces[2]),tan_norm(spin_forces[3])))
        zscale=scale#np.max((z_norm(grad_forces[0]),z_norm(grad_forces[1]),z_norm(grad_forces[2]),z_norm(grad_forces[3]),z_norm(pres_forces[0]),z_norm(pres_forces[1]),z_norm(pres_forces[2]),z_norm(pres_forces[3]),z_norm(spin_forces[0]),z_norm(spin_forces[1]),z_norm(spin_forces[2]),z_norm(spin_forces[3])))
        fig, ax = plt.subplots(ncols=len(grad_titles),nrows=3,figsize=(10,7.9),constrained_layout=True, sharex=True, sharey=True)
        for i in range(3):
            for j in range(len(grad_titles)):
                (pcol,quiv)=pltvector(X,Y,zix,b[0,:,:,:]*forces[i][j],ax[i,j],r0,λ0,scale=w*tscale,zscale=zscale,step=step,title=titles[i][j],out=True,mask=b[0,:,:,:])
                # fibre=plt.Circle((0, 0), r0/λ0, color='white', ec='black',fill=True,linestyle='--'); ax[i,j].add_artist(fibre)
            cbar=fig.colorbar(pcol, ticks=[-zscale,0,zscale],aspect=11,shrink=.9,pad=0.08)
            cbar.ax.set_yticklabels(['min', '0', 'max'])
    fig.supxlabel(r'$x/\lambda_0$');fig.supylabel(r'$y/\lambda_0$')
    if pdfs:plt.savefig("figures/FE_"+pdfname+".pdf",format='pdf')
    else:plt.show(fig)

def plot_max_F_enantiomer(r0,λ0,εp,μp,κp,ϵ,μ,E,H,X,Y,Z,zix=1,pdfs=False,pdfname="figure",out=False,a0=None,k0amin=0,k0amax=2,samples=25):
    a_per_λ0=np.linspace(k0amin/(2*np.pi),k0amax/(2*np.pi),samples)
    a=a_per_λ0*λ0
    Famax=np.zeros(len(a_per_λ0))
    Fcmax=np.zeros(len(a_per_λ0))
    α=get_α(λ0,a,εp,μp,κp)
    unit=1e12 # convert to fN/mW
    for i in tqdm(range(len(a_per_λ0))):
        c=1/np.sqrt(ϵ*μ)            # speed of light in the medium [m/s]
        ω=2*np.pi*c0/λ0             # frequency [Hz]
        k=ω/c                       # wave number [1/m]
        R=np.sqrt(X**2+Y**2)        # radial coordinate
        b=np.tile(np.reshape(1*(R>(r0+a[i])),(1,np.shape(X)[0],np.shape(Y)[1],np.shape(Z)[2])),(3,1,1,1))
        (W0,W1,W2,W3,grad_W0,grad_W1,grad_W2,grad_W3,ωp0,ωp1,ωp2,ωp3,ωS0,Im_Π,Re_Π,ωS1)=observables(ϵ,E,μ,H,X,Y,Z)
        βe=k**3*np.real(np.conj(α[0][i])*α[2][i])/(3*np.pi)
        βm=k**3*np.real(np.conj(α[1][i])*α[2][i])/(3*np.pi)
        βr=k**3*np.real(np.conj(α[0][i])*α[1][i]+np.conj(α[2][i])*α[2][i])/(6*np.pi)
        βi=k**3*np.imag(np.conj(α[0][i])*α[1][i])/(6*np.pi)
        grad_We=(grad_W0+grad_W1)/2
        grad_Wm=(grad_W0-grad_W1)/2
        ωpe=(ωp0+ωp1)/2
        ωpm=(ωp0-ωp1)/2
        ωSe=(ωS0+ωS1)/2
        ωSm=(ωS0-ωS1)/2
        Fa=b*(np.real(α[0][i])*grad_We+np.real(α[1][i])*grad_Wm+2*np.imag(α[0][i])*ωpe+2*np.imag(α[1][i])*ωpm-k*(βr*Re_Π/c+βi*Im_Π/c))
        Fc=b*(np.real(α[2][i])*grad_W3+2*np.imag(α[2][i])*ωp3-k*(βe*ωSe+βm*ωSm))
        Famax[i]=unit*np.max((v_norm(Fa)))
        Fcmax[i]=unit*np.max((v_norm(Fc)))
    fig, ax = plt.subplots(ncols=1,figsize=(5,4),nrows=1, sharex=True, sharey=True,constrained_layout=True)
    ax.plot(2*np.pi*a_per_λ0,Fcmax,label=f'chiral force' ,linestyle='solid')
    ax.plot(2*np.pi*a_per_λ0,Famax,label=f'achiral force',linestyle='dashed')
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ax.set_yscale('log')
    ax.set_ylim(1e-3,1e3)
    ax.set_xlim(k0amin,k0amax)	
    ax.grid(color='#bfbfbf',linestyle='-')#color='black', linestyle='--')   
    if a0 is not None: ax.axvline(2*np.pi*a0/λ0,linestyle='dashed',color='black', linewidth=1)
    fig.supxlabel(r'Radius of the particle in units of reduced wavelength $k_0a$');
    fig.supylabel(r'Power normalised force density [$\mathrm{fN}/(\mathrm{mW}$)]')
    plt.legend()
    if pdfs:
        # tpl.save("figures/EF_"+pdfname+".tikz", flavor="latex")
        plt.savefig("figures/EF_"+pdfname+".pdf",format='pdf')
    else: plt.show()
    if out: return (a_per_λ0,Famax,Fcmax)
    else: return None

def plot_F_vs_a(r0,λ0,εp,μp,κp,ϵ,μ,E,H,X,Y,Z,zix=1,pdfs=False,pdfname="figure",out=False,a0=None,k0amin=0,k0amax=2,samples=25,x=None,y=None,ρ=None,φ=None,comp=None):
    a_per_λ0=np.linspace(k0amin/(2*np.pi),k0amax/(2*np.pi),samples)
    a=a_per_λ0*λ0
    Famax=np.zeros(len(a_per_λ0))
    Fcmax=np.zeros(len(a_per_λ0))
    α=get_α(λ0,a,εp,μp,κp)
    unit=1e12 # convert to fN/mW
    dx=np.diff(X[:,0,0])[0]
    dy=np.diff(Y[0,:,0])[0]
    dr=np.sqrt(dx**2+dy**2)     # radial step size 
    for i in tqdm(range(len(a_per_λ0))):
        ρ=(r0+a[i]+dr)
        c=1/np.sqrt(ϵ*μ)            # speed of light in the medium [m/s]
        ω=2*np.pi*c0/λ0             # frequency [Hz]
        k=ω/c                       # wave number [1/m]
        R=np.sqrt(X**2+Y**2)        # radial coordinate
        b=np.tile(np.reshape(1*(R>(r0+a[i])),(1,np.shape(X)[0],np.shape(Y)[1],np.shape(Z)[2])),(3,1,1,1))
        (W0,W1,W2,W3,grad_W0,grad_W1,grad_W2,grad_W3,ωp0,ωp1,ωp2,ωp3,ωS0,Im_Π,Re_Π,ωS1)=observables(ϵ,E,μ,H,X,Y,Z)
        βe=k**3*np.real(np.conj(α[0][i])*α[2][i])/(3*np.pi)#here I need to change indices
        βm=k**3*np.real(np.conj(α[1][i])*α[2][i])/(3*np.pi)
        βr=k**3*np.real(np.conj(α[0][i])*α[1][i]+np.conj(α[2][i])*α[2][i])/(6*np.pi)
        βi=k**3*np.imag(np.conj(α[0][i])*α[1][i])/(6*np.pi)
        grad_We=(grad_W0+grad_W1)/2
        grad_Wm=(grad_W0-grad_W1)/2
        ωpe=(ωp0+ωp1)/2
        ωpm=(ωp0-ωp1)/2
        ωSe=(ωS0+ωS1)/2
        ωSm=(ωS0-ωS1)/2
        Fa=b*(np.real(α[0][i])*grad_We+np.real(α[1][i])*grad_Wm+2*np.imag(α[0][i])*ωpe+2*np.imag(α[1][i])*ωpm-k*(βr*Re_Π/c+βi*Im_Π/c))
        Fc=b*(np.real(α[2][i])*grad_W3+2*np.imag(α[2][i])*ωp3-k*(βe*ωSe+βm*ωSm))
        if x is not None and y is not None or ρ is not None and φ is not None:
            Fa_r=vector_field_at(X,Y,Z,Fa,x=x,y=y,z=0,ρ=ρ,φ=φ)
            Fc_r=vector_field_at(X,Y,Z,Fc,x=x,y=y,z=0,ρ=ρ,φ=φ)
            if comp=='x': Famax[i]=unit*(np.abs(Fa_r[0])); Fcmax[i]=unit*(np.abs(Fc_r[0]))
            elif comp=='y': Famax[i]=unit*(np.abs(Fa_r[1])); Fcmax[i]=unit*(np.abs(Fc_r[1]))
            elif comp=='z': Famax[i]=unit*(z_norm(Fa_r)); Fcmax[i]=unit*(z_norm(Fc_r))
            elif comp=='tan': Famax[i]=unit*(tan_norm(Fa_r)); Fcmax[i]=unit*(tan_norm(Fc_r))
            elif comp=='r': Famax[i]=unit*np.abs(cart2cylin(Fa_r[0],Fa_r[1],Fa_r[2],φ)[0]); Fcmax[i]=unit*np.abs(cart2cylin(Fc_r[0],Fc_r[1],Fc_r[2],φ)[0])
            elif comp=='φ': Famax[i]=unit*np.abs(cart2cylin(Fa_r[0],Fa_r[1],Fa_r[2],φ)[1]); Fcmax[i]=unit*np.abs(cart2cylin(Fc_r[0],Fc_r[1],Fc_r[2],φ)[1])
            else: Famax[i]=unit*(v_norm(Fa_r)); Fcmax[i]=unit*(v_norm(Fc_r))
        else:
            Famax[i]=unit*np.max((v_norm(Fa)))
            Fcmax[i]=unit*np.max((v_norm(Fc)))
    fig, ax = plt.subplots(ncols=1,figsize=(5,4),nrows=1, sharex=True, sharey=True,constrained_layout=True)
    ax.plot(2*np.pi*a_per_λ0,Fcmax,label=f'chiral force' ,linestyle='solid')
    ax.plot(2*np.pi*a_per_λ0,Famax,label=f'achiral force',linestyle='dashed')
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ax.set_yscale('log')
    ax.set_ylim(1e-3,1e3)
    ax.set_xlim(k0amin,k0amax)	
    ax.grid(color='#bfbfbf',linestyle='-')#color='black', linestyle='--')   
    if a0 is not None: ax.axvline(2*np.pi*a0/λ0,linestyle='dashed',color='black', linewidth=1)
    fig.supxlabel(r'Radius of the particle in units of reduced wavelength $k_0a$');
    fig.supylabel(r'Power normalised force density [$\mathrm{fN}/(\mathrm{mW}$)]')
    plt.legend()
    if pdfs:
        # tpl.save("figures/EF_"+pdfname+".tikz", flavor="latex")
        plt.savefig("figures/EF_"+pdfname+".pdf",format='pdf')
    else: plt.show()
    if out: return (a_per_λ0,Famax,Fcmax)
    else: return None


def plot_max_forces(λ0,r0,R=1,L=0,k0r0min=1,k0r0max=6,N=500,resolution=250/3,samples=100,ℓ=1,zix=1,inside=False,plot_all=False,mode=0,arrows=8,basis='e-m-c',out=False,pdfs=False,pdfname="figure",rescale=False,low=4,high=6):
    if samples>N: N=samples
    (r0_per_λ0,nr,Vr)=get_neff(rmin=k0r0min/(2*np.pi),rmax=k0r0max/(2*np.pi),N=N,ℓ=ℓ,mode=mode)
    (r0_per_λ0,nl,Vl)=get_neff(rmin=k0r0min/(2*np.pi),rmax=k0r0max/(2*np.pi),N=N,ℓ=-ℓ,mode=mode)
    # reduce number of points
    sampling=int(N/samples) #number of points to sample
    r0_per_λ0=r0_per_λ0[::sampling]
    nr=nr[::sampling]
    nl=nl[::sampling]
    Vr=Vr[::sampling]
    Vl=Vl[::sampling]
    # allocate memory
    f_grad_W0=np.zeros(len(r0_per_λ0))
    f_grad_W3=np.zeros(len(r0_per_λ0))
    f_grad_W2=np.zeros(len(r0_per_λ0))
    f_grad_W1=np.zeros(len(r0_per_λ0))
    f_2ωp0=np.zeros(len(r0_per_λ0))
    f_2ωp3=np.zeros(len(r0_per_λ0))
    f_2ωp2=np.zeros(len(r0_per_λ0))
    f_2ωp1=np.zeros(len(r0_per_λ0))
    f_kRe_Π_per_c=np.zeros(len(r0_per_λ0))
    f_kIm_Π_per_c=np.zeros(len(r0_per_λ0))
    f_kωS0=np.zeros(len(r0_per_λ0))
    f_kωS1=np.zeros(len(r0_per_λ0))
    zf_grad_W0=np.zeros(len(r0_per_λ0))
    zf_grad_W3=np.zeros(len(r0_per_λ0))
    zf_grad_W2=np.zeros(len(r0_per_λ0))
    zf_grad_W1=np.zeros(len(r0_per_λ0))
    zf_2ωp0=np.zeros(len(r0_per_λ0))
    zf_2ωp3=np.zeros(len(r0_per_λ0))
    zf_2ωp2=np.zeros(len(r0_per_λ0))
    zf_2ωp1=np.zeros(len(r0_per_λ0))
    zf_kRe_Π_per_c=np.zeros(len(r0_per_λ0))
    zf_kIm_Π_per_c=np.zeros(len(r0_per_λ0))
    zf_kωS0=np.zeros(len(r0_per_λ0))
    zf_kωS1=np.zeros(len(r0_per_λ0))
    f_grad_We=np.zeros(len(r0_per_λ0))
    f_grad_Wm=np.zeros(len(r0_per_λ0))
    zf_grad_We=np.zeros(len(r0_per_λ0))
    zf_grad_Wm=np.zeros(len(r0_per_λ0))
    f_2ωpe=np.zeros(len(r0_per_λ0))
    f_2ωpm=np.zeros(len(r0_per_λ0))
    zf_2ωpe=np.zeros(len(r0_per_λ0))
    zf_2ωpm=np.zeros(len(r0_per_λ0))
    f_kωSe=np.zeros(len(r0_per_λ0))
    f_kωSm=np.zeros(len(r0_per_λ0))
    zf_kωSe=np.zeros(len(r0_per_λ0))
    zf_kωSm=np.zeros(len(r0_per_λ0))
    # Powers=np.zeros(len(r0_per_λ0))
    # Powers_out=np.zeros(len(r0_per_λ0))

    for i in tqdm(range(len(r0_per_λ0))):
        # calculate the power
        ω=2*np.pi*c0/λ0 #frequency [Hz]
        x=λ0*np.linspace(-1,1,int(N))   #x coordinates
        y=λ0*np.linspace(-1,1,int(N))   #y coordinates
        X, Y = np.meshgrid(x, y, indexing='ij')  #3D grid of coordinates
        dx=np.diff(X[:,0])[0]
        dy=np.diff(Y[0,:])[0]
        E=get_E(X,Y,0,λ0,r0_per_λ0[i]*λ0,nr[i],Vr[i],nl[i],Vl[i],R=R,L=L,ℓ=ℓ)   #electric field
        H=get_H(X,Y,0,λ0,r0_per_λ0[i]*λ0,nr[i],Vr[i],nl[i],Vl[i],R=R,L=L,ℓ=ℓ)   #magnetic field
        Power=calculate_power(E,H,dx,dy)

        # set up the grid
        m=1.1*r0_per_λ0[i]
        N=resolution   #number of points in each direction
        x=λ0*np.linspace(-m,m,int(N))   #x coordinates
        y=λ0*np.linspace(-m,m,int(N))   #y coordinates
        z=λ0*np.linspace(-np.diff(x)[0],np.diff(x)[0],int(3))   #z coordinates
        X, Y, Z = np.meshgrid(x, y, z,indexing='ij')  #3D grid of coordinates
        # generate fields 
        ϵ=get_ε(X,Y,r0_per_λ0[i]*λ0)   #permittivity
        μ=get_μ(X,Y,r0_per_λ0[i]*λ0)   #permeability
        E=get_E(X,Y,Z,λ0,r0_per_λ0[i]*λ0,nr[i],Vr[i],nl[i],Vl[i],R=R,L=L,ℓ=ℓ)   #electric field
        H=get_H(X,Y,Z,λ0,r0_per_λ0[i]*λ0,nr[i],Vr[i],nl[i],Vl[i],R=R,L=L,ℓ=ℓ)   #magnetic field
        dx=np.diff(X[:,0,0])[0]
        dy=np.diff(Y[0,:,0])[0]
        dz=np.diff(Z[0,0,:])[0]
        c=1/np.sqrt(ϵ*μ)    #speed of light
        k=ω/c               #wave number
        dr=np.sqrt(dx**2+dy**2)     # radial step size 
        r=np.sqrt(X**2+Y**2)        # radial coordinate

        #find maximums
        if inside: factor=np.tile(np.reshape(1*(r>(r0_per_λ0[i]*λ0+dr))+1*(r<(r0_per_λ0[i]*λ0-dr)),(1,np.shape(X)[0],np.shape(Y)[1],np.shape(Z)[2])),(3,1,1,1))
        else:
            factor=np.tile(np.reshape(1*(r>(r0_per_λ0[i]*λ0+dr)),(1,np.shape(X)[0],np.shape(Y)[1],np.shape(Z)[2])),(3,1,1,1))
        # Powers[i]=Power
        # Powers_out[i]=calculate_power(factor*E,factor*H,dx,dy)
        
        #calculate energy densities
        W0=get_W0(ϵ,E,μ,H)   #total energy density
        W3=get_W3(ϵ,E,μ,H)   #difference between electric and magnetic energy densities
        W2=get_W2(ϵ,E,μ,H)   #circularly polarized energy density
        W1=get_W1(ϵ,E,μ,H)   #diagonal energy density

        #gradients of energy densities
        grad_W0=factor*np.asarray(np.gradient(W0,dx,dy,dz))/Power  #gradient of total energy density
        grad_W3=factor*np.asarray(np.gradient(W3,dx,dy,dz))/Power  #gradient of difference between electric and magnetic energy densities
        grad_W2=factor*np.asarray(np.gradient(W2,dx,dy,dz))/Power  #gradient of circularly polarized energy density
        grad_W1=factor*np.asarray(np.gradient(W1,dx,dy,dz))/Power  #gradient of diagonal energy density
        grad_We=(grad_W0+grad_W1)/2 #electric gradient
        grad_Wm=(grad_W0-grad_W1)/2 #magnetic gradient

        #calculate momentum densities
        ωp0=factor*get_ωp0(ϵ,E,μ,H,dx,dy,dz)/Power   #total momentum density
        ωp3=factor*get_ωp3(ϵ,E,μ,H,dx,dy,dz)/Power   #difference between electric and magnetic momentum densities
        ωp2=factor*get_ωp2(ϵ,E,μ,H,dx,dy,dz)/Power   #circularly polarized momentum density
        ωp1=factor*get_ωp1(ϵ,E,μ,H,dx,dy,dz)/Power   #diagonal momentum density
        ωpe=(ωp0+ωp1)/2 #electric momentum
        ωpm=(ωp0-ωp1)/2 #magnetic momentum

        #calculate poynting vector and spin densities
        Re_Π=factor*get_Re_Π(E,H)/Power  #real part of poynting vector
        Im_Π=factor*get_Im_Π(E,H)/Power  #imaginary part of poynting vector
        ωS0=factor*get_ωS0(ϵ,E,μ,H)/Power  #total spin density
        ωS1=factor*get_ωS1(ϵ,E,μ,H)/Power  #difference between electric and magnetic spin densities
        ωSe=(ωS0+ωS1)/2 #electric spin
        ωSm=(ωS0-ωS1)/2 #magnetic spin
        
        f_grad_W0[i]=np.max(tan_norm(grad_W0))
        f_grad_W3[i]=np.max(tan_norm(grad_W3))
        f_grad_W2[i]=np.max(tan_norm(grad_W2))
        f_grad_W1[i]=np.max(tan_norm(grad_W1))
        f_2ωp0[i]=np.max(tan_norm(2*ωp0))
        f_2ωp3[i]=np.max(tan_norm(2*ωp3))
        f_2ωp2[i]=np.max(tan_norm(2*ωp2))
        f_2ωp1[i]=np.max(tan_norm(2*ωp1))
        f_kRe_Π_per_c[i]=np.max(tan_norm(k*Re_Π/c))
        f_kIm_Π_per_c[i]=np.max(tan_norm(k*Im_Π/c))
        f_kωS0[i]=np.max(tan_norm(k*ωS0))
        f_kωS1[i]=np.max(tan_norm(k*ωS1))
        f_grad_We[i]=np.max(tan_norm(grad_We))
        f_grad_Wm[i]=np.max(tan_norm(grad_Wm))
        f_2ωpe[i]=np.max(tan_norm(2*ωpe))
        f_2ωpm[i]=np.max(tan_norm(2*ωpm))
        f_kωSe[i]=np.max(tan_norm(k*ωSe))
        f_kωSm[i]=np.max(tan_norm(k*ωSm))
        zf_grad_W0[i]=np.max(z_norm(grad_W0))
        zf_grad_W3[i]=np.max(z_norm(grad_W3))
        zf_grad_W2[i]=np.max(z_norm(grad_W2))
        zf_grad_W1[i]=np.max(z_norm(grad_W1))
        zf_2ωp0[i]=np.max(z_norm(2*ωp0))
        zf_2ωp3[i]=np.max(z_norm(2*ωp3))
        zf_2ωp2[i]=np.max(z_norm(2*ωp2))
        zf_2ωp1[i]=np.max(z_norm(2*ωp1))
        zf_kRe_Π_per_c[i]=np.max(z_norm(k*Re_Π/c))
        zf_kIm_Π_per_c[i]=np.max(z_norm(k*Im_Π/c))
        zf_kωS0[i]=np.max(z_norm(k*ωS0))
        zf_kωS1[i]=np.max(z_norm(k*ωS1))
        zf_grad_We[i]=np.max(z_norm(grad_We))
        zf_grad_Wm[i]=np.max(z_norm(grad_Wm))
        zf_2ωpe[i]=np.max(z_norm(2*ωpe))
        zf_2ωpm[i]=np.max(z_norm(2*ωpm))
        zf_kωSe[i]=np.max(z_norm(k*ωSe))
        zf_kωSm[i]=np.max(z_norm(k*ωSm))

        if plot_all:
            plot_forces(r0_per_λ0[i]*λ0,λ0,ϵ,μ,E,H,X,Y,Z,zix=1,w=4,arrows=arrows,basis=basis,inside=inside)
    if rescale: 
        unit = 1e12*λ0**3/NA**3 #conversion factor to fN/(mW·λ0³/√(n1**2-n2**2)³)
        # xuni = n1
        # xlab = r'Radius of the fibre in units of reduced wavelength $n_1k_0r_0$'
        xuni = NA
        xlab = r'Normalised radius of the fibre (or frequency) $V=r_0\sqrt{k_1^2-k^2_2}$'
        ylab = r'Power normalised force density [$\mathrm{fN}/(\mathrm{mW}\cdot\lambda_0^3/\sqrt{n_1^2-n_2^2}^3$)]'
    else:       
        unit = 1e12*λ0**3 #conversion factor to fN/(mW·λ0³)
        xuni = NA
        xlab = r'Normalised radius of the fibre (or frequency) $V=r_0\sqrt{k_1^2-k^2_2}$'
        ylab = r'Power normalised force density $\vec{V}\!\!_i$ [$\mathrm{fN}/(\mathrm{mW}\cdot\lambda_0^3$)]'
    
    if basis=='pauli':
        grad_titles = (r'$\grad (W_\mathrm{e}+W_\mathrm{m})$',r'$\grad (W_\mathrm{e}-W_\mathrm{m})$',r'$\grad \Re W_c$',r'$\grad \Im W_c$')
        pres_titles = (r'$2\omega(\vec{p}_\mathrm{e}+\vec{p}_\mathrm{m})$',r'$2\omega(\vec{p}_\mathrm{e}-\vec{p}_\mathrm{m})$',r'$2\omega\Re\vec{p}_\mathrm{c}$',r'$2\omega\Im\vec{p}_\mathrm{c}$')
        spin_titles = (r'$k\omega(\vec{S}_\mathrm{e}+\vec{S}_\mathrm{m})$',r'$k\omega(\vec{S}_\mathrm{e}-\vec{S}_\mathrm{m})$',r'$k\Re\vec{\varPi}/c$',r'$k\Im\vec{\varPi}/c$')
        grad_forces = (unit*f_grad_W0,unit*f_grad_W1,unit*f_grad_W3,unit*f_grad_W2)
        pres_forces = (unit*f_2ωp0,unit*f_2ωp1,unit*f_2ωp3,unit*f_2ωp2)
        spin_forces = (unit*f_kωS0,unit*f_kωS1,unit*f_kRe_Π_per_c,unit*f_kIm_Π_per_c)
        zgrad_forces = (unit*zf_grad_W0,unit*zf_grad_W1,unit*zf_grad_W3,unit*zf_grad_W2)
        zpres_forces = (unit*zf_2ωp0,unit*zf_2ωp1,unit*zf_2ωp3,unit*zf_2ωp2)
        zspin_forces = (unit*zf_kωS0,unit*zf_kωS1,unit*zf_kRe_Π_per_c,unit*zf_kIm_Π_per_c)
    elif basis=='e-m-c':
        grad_titles = (r'$\grad W_\mathrm{e}$',r'$\grad W_\mathrm{m}$',r'$\grad\Re W_\mathrm{c}$',r'$\grad \Im W_\mathrm{c}$')
        pres_titles = (r'$2\omega\vec{p}_\mathrm{e}$',r'$2\omega\vec{p}_\mathrm{m}$',r'$2\omega\Re\vec{p}_\mathrm{c}$',r'$2\omega\Im\vec{p}_\mathrm{c}$')
        spin_titles = (r'$k\omega\vec{S}_\mathrm{e}$',r'$k\omega\vec{S}_\mathrm{m}$',r'$k\Re\vec{\varPi}/c$',r'$k\Im\vec{\varPi}/c$')
        grad_forces = (unit*f_grad_We,unit*f_grad_Wm,unit*f_grad_W2,unit*f_grad_W1)
        pres_forces = (unit*f_2ωpe,unit*f_2ωpm,unit*f_2ωp3,unit*f_2ωp2)
        spin_forces = (unit*f_kωSe,unit*f_kωSm,unit*f_kRe_Π_per_c,unit*f_kIm_Π_per_c)
        zgrad_forces = (unit*zf_grad_We,unit*zf_grad_Wm,unit*zf_grad_W3,unit*zf_grad_W2)
        zpres_forces = (unit*zf_2ωpe,unit*zf_2ωpm,unit*zf_2ωp3,unit*zf_2ωp2)
        zspin_forces = (unit*zf_kωSe,unit*zf_kωSm,unit*zf_kRe_Π_per_c,unit*zf_kIm_Π_per_c)
    titles = (grad_titles,pres_titles,spin_titles)
    tforces = (grad_forces,pres_forces,spin_forces)
    zforces = (zgrad_forces,zpres_forces,zspin_forces)
    forces = (tforces,zforces)
    axtitles = ('transverse force','longitudinal force')
    xaxis = 2*np.pi*r0_per_λ0*xuni
    r_opt = (2*np.pi)*r0/λ0*xuni

    if out:
        return (titles,forces,xaxis,r_opt,(xlab,ylab),k0r0min*xuni,k0r0max*xuni)
    else: 
        fig, ax = plt.subplots(ncols=2,figsize=(4.5,5),nrows=3, sharex=True, sharey=True,constrained_layout=True)
        for i in range(2):
            ax[0,i].set_title(axtitles[i])
            for j in range(3):
                for l in range(len(grad_titles)):
                    color = next(ax[j,i]._get_lines.prop_cycler)['color']
                    # ax[j,i].plot(2*np.pi*r0_per_λ0,forces[i][j][l],label=titles[j][l])
                    if j<2 and l<2:     ax[j,i].plot(xaxis,forces[i][j][l],label=titles[j][l],color=color,linestyle='dashed')
                    elif j>1 and l>1:   ax[j,i].plot(xaxis,forces[i][j][l],label=titles[j][l],color=color,linestyle='dashed')
                    else:   
                        ax[j,i].plot(xaxis,forces[i][j][l],label=titles[j][l],color=color)
                        # if l!=1:
                        #     print(2*np.pi*r0_per_λ0[signal.find_peaks(forces[i][j][l],width=5,prominence=250)[0]]*xuni)
                        #     ax[j,i].vlines(2*np.pi*r0_per_λ0[signal.find_peaks(forces[i][j][l],width=5,prominence=250)[0]]*xuni,1e4,1e6,linestyle='dotted', linewidth=1,color=color)
                ax[j,i].axvline(r_opt,linestyle='dashed',color='black', linewidth=1)
                ax[j,i].set_xlim(xmin=np.min(xaxis),xmax=k0r0max*xuni)
                ax[j,i].set_ylim(ymin=10**low,ymax=10**high)
                if   i==0 and j==1:ax[j,i].legend()
                elif i==1 and j!=1:ax[j,i].legend()
                ax[j,i].set_yscale('log')
                ax[j,i].grid(color='#bfbfbf',linestyle='-')#color='black', linestyle='--')   
        fig.supxlabel(xlab)
        fig.supylabel(ylab)
        if pdfs:
            # tpl.save("figures/FM_"+pdfname+".tikz", flavor="latex")
            plt.savefig("figures/FM_"+pdfname+".pdf",format='pdf')
        else: plt.show()
    return None
    
def radius_figure(CP,LP,pdfs=False,pdfname="radius",low=4,high=6):
    axtitles=('transverse force','longitudinal force','transverse force','longitudinal force')
    titles = CP[0]
    forces = (CP[1][0],CP[1][1],LP[1][0],LP[1][1])
    xaxis = CP[2]
    r_opt = CP[3]
    # powers = (CP[7],CP[7],LP[7],LP[7])
    fig, ax = plt.subplots(ncols=4,figsize=(8,3.5),nrows=3, sharex=True, sharey=True,constrained_layout=True)
    for i in range(4):
        ax[0,i].set_title(axtitles[i])
        for j in range(3):
            for l in range(len(titles[0])):
                color = next(ax[j,i]._get_lines.prop_cycler)['color']
                # if j==0 and (l==1 or l==3): ax[j,i].plot(xaxis,powers[i],label='power',color='black')
                if j<2 and l<2:     ax[j,i].plot(xaxis,forces[i][j][l],label=titles[j][l],color=color,linestyle='dashed')
                elif j>1 and l>1:   ax[j,i].plot(xaxis,forces[i][j][l],label=titles[j][l],color=color,linestyle='dashed')
                else:   
                    ax[j,i].plot(xaxis,forces[i][j][l],label=titles[j][l],color=color)
            ax[j,i].axvline(r_opt,linestyle='dashed',color='black', linewidth=1)
            ax[j,i].set_xlim(xmin=CP[5],xmax=CP[6])
            ax[j,i].set_ylim(ymin=10**low,ymax=10**high)
            if i==3: ax[j,i].legend(loc='center left', bbox_to_anchor=(1.05, 0.5)) 
            # if   i==2 and j==1:ax[j,i].legend()
            # elif i==3 and j!=1:ax[j,i].legend()
            ax[j,i].set_yscale('log')
            ax[j,i].grid(color='#bfbfbf',linestyle='-')#color='black', linestyle='--')  
    plt.figtext(0.25,1.03,"Circular polarisation", va="center", ha="center", size=14)
    plt.figtext(.65,1.03,"Linear polarisation", va="center", ha="center", size=14)

        
    fig.supxlabel(CP[4][0])
    fig.supylabel(CP[4][1])
    if pdfs:
        # tpl.save("figures/"+pdfname+".tikz", flavor="latex")
        plt.savefig("figures/"+pdfname+".pdf",format='pdf', bbox_inches = "tight")
    else: plt.show()