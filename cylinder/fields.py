import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import signal
from scipy.special import jv, hankel1 as hv,jvp, h1vp as hvp
from scipy.constants import hbar,c as c0, epsilon_0 as ϵ0, mu_0 as μ0
from tqdm.notebook import tqdm
# import tikzplotlib as tpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
    "text.latex.preamble": r"\usepackage{physics}\renewcommand{\vec}{\vb*}\usepackage{siunitx}"
}
)

def material(n_in=1.45,n_out=1,Δχ_in=0,Δχ_out=0): 
    '''Sets the refractive index of the material inside and outside the cylinder'''
    global n1; n1=n_in  # Refractive index inside
    global n2; n2=n_out # Refractive index outside


def find_nearest(array, value):     # Finds the nearest value in an array
    array = np.asarray(array)       # https://stackoverflow.com/a/2566508
    idx = (np.abs(array - value)).argmin()  
    return array[idx]   

def v_norm(F):                # Returns the norm of a vector field
    return np.max(np.real(np.sqrt(np.conj(F[0])*(F[0])+np.conj(F[1])*(F[1])+np.conj(F[2])*(F[2]))))
def tan_norm(F):            # Returns the norm of a vector field in the transverse plane
    return np.max(np.real(np.sqrt(np.conj(F[0])*(F[0])+np.conj(F[1])*(F[1]))))
def z_norm(F):              # Returns the norm of a vector field in the longitudinal direction
    return np.max(np.real(np.sqrt(np.conj(F[2])*(F[2]))))

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
        a11=n2*J
        a12=0j*r0_per_λ0
        a13=-n1*H
        a14=0j*r0_per_λ0
        a21=n2*(ℓ*kz_r0/((κ1_r0)**2))*J
        a22=1j*n2*(k1_r0/κ1_r0)*Jp
        a23=-n1*(ℓ*kz_r0/((κ2_r0)**2))*H
        a24=-1j*n1*(k2_r0/κ2_r0)*Hp
        a31=0j*r0_per_λ0
        a32=J
        a33=0j*r0_per_λ0
        a34=-H
        a41=-1j*(k1_r0/κ1_r0)*Jp
        a42=(ℓ*kz_r0/((κ1_r0)**2))*J
        a43=1j*(k2_r0/κ2_r0)*Hp
        a44=-(ℓ*kz_r0/((κ2_r0)**2))*H
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
    return np.where((neff<n1) & (neff>n2),output,0)


def get_neff(rmin=0,rmax=2,N=500,neff_fig=False,r0_over_λ0=None,ℓ=1):  
    '''Returns the effective refractive index as function r0/λ0 for a given ℓ mode and 
    eigenvector V of magnitudes of the fields'''  
    r0_per_λ0=np.linspace(rmin,rmax,N+1,endpoint=False)[1:] # r0/λ0
    neff=np.linspace(n2,n1,N+1,endpoint=False)[1:]          # n_eff
    Ro, Ne = np.meshgrid(r0_per_λ0, neff)                   # r0/λ0, n_eff meshgrid
    logdetM=logdetA(Ro,Ne,ℓ=ℓ)                              # log10|detM|

    # Finds n_eff for which log10|detM|=0
    n=np.zeros_like(r0_per_λ0)  
    for i in range(len(r0_per_λ0)):     
        try: # Finds the first peak of -log10|detM| and sets n_eff to the corresponding value
            n[i]=neff[np.max(signal.find_peaks(-logdetM[:,i])[0])]
        except ValueError: n[i]=0 #n_eff=0 if no peaks

    # Calculates amplitudes    
    M=np.transpose(boundary_conditions(r0_per_λ0,n,ℓ=ℓ),(2,0,1))
    (w,v)=np.linalg.eig(M)              # w,v are eigenvalues and eigenvectors of M
    argus=np.argmin(np.abs(w),axis=1)   # argus is the index of the eigenvalue closest to zero
    V=v[np.arange(N),:,argus]           # V is the eigenvector corresponding to the eigenvalue closest to zero

    # Plots n_eff
    if neff_fig:
        plt.pcolormesh(2*np.pi*Ro,Ne,-logdetM,cmap='Greys')
        plt.colorbar(label=r'$-\log_{10}|\det{M}|$')
        plt.xlim(xmax=2*np.pi*rmax,xmin=2*np.pi*rmin) 
        plt.plot(2*np.pi*r0_per_λ0,n,label=f'$n_\\mathrm{{eff}}(k_0r_0)$')
        if r0_over_λ0 is None: 
            pass
        else:
            idx=int(np.argwhere(r0_per_λ0==find_nearest(r0_per_λ0,r0_over_λ0)))
            plt.scatter(2*np.pi*r0_per_λ0[idx],n[idx])
        plt.ylim(n2,n1)
        
        plt.grid(color='black', linestyle='--')
        plt.ylabel(r'$n_\mathrm{eff}=\lambda_0/\lambda_z$')
        plt.xlabel(r'$k_0r_0=2\pi r_0/\lambda_0$')
        plt.legend()
        plt.show()

    if r0_over_λ0 is None: # Returns all values
        return (r0_per_λ0,n,V) 
    else: # Returns only the value closest to r0/λ0
        idx=int(np.argwhere(r0_per_λ0==find_nearest(r0_per_λ0,r0_over_λ0))) 
        return (r0_per_λ0[idx],n[idx],V[idx])

def get_Eℓ(x,y,z,λ0,r0,n,V,ℓ=1):
    '''returns ℓ mode of (Ex,Ey,Ez,Eφ) and ϵ in SI units''' 
    r=np.sqrt(x**2+y**2)        # radial coordinate
    φ=np.arctan2(y,x)           # azimuthal coordinate
    k0_r=2*np.pi*r/λ0           # k0·r
    kz_z= 2*np.pi*n*z/λ0
    κ1_per_k0=np.emath.sqrt(n1**2-n**2) # κ1/k0
    κ2_per_k0=np.emath.sqrt(n2**2-n**2) # κ2/k0
    ϵ1=ϵ0*n1**2; ϵ2=ϵ0*n2**2    # ϵ1,ϵ2
    Ep=np.where(r/r0<1,
                (-1j/(np.sqrt(ϵ1)*κ1_per_k0))*(n*V[0]+1j*n1*V[1])*jv(ℓ-1,k0_r*κ1_per_k0)*np.exp(1j*((ℓ-1)*φ+kz_z)),
                (-1j/(np.sqrt(ϵ2)*κ2_per_k0))*(n*V[2]+1j*n2*V[3])*hv(ℓ-1,k0_r*κ2_per_k0)*np.exp(1j*((ℓ-1)*φ+kz_z)))
    Em=np.where(r/r0<1,
                (-1j/(np.sqrt(ϵ1)*κ1_per_k0))*(-n*V[0]+1j*n1*V[1])*jv(ℓ+1,k0_r*κ1_per_k0)*np.exp(1j*((ℓ+1)*φ+kz_z)),
                (-1j/(np.sqrt(ϵ2)*κ2_per_k0))*(-n*V[2]+1j*n2*V[3])*hv(ℓ+1,k0_r*κ2_per_k0)*np.exp(1j*((ℓ+1)*φ+kz_z)))
    Ez=np.where(r/r0<1,
                (np.sqrt(2/ϵ1))*V[0]*jv(ℓ,k0_r*κ1_per_k0)*np.exp(1j*(ℓ*φ+kz_z)),
                (np.sqrt(2/ϵ2))*V[2]*hv(ℓ,k0_r*κ2_per_k0)*np.exp(1j*(ℓ*φ+kz_z)))
    Eφ=1j/np.sqrt(2)*(Ep*np.exp(1j*(φ))-Em*np.exp(-1j*(φ)))
    return ((Ep+Em)/np.sqrt(2),1j*(Ep-Em)/np.sqrt(2),Ez,Eφ,np.where(r/r0<1,ϵ1,ϵ2))

def get_Hℓ(x,y,z,λ0,r0,n,V,ℓ=1):
    '''returns ℓ mode of (H_x,H_y,H_z) and μ0 in SI units'''
    r=np.sqrt(x**2+y**2)    # radial coordinate
    φ=np.arctan2(y,x)       # azimuthal coordinate  
    k0_r=2*np.pi*r/λ0       # k0·r
    kz_z= 2*np.pi*n*z/λ0
    κ1_per_k0=np.emath.sqrt(n1**2-n**2) # κ1/k0 
    κ2_per_k0=np.emath.sqrt(n2**2-n**2) # κ2/k0
    Hp=np.where(r/r0<1,
                (-1j/(np.sqrt(μ0)*κ1_per_k0))*(n*V[1]-1j*n1*V[0])*jv(ℓ-1,k0_r*κ1_per_k0)*np.exp(1j*((ℓ-1)*φ+kz_z)),
                (-1j/(np.sqrt(μ0)*κ2_per_k0))*(n*V[3]-1j*n2*V[2])*hv(ℓ-1,k0_r*κ2_per_k0)*np.exp(1j*((ℓ-1)*φ+kz_z)))
    Hm=np.where(r/r0<1,
                (-1j/(np.sqrt(μ0)*κ1_per_k0))*(-n*V[1]-1j*n1*V[0])*jv(ℓ+1,k0_r*κ1_per_k0)*np.exp(1j*((ℓ+1)*φ+kz_z)),
                (-1j/(np.sqrt(μ0)*κ2_per_k0))*(-n*V[3]-1j*n2*V[2])*hv(ℓ+1,k0_r*κ2_per_k0)*np.exp(1j*((ℓ+1)*φ+kz_z)))
    Hz=np.where(r/r0<1,
                (np.sqrt(2/μ0))*V[1]*jv(ℓ,k0_r*κ1_per_k0)*np.exp(1j*(ℓ*φ+kz_z)),
                (np.sqrt(2/μ0))*V[3]*hv(ℓ,k0_r*κ2_per_k0)*np.exp(1j*(ℓ*φ+kz_z)))
    Hφ=1j/np.sqrt(2)*(Hp*np.exp(1j*(φ))-Hm*np.exp(-1j*(φ)))
    return ((Hp+Hm)/np.sqrt(2),1j*(Hp-Hm)/np.sqrt(2),Hz,Hφ,μ0)

def get_E(x,y,z,λ0,r0,nr,Vr,nl,Vl,R=1,L=0,ℓ=1):
    pE=get_Eℓ(x,y,z,λ0,r0,nr,Vr,ℓ=ℓ)
    mE=get_Eℓ(x,y,z,λ0,r0,nl,Vl,ℓ=-ℓ)
    return ((R*pE[0]+L*mE[0]),(R*pE[1]+L*mE[1]),(R*pE[2]+L*mE[2]),(R*pE[3]+L*mE[3]),pE[4])
def get_H(x,y,z,λ0,r0,nr,Vr,nl,Vl,R=1,L=0,ℓ=1):
    pH=get_Hℓ(x,y,z,λ0,r0,nr,Vr,ℓ=ℓ)
    mH=get_Hℓ(x,y,z,λ0,r0,nl,Vl,ℓ=-ℓ)
    return ((R*pH[0]+L*mH[0]),(R*pH[1]+L*mH[1]),(R*pH[2]+L*mH[2]),(R*pH[3]+L*mH[3]),pH[4])

def curl(A,dx,dy,dz):           
    '''returns the curl of A'''
    dAx=np.gradient(A[0],dx,dy,dz) 
    dAy=np.gradient(A[1],dx,dy,dz)
    dAz=np.gradient(A[2],dx,dy,dz)
    return np.array([dAy[2]-dAz[1],dAz[0]-dAx[2],dAx[1]-dAy[0]])

# Poynting vector
def get_Re_Π(E,H):
    '''returns the real Poynting vector in SI units'''
    return np.real(np.cross(E,np.conj(H),0,0,0))/2
def get_Im_Π(E,H):
    '''returns the imaginary Poynting vector in SI units'''
    return np.imag(np.cross(E,np.conj(H),0,0,0))/2
def calculate_power(Re_Π_z,dx,dy):
    '''returns the power in SI units'''
    return np.sum(Re_Π_z*dx*dy,axis=(0,1))

# Spin densities
def get_ωSt(ϵ,E,μ,H):
    ''' returns ω times total spin density in SI units'''
    return np.imag(ϵ*np.cross(np.conj(E),E,0,0,0)+μ*np.cross(np.conj(H),H,0,0,0))/4
def get_ωΔS(ϵ,E,μ,H):
    ''' returns ω times dual-odd spin density in SI units'''
    return np.imag(ϵ*np.cross(np.conj(E),E,0,0,0)-μ*np.cross(np.conj(H),H,0,0,0))/4

# Energy densities
def get_Wt(ϵ,E,μ,H):
    ''' returns total energy density in SI units'''
    Wt=0
    for i in range(3):
        Wt+=np.real(ϵ*np.conj(E[i])*E[i]+μ*np.conj(H[i])*H[i])/4
    return Wt
def get_ΔW(ϵ,E,μ,H):
    ''' returns dual-odd energy density in SI units'''
    ΔW=0
    for i in range(3):
        ΔW+=np.real(ϵ*np.conj(E[i])*E[i]-μ*np.conj(H[i])*H[i])/4
    return ΔW
def get_Wc(ϵ,E,μ,H):
    ''' returns parity-odd energy density in SI units'''
    Wc=0
    for i in range(3):
        Wc+=np.sqrt(ϵ*μ)*np.imag(np.conj(E[i])*H[i])/2
    return Wc
def get_Wr(ϵ,E,μ,H):
    ''' returns time-odd energy density in SI units'''
    Wr=0
    for i in range(3):
        Wr+=np.sqrt(ϵ*μ)*np.real(np.conj(E[i])*H[i])/2
    return Wr

# Orbital/canonical linear momentum densities
def get_ωpt(ω,ϵ,E,μ,H,dx,dy,dz):
    ''' returns linear momentum density in SI units'''
    Re_Π_per_c=np.sqrt(ϵ*μ)*get_Re_Π(E,H)
    ωSt=get_ωSt(ϵ,E,μ,H)
    k=ω*np.sqrt(ϵ*μ)
    return k*Re_Π_per_c-curl(ωSt,dx,dy,dz)/2
def get_ωΔp(ω,ϵ,E,μ,H,dx,dy,dz):
    ''' returns dual-odd linear momentum density in SI units'''
    ωΔS=get_ωΔS(ϵ,E,μ,H)
    return -curl(ωΔS,dx,dy,dz)/2
def get_ωpc(ω,ϵ,E,μ,H,dx,dy,dz):
    ''' returns parity-even linear momentum density in SI units'''
    Re_Π_per_c=np.sqrt(ϵ*μ)*get_Re_Π(E,H)
    ωSt=get_ωSt(ϵ,E,μ,H)
    k=ω*np.sqrt(ϵ*μ)
    return k*ωSt-curl(Re_Π_per_c,dx,dy,dz)/2
def get_ωpr(ω,ϵ,E,μ,H,dx,dy,dz):
    ''' returns time-even linear momentum density in SI units'''
    Im_Π_per_c=np.sqrt(ϵ*μ)*get_Im_Π(E,H)
    ωΔS=get_ωΔS(ϵ,E,μ,H)
    return -curl(Im_Π_per_c,dx,dy,dz)/2

# def pltscalar(X,Y,zix,W,ax,r0,λ0,scale=None,title=''):
#     if scale is None: scale=np.max(np.abs(W[:,:,zix]))
#     '''Plot the field in the x-y plane at z=zix'''
#     ax.pcolormesh(X[:,:,zix]/λ0,Y[:,:,zix]/λ0,W[:,:,zix],cmap='bwr',vmin=-scale,vmax=scale) #longitudinal component
#     circle=plt.Circle((0, 0), r0/λ0, color='black',fill=False,linestyle='--');ax.add_artist(circle);ax.set_aspect('equal')  #fibre boundary
#     ax.set_title(title)

def pltvector(X,Y,zix,V,ax,r0,λ0,W=None,scale=None,zscale=None,step=1,title='',out=False):
    '''Plot the field in the x-y plane at z=zix'''
    if W is None: 
        if zscale is None: zscale=np.max(np.abs(V[2][:,:,zix]))
        pcol=ax.pcolormesh(X[:,:,zix]/λ0,Y[:,:,zix]/λ0,V[2][:,:,zix],cmap='bwr',vmin=-zscale,vmax=zscale) #longitudinal component
    else: 
        if zscale is None: zscale=np.max(np.abs(W[:,:,zix]))
        pcol=ax.pcolormesh(X[:,:,zix]/λ0,Y[:,:,zix]/λ0,W[:,:,zix],cmap='bwr',vmin=-zscale,vmax=zscale)
    norm=np.real(np.sqrt(np.conj(V[0][:,:,zix])*(V[0][:,:,zix])+np.conj(V[1][:,:,zix])*(V[1][:,:,zix])))
    if scale is None: scale=10*np.max(norm)
    norm=np.where(norm>0,norm,0)
    '''Plot the field in the x-y plane at z=zix'''
    quiv=ax.quiver(X[::step,::step,zix]/λ0,Y[::step,::step,zix]/λ0,V[0][::step,::step,zix],V[1][::step,::step,zix],scale=scale) #transverse components
    # quiv=ax.quiver(X[::step,::step,zix]/λ0,Y[::step,::step,zix]/λ0,V[0][::step,::step,zix],V[1][::step,::step,zix],norm[::step,::step],scale=scale,cmap='bwr') #transverse components
    # ax.quiver(X[::step,::step,zix]/λ0,Y[::step,::step,zix]/λ0,V[0][::step,::step,zix],V[1][::step,::step,zix],scale=scale, edgecolor='k', facecolor='None', linewidth=.4)
    circle=plt.Circle((0, 0), r0/λ0, color='black',fill=False,linestyle='--');ax.add_artist(circle);ax.set_aspect('equal')  #fibre boundary
    ax.set_title(title) 
    if out: return pcol,quiv


def plot_max_forces(λ0,R=1,L=0,k0r0min=1,k0r0max=6,N=500,resolution=250/3,samples=100,window=4,ℓ=1,zix=0,inside=False,plot_all=False, output=False,unscaled=True,scaled=False ):
    sampling=int(N/samples) #number of points to sample
    (r0_per_λ0,nr,Vr)=get_neff(rmin=k0r0min/(2*np.pi),rmax=k0r0max/(2*np.pi),N=N,ℓ=1)
    (r0_per_λ0,nl,Vl)=get_neff(rmin=k0r0min/(2*np.pi),rmax=k0r0max/(2*np.pi),N=N,ℓ=-1)
    r0_per_λ0=r0_per_λ0[::sampling]
    nr=nr[::sampling]
    nl=nl[::sampling]
    Vr=Vr[::sampling]
    Vl=Vl[::sampling]

    f_grad_Wt=np.zeros(len(r0_per_λ0))
    f_grad_ΔW=np.zeros(len(r0_per_λ0))
    f_grad_Wc=np.zeros(len(r0_per_λ0))
    f_grad_Wr=np.zeros(len(r0_per_λ0))
    f_Wt=np.zeros(len(r0_per_λ0))
    f_ΔW=np.zeros(len(r0_per_λ0))
    f_Wc=np.zeros(len(r0_per_λ0))
    f_Wr=np.zeros(len(r0_per_λ0))
    f_2ωpt=np.zeros(len(r0_per_λ0))
    f_2ωΔp=np.zeros(len(r0_per_λ0))
    f_2ωpc=np.zeros(len(r0_per_λ0))
    f_2ωpr=np.zeros(len(r0_per_λ0))
    f_kRe_Π_per_c=np.zeros(len(r0_per_λ0))
    f_kIm_Π_per_c=np.zeros(len(r0_per_λ0))
    f_kωSt=np.zeros(len(r0_per_λ0))
    f_kωΔS=np.zeros(len(r0_per_λ0))
    Powers=np.zeros(len(r0_per_λ0))
    zf_2ωpt=np.zeros(len(r0_per_λ0))
    zf_2ωΔp=np.zeros(len(r0_per_λ0))
    zf_2ωpc=np.zeros(len(r0_per_λ0))
    zf_2ωpr=np.zeros(len(r0_per_λ0))
    zf_kRe_Π_per_c=np.zeros(len(r0_per_λ0))
    zf_kIm_Π_per_c=np.zeros(len(r0_per_λ0))
    zf_kωSt=np.zeros(len(r0_per_λ0))
    zf_kωΔS=np.zeros(len(r0_per_λ0))
    # n_grad_Wt=np.zeros(len(r0_per_λ0))
    n_grad_ΔW=np.zeros(len(r0_per_λ0))
    n_grad_Wc=np.zeros(len(r0_per_λ0))
    n_grad_Wr=np.zeros(len(r0_per_λ0))
    # n_2ωpt=np.zeros(len(r0_per_λ0))
    n_2ωΔp=np.zeros(len(r0_per_λ0))
    n_2ωpc=np.zeros(len(r0_per_λ0))
    n_2ωpr=np.zeros(len(r0_per_λ0))
    # n_kRe_Π_per_c=np.zeros(len(r0_per_λ0))
    n_kIm_Π_per_c=np.zeros(len(r0_per_λ0))
    n_kωSt=np.zeros(len(r0_per_λ0))
    n_kωΔS=np.zeros(len(r0_per_λ0))
    # nz_2ωpt=np.zeros(len(r0_per_λ0))
    nz_2ωΔp=np.zeros(len(r0_per_λ0))
    nz_2ωpc=np.zeros(len(r0_per_λ0))
    nz_2ωpr=np.zeros(len(r0_per_λ0))
    # nz_kRe_Π_per_c=np.zeros(len(r0_per_λ0))
    nz_kIm_Π_per_c=np.zeros(len(r0_per_λ0))
    nz_kωSt=np.zeros(len(r0_per_λ0))
    nz_kωΔS=np.zeros(len(r0_per_λ0))
    
    n_grad_Wt=np.zeros(len(r0_per_λ0))
    n_2ωpt=np.zeros(len(r0_per_λ0))
    n_kRe_Π_per_c=np.zeros(len(r0_per_λ0))
    nz_2ωpt=np.zeros(len(r0_per_λ0))
    nz_kRe_Π_per_c=np.zeros(len(r0_per_λ0))


    for i in tqdm(range(len(r0_per_λ0))):
        m=window #size of plot
        N=resolution   #number of points in each direction
        ω=2*np.pi*c0/λ0 #frequency [Hz]

        x=λ0*np.linspace(-m,m,int(N))   #x coordinates
        y=λ0*np.linspace(-m,m,int(N))   #y coordinates
        z=λ0*np.linspace(0,m,int(3))   #z coordinates
        # dx=x[1]-x[0]    #x step size
        # dy=y[1]-y[0]    #y step size
        # dz=z[1]-z[0]    #z step size
        X, Y, Z = np.meshgrid(x, y, z,indexing='ij')  #3D grid of coordinates
        # generate fields 
        (Ex,Ey,Ez,Eφ,ϵ)=get_E(X,Y,Z,λ0,r0_per_λ0[i]*λ0,nr[i],Vr[i],nl[i],Vl[i],R=R,L=L,ℓ=1)   #electric field
        (Hx,Hy,Hz,Hφ,μ)=get_H(X,Y,Z,λ0,r0_per_λ0[i]*λ0,nr[i],Vr[i],nl[i],Vl[i],R=R,L=L,ℓ=1)   #magnetic field

        E=(Ex,Ey,Ez)        #electric field vector
        H=(Hx,Hy,Hz)        #magnetic field vector
        # (ϵ,E,μ,H,X,Y,Z)=get(r0_per_λ0[i]*λ0,λ0,n1,n2,R=R,L=L,fld_plot=False,zix=0,window=window,resolution=resolution)
        dx=np.diff(X[:,0,0])[0]
        dy=np.diff(Y[0,:,0])[0]
        dz=np.diff(Z[0,0,:])[0]
        c=1/np.sqrt(ϵ*μ)    #speed of light
        k=ω/c               #wave number
        #calculate energy densities
        Wt=get_Wt(ϵ,E,μ,H)   #total energy density
        ΔW=get_ΔW(ϵ,E,μ,H)   #difference between electric and magnetic energy densities
        Wc=get_Wc(ϵ,E,μ,H)   #circularly polarized energy density
        Wr=get_Wr(ϵ,E,μ,H)   #diagonal energy density

        #gradients of energy densities
        grad_Wt=np.asarray(np.gradient(Wt,dx,dy,dz))  #gradient of total energy density
        grad_ΔW=np.asarray(np.gradient(ΔW,dx,dy,dz))  #gradient of difference between electric and magnetic energy densities
        grad_Wc=np.asarray(np.gradient(Wc,dx,dy,dz))  #gradient of circularly polarized energy density
        grad_Wr=np.asarray(np.gradient(Wr,dx,dy,dz))  #gradient of diagonal energy density

        #calculate momentum densities
        ωpt=get_ωpt(ω,ϵ,E,μ,H,dx,dy,dz)   #total momentum density
        ωΔp=get_ωΔp(ω,ϵ,E,μ,H,dx,dy,dz)   #difference between electric and magnetic momentum densities
        ωpc=get_ωpc(ω,ϵ,E,μ,H,dx,dy,dz)   #circularly polarized momentum density
        ωpr=get_ωpr(ω,ϵ,E,μ,H,dx,dy,dz)   #diagonal momentum density

        #calculate poynting vector and spin densities
        Re_Π=get_Re_Π(E,H)  #real part of poynting vector
        Im_Π=get_Im_Π(E,H)  #imaginary part of poynting vector
        ωSt=get_ωSt(ϵ,E,μ,H)  #total spin density
        ωΔS=get_ωΔS(ϵ,E,μ,H)  #difference between electric and magnetic spin densities
        Power=np.sum(Re_Π[2][:,:,zix]*dx*dy)

        #find maximums
        if inside: factor=1
        else:
            R_squared=X**2+Y**2
            index=np.tile(np.reshape(R_squared>(r0_per_λ0[i]*λ0)**2,(1,np.shape(X)[0],np.shape(Y)[1],np.shape(Z)[2])),(3,1,1,1))
            factor=np.where(index,1,0)
        f_Wt[i]=np.max(np.abs(factor*Wt))/Power
        f_ΔW[i]=np.max(np.abs(factor*ΔW))/Power
        f_Wc[i]=np.max(np.abs(factor*Wc))/Power
        f_Wr[i]=np.max(np.abs(factor*Wr))/Power
        f_grad_Wt[i]=np.max(tan_norm(factor*grad_Wt))/Power
        f_grad_ΔW[i]=np.max(tan_norm(factor*grad_ΔW))/Power
        f_grad_Wc[i]=np.max(tan_norm(factor*grad_Wc))/Power
        f_grad_Wr[i]=np.max(tan_norm(factor*grad_Wr))/Power
        f_2ωpt[i]=np.max(tan_norm(factor*ωpt))/Power
        f_2ωΔp[i]=np.max(tan_norm(factor*ωΔp))/Power
        f_2ωpc[i]=np.max(tan_norm(factor*ωpc))/Power
        f_2ωpr[i]=np.max(tan_norm(factor*ωpr))/Power
        f_kRe_Π_per_c[i]=np.max(tan_norm(factor*k*Re_Π/c))/Power
        f_kIm_Π_per_c[i]=np.max(tan_norm(factor*k*Im_Π/c))/Power
        f_kωSt[i]=np.max(tan_norm(factor*k*ωSt))/Power
        f_kωΔS[i]=np.max(tan_norm(factor*k*ωΔS))/Power
        # Powers[i]=nr[i]*np.max(np.abs((Wt/c)[:,:,zix]))/Power
        zf_2ωpt[i]=np.max((factor*ωpt[2])*(factor*ωpt[2]))/Power
        zf_2ωΔp[i]=np.max((factor*ωΔp[2])*(factor*ωΔp[2]))/Power
        zf_2ωpc[i]=np.max((factor*ωpc[2])*(factor*ωpc[2]))/Power
        zf_2ωpr[i]=np.max((factor*ωpr[2])*(factor*ωpr[2]))/Power
        zf_kRe_Π_per_c[i]=np.max((factor*k*Re_Π[2]/c)*(factor*k*Re_Π[2]/c))/Power
        zf_kIm_Π_per_c[i]=np.max((factor*k*Im_Π[2]/c)*(factor*k*Im_Π[2]/c))/Power
        zf_kωSt[i]=np.max((factor*k*ωSt[2])*(factor*k*ωSt[2]))/Power
        zf_kωΔS[i]=np.max((factor*k*ωΔS[2])*(factor*k*ωΔS[2]))/Power
        # n_grad_Wt[i]=np.max(tan_norm(factor*grad_Wt))/Power
        Powers[i]=np.max(tan_norm(factor*grad_Wt))
        n_grad_Wt[i]=np.max(tan_norm(factor*grad_Wt))/Powers[i]#/np.max(tan_norm(factor*grad_Wt))
        n_grad_ΔW[i]=np.max(tan_norm(factor*grad_ΔW))/Powers[i]#/np.max(tan_norm(factor*grad_Wt))
        n_grad_Wc[i]=np.max(tan_norm(factor*grad_Wc))/Powers[i]#/np.max(tan_norm(factor*grad_Wt))
        n_grad_Wr[i]=np.max(tan_norm(factor*grad_Wr))/Powers[i]#/np.max(tan_norm(factor*grad_Wt))
        # n_2ωpt[i]=np.max(v_norm(factor*ωpt))/Power
        Powers[i]=np.max(v_norm(factor*ωpt))
        n_2ωpt[i]=np.max(v_norm(factor*ωpt))/Powers[i]#/np.max(v_norm(factor*ωpt))
        n_2ωΔp[i]=np.max(v_norm(factor*ωΔp))/Powers[i]#/np.max(v_norm(factor*ωpt))
        n_2ωpc[i]=np.max(v_norm(factor*ωpc))/Powers[i]#/np.max(v_norm(factor*ωpt))
        n_2ωpr[i]=np.max(v_norm(factor*ωpr))/Powers[i]#/np.max(v_norm(factor*ωpt))
        nz_2ωpt[i]=np.max(z_norm(factor*ωpt))/Powers[i]#/np.max(v_norm(factor*ωpt))
        nz_2ωΔp[i]=np.max(z_norm(factor*ωΔp))/Powers[i]#/np.max(v_norm(factor*ωpt))
        nz_2ωpc[i]=np.max(z_norm(factor*ωpc))/Powers[i]#/np.max(v_norm(factor*ωpt))
        nz_2ωpr[i]=np.max(z_norm(factor*ωpr))/Powers[i]#/np.max(v_norm(factor*ωpt))
        # n_kRe_Π_per_c[i]=np.max(v_norm(factor*k*Re_Π/c))/Power
        Powers[i]=np.max(v_norm(factor*k*Re_Π/c))
        n_kRe_Π_per_c[i]=np.max(v_norm(factor*k*Re_Π/c))/Powers[i]#/np.max(v_norm(factor*k*Re_Π))
        n_kIm_Π_per_c[i]=np.max(v_norm(factor*k*Im_Π/c))/Powers[i]#/n_kRe_Π_per_c[i]/np.max(v_norm(factor*Re_Π))
        n_kωSt[i]=np.max(v_norm(factor*k*ωSt))/Powers[i]#/np.max(v_norm(factor*Re_Π/c))
        n_kωΔS[i]=np.max(v_norm(factor*k*ωΔS))/Powers[i]#/np.max(v_norm(factor*Re_Π/c))
        nz_kRe_Π_per_c[i]=np.max(z_norm(factor*k*Re_Π/c))/Powers[i]#/np.max(v_norm(factor*k*Re_Π/c))
        # nz_kIm_Π_per_c[i]=np.max(v_norm(factor*Im_Π[2]))/np.max(v_norm(factor*Re_Π[2]/c))
        nz_kωSt[i]=np.max(z_norm(factor*k*ωSt))/Powers[i]#/nz_kRe_Π_per_c[i]#/np.max(v_norm(factor*Re_Π[2]/c))
        nz_kωΔS[i]=np.max(z_norm(factor*k*ωΔS))/Powers[i]#/nz_kRe_Π_per_c[i]#/np.max(v_norm(factor*Re_Π[2]/c))
        Powers[i]=Power

        if plot_all:
            (Wt,ΔW,Wc,Wr,grad_Wt,grad_ΔW,grad_Wc,grad_Wr,ωpt,ωΔp,ωpc,ωpr,Re_Π,Im_Π,ωSt,ωΔS)=observables(λ0,ϵ,E,μ,H,X,Y,Z)
            plot_forces(r0_per_λ0[i]*λ0,λ0,ϵ,μ,Wt,ΔW,Wc,Wr,grad_Wt,grad_ΔW,grad_Wc,grad_Wr,ωpt,ωΔp,ωpc,ωpr,Re_Π,Im_Π,ωSt,ωΔS,X,Y,Z,w=3)
    
        
    unit = 1e-30*1e12 #conversion factor to fN/(mW·Å³)
    if unscaled:
        # plot transverse components of force densities
        fig, ax = plt.subplots(ncols=3,figsize=(16,4))  
        ax[0].set_title('maximal gradient force densities')
        ax[0].plot(2*np.pi*r0_per_λ0,unit*f_grad_Wt,label=r'$\grad W$')
        ax[0].plot(2*np.pi*r0_per_λ0,unit*f_grad_ΔW,label=r'$\grad (W_\mathrm{e}-W_\mathrm{m})$')
        ax[0].plot(2*np.pi*r0_per_λ0,unit*f_grad_Wc,label=r'$\grad W_c$')
        ax[0].plot(2*np.pi*r0_per_λ0,unit*f_grad_Wr,label=r'$\grad W_\chi$')

        ax[1].set_title('maximal momentum force densities')
        ax[1].plot(2*np.pi*r0_per_λ0,unit*f_2ωpt,label=r'$2\omega\vec{p}_t$')
        ax[1].plot(2*np.pi*r0_per_λ0,unit*f_2ωΔp,label=r'$2\omega(\vec{p}_e-\vec{p}_m)$')
        ax[1].plot(2*np.pi*r0_per_λ0,unit*f_2ωpc,label=r'$2\omega\vec{p}_c$')
        ax[1].plot(2*np.pi*r0_per_λ0,unit*f_2ωpr,label=r'$2\omega\vec{p}_\chi$')

        ax[2].set_title('maximal recoil force densities')
        ax[2].plot(2*np.pi*r0_per_λ0,unit*f_kRe_Π_per_c,label=r'$k\Re(\vec{\varPi})/c$')
        ax[2].plot(2*np.pi*r0_per_λ0,unit*f_kωΔS,label=r'$k\omega(\vec{S}_e-\vec{S}_m)$')
        ax[2].plot(2*np.pi*r0_per_λ0,unit*f_kωSt,label=r'$k\omega\vec{S}$')
        ax[2].plot(2*np.pi*r0_per_λ0,unit*f_kIm_Π_per_c,label=r'$k\Im(\vec{\varPi})/c$')

        for i in range(3):
            ax[i].set_xlim(xmin=k0r0min,xmax=k0r0max)
            ax[i].set_ylim(ymin=1e-9,ymax=1e-5)
            ax[i].legend()
            ax[i].set_yscale('log')
            ax[i].grid(color='black', linestyle='--')   
        fig.supxlabel(r'$k_0r_0$',y=-0.05);fig.supylabel(r'force density [fN/(mW·Å³)]',x=0.08)
        plt.show(fig)

        #plot longitudinal components
        fig, ax = plt.subplots(ncols=2,figsize=(14,4))

        ax[0].set_title('maximal longitudinal force densities')
        ax[0].plot(2*np.pi*r0_per_λ0,unit*zf_2ωpt,label=r'$2\omega p_{t,z}$')
        ax[0].plot(2*np.pi*r0_per_λ0,unit*zf_2ωΔp,label=r'$2\omega(p_{e,z}-p_{m,z})$')
        ax[0].plot(2*np.pi*r0_per_λ0,unit*zf_2ωpc,label=r'$2\omega p_{c,z}$')
        ax[0].plot(2*np.pi*r0_per_λ0,unit*zf_2ωpr,label=r'$2\omega p_{\chi,z}$')

        ax[1].set_title('maximal longitudinal recoil force densities')
        ax[1].plot(2*np.pi*r0_per_λ0,unit*zf_kRe_Π_per_c,label=r'$k\Re(\vec{\varPi})/c_z$')
        ax[1].plot(2*np.pi*r0_per_λ0,unit*zf_kωΔS,label=r'$k\omega(S_{e,z}-S_{m,z})$')
        ax[1].plot(2*np.pi*r0_per_λ0,unit*zf_kωSt,label=r'$k\omega S_z$')
        ax[1].plot(2*np.pi*r0_per_λ0,unit*zf_kIm_Π_per_c,label=r'$k\Im(\vec{\varPi})/c_z$')

        for i in range(2):
            ax[i].set_xlim(xmin=k0r0min,xmax=k0r0max)
            ax[i].set_ylim(ymin=1e-8,ymax=1e2)
            ax[i].legend()
            ax[i].set_yscale('log')
            ax[i].grid(color='black', linestyle='--')
        fig.supxlabel(r'$k_0r_0$',y=-0.05);fig.supylabel(r'$force density [fN/(mW·Å³)]',x=0.08)
        plt.show(fig)
    if scaled:
        # figure of normalized forces
        fig, ax = plt.subplots(ncols=3,figsize=(16,4))
        ax[0].set_title('maximal gradient force densities')
        ax[0].plot(2*np.pi*r0_per_λ0,n_grad_Wt,label=r'$|\grad W|/|\grad W|$')
        ax[0].plot(2*np.pi*r0_per_λ0,n_grad_ΔW,label=r'$|\grad (W_\mathrm{e}-W_\mathrm{m})|/|\grad W|$')
        ax[0].plot(2*np.pi*r0_per_λ0,n_grad_Wc,label=r'$|\grad W_c|/|\grad W|$')
        ax[0].plot(2*np.pi*r0_per_λ0,n_grad_Wr,label=r'$|\grad W_\chi|/|\grad W|$')
        # ax[0].plot(2*np.pi*r0_per_λ0,n_grad_ΔW+n_grad_Wc+n_grad_Wr,label=r'$|\grad (\sum|W_i|)/|\grad W|$')

        ax[1].set_title('maximal momentum force densities')
        ax[1].plot(2*np.pi*r0_per_λ0,n_2ωpt,label=r'$|\vec{p}_t|/|\vec{p}_t|$')
        # ax[1].plot(2*np.pi*r0_per_λ0,nz_2ωpt,color=ax[1].lines[-1].get_color(),linestyle='--')
        ax[1].plot(2*np.pi*r0_per_λ0,n_2ωΔp,label=r'$|\vec{p}_e-\vec{p}_m|/|\vec{p}_t|$')
        # ax[1].plot(2*np.pi*r0_per_λ0,nz_2ωΔp,color=ax[1].lines[-1].get_color(),linestyle='--')
        ax[1].plot(2*np.pi*r0_per_λ0,n_2ωpc,label=r'$|\vec{p}_c|/|\vec{p}_t|$')
        # ax[1].plot(2*np.pi*r0_per_λ0,nz_2ωpc,color=ax[1].lines[-1].get_color(),linestyle='--')
        ax[1].plot(2*np.pi*r0_per_λ0,n_2ωpr,label=r'$|\vec{p}_\chi|/|\vec{p}_t|$')
        # ax[1].plot(2*np.pi*r0_per_λ0,nz_2ωpr,color=ax[1].lines[-1].get_color(),linestyle='--')
        # ax[1].plot(2*np.pi*r0_per_λ0,Powers,label=r'$P$')
        # ax[1].plot(2*np.pi*r0_per_λ0,n_2ωΔp+n_2ωpc+n_2ωpr,label=r'$|\sum|\vec{p}_i||/|\vec{p}_t|$')

        ax[2].set_title('maximal recoil force densities')
        ax[2].plot(2*np.pi*r0_per_λ0,n_kRe_Π_per_c,label=r'$|\Re\{\vec{\varPi}\}|/|\Re\{\vec{\varPi}\}|$')
        # ax[2].plot(2*np.pi*r0_per_λ0,nz_kRe_Π_per_c,color=ax[2].lines[-1].get_color(),linestyle='--')
        ax[2].plot(2*np.pi*r0_per_λ0,n_kωΔS,label=r'$|\vec{S}_e-\vec{S}_m|/|\Re\{\vec{\varPi}\}|$')
        # ax[2].plot(2*np.pi*r0_per_λ0,nz_kωΔS,color=ax[2].lines[-1].get_color(),linestyle='--')
        ax[2].plot(2*np.pi*r0_per_λ0,n_kωSt,label=r'$|\vec{S}|/|\Re\{\vec{\varPi}\}|$')
        # ax[2].plot(2*np.pi*r0_per_λ0,nz_kωSt,color=ax[2].lines[-1].get_color(),linestyle='--')
        ax[2].plot(2*np.pi*r0_per_λ0,n_kIm_Π_per_c,label=r'$|\Im\{\vec{\varPi}\}|/|\Re\{\vec{\varPi}\}|$')
        # ax[2].plot(2*np.pi*r0_per_λ0,nz_kIm_Π_per_c,color=ax[2].lines[-1].get_color(),linestyle='--')
        # ax[2].plot(2*np.pi*r0_per_λ0,Powers,label=r'$P$')

        for i in range(3):
            ax[i].set_xlim(xmin=k0r0min,xmax=k0r0max)
            ax[i].set_ylim(ymin=0,ymax=1.2)
            ax[i].legend()
            # ax[i].set_yscale('log')
            ax[i].grid(color='black', linestyle='--')
        fig.supxlabel(r'$k_0r_0$',y=-0.05);fig.supylabel(r'relative strength of the components',x=0.08)
        plt.show(fig)
    if output:return (np.mean(n_grad_ΔW),np.mean(n_grad_Wc),np.mean(n_grad_Wr),np.mean(n_2ωΔp),np.mean(n_2ωpc),np.mean(n_2ωpr),np.mean(n_kωSt),np.mean(n_kωΔS),np.mean(n_kIm_Π_per_c))
    else: return


def get(r0,λ0,n_in,n_out,R=1,L=0,fld_plot=False,neff_fig=False,zix=0,window=None,resolution=250,ℓ=1):
    if window is None: m=1.5*r0/λ0
    else: m=window #size of plot
    N=resolution   #number of points in each direction
    step=int(N/8)  #step size for quiver plot

    material(n_in=n_in,n_out=n_out)
    (_,nr,Vr)=get_neff(rmin=0,rmax=2,N=500,r0_over_λ0=r0/λ0,neff_fig=neff_fig,ℓ=ℓ)
    (_,nl,Vl)=get_neff(rmin=0,rmax=2,N=500,r0_over_λ0=r0/λ0,neff_fig=False,ℓ=-ℓ)
    x=λ0*np.linspace(-m,m,int(N))   #x coordinates
    y=λ0*np.linspace(-m,m,int(N))   #y coordinates
    z=λ0*np.linspace(0,m,int(3))   #z coordinates
    X, Y, Z = np.meshgrid(x, y, z,indexing='ij')  #3D grid of coordinates

    # generate fields 
    (Ex,Ey,Ez,_,ϵ)=get_E(X,Y,Z,λ0,r0,nr,Vr,nl,Vl,R=R,L=L,ℓ=ℓ)   #electric field
    (Hx,Hy,Hz,_,μ)=get_H(X,Y,Z,λ0,r0,nr,Vr,nl,Vl,R=R,L=L,ℓ=ℓ)   #magnetic field

    E=(Ex,Ey,Ez)        #electric field vector
    H=(Hx,Hy,Hz)        #magnetic field vector
    if fld_plot: plt(r0,λ0,E,H,X,Y,zix=zix,step=step)
    return (ϵ,E,μ,H,X,Y,Z)

def plot_EH(r0,λ0,E,H,X,Y,zix=0,step=None,w=10,pdf=False,pdfname="figure",c_bar=True,arrows=8):
    step=int(len(X)/arrows)
    fig, ax = plt.subplots(ncols=4,figsize=(11,3),constrained_layout=True)  #create figure with plots of E/|E| and H/|H| 
    pltvector(X,Y,zix,np.real(E)/v_norm(E),ax[0],r0,λ0,zscale=1,scale=w,step=step,title=r'$\Re(\vec{E})/|\vec{E}|$')
    pltvector(X,Y,zix,np.imag(E)/v_norm(E),ax[1],r0,λ0,zscale=1,scale=w,step=step,title=r'$\Im(\vec{E})/|\vec{E}|$')
    pltvector(X,Y,zix,np.real(H)/v_norm(H),ax[2],r0,λ0,zscale=1,scale=w,step=step,title=r'$\Re(\vec{H})/|\vec{H}|$')
    (pcol,_)=pltvector(X,Y,zix,np.imag(H)/v_norm(H),ax[3],r0,λ0,zscale=1,scale=w,step=step,title=r'$\Im(\vec{H})/|\vec{H}|$',out=True)
    fig.supxlabel(r'$x/\lambda_0$');fig.supylabel(r'$y/\lambda_0$')#,x=0.08)
    if c_bar:
        fig.colorbar(pcol, ticks=[-1,0,1],pad=0.08,aspect=11,shrink=0.8)
    if pdf:plt.savefig("EH_"+pdfname,format='pdf')
    else:plt.show(fig)

def plot_forces(r0,λ0,ϵ,μ,Wt,ΔW,Wc,Wr,grad_Wt,grad_ΔW,grad_Wc,grad_Wr,ωpt,ωΔp,ωpc,ωpr,Re_Π,Im_Π,ωSt,ωΔS,X,Y,Z,zix=0,w=4,arrows=None):
    dx=np.diff(X[:,0,0])[0]
    dy=np.diff(Y[0,:,0])[0]
    dz=np.diff(Z[0,0,:])[0]
    c=1/np.sqrt(ϵ*μ)            #speed of light in the medium [m/s]
    ω=2*np.pi*c0/λ0             #frequency [Hz]
    k=ω/c                       #wave number [1/m]
    scale=w*np.max((tan_norm(grad_Wt),tan_norm(grad_ΔW),tan_norm(grad_Wc),tan_norm(grad_Wr),tan_norm(ωpt),tan_norm(ωΔp),tan_norm(ωpc),tan_norm(ωpr),tan_norm(k*Re_Π/c),tan_norm(k*Im_Π/c),tan_norm(k*ωSt),tan_norm(k*ωΔS)))
    zscale=np.max((ωpt[2],ωΔp[2],ωpc[2],ωpr[2],k*Re_Π[2]/c,k*Im_Π[2]/c,k*ωSt[2],k*ωΔS[2]))
    if arrows is None:step=int(len(X)/8)
    else: step=int(len(X)/arrows)
    Wmax=np.max((Wt,ΔW,Wc,Wr)) #maximum value of energy densities
    # scale=3*np.max((tan_norm(grad_Wt),tan_norm(grad_ΔW),tan_norm(grad_Wc),tan_norm(grad_Wr),tan_norm(ωpt),tan_norm(ωΔp),tan_norm(ωpc),tan_norm(ωpr)))
    #Figures gradients of energy densities
    fig, ax = plt.subplots(ncols=4,figsize=(16,4)) 
    pltvector(X,Y,zix,grad_Wt,ax[0],r0,λ0,W=Wt,zscale=Wmax,scale=scale,step=step,title=r'$\grad W$')
    pltvector(X,Y,zix,grad_ΔW,ax[1],r0,λ0,W=ΔW,zscale=Wmax,scale=scale,step=step,title=r'$\grad (W_\mathrm{e}-W_\mathrm{m})$')
    pltvector(X,Y,zix,grad_Wc,ax[2],r0,λ0,W=Wc,zscale=Wmax,scale=scale,step=step,title=r'$\grad W_\mathrm{c}$')
    pltvector(X,Y,zix,grad_Wr,ax[3],r0,λ0,W=Wr,zscale=Wmax,scale=scale,step=step,title=r'$\grad W_\mathrm{\chi}$')
    fig.supxlabel(r'$x/\lambda_0$');fig.supylabel(r'$y/\lambda_0$',x=0.08)
    plt.show(fig)

    #Figures of momentum densities
    fig, ax = plt.subplots(ncols=4,figsize=(16,4))
    pltvector(X,Y,zix,2*ωpt,ax[0],r0,λ0,scale=scale,zscale=zscale,step=step,title=r'$2\omega\vec{p}_\mathrm{o}$')
    pltvector(X,Y,zix,2*ωΔp,ax[1],r0,λ0,scale=scale,zscale=zscale,step=step,title=r'$2\omega(\vec{p}_\mathrm{e}-\vec{p}_\mathrm{m})$')
    pltvector(X,Y,zix,2*ωpc,ax[2],r0,λ0,scale=scale,zscale=zscale,step=step,title=r'$2\omega\vec{p}_\mathrm{c}$')
    pltvector(X,Y,zix,2*ωpr,ax[3],r0,λ0,scale=scale,zscale=zscale,step=step,title=r'$2\omega\vec{p}_\mathrm{\chi}$')
    fig.supxlabel(r'$x/\lambda_0$');fig.supylabel(r'$y/\lambda_0$',x=0.08)
    plt.show(fig)

    #figures of poynting vector and spin densities
    fig, ax = plt.subplots(ncols=4,figsize=(16,4))
    pltvector(X,Y,zix,k*Re_Π/c,ax[0],r0,λ0,scale=scale,zscale=zscale,step=step,title=r'$k\Re(\vec{\varPi})/c$')
    pltvector(X,Y,zix,k*Im_Π/c,ax[1],r0,λ0,scale=scale,zscale=zscale,step=step,title=r'$k\Im(\vec{\varPi})/c$')
    pltvector(X,Y,zix,k*ωSt,ax[2],r0,λ0,scale=scale,zscale=zscale,step=step,title=r'$k\omega\vec{S}$')
    pltvector(X,Y,zix,k*ωΔS,ax[3],r0,λ0,scale=scale,zscale=zscale,step=step,title=r'$k\omega(\vec{S}_\mathrm{e}-\vec{S}_\mathrm{m})$')
    fig.supxlabel(r'$x/\lambda_0$');fig.supylabel(r'$y/\lambda_0$',x=0.08)
    plt.show(fig)

    #figures of curl of poynting vector and spin densities
    fig, ax = plt.subplots(ncols=4,figsize=(16,4))
    pltvector(X,Y,zix,curl(Re_Π/c,dx,dy,dz),ax[0],r0,λ0,scale=scale,zscale=zscale,step=step,title=r'$\grad\times\Re(\vec{\varPi})/c$')
    pltvector(X,Y,zix,curl(Im_Π/c,dx,dy,dz),ax[1],r0,λ0,scale=scale,zscale=zscale,step=step,title=r'$\grad\times\Im(\vec{\varPi})/c$')
    pltvector(X,Y,zix,curl(ωSt,dx,dy,dz),ax[2],r0,λ0,scale=scale,zscale=zscale,step=step,title=r'$\omega\grad\times\vec{S}$')
    pltvector(X,Y,zix,curl(ωΔS,dx,dy,dz),ax[3],r0,λ0,scale=scale,zscale=zscale,step=step,title=r'$\omega\grad\times(\vec{S}_\mathrm{e}-\vec{S}_\mathrm{m})$')
    fig.supxlabel(r'$x/\lambda_0$');fig.supylabel(r'$y/\lambda_0$',x=0.08)
    plt.show(fig)

    # electric and magnetic basis
    fig, ax = plt.subplots(ncols=4,figsize=(16,4))
    pltvector(X,Y,zix,grad_Wt+grad_ΔW,ax[0],r0,λ0,W=Wt+ΔW,zscale=Wmax,scale=scale,step=step,title=r'$\grad W_\mathrm{e}$')
    pltvector(X,Y,zix,grad_Wt-grad_ΔW,ax[1],r0,λ0,W=Wt-ΔW,zscale=Wmax,scale=scale,step=step,title=r'$\grad W_\mathrm{m}$')
    pltvector(X,Y,zix,2*(ωpt+ωΔp),ax[2],r0,λ0,scale=scale,zscale=zscale,step=step,title=r'$2\omega\vec{p}_\mathrm{e}$')
    pltvector(X,Y,zix,2*(ωpt-ωΔp),ax[3],r0,λ0,scale=scale,zscale=zscale,step=step,title=r'$2\omega\vec{p}_\mathrm{m}$')
    fig.supxlabel(r'$x/\lambda_0$');fig.supylabel(r'$y/\lambda_0$',x=0.08)
    plt.show(fig)

    fig, ax = plt.subplots(ncols=4,figsize=(16,4))
    pltvector(X,Y,zix,k*(ωSt+ωΔS),ax[0],r0,λ0,scale=scale,zscale=zscale,step=step,title=r'$k\omega\vec{S}_\mathrm{e}$')
    pltvector(X,Y,zix,k*(ωSt-ωΔS),ax[1],r0,λ0,scale=scale,zscale=zscale,step=step,title=r'$k\omega\vec{S}_\mathrm{m}$')
    pltvector(X,Y,zix,curl(ωSt+ωΔS,dx,dy,dz),ax[2],r0,λ0,scale=scale,zscale=zscale,step=step,title=r'$\grad\times \omega\vec{S}_\mathrm{e}$')
    pltvector(X,Y,zix,curl(ωSt-ωΔS,dx,dy,dz),ax[3],r0,λ0,scale=scale,zscale=zscale,step=step,title=r'$\grad\times \omega\vec{S}_\mathrm{m}$')
    fig.supxlabel(r'$x/\lambda_0$');fig.supylabel(r'$y/\lambda_0$',x=0.08)
    plt.show(fig)



def observables(λ0,ϵ,E,μ,H,X,Y,Z):
    ω=2*np.pi*c0/λ0 #frequency [Hz]
    dx=np.diff(X[:,0,0])[0]
    dy=np.diff(Y[0,:,0])[0]
    dz=np.diff(Z[0,0,:])[0]
    #calculate energy densities
    Wt=get_Wt(ϵ,E,μ,H)   #total energy density
    ΔW=get_ΔW(ϵ,E,μ,H)   #difference between electric and magnetic energy densities
    Wc=get_Wc(ϵ,E,μ,H)   #circularly polarized energy density
    Wr=get_Wr(ϵ,E,μ,H)   #diagonal energy density

    #gradients of energy densities
    grad_Wt=np.asarray(np.gradient(Wt,dx,dy,dz))  #gradient of total energy density
    grad_ΔW=np.asarray(np.gradient(ΔW,dx,dy,dz))  #gradient of difference between electric and magnetic energy densities
    grad_Wc=np.asarray(np.gradient(Wc,dx,dy,dz))  #gradient of circularly polarized energy density
    grad_Wr=np.asarray(np.gradient(Wr,dx,dy,dz))  #gradient of diagonal energy density

    #calculate momentum densities
    ωpt=get_ωpt(ω,ϵ,E,μ,H,dx,dy,dz)   #total momentum density
    ωΔp=get_ωΔp(ω,ϵ,E,μ,H,dx,dy,dz)   #difference between electric and magnetic momentum densities
    ωpc=get_ωpc(ω,ϵ,E,μ,H,dx,dy,dz)   #circularly polarized momentum density
    ωpr=get_ωpr(ω,ϵ,E,μ,H,dx,dy,dz)   #diagonal momentum density

    #calculate poynting vector and spin densities
    Re_Π=get_Re_Π(E,H)  #real part of poynting vector
    Im_Π=get_Im_Π(E,H)  #imaginary part of poynting vector
    ωSt=get_ωSt(ϵ,E,μ,H)  #total spin density
    ωΔS=get_ωΔS(ϵ,E,μ,H)  #difference between electric and magnetic spin densities
    return (Wt,ΔW,Wc,Wr,grad_Wt,grad_ΔW,grad_Wc,grad_Wr,ωpt,ωΔp,ωpc,ωpr,Re_Π,Im_Π,ωSt,ωΔS)

def get_amplitudes_from_stokes(P,θ,V):
    Q=np.real(P*np.exp(2j*θ))
    U=np.imag(P*np.exp(2j*θ))
    Q=Q/np.sqrt(Q**2+U**2+V**2)
    U=U/np.sqrt(Q**2+U**2+V**2)
    V=V/np.sqrt(Q**2+U**2+V**2)
    A=np.sqrt((1+Q)/2)
    if A==0:
        B=1
    else:
        B=U/(2*A)-1j*V/(2*A)
    R=(A+1j*B)/np.sqrt(2)
    L=(A-1j*B)/np.sqrt(2)
    return (R,L)



def plot_forces_rescaled(r0,λ0,ϵ,μ,Wt,ΔW,Wc,Wr,grad_Wt,grad_ΔW,grad_Wc,grad_Wr,ωpt,ωΔp,ωpc,ωpr,Re_Π,Im_Π,ωSt,ωΔS,X,Y,Z,zix=0,w=4,nΔW=1,nWc=1,nWr=1,n_2ωΔp=1,n_2ωpc=1,n_2ωpr=1,n_kωSt=1,n_kωΔS=1,n_kIm_Π_per_c=1,arrows=None,basis='pauli',pdf=False,pdfname="figure"):
    dx=np.diff(X[:,0,0])[0]
    dy=np.diff(Y[0,:,0])[0]
    dz=np.diff(Z[0,0,:])[0]
    c=1/np.sqrt(ϵ*μ)            #speed of light in the medium [m/s]
    ω=2*np.pi*c0/λ0             #frequency [Hz]
    k=ω/c                       #wave number [1/m]
    scale=w*np.max((tan_norm(grad_Wt),tan_norm(grad_ΔW)/nΔW,tan_norm(grad_Wc)/nWc,tan_norm(grad_Wr)/nWr,tan_norm(2*ωpt),tan_norm(2*ωΔp)/n_2ωΔp,tan_norm(2*ωpc)/n_2ωpc,tan_norm(2*ωpr)/n_2ωpr,tan_norm(k*Re_Π/c),tan_norm(k*Im_Π/c)/n_kIm_Π_per_c,tan_norm(k*ωSt),tan_norm(k*ωΔS)/n_kωΔS))
    zscale=np.max((np.abs(2*ωpt[2]),np.abs(2*ωΔp[2]/n_2ωΔp),np.abs(2*ωpc[2]/n_2ωpc),np.abs(2*ωpr[2]/n_2ωpr),np.abs(k*ωSt[2]),np.abs(k*ωΔS[2]/n_kωΔS),np.abs(k*Re_Π[2]/c),np.abs(k*Im_Π[2]/c)/n_kIm_Π_per_c))
    if arrows is None:step=int(len(X)/8)
    else: step=int(len(X)/arrows)
    Wmax=np.max((Wt,ΔW,Wc,Wr)) #maximum value of energy densities

    if basis=='pauli':
        grad_titles=(r'$\grad W_0=\grad W$',r'$\grad W_1=\grad (W_\mathrm{e}-W_\mathrm{m})$',r'$-\grad W_2=\grad W_\mathrm{c}$',r'$\grad W_3$')
        pres_titles=(r'$2\omega\vec{p}_0=2\omega\vec{p}_\mathrm{o}$',r'$2\omega\vec{p}_1=2\omega(\vec{p}_\mathrm{e}-\vec{p}_\mathrm{m})$',r'$-2\omega\vec{p}_2=2\omega\vec{p}_\mathrm{c}$',r'$2\omega\vec{p}_3$')
        spin_titles=(r'$k\omega\vec{S}_0=k\omega\vec{S}$',r'$k\omega\vec{S}_1=k\omega(\vec{S}_\mathrm{e}-\vec{S}_\mathrm{m})$',r'$-k\omega\vec{S}_2=k\Re(\vec{\varPi})/c$',r'$k\omega\vec{S}_3=k\Im(\vec{\varPi})/c$')
        grad_forces=(grad_Wt,grad_ΔW/nΔW,grad_Wc/nWc,grad_Wr/nWr)
        energy_dens=(Wt,ΔW/nΔW,Wc/nWc,Wr/nWr)
        pres_forces=(2*ωpt,2*ωΔp/n_2ωΔp,2*ωpc/n_2ωpc,2*ωpr/n_2ωpr)
        spin_forces=(k*ωSt/n_kωSt,k*ωΔS/n_kωΔS,k*Re_Π/c,k*Im_Π/c/n_kIm_Π_per_c)
    elif basis=='em':
        grad_titles=(r'$\grad W_\mathrm{e}$',r'$\grad W_\mathrm{m}$',r'$\grad W_\mathrm{c}$',r'$\grad W_\chi$')
        pres_titles=(r'$2\omega\vec{p}_\mathrm{e}$',r'$2\omega\vec{p}_\mathrm{m}$',r'$2\omega\vec{p}_\mathrm{c}$',r'$2\omega\vec{p}_\mathrm{\chi}$')
        spin_titles=(r'$k\Re(\vec{\varPi})/c$',r'$k\Im(\vec{\varPi})/c$',r'$k\omega\vec{S}_\mathrm{e}$',r'$k\omega\vec{S}_\mathrm{m}$')
        grad_forces=(grad_Wt+grad_ΔW,grad_Wt-grad_ΔW,grad_Wc,grad_Wr)
        energy_dens=(Wt+ΔW,Wt-ΔW,Wc,Wr)
        pres_forces=(2*(ωpt+ωΔp),2*(ωpt-ωΔp),2*ωpc,2*ωpr)
        spin_forces=(k*Re_Π/c,k*Im_Π/c,k*(ωSt+ωΔS),k*(ωSt-ωΔS))
    elif basis=='em_reci':
        grad_titles=(r'$\grad W_\mathrm{e}$',r'$\grad W_\mathrm{m}$',r'$\omega\grad \mathfrak{S}$')
        pres_titles=(r'$2\omega\vec{p}_\mathrm{e}$',r'$2\omega\vec{p}_\mathrm{m}$',r'$2\omega\vec{p}_\mathrm{c}$')
        spin_titles=(r'$k\Re(\vec{\varPi})/c$',r'$k\Im(\vec{\varPi})/c$',r'$k\omega\vec{S}_\mathrm{e}$',r'$k\omega\vec{S}_\mathrm{m}$')
        grad_forces=(grad_Wt+grad_ΔW,grad_Wt-grad_ΔW,grad_Wc)
        energy_dens=(Wt+ΔW,Wt-ΔW,Wc)
        pres_forces=(2*(ωpt+ωΔp),2*(ωpt-ωΔp),2*ωpc)
        spin_forces=(k*Re_Π/c,k*Im_Π/c,k*(ωSt+ωΔS),k*(ωSt-ωΔS))
    

    #Figures gradients of energy densities
    fig, ax = plt.subplots(ncols=len(grad_titles),figsize=(11,3),constrained_layout=True)
    for i in range(len(grad_titles)):
        (pcol,_)=pltvector(X,Y,zix,grad_forces[i],ax[i],r0,λ0,W=energy_dens[i],zscale=Wmax,scale=scale,step=step,title=grad_titles[i],out=True)
    fig.supxlabel(r'$x/\lambda_0$');fig.supylabel(r'$y/\lambda_0$')#,x=0.08)
    cbar=fig.colorbar(pcol, ticks=[-Wmax,0,Wmax],pad=0.08,aspect=11,shrink=0.8)
    cbar.ax.set_yticklabels(['min', '0', 'max'])
    if pdf:plt.savefig("g_F_"+pdfname,format='pdf')
    else:plt.show(fig)

    #Figures of momentum densities
    fig, ax = plt.subplots(ncols=len(pres_titles),figsize=(11,3),constrained_layout=True)
    for i in range(len(pres_titles)):   
        (pcol,_)=pltvector(X,Y,zix,pres_forces[i],ax[i],r0,λ0,scale=scale,zscale=zscale,step=step,title=pres_titles[i],out=True)
    fig.supxlabel(r'$x/\lambda_0$');fig.supylabel(r'$y/\lambda_0$')#,x=0.08)
    cbar=fig.colorbar(pcol, ticks=[-zscale,0,zscale],pad=0.08,aspect=11,shrink=0.8)
    cbar.ax.set_yticklabels(['min', '0', 'max'])
    if pdf:plt.savefig("p_F_"+pdfname,format='pdf')
    else:plt.show(fig)

    #figures of poynting vector and spin densities
    fig, ax = plt.subplots(ncols=len(spin_titles),figsize=(11,3),constrained_layout=True)
    for i in range(len(spin_titles)):
        (pcol,_)=pltvector(X,Y,zix,spin_forces[i],ax[i],r0,λ0,scale=scale,zscale=zscale,step=step,title=spin_titles[i],out=True)
    fig.supxlabel(r'$x/\lambda_0$');fig.supylabel(r'$y/\lambda_0$')#,x=0.08)
    cbar=fig.colorbar(pcol, ticks=[-zscale,0,zscale],pad=0.08,aspect=11,shrink=0.8)
    cbar.ax.set_yticklabels(['min', '0', 'max'])
    if pdf:plt.savefig("r_F_"+pdfname,format='pdf')
    else:plt.show(fig)