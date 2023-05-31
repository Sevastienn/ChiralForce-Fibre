import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, optimize
from scipy.special import jv, hankel1 as hv,jvp, h1vp as hvp
from scipy.linalg import null_space

# One can set different material default glass and air
def material(n_in=1.45,n_out=1):
    global n1; n1=n_in  # Refractive index or sqrt(ϵ₁) inside
    global n2; n2=n_out # Refractive index or sqrt(ϵ₂) outside

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# Boundary condition matrix
def A(neff,r0_per_λ0):
    k0_r0=2*np.pi*r0_per_λ0 #k0·r0
    kz_r0=neff*k0_r0        #kz·r0
    k1_r0=n1*k0_r0          #k1·r0
    k2_r0=n2*k0_r0          #k2·r0
    κ1_r0=np.emath.sqrt(n1**2-neff**2)*k0_r0    #κ1·r0
    κ2_r0=np.emath.sqrt(n2**2-neff**2)*k0_r0    #κ2·r0
    with np.errstate(divide='ignore', invalid='ignore'): 
        J=jv(1,κ1_r0)
        Jp=jvp(1,κ1_r0)
        H=hv(1,κ2_r0)
        Hp=hvp(1,κ2_r0)
        a11=n2*J
        a12=0j*r0_per_λ0
        a13=-n1*H
        a14=0j*r0_per_λ0
        a21=n2*(kz_r0/((κ1_r0)**2))*J
        a22=1j*n2*(k1_r0/κ1_r0)*Jp
        a23=-n1*(kz_r0/((κ2_r0)**2))*H
        a24=-1j*n1*(k2_r0/κ2_r0)*Hp
        a31=0j*r0_per_λ0
        a32=J
        a33=0j*r0_per_λ0
        a34=-H
        a41=-1j*(k1_r0/κ1_r0)*Jp
        a42=(kz_r0/((κ1_r0)**2))*J
        a43=1j*(k2_r0/κ2_r0)*Hp
        a44=-(kz_r0/((κ2_r0)**2))*H
    output=np.array([[a11,a12,a13,a14],[a21,a22,a23,a24],[a31,a32,a33,a34],[a41,a42,a43,a44]],dtype=complex)
    # output=np.array([[a11,a21,a31,a41],[a12,a22,a32,a42],[a31,a23,a33,a43],[a14,a24,a34,a44]],dtype=complex)
    return output
from IPython.display import display, Math
def print_matrix(array):
    matrix = ''
    for row in array:
        try:
            for number in row:
                matrix += f'{number}&'
        except TypeError:
            matrix += f'{row}&'
        matrix = matrix[:-1] + r'\\'
    display(Math(r'\begin{bmatrix}'+matrix+r'\end{bmatrix}'))
# log det of the matrix
def logdetA(neff,r0_per_λ0):
    a=A(neff,r0_per_λ0)
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

def get_neff(rmin=0,rmax=2,N=1000,neff_fig=False,r0_over_λ0=None):    
    r0_per_λ0=np.linspace(rmin,rmax,N+1,endpoint=False)[1:]
    neff=np.linspace(n2,n1,N+1,endpoint=False)[1:]
    Ro, Ne = np.meshgrid(r0_per_λ0, neff)
    logdetM=logdetA(Ne,Ro)

    # Finds n_eff
    n=np.zeros_like(r0_per_λ0)
    for i in range(len(r0_per_λ0)):
        try:
            n[i]=neff[np.max(signal.find_peaks(-logdetM[:,i])[0])]
        except ValueError:  #raised if there are no peaks
            n[i]=0

    # Calculates amplitudes    
    M=np.transpose(A(n,r0_per_λ0),(2,0,1))
    (w,v)=np.linalg.eig(M)
    argus=np.argmin(np.abs(w),axis=1)
    W=w[np.arange(N),argus]
    V=v[np.arange(N),:,argus]


    # Plots n_eff
    if neff_fig:
        plt.pcolormesh(2*np.pi*Ro,Ne,-logdetM,cmap='Greys')
        plt.xlim(xmax=2*np.pi*rmax,xmin=rmin) 
        plt.plot(2*np.pi*r0_per_λ0,n,label=r'$n_\mathrm{eff}(k_0r_0)$')
        # plt.plot(2*np.pi*r0_per_λ0,n*g,label=r'has eigenvector')
        plt.ylim(n2,n1)
        plt.colorbar(label=r'$-\log_{10}|\det{M}|$')
        plt.grid(color='black', linestyle='--')
        plt.ylabel(r'$n_\mathrm{eff}=\lambda_0/\lambda_z$')
        plt.xlabel(r'$k_0r_0=2\pi r_0/\lambda_0$')
        plt.legend()
        plt.show()

    
    
    # plt.grid(color='black', linestyle='--')
    # plt.plot(2*np.pi*r0_per_λ0,(np.abs(np.linalg.det(M))),label=r'$|\det{M}|$')
    # plt.plot(2*np.pi*r0_per_λ0,(np.abs(w)),label=[f"$|\lambda_1|$",f"$|\lambda_2|$",f"$|\lambda_3|$",f"$|\lambda_4|$",])
    # plt.plot(2*np.pi*r0_per_λ0,(np.abs(W)),label=r"$|\lambda_\mathrm{min}|$", linestyle='--')
    # plt.yscale('log')
    # # plt.ylim(ymin=-10,ymax=5) 
    # plt.legend()
    # plt.show()
    if r0_over_λ0 is None: 
        return (r0_per_λ0,n,V)
    else:
        idx=int(np.argwhere(r0_per_λ0==find_nearest(r0_per_λ0,r0_over_λ0)))
        return (r0_per_λ0[idx],n[idx],V[idx])

def get_E(x,y,z,λ0,r0,n_in=1.45,n_out=1):
    material(n_in,n_out)
    (r0_per_λ0,n,V)=get_neff(rmin=0,rmax=int(np.ceil(r0/λ0)),N=1000,r0_over_λ0=r0/λ0)

    r=np.sqrt(x**2+y**2)
    φ=np.arctan2(y,x)
    k0_r=2*np.pi*r/λ0
    kz_z=2*np.pi*n*z/λ0
    κ1_per_k0=np.emath.sqrt(n1**2-n**2) 
    κ2_per_k0=np.emath.sqrt(n2**2-n**2)

    Ep=np.where(r/r0<1,
                (-1j/(n1*κ1_per_k0))*(n*V[0]+1j*n1*V[1])*jv(0,k0_r*κ1_per_k0)*np.exp(1j*kz_z),
                (-1j/(n2*κ2_per_k0))*(n*V[2]+1j*n2*V[3])*hv(0,k0_r*κ2_per_k0)*np.exp(1j*kz_z))
    Em=np.where(r/r0<1,
                (-1j/(n1*κ1_per_k0))*(-n*V[0]+1j*n1*V[1])*jv(2,k0_r*κ1_per_k0)*np.exp(1j*(2*φ+kz_z)),
                (-1j/(n2*κ2_per_k0))*(-n*V[2]+1j*n2*V[3])*hv(2,k0_r*κ2_per_k0)*np.exp(1j*(2*φ+kz_z)))
    Ez=np.where(r/r0<1,
                (np.sqrt(2)/n1)*V[0]*jv(1,k0_r*κ1_per_k0)*np.exp(1j*(φ+kz_z)),
                (np.sqrt(2)/n2)*V[2]*hv(1,k0_r*κ2_per_k0)*np.exp(1j*(φ+kz_z)))

    return ((Ep+Em)/np.sqrt(2),1j*(Ep-Em)/np.sqrt(2),Ez)

def get_H(x,y,z,λ0,r0,n_in=1.45,n_out=1):
    material(n_in,n_out)
    (r0_per_λ0,n,V)=get_neff(rmin=0,rmax=int(np.ceil(r0/λ0)),N=1000,r0_over_λ0=r0/λ0)

    r=np.sqrt(x**2+y**2)
    φ=np.arctan2(y,x)
    k0_r=2*np.pi*r/λ0
    kz_z=2*np.pi*n*z/λ0
    κ1_per_k0=np.emath.sqrt(n1**2-n**2) 
    κ2_per_k0=np.emath.sqrt(n2**2-n**2)

    Hp=np.where(r/r0<1,
                (-1j/(κ1_per_k0))*(n*V[1]-1j*n1*V[0])*jv(0,k0_r*κ1_per_k0)*np.exp(1j*kz_z),
                (-1j/(κ2_per_k0))*(n*V[3]-1j*n1*V[2])*hv(0,k0_r*κ2_per_k0)*np.exp(1j*kz_z))
    Hm=np.where(r/r0<1,
                (-1j/(κ1_per_k0))*(-n*V[1]-1j*n1*V[0])*jv(2,k0_r*κ1_per_k0)*np.exp(1j*(2*φ+kz_z)),
                (-1j/(κ2_per_k0))*(-n*V[3]-1j*n1*V[2])*hv(2,k0_r*κ2_per_k0)*np.exp(1j*(2*φ+kz_z)))
    Hz=np.where(r/r0<1,
                np.sqrt(2)*V[1]*jv(1,k0_r*κ1_per_k0)*np.exp(1j*(φ+kz_z)),
                np.sqrt(2)*V[3]*hv(1,k0_r*κ2_per_k0)*np.exp(1j*(φ+kz_z)))

    return ((Hp+Hm)/np.sqrt(2),1j*(Hp-Hm)/np.sqrt(2),Hz)

# E and H fields
def E(V,n,r0_per_λ0,x_per_r0,y_per_r0,z_per_r0):
    # x,y,z in units of r0
    r_per_r0=np.sqrt(x_per_r0**2+y_per_r0**2)
    k0_r=2*np.pi*r0_per_λ0*r_per_r0
    kz_z=2*np.pi*n*z_per_r0*r0_per_λ0
    φ=np.arctan2(y_per_r0, x_per_r0)
    κ1_per_k0=np.emath.sqrt(n1**2-n**2) 
    κ2_per_k0=np.emath.sqrt(n2**2-n**2)    
    Ep=np.where(r_per_r0<1,
                (-1j/(n1*κ1_per_k0))*(n*V[0]+1j*n1*V[1])*jv(0,k0_r*κ1_per_k0)*np.exp(1j*kz_z),
                (-1j/(n2*κ2_per_k0))*(n*V[2]+1j*n2*V[3])*hv(0,k0_r*κ2_per_k0)*np.exp(1j*kz_z))
    Em=np.where(r_per_r0<1,
                (-1j/(n1*κ1_per_k0))*(-n*V[0]+1j*n1*V[1])*jv(2,k0_r*κ1_per_k0)*np.exp(1j*(2*φ+kz_z)),
                (-1j/(n2*κ2_per_k0))*(-n*V[2]+1j*n2*V[3])*hv(2,k0_r*κ2_per_k0)*np.exp(1j*(2*φ+kz_z)))
    Ez=np.where(r_per_r0<1,
                (np.sqrt(2)/n1)*V[0]*jv(1,k0_r*κ1_per_k0)*np.exp(1j*(φ+kz_z)),
                (np.sqrt(2)/n2)*V[2]*hv(1,k0_r*κ2_per_k0)*np.exp(1j*(φ+kz_z)))
    print((np.sqrt(2)/n1)*V[0]*jv(1,2*np.pi*r0_per_λ0*κ1_per_k0)*np.exp(1j*(kz_z))-(np.sqrt(2)/n2)*V[2]*hv(1,2*np.pi*r0_per_λ0*κ2_per_k0)*np.exp(1j*(kz_z)))
    return ((Ep+Em)/np.sqrt(2),1j*(Ep-Em)/np.sqrt(2),Ez)

def H(V,n,r0_per_λ0,x_per_r0,y_per_r0,z_per_r0):
    # x,y,z in units of r0
    r_per_r0=np.sqrt(x_per_r0**2+y_per_r0**2)
    k0_r=2*np.pi*r0_per_λ0*r_per_r0
    kz_z=2*np.pi*n*z_per_r0*r0_per_λ0
    φ=np.arctan2(y_per_r0, x_per_r0)
    κ1_per_k0=np.emath.sqrt((n1**2-n**2))   
    κ2_per_k0=np.emath.sqrt((n2**2-n**2))   
    Hp=np.where((r_per_r0<1),
                (-1j/(κ1_per_k0))*(n*V[1]-1j*n1*V[0])*jv(0,k0_r*κ1_per_k0)*np.exp(1j*kz_z),
                (-1j/(κ2_per_k0))*(n*V[3]-1j*n1*V[2])*hv(0,k0_r*κ2_per_k0)*np.exp(1j*kz_z))
    Hm=np.where((r_per_r0<1),
                (-1j/(κ1_per_k0))*(-n*V[1]-1j*n1*V[0])*jv(2,k0_r*κ1_per_k0)*np.exp(1j*(2*φ+kz_z)),
                (-1j/(κ2_per_k0))*(-n*V[3]-1j*n1*V[2])*hv(2,k0_r*κ2_per_k0)*np.exp(1j*(2*φ+kz_z)))
    Hz=np.where((r_per_r0<1),
                np.sqrt(2)*V[1]*jv(1,k0_r*κ1_per_k0)*np.exp(1j*(φ+kz_z)),
                np.sqrt(2)*V[3]*hv(1,k0_r*κ2_per_k0)*np.exp(1j*(φ+kz_z)))
    return ((Hp+Hm)/np.sqrt(2),1j*(Hp-Hm)/np.sqrt(2),Hz)

def get(x,y,z,λ0=189e-9,r0=250e-9,n_in=1.45,n_out=1,rmin=0,rmax=2,neff_samples=1000):
    x_per_r0=x/r0
    y_per_r0=y/r0
    z_per_r0=z/r0
    material(n_in,n_out)
    (r0_per_λ0,n,V)=get_neff(rmin,rmax,neff_samples)
    X, Y, Z = np.meshgrid(x_per_r0, y_per_r0, z_per_r0)
    # E_grid=np.zeros((neff_samples,3,np.shape(x)[0],np.shape(y)[0],np.shape(z)[0]),dtype=complex)
    # H_grid=np.zeros((neff_samples,3,np.shape(x)[0],np.shape(y)[0],np.shape(z)[0]),dtype=complex)
    # np.zeros((100,3,10,10,10))
    # for i in range(neff_samples):
    #     E_grid[i]=E(V[i],r0_per_λ0[i],n[i],X,Y,Z)
    #     H_grid[i]=H(V[i],r0_per_λ0[i],n[i],X,Y,Z)
    i=np.abs(r0_per_λ0 - r0/λ0).argmin()
    E_grid=E(V[i],r0_per_λ0[i],n[i],X,Y,Z)
    H_grid=H(V[i],r0_per_λ0[i],n[i],X,Y,Z)
    kz=2*np.pi*n[i]/λ0
    return (E_grid,H_grid,kz)