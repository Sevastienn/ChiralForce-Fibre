import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, optimize
from scipy.special import jv, hankel1 as hv,jvp, h1vp as hvp

# # Henkel function of first kind and derivatives
# def jvp(l,z):
#     return jv(l-1,z)-l*jv(l,z)/z
# def hv(l,z):
#     return jv(l,z)+1j*yv(l,z)
# def hvp(l,z):
#     return (jv(l-1,z)-l*jv(l,z)/z)+1j*(yv(l-1,z)-l*yv(l,z)/z)

# One can set different material default glass and air
def material(n_in=1.45,n_out=1):
    global n1; n1=n_in  # Refractive index or sqrt(ϵ₁) inside
    global n2; n2=n_out # Refractive index or sqrt(ϵ₂) outside

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# Boundary condition matrix
def A(neff,r0_per_λ0,ℓ=1):
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
def logdetA(neff,r0_per_λ0,ℓ=1):
    a=A(neff,r0_per_λ0,ℓ=ℓ)
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

def get_neff(rmin=0,rmax=2,N=1000,neff_fig=False,r0_over_λ0=None,ℓ=1):    
    r0_per_λ0=np.linspace(rmin,rmax,N+1,endpoint=False)[1:]
    neff=np.linspace(n2,n1,N+1,endpoint=False)[1:]
    Ro, Ne = np.meshgrid(r0_per_λ0, neff)
    logdetM=logdetA(Ne,Ro,ℓ=ℓ)

    # Finds n_eff
    n=np.zeros_like(r0_per_λ0)
    for i in range(len(r0_per_λ0)):
        try:
            n[i]=neff[np.max(signal.find_peaks(-logdetM[:,i])[0])]
        except ValueError:  #raised if there are no peaks
            n[i]=0

    # Calculates amplitudes    
    M=np.transpose(A(n,r0_per_λ0,ℓ=ℓ),(2,0,1))
    (w,v)=np.linalg.eig(M)
    argus=np.argmin(np.abs(w),axis=1)
    W=w[np.arange(N),argus]
    V=v[np.arange(N),:,argus]


    # Plots n_eff
    if neff_fig:
        plt.pcolormesh(2*np.pi*Ro,Ne,-logdetM,cmap='Greys')
        plt.colorbar(label=r'$-\log_{10}|\det{M}|$')
        plt.xlim(xmax=2*np.pi*rmax,xmin=rmin) 
        plt.plot(2*np.pi*r0_per_λ0,n,label=r'$n_\mathrm{eff}(k_0r_0)$')
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

        # return (r0_per_λ0[idx],n[idx],V[idx])

        # a=np.zeros(4)
        # eM=np.zeros_like(M[idx])
        # # # print_matrix(M[idx])
        # for i in range(4):
        # #     # print(np.inner(np.conj(eM[:,i]),eM[:,i]))
        #     a[i]=np.sqrt(np.inner(np.conj(M[idx,:,i]),M[idx,:,i]))
        # #     # print(a[i])
        #     eM[:,i]=M[idx,:,i]/a[i]
        #     # print_matrix(eM[:,i])
        #     # print(np.inner(np.conj(eM[:,i]),eM[:,i]))
        # # print_matrix(eM)
        # # print(print(np.inner(np.conj(eM[:,0]),eM[:,0])))
        # (w,v)=np.linalg.eig(eM)
        # argus=np.argmin(np.abs(w))
        # W=w[argus]
        # # print(w)
        # # V=v[:,argus]
        # # print_matrix(v[:,argus])
        # V=np.zeros(4,dtype=complex)
        # for i in range(4):
        #     # V[i]=a[i]*V[i]
        #     V[i]=v[i,argus]/a[i]
        # # print_matrix(V)
        # # print_matrix(M[idx])
        # # print(np.inner(np.conj(M[idx,:,0]),M[idx,:,0]))
        # # print(np.inner(np.conj(M[idx,:,1]),M[idx,:,1]))
        # # print(np.inner(np.conj(M[idx,:,2]),M[idx,:,2]))
        # # print(np.inner(np.conj(M[idx,:,3]),M[idx,:,3]))

        a=np.zeros(4)
        eM=np.zeros_like(M[idx])
        for i in range(4):
            a[i]=np.sqrt(np.inner(np.conj(M[idx,i,:]),M[idx,i,:]))
            eM[i,:]=M[idx,i,:]/a[i]
        (w,v)=np.linalg.eig(eM)
        argus=np.argmin(np.abs(w))
        W=w[argus]
        V=v[:,argus]
        return (r0_per_λ0[idx],n[idx],V)

def get_E(x,y,z,λ0,r0,n_in=1.45,n_out=1,neff_fig=False,ℓ=1,times_ϵ=False):
    '''returns sqrt(ϵ₀)(E_x,E_y,E_z) in units of sqrt(J/m³)'''

    material(n_in,n_out)
    (r0_per_λ0,n,V)=get_neff(rmin=0,rmax=int(np.ceil(r0/λ0))+1,N=1000,r0_over_λ0=r0/λ0,neff_fig=neff_fig,ℓ=ℓ)

    r=np.sqrt(x**2+y**2)
    φ=np.arctan2(y,x)
    k0_r=2*np.pi*r/λ0
    kz_z=2*np.pi*n*z/λ0
    κ1_per_k0=np.emath.sqrt(n1**2-n**2) 
    κ2_per_k0=np.emath.sqrt(n2**2-n**2)

    if times_ε: root_ϵ1=1;root_ϵ2=1
    else: root_ϵ1=n1;root_ϵ2=n2
    Ep=np.where(r/r0<1,
                (-1j/(root_ϵ1*κ1_per_k0))*(n*V[0]+1j*n1*V[1])*jv(ℓ-1,k0_r*κ1_per_k0)*np.exp(1j*((ℓ-1)*φ+kz_z)),
                (-1j/(root_ϵ2*κ2_per_k0))*(n*V[2]+1j*n2*V[3])*hv(ℓ-1,k0_r*κ2_per_k0)*np.exp(1j*((ℓ-1)*φ+kz_z)))
    Em=np.where(r/r0<1,
                (-1j/(root_ϵ1*κ1_per_k0))*(-n*V[0]+1j*n1*V[1])*jv(ℓ+1,k0_r*κ1_per_k0)*np.exp(1j*((ℓ+1)*φ+kz_z)),
                (-1j/(root_ϵ2*κ2_per_k0))*(-n*V[2]+1j*n2*V[3])*hv(ℓ+1,k0_r*κ2_per_k0)*np.exp(1j*((ℓ+1)*φ+kz_z)))
    Ez=np.where(r/r0<1,
                (np.sqrt(2)/root_ϵ1)*V[0]*jv(ℓ,k0_r*κ1_per_k0)*np.exp(1j*(ℓ*φ+kz_z)),
                (np.sqrt(2)/root_ϵ2)*V[2]*hv(ℓ,k0_r*κ2_per_k0)*np.exp(1j*(ℓ*φ+kz_z)))
    
    Eφ=1j/np.sqrt(2)*(Ep*np.exp(1j*(φ))-Em*np.exp(-1j*(φ)))

    return ((Ep+Em)/np.sqrt(2),1j*(Ep-Em)/np.sqrt(2),Ez,Eφ)

def E(x,y,z,λ0,r0,R,L,n_in=1.45,n_out=1,ℓ=1,times_ϵ=False):
    pE=get_E(x,y,z,λ0,r0,n_in=n_in,n_out=n_out,ℓ=ℓ,times_ϵ=times_ϵ)
    mE=get_E(x,y,z,λ0,r0,n_in=n_in,n_out=n_out,ℓ=-ℓ,times_ϵ=times_ϵ)
    return ((R*pE[0]+L*mE[0])/np.sqrt(2),(R*pE[1]+L*mE[1])/np.sqrt(2),(R*pE[2]+L*mE[2])/np.sqrt(2),(R*pE[3]+L*mE[3])/np.sqrt(2))
def H(x,y,z,λ0,r0,R,L,n_in=1.45,n_out=1,ℓ=1):
    pH=get_H(x,y,z,λ0,r0,n_in=n_in,n_out=n_out,ℓ=ℓ)
    mH=get_H(x,y,z,λ0,r0,n_in=n_in,n_out=n_out,ℓ=-ℓ)
    return (1j*(R*pH[0]+L*mH[0])/np.sqrt(2),1j*(R*pH[1]+L*mH[1])/np.sqrt(2),1j*(R*pH[2]+L*mH[2])/np.sqrt(2),1j*(R*pH[3]+L*mH[3])/np.sqrt(2))

def get_H(x,y,z,λ0,r0,n_in=1.45,n_out=1,neff_fig=False,ℓ=1):
    '''returns sqrt(ϵ₀)(E_x,E_y,E_z) in units of sqrt(J/m³)'''

    material(n_in,n_out)
    (r0_per_λ0,n,V)=get_neff(rmin=0,rmax=int(np.ceil(r0/λ0))+1,N=1000,r0_over_λ0=r0/λ0,neff_fig=neff_fig,ℓ=ℓ)

    r=np.sqrt(x**2+y**2)
    φ=np.arctan2(y,x)
    k0_r=2*np.pi*r/λ0
    kz_z=2*np.pi*n*z/λ0
    κ1_per_k0=np.emath.sqrt(n1**2-n**2) 
    κ2_per_k0=np.emath.sqrt(n2**2-n**2)

    Hp=np.where(r/r0<1,
                (-1j/(κ1_per_k0))*(n*V[1]-1j*n1*V[0])*jv(ℓ-1,k0_r*κ1_per_k0)*np.exp(1j*((ℓ-1)*φ+kz_z)),
                (-1j/(κ2_per_k0))*(n*V[3]-1j*n2*V[2])*hv(ℓ-1,k0_r*κ2_per_k0)*np.exp(1j*((ℓ-1)*φ+kz_z)))
    Hm=np.where(r/r0<1,
                (-1j/(κ1_per_k0))*(-n*V[1]-1j*n1*V[0])*jv(ℓ+1,k0_r*κ1_per_k0)*np.exp(1j*((ℓ+1)*φ+kz_z)),
                (-1j/(κ2_per_k0))*(-n*V[3]-1j*n2*V[2])*hv(ℓ+1,k0_r*κ2_per_k0)*np.exp(1j*((ℓ+1)*φ+kz_z)))
    Hz=np.where(r/r0<1,
                (np.sqrt(2))*V[1]*jv(ℓ,k0_r*κ1_per_k0)*np.exp(1j*(ℓ*φ+kz_z)),
                (np.sqrt(2))*V[3]*hv(ℓ,k0_r*κ2_per_k0)*np.exp(1j*(ℓ*φ+kz_z)))
    
    Hφ=1j/np.sqrt(2)*(Hp*np.exp(1j*(φ))-Hm*np.exp(-1j*(φ)))

    return ((Hp+Hm)/np.sqrt(2),1j*(Hp-Hm)/np.sqrt(2),Hz,Hφ)

# def get_H(x,y,z,λ0,r0,n_in=1.45,n_out=1,neff_fig=False,ℓ=1):
#     '''returns sqrt(μ₀)(H_x,H_y,H_z) in units of sqrt(J/m³)'''
#     material(n_in,n_out)
#     (r0_per_λ0,n,V)=get_neff(rmin=0,rmax=int(np.ceil(r0/λ0))+1,N=1000,r0_over_λ0=r0/λ0,neff_fig=neff_fig,ℓ=ℓ)

#     r=np.sqrt(x**2+y**2)
#     φ=np.arctan2(y,x)
#     k0_r=2*np.pi*r/λ0
#     kz_z=2*np.pi*n*z/λ0
#     κ1_per_k0=np.emath.sqrt(n1**2-n**2) 
#     κ2_per_k0=np.emath.sqrt(n2**2-n**2)

#     Hp=np.where(r/r0<1,
#                 (-1j/(κ1_per_k0))*(n*V[1]-1j*n1*V[0])*jv(ℓ-1,k0_r*κ1_per_k0)*np.exp(1j*((ℓ-1)*φ+kz_z)),
#                 (-1j/(κ2_per_k0))*(n*V[3]-1j*n1*V[2])*hv(ℓ-1,k0_r*κ2_per_k0)*np.exp(1j*((ℓ-1)*φ+kz_z)))
#     Hm=np.where(r/r0<1,
#                 (-1j/(κ1_per_k0))*(-n*V[1]-1j*n1*V[0])*jv(ℓ+1,k0_r*κ1_per_k0)*np.exp(1j*((ℓ+1)*φ+kz_z)),
#                 (-1j/(κ2_per_k0))*(-n*V[3]-1j*n1*V[2])*hv(ℓ+1,k0_r*κ2_per_k0)*np.exp(1j*((ℓ+1)*φ+kz_z)))
#     Hz=np.where(r/r0<1,
#                 (np.sqrt(2))*V[1]*jv(ℓ,k0_r*κ1_per_k0)*np.exp(1j*(ℓ*φ+φ+kz_z)),
#                 (np.sqrt(2))*V[3]*hv(ℓ,k0_r*κ2_per_k0)*np.exp(1j*(ℓ*φ+φ+kz_z)))
    
#     Hφ=1j/np.sqrt(2)*(Hp*np.exp(1j*(φ))-Hm*np.exp(-1j*(φ)))

#     return ((Hp+Hm)/np.sqrt(2),1j*(Hp-Hm)/np.sqrt(2),Hz,Hφ)

# # E and H fields
# def E(V,n,r0_per_λ0,x_per_r0,y_per_r0,z_per_r0):
#     # x,y,z in units of r0
#     r_per_r0=np.sqrt(x_per_r0**2+y_per_r0**2)
#     k0_r=2*np.pi*r0_per_λ0*r_per_r0
#     kz_z=2*np.pi*n*z_per_r0*r0_per_λ0
#     φ=np.arctan2(y_per_r0, x_per_r0)
#     κ1_per_k0=np.emath.sqrt(n1**2-n**2) 
#     κ2_per_k0=np.emath.sqrt(n2**2-n**2)    
#     Ep=np.where(r_per_r0<1,
#                 (-1j/(n1*κ1_per_k0))*(n*V[0]+1j*n1*V[1])*jv(0,k0_r*κ1_per_k0)*np.exp(1j*kz_z),
#                 (-1j/(n2*κ2_per_k0))*(n*V[2]+1j*n2*V[3])*hv(0,k0_r*κ2_per_k0)*np.exp(1j*kz_z))
#     Em=np.where(r_per_r0<1,
#                 (-1j/(n1*κ1_per_k0))*(-n*V[0]+1j*n1*V[1])*jv(2,k0_r*κ1_per_k0)*np.exp(1j*(2*φ+kz_z)),
#                 (-1j/(n2*κ2_per_k0))*(-n*V[2]+1j*n2*V[3])*hv(2,k0_r*κ2_per_k0)*np.exp(1j*(2*φ+kz_z)))
#     Ez=np.where(r_per_r0<1,
#                 (np.sqrt(2)/n1)*V[0]*jv(1,k0_r*κ1_per_k0)*np.exp(1j*(φ+kz_z)),
#                 (np.sqrt(2)/n2)*V[2]*hv(1,k0_r*κ2_per_k0)*np.exp(1j*(φ+kz_z)))
#     print((np.sqrt(2)/n1)*V[0]*jv(1,2*np.pi*r0_per_λ0*κ1_per_k0)*np.exp(1j*(kz_z))-(np.sqrt(2)/n2)*V[2]*hv(1,2*np.pi*r0_per_λ0*κ2_per_k0)*np.exp(1j*(kz_z)))
#     return ((Ep+Em)/np.sqrt(2),1j*(Ep-Em)/np.sqrt(2),Ez)

# def H(V,n,r0_per_λ0,x_per_r0,y_per_r0,z_per_r0):
#     # x,y,z in units of r0
#     r_per_r0=np.sqrt(x_per_r0**2+y_per_r0**2)
#     k0_r=2*np.pi*r0_per_λ0*r_per_r0
#     kz_z=2*np.pi*n*z_per_r0*r0_per_λ0
#     φ=np.arctan2(y_per_r0, x_per_r0)
#     κ1_per_k0=np.emath.sqrt((n1**2-n**2))   
#     κ2_per_k0=np.emath.sqrt((n2**2-n**2))   
#     Hp=np.where((r_per_r0<1),
#                 (-1j/(κ1_per_k0))*(n*V[1]-1j*n1*V[0])*jv(0,k0_r*κ1_per_k0)*np.exp(1j*kz_z),
#                 (-1j/(κ2_per_k0))*(n*V[3]-1j*n1*V[2])*hv(0,k0_r*κ2_per_k0)*np.exp(1j*kz_z))
#     Hm=np.where((r_per_r0<1),
#                 (-1j/(κ1_per_k0))*(-n*V[1]-1j*n1*V[0])*jv(2,k0_r*κ1_per_k0)*np.exp(1j*(2*φ+kz_z)),
#                 (-1j/(κ2_per_k0))*(-n*V[3]-1j*n1*V[2])*hv(2,k0_r*κ2_per_k0)*np.exp(1j*(2*φ+kz_z)))
#     Hz=np.where((r_per_r0<1),
#                 np.sqrt(2)*V[1]*jv(1,k0_r*κ1_per_k0)*np.exp(1j*(φ+kz_z)),
#                 np.sqrt(2)*V[3]*hv(1,k0_r*κ2_per_k0)*np.exp(1j*(φ+kz_z)))
#     return ((Hp+Hm)/np.sqrt(2),1j*(Hp-Hm)/np.sqrt(2),Hz)

# def get(x,y,z,λ0=189e-9,r0=250e-9,n_in=1.45,n_out=1,rmin=0,rmax=2,neff_samples=1000):
#     x_per_r0=x/r0
#     y_per_r0=y/r0
#     z_per_r0=z/r0
#     material(n_in,n_out)
#     (r0_per_λ0,n,V)=get_neff(rmin,rmax,neff_samples)
#     X, Y, Z = np.meshgrid(x_per_r0, y_per_r0, z_per_r0)
#     # E_grid=np.zeros((neff_samples,3,np.shape(x)[0],np.shape(y)[0],np.shape(z)[0]),dtype=complex)
#     # H_grid=np.zeros((neff_samples,3,np.shape(x)[0],np.shape(y)[0],np.shape(z)[0]),dtype=complex)
#     # np.zeros((100,3,10,10,10))
#     # for i in range(neff_samples):
#     #     E_grid[i]=E(V[i],r0_per_λ0[i],n[i],X,Y,Z)
#     #     H_grid[i]=H(V[i],r0_per_λ0[i],n[i],X,Y,Z)
#     i=np.abs(r0_per_λ0 - r0/λ0).argmin()
#     E_grid=E(V[i],r0_per_λ0[i],n[i],X,Y,Z)
#     H_grid=H(V[i],r0_per_λ0[i],n[i],X,Y,Z)
#     kz=2*np.pi*n[i]/λ0
#     return (E_grid,H_grid,kz)