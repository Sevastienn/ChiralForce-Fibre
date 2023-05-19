import numpy as np
from scipy.special import jve as jv, yve as yv

def detA(n0,ρ):
    n1=1.45
    n2=1
    k0=2*np.pi*ρ    
    kz=n0*k0
    k1=n1*k0
    k2=n2*k0
    κ1=np.emath.sqrt(k1**2-kz**2)
    κ2=np.emath.sqrt(k2**2-kz**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        J=jv(1,κ1)
        Jp=jv(0,κ1)-jv(1,κ1)/κ1
        Yp=yv(0,κ1)-yv(1,κ1)/κ1
        H=(jv(1,κ2)+1j*yv(1,κ2))
        Hp=(Jp+1j*Yp)
        a11=n2*J
        a12=0
        a13=-n1*H
        a14=0
        a21=n2*(kz/(κ1)**2)*J
        a22=1j*n2*(k1/κ1)*Jp
        a23=-n1*(kz/(κ2)**2)*H
        a24=-1j*n1*(k2/κ2)*Hp
        a31=0
        a32=J
        a33=0
        a34=-H
        a41=-1j*(k1/κ1)*Jp
        a42=(kz/(κ1)**2)*J
        a43=1j*(k2/κ2)*Hp
        a44=-(kz/(κ2)**2)*H
    output=np.log10(np.abs(a14*a23*a32*a41 - a13*a24*a32*a41 - a14*a22*a33*a41 + 
        a12*a24*a33*a41 + a13*a22*a34*a41 - a12*a23*a34*a41 - 
        a14*a23*a31*a42 + a13*a24*a31*a42 + a14*a21*a33*a42 - 
        a11*a24*a33*a42 - a13*a21*a34*a42 + a11*a23*a34*a42 + 
        a14*a22*a31*a43 - a12*a24*a31*a43 - a14*a21*a32*a43 + 
        a11*a24*a32*a43 + a12*a21*a34*a43 - a11*a22*a34*a43 - 
        a13*a22*a31*a44 + a12*a23*a31*a44 + a13*a21*a32*a44 - 
        a11*a23*a32*a44 - a12*a21*a33*a44 + a11*a22*a33*a44))
    return np.where((n0<n1) & (n0>n2),output,0)
    # if (n0<n1) & (n0>n2): return np.log10(np.abs(a14*a23*a32*a41 - a13*a24*a32*a41 - a14*a22*a33*a41 + 
    #     a12*a24*a33*a41 + a13*a22*a34*a41 - a12*a23*a34*a41 - 
    #     a14*a23*a31*a42 + a13*a24*a31*a42 + a14*a21*a33*a42 - 
    #     a11*a24*a33*a42 - a13*a21*a34*a42 + a11*a23*a34*a42 + 
    #     a14*a22*a31*a43 - a12*a24*a31*a43 - a14*a21*a32*a43 + 
    #     a11*a24*a32*a43 + a12*a21*a34*a43 - a11*a22*a34*a43 - 
    #     a13*a22*a31*a44 + a12*a23*a31*a44 + a13*a21*a32*a44 - 
    #     a11*a23*a32*a44 - a12*a21*a33*a44 + a11*a22*a33*a44))
    # else: return None

def M(n0,ρ):
    n1=1.45
    n2=1
    k0=2*np.pi*ρ    
    kz=n0*k0
    k1=n1*k0
    k2=n2*k0
    κ1=np.emath.sqrt(k1**2-kz**2)
    κ2=np.emath.sqrt(k2**2-kz**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        J=jv(1,κ1)
        Jp=jv(0,κ1)-jv(1,κ1)/κ1
        Yp=yv(0,κ1)-yv(1,κ1)/κ1
        H=(jv(1,κ2)+1j*yv(1,κ2))
        Hp=(Jp+1j*Yp)
        a11=n2*J
        a12=0
        a13=-n1*H
        a14=0
        a21=n2*(kz/(κ1)**2)*J
        a22=1j*n2*(k1/κ1)*Jp
        a23=-n1*(kz/(κ2)**2)*H
        a24=-1j*n1*(k2/κ2)*Hp
        a31=0
        a32=J
        a33=0
        a34=-H
        a41=-1j*(k1/κ1)*Jp
        a42=(kz/(κ1)**2)*J
        a43=1j*(k2/κ2)*Hp
        a44=-(kz/(κ2)**2)*H
    return np.array([[a11,a12,a13,a14],[a21,a22,a23,a24],[a31,a32,a33,a34],[a41,a42,a43,a44]])