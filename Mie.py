# -*- coding: utf-8 -*-
"""
Created on Mon May 16 12:41:26 2022

@author: Philipp
"""

from scipy.special import spherical_jn # spherical bessel of first kind
from scipy.special import spherical_yn # spherical bessel of second kind
from scipy.special import lpmn #associated Legendre functions of the first kind

def spherical_hn1(n,z,derivative=False):
    """ Spherical Hankel Function of the First Kind """
    return spherical_jn(n,z,derivative=derivative)+1j*spherical_yn(n,z,derivative=derivative)

def lfac(l):
    return (2*l + 1) / ( l * (l+1) )

def E(l):
    return 1j**l * lfac(l) # some prefactor used in the expansion

import numpy as np


class Mie:
    def __init__(self,lmax,R_sphere_nm,epsilon_r,wavelength_nm):
        self.lmax = lmax # number of expansion coefficients
        self.R_nm = R_sphere_nm # radius of sphere in nanometers
        self.wavelength_nm = wavelength_nm # wavelength in nanometers
        self.k=2*np.pi/wavelength_nm # k vector in vacuum 
        self.epsilon_r = epsilon_r # relative permittivity
        # the expansion coefficients only depend on the parameters above
        # so we can calculate them right away
        self.calc_coeffs() 
    
    def __call__(self,x,y,z):
        ''' overload of the call operator 
            
            takes 2D arrays for x,y and z.
            
            returns:    Ei0: the incident field
                        Ei:  the expansion of the incident field (outside)
                        Er:  the expansion of the reflected field (outside)
                        Et:  the expansion of the transmitted field (inside)
        '''
        r=self.r = np.sqrt(x**2+y**2+z**2)
        phi=self.phi = np.arctan2(y,x)
        theta = self.theta = np.arccos(z/r)
        print(phi.shape,theta.shape,r.shape)
        print('range phi',np.min(phi),np.max(phi))
        print('range theta',np.min(theta),np.max(theta))
        sin = np.sin
        cos = np.cos
        e_r = lambda theta,phi: [sin(theta)*cos(phi),
                                 sin(theta)*sin(phi),
                                 cos(theta)]

        e_theta = lambda theta,phi: [cos(theta)*cos(phi),
                                     cos(theta)*sin(phi),
                                     -sin(theta)]
        
        e_phi = lambda theta,phi: [ -sin(phi),
                                     cos(phi),
                                     np.zeros_like(phi) ]

        
        self.e_r =      e_r(theta,phi)
        self.e_theta =  e_theta(theta,phi)
        self.e_phi =    e_phi(theta,phi)

        self.sin_theta = sin(theta)
        self.cos_theta = z/r
        self.cos_phi = cos(phi)
        self.sin_phi = sin(phi)
        
        self.P = np.zeros((self.lmax+1,*self.cos_theta.shape))
        self.dP = np.zeros((self.lmax+1,*self.cos_theta.shape))
        
        for iy, ix in np.ndindex(self.cos_theta.shape):
            P,dP = lpmn(1,self.lmax,self.cos_theta[iy,ix])
            self.P[:,iy,ix] = P[1,:]
            self.dP[:,iy,ix] = dP[1,:]
        
        self.dP *= -self.sin_theta
        Ei1 = np.array([np.exp(1j*self.k*z),np.zeros_like(z),np.zeros_like(z)])
        return Ei1,self.get_Ei(),self.get_Er(),self.get_Et()
        
    def calc_coeffs(self):
        eps = self.epsilon_r
        l = np.arange(0,self.lmax+1)
        rho = self.k*self.R_nm
        rho_eps = rho*np.sqrt(eps)
        
        J = spherical_jn(l,rho)
        Jeps = spherical_jn(l,rho_eps)
        H = spherical_hn1(l,rho)
        product_derivative = lambda l,x,fun: x*fun(l,x,derivative=True) + fun(l,x)
        der_rhoH = product_derivative(l,rho,spherical_hn1)
        der_rhoJ = product_derivative(l,rho,spherical_jn)
        der_rhoepsJ = product_derivative(l,rho_eps,spherical_jn)
        
        self.a = (eps*der_rhoJ*Jeps - der_rhoepsJ*J) / (der_rhoH*Jeps*eps - H*der_rhoepsJ)
        self.b = (der_rhoJ*Jeps - der_rhoepsJ*J) / (der_rhoH*Jeps - der_rhoepsJ*H)
        self.c = (der_rhoH*J-der_rhoJ*H) / (der_rhoH*Jeps - der_rhoepsJ*H)
        self.d = np.sqrt(eps)*( der_rhoH*J - der_rhoJ*H ) / (eps*der_rhoH*Jeps - der_rhoepsJ*H)
        
        
        
    def get_J(self,kind,derivative=False):
        if kind == '1st':
            J = lambda l: spherical_jn(l,self.k*self.r,derivative=derivative)
        if kind == '3rd':
            J = lambda l: spherical_hn1(l,self.k*self.r,derivative=derivative)  
        return J
        
    def Mo(self,l,kind='1st'):
        ''' calculates the odd vector harmonic M for m=1 '''
        sin_theta = self.sin_theta
        cos_theta = self.cos_theta
        cos_phi = self.cos_phi
        sin_phi = self.sin_phi
        e_r = self.e_r
        e_theta = self.e_theta
        e_phi = self.e_phi
        J = self.get_J(kind)
        
        P_l = self.P[l] 
        dP_l = self.dP[l] 
        theta_comp =  np.einsum('ajk,jk->ajk',e_theta, cos_phi/sin_theta*J(l)*P_l)
        phi_comp = np.einsum('ajk,jk->ajk',e_phi,+J(l)*dP_l*sin_phi)
        return  theta_comp+phi_comp
    
       
    def Ne(self,l,kind='1st'):
        ''' calculates the even vector harmonic N for m=1 '''
        r = self.r
        k = self.k
        sin_theta = self.sin_theta
        cos_theta = self.cos_theta
        cos_phi = self.cos_phi
        sin_phi = self.sin_phi
        e_r = self.e_r
        e_theta = self.e_theta
        e_phi = self.e_phi
        
        J = self.get_J(kind)
        Jder = self.get_J(kind,derivative=True)
        
        der_krJ = lambda l: k*r*Jder(l)+J(l)
        P_l = self.P[l] 
        dP_l = self.dP[l] 
        
        r_comp = np.einsum('ajk,jk->ajk',e_r, l*(l+1) * J(l)* P_l * cos_phi)
        theta_comp = np.einsum('ajk,jk->ajk',e_theta, der_krJ(l)*dP_l*cos_phi)
        phi_comp = np.einsum('ajk,jk->ajk',e_phi,+1/sin_theta*der_krJ(l)*P_l*sin_phi)
        return (r_comp + theta_comp + phi_comp)/(k*r)

    def get_Ei(self):
        ''' expand the incident field (valid outside of sphere)'''
        lmax=self.lmax
        Mo=self.Mo
        Ne=self.Ne
        Ei=0
        for l in range(1,lmax+1):
            Ei -= E(l) *  (Mo(l) - 1j*Ne(l))
        return Ei
    
    def get_Er(self):
        ''' expand the reflected field (valid outside of sphere)'''
        lmax=self.lmax
        M3o=lambda l: self.Mo(l,kind='3rd')
        N3e=lambda l: self.Ne(l,kind='3rd')
        a = lambda l:self.a[l]
        b = lambda l:self.b[l]
        Er = 0 
        for l in range(1,lmax+1):
            Er -= E(l)*(1j*a(l)*N3e(l) - b(l)*M3o(l))
            
        return Er
    
    def get_Et(self):
        ''' expand the transmitted field (valid inside the sphere)'''
        lmax=self.lmax
        Mo=self.Mo
        Ne=self.Ne
        c = lambda l:self.c[l]
        d = lambda l:self.d[l]
        Et = 0 
        for l in range(1,lmax+1):
            Et-= E(l)*(c(l)*Mo(l) - 1j*d(l)*Ne(l) )
            
        return Et




if __name__ == '__main__':
    r0 = 300
    lmax = 5
    n = 1.43
    wavelength_nm = 2000
    
    ''' setup the Mie Object (calculates the expansion coefficients): '''
    mie = Mie(lmax=lmax,
              R_sphere_nm=r0,
              epsilon_r=n**2,
              wavelength_nm=wavelength_nm)
    
    ''' generate a grid of points '''
    lim = 2*r0
    x = np.linspace(-lim,lim,200)
    z = np.linspace(-lim,lim,204)
    X,Z=np.meshgrid(x,z)
    Y=np.zeros_like(X)
    R = np.sqrt(X**2+Y**2+Z**2)
    vecabs = lambda vec : np.sqrt(np.sum( [np.abs(x)**2 for x in vec] ,axis=0))
    
    
    ''' calculate the fields: '''
    Ei0,Ei,Er,Et = mie(X,Y,Z)
    
    
    Eabs = vecabs(Ei+Er)
    Eabs[R<r0] = vecabs(Et)[R<r0]
    import matplotlib.pyplot as plt
    

    fig=plt.figure(figsize=[12,3])
    fig.suptitle('l$_{max}$: %i,  r :%inm,  n: %.2f,  wavelength: %inm'%(lmax,r0,n,wavelength_nm))
    cm=plt.cm.nipy_spectral
    ax1 = plt.subplot(141,title='incoming field E$_{inc}$')
    im1 = ax1.pcolormesh(x,z,np.zeros_like(Ei[0].real),vmin=-1.6,vmax=1.6,shading='auto',cmap=cm)
    ax2 = plt.subplot(142,title='expansion of E$_{inc}$')
    im2 = ax2.pcolormesh(x,z,np.zeros_like(Ei[0].real),vmin=-1.6,vmax=1.6,shading='auto',cmap=cm)
    ax3 = plt.subplot(143,title='field x-component E$_x$')
    im3 = ax3.pcolormesh(x,z,np.zeros_like(Ei[0].real),vmin=-1.6,vmax=1.6,shading='auto',cmap=cm)
    plt.colorbar(im3)
    ax4 = plt.subplot(144,title='Enhancement Factor')
    im4 = ax4.pcolormesh(x,z,Eabs,shading='auto',cmap='inferno')
    plt.colorbar(im4)
    plt.tight_layout()
    
    
    from matplotlib.animation import FuncAnimation
    def update(t):
        Eabs = vecabs((Ei+Er)*np.exp(-1j*t))
        Ex=((Ei+Er)[0]*np.exp(-1j*t)).real
        Eabs[R<r0] = vecabs(Et*np.exp(-1j*t))[R<r0]
        Ex[R<r0] = ((Et[0]*np.exp(-1j*t)).real)[R<r0]
        im1.set_array((Ei0[0]*np.exp(-1j*t)).real)
        im2.set_array((Ei[0]*np.exp(-1j*t)).real)
        im3.set_array(Ex)
        return im1,im2,im3

    
    ani = FuncAnimation(fig, update, frames=np.linspace(0, 20*np.pi, 128),blit=True)
    plt.show()
    
    

        






