#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Dorian Bouchet - dorianf.bouchet@gmail.com
"""

# =============================================================================
# import modules
# =============================================================================

import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image   
from scipy import linalg
    
# =============================================================================
# define parameters
# =============================================================================

Nphotons=1 # number of photons per sampling point in the object plane (in counts)
Radius1=1 # radius of the phase mask in Fourier space (in pixels)
Radius2=np.inf # radius of the objective in Fourier space (in pixels)
delta_param=1e-8 # phase increment for the finite difference scheme (in rad)
phase_shifter=np.pi/2 # phase shift applied by the phase mask (in rad)
Eref=np.sqrt(0)*np.exp(1j*np.pi/2) # external reference field (here set to zero)
sigma_gauss=50 # standard deviation of the Gaussian enveloppe of the incident field (in pixels)

# =============================================================================
# import synthetic object
# =============================================================================

downsampling=2 # downsampling of the imported synthetic object
phase_contrast=2 # scaling factor for the phase contrast of the synthetic object

image_file_phase=np.array(Image.open('cameraman.png').convert('L')).astype(np.float64)
imarray_ph=image_file_phase[0::downsampling,0::downsampling]
imarray_ph=imarray_ph.astype(float)/255*phase_contrast*np.pi-phase_contrast*np.pi/2

Npix=np.shape(imarray_ph)[0]
XX,YY=np.meshgrid(np.arange(Npix),np.arange(Npix))
RR=np.sqrt((XX-int(Npix/2))**2+(YY-int(Npix/2))**2)

Eexc=np.ones((Npix,Npix),dtype=np.complex128)*np.sqrt(Nphotons)*np.exp(-RR**2/(2*sigma_gauss**2))
Eexc=Eexc*np.exp(1j*imarray_ph)

# =============================================================================
# calculate the intensity at the camera plane
# =============================================================================

Efourier=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Eexc)))
XX,YY=np.meshgrid(np.arange(Npix),np.arange(Npix))
idx1=np.where((XX-int(Npix/2))**2+(YY-int(Npix/2))**2<Radius1**2)
Efourier[idx1]=Efourier[idx1]*np.exp(1j*phase_shifter) # effect of the phase mask
idx2=np.where((XX-int(Npix/2))**2+(YY-int(Npix/2))**2>Radius2**2)
Efourier[idx2]=0. # effect of the finite NA
Ntrunc=np.shape(idx1)[1]+np.shape(idx2)[1]

Ereal=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Efourier)))+Eref*np.ones((Npix,Npix),dtype=np.complex128)
intensity=np.reshape(np.abs(Ereal)**2,Npix**2)
ratio_fourier=np.sum(np.abs(Efourier[idx1])**2)/np.sum(np.abs(Efourier)**2)

# =============================================================================
# calculate the Fisher information at the camera plane
# =============================================================================

t1=time.time()

Eexc_temp=np.copy(np.reshape(Eexc,Npix**2))
field_mod=np.empty((Npix**2,2),dtype=np.complex128)
intensity_mod=np.empty((Npix**2,Npix**2,2),dtype=np.float64)
J=np.empty(Npix**2,dtype=np.float64)
for ii in range(Npix**2):
  for jj in range(2):
    Eexc_temp[ii]=Eexc_temp[ii]*np.exp(1j*delta_param*(-1)**jj)
    Efourier_temp=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(np.reshape(Eexc_temp,(Npix,Npix)))))
    Efourier_temp[idx1]=Efourier_temp[idx1]*np.exp(1j*phase_shifter) 
    Efourier_temp[idx2]=0.
    Ereal_temp=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Efourier_temp)))+Eref*np.ones((Npix,Npix),dtype=np.complex128)
    field_mod[:,jj]=np.reshape(Ereal_temp,Npix**2)
    intensity_mod[:,ii,jj]=np.reshape(np.abs(Ereal_temp)**2,Npix**2)
    Eexc_temp[ii]=Eexc_temp[ii]*np.exp(-1j*delta_param*(-1)**jj)
  J[ii]=4*np.sum(np.abs((field_mod[:,0]-field_mod[:,1])/(2*delta_param))**2)
                                  
H=np.empty((Npix**2,Npix**2))
for kk in range(Npix**2):
    H[:,kk]=(1/np.sqrt(intensity[:]))*(intensity_mod[:,kk,0]-intensity_mod[:,kk,1])/(2*delta_param)
   
Ifisher=np.matmul(np.transpose(H),H)

print('Construction of the Fisher information matrix: '+str(np.round(time.time()-t1,1)) + ' s')

# =============================================================================
# truncate the Fisher information matrix and invert it
# =============================================================================

t1=time.time()

Fm=linalg.dft(Npix)
W=np.kron(Fm,Fm)/Npix
Wdagger=np.transpose(np.conjugate(W))

Ifisher_fourier=np.matmul(W,np.matmul(Ifisher,Wdagger))
fourier_fisher_diag_map=np.fft.fftshift(np.reshape(np.real(np.diag(Ifisher_fourier)),(Npix,Npix)))

# here is if we want to know the indexes of the points where the phase mask lies
list_indexes=np.fft.fftshift(np.reshape(np.arange(Npix**2),(Npix,Npix)))
indexes_badguys=np.concatenate((list_indexes[idx1],list_indexes[idx2]))

# truncating the matrices accordingly 
Ifisher_fourier_trunc=np.delete(np.delete(Ifisher_fourier,indexes_badguys,axis=1),indexes_badguys,axis=0)
Wdagger_trunc=np.delete(Wdagger,indexes_badguys,axis=1)
W_trunc=np.delete(W,indexes_badguys,axis=0)
sigma_trunc=np.real(np.matmul(Wdagger_trunc,np.matmul(np.linalg.inv(Ifisher_fourier_trunc),W_trunc)))
crlb_prior=np.reshape(np.diag(sigma_trunc),(Npix,Npix))

print('Inversion of the Fisher information matrix: '+str(np.round(time.time()-t1,1))+ ' s')

# =============================================================================
# plot figure
# =============================================================================

plt.figure(figsize=(12,6))
plt.subplot(231)
plt.imshow(np.abs(Eexc)**2,cmap='gray')
plt.clim(0,1)
plt.axis('off')
plt.title('Intensity (object)')
plt.colorbar()
plt.subplot(234)
plt.imshow(np.angle(Eexc),cmap='hsv')
plt.clim(-np.pi,np.pi)
plt.axis('off')
plt.title('Phase (object)')
plt.colorbar()

plt.subplot(232)
plt.imshow(np.abs(Efourier)**2,cmap='gray')
plt.axis('off')
plt.title('Intensity (Fourier)')
plt.colorbar()
plt.subplot(235)
plt.imshow(np.angle(Efourier),cmap='hsv')
plt.clim(-np.pi,np.pi)
plt.axis('off')
plt.title('Phase (Fourier)')
plt.colorbar()

plt.subplot(233)
plt.imshow(np.abs(Ereal)**2,cmap='gray')
plt.clim(0,1.5)
plt.axis('off')
plt.title('Intensity (image)')
plt.colorbar()
plt.subplot(236)
plt.imshow(np.angle(Ereal),cmap='hsv')
plt.clim(-np.pi,np.pi)
plt.title('Phase (image)')
plt.axis('off')
plt.colorbar()
plt.tight_layout()

# =============================================================================
# plot figure
# =============================================================================

plt.figure(figsize=(11,4))
plt.subplot(121)
plt.title('Diagonal elements of FI matrix in Fourier basis')
plt.imshow(fourier_fisher_diag_map)
plt.axis('off')
plt.colorbar()
plt.subplot(122)
plt.title('Resulting CRLB')
plt.imshow(np.log10(crlb_prior),cmap='hot')
plt.clim(-1,3)
plt.axis('off')
plt.colorbar()
plt.tight_layout()


