import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
import scipy.ndimage
import os
import random
from scipy import spatial
from compute_deformation_tensor import *
import copy
import time

def parabola_pdf(x, mean):
	a = 12*mean-6
	return a*x**2. + 1.0 - 1./3.*a

def rebin(a, *args):
	# rebins a 3D array to a new shape
    shape = a.shape
    lenShape = len(shape)
    factor = np.asarray(shape)/np.asarray(args)
    evList = ['a.reshape('] + \
             ['args[%d],factor[%d],'%(i,i) for i in range(lenShape)] + \
             [')'] + ['.mean(%d)'%(i+1) for i in range(lenShape)]
    return eval(''.join(evList))

#WMAP cosmology from Zarija's paper
omegam = 0.3
omegalambda = 1 - omegam
h = 0.685
redshift = 2.4 # Redshift of sims
hz = 100*h*np.sqrt(omegam*(1+redshift)**3.+omegalambda)

boxsize = 100.0 # in Mpc/h
boxsize_mpc = 100.0/h
npix_sim = 512.

# Read in tau and velocity
f = h5py.File("/global/cscratch1/sd/akrolew/Alex/4096-512bin.h5")
tau_red = f["derived_fields/tau_red"][:,:,:]
vel_z = f["native_fields/velocity_z"][:,:,:]/10**5 #Convert to km/s
deltachi = vel_z*(1+redshift)/hz

# Read in the redshift-space density
f = h5py.File("/global/cscratch1/sd/akrolew/Alex/4096-512bin_tot_density_rs.h5")
dm_density_rs = f["tot_density_rs"][:,:,:]

# Rebin the DM density and tau; define deltaf
rebin_size = 128
dm_density_bin = rebin(dm_density_rs, rebin_size, rebin_size, rebin_size)
tau_bin = rebin(tau_red, rebin_size, rebin_size, rebin_size)
flux = np.exp(-tau_bin)
deltaf = flux/np.mean(flux)-1.

# Smooth the dark matter and deltaf
kernel_size = 2.0 # h^-1 Mpc

# Mode=wrap means this is consistent with PBC
dm_density_smoothed = scipy.ndimage.filters.gaussian_filter(dm_density_bin,kernel_size*(float(rebin_size)/100.0),mode='wrap')
deltaf_smoothed = scipy.ndimage.filters.gaussian_filter(deltaf,kernel_size*(float(rebin_size)/100.0),mode='wrap')

e_dm = compute_deformation_tensor(dm_density_smoothed-1.)

evals_dm = e_dm[0]
evecs_dm = e_dm[1]

e1_dm = evecs_dm[:,:,:,:,0]
e2_dm = evecs_dm[:,:,:,:,1]
e3_dm = evecs_dm[:,:,:,:,2]

e_deltaf = compute_deformation_tensor(deltaf_smoothed)

evals_deltaf = e_deltaf[0]
evecs_deltaf = e_deltaf[1]

e1_deltaf = evecs_deltaf[:,:,:,:,2]
e2_deltaf = evecs_deltaf[:,:,:,:,1]
e3_deltaf = evecs_deltaf[:,:,:,:,0]

# Threshold adjusted so that 19% of the volume is in voids, following the findings of
# Casey Stark's voids paper (this is the criteria that was used to define lambda_thresh)
# in Lee & White (2016)
lambda_thresh = 0.037

evals_dm = evals_dm - lambda_thresh

# Halos
nodes_dm = np.where(((evals_dm[:,:,:,0]*evals_dm[:,:,:,1]*evals_dm[:,:,:,2] > 0) &
(evals_dm[:,:,:,0] > 0) & (evals_dm[:,:,:,1] > 0) & (evals_dm[:,:,:,2] > 0)))
# Filaments
filaments_dm = np.where(((evals_dm[:,:,:,0]*evals_dm[:,:,:,1]*evals_dm[:,:,:,2] < 0) &
np.logical_not((evals_dm[:,:,:,0] < 0) & (evals_dm[:,:,:,1] < 0) & (evals_dm[:,:,:,2] < 0))))
# Sheets
sheets_dm = np.where(((evals_dm[:,:,:,0]*evals_dm[:,:,:,1]*evals_dm[:,:,:,2] > 0) &
np.logical_not((evals_dm[:,:,:,0] > 0) & (evals_dm[:,:,:,1] > 0) & (evals_dm[:,:,:,2] > 0))))
# Voids
voids_dm = np.where(((evals_dm[:,:,:,0]*evals_dm[:,:,:,1]*evals_dm[:,:,:,2] < 0) &
(evals_dm[:,:,:,0] < 0) & (evals_dm[:,:,:,1] < 0) & (evals_dm[:,:,:,2] < 0)))