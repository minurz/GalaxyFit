#!/usr/bin/env python
# coding: utf-8

# In[7]:


#import import_ipynb
#from statmorph_lib import *

from lib.statmorph_lib import *

get_ipython().run_line_magic('matplotlib', 'widget')

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.max_open_warning'] = 200
#matplotlib.__version__
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
import argparse
from scipy.ndimage import zoom
from astropy.visualization import (ImageNormalize, LogStretch, AsinhStretch, PercentileInterval, make_lupton_rgb)
#import pandas as pd
import IPython
from IPython.display import display


# In[2]:


NH153_1 = 'NH_153_1_i0.00_a0.00_nodust_i0.00_a0.00_broad_NIRCam_super_total.fits'
NH153_2 = 'NH_153_2_i0.00_a0.00_nodust_i0.00_a0.00_broad_NIRCam_super_total.fits'
NH153_3 = 'NH_153_3_i0.00_a0.00_nodust_i0.00_a0.00_broad_NIRCam_super_total.fits'
NH153_4 = 'NH_153_4_i0.00_a0.00_nodust_i0.00_a0.00_broad_NIRCam_super_total.fits'
NH153_14 = 'NH_153_14_i0.00_a0.00_nodust_i0.00_a0.00_broad_NIRCam_super_total.fits'
#z =  3.10
#datname = 'fits/NH_153_4_i0.00_a0.00_luminosity_luminosities.dat'

NH201_1 = 'NH_201_1_i0.00_a0.00_nodust_i0.00_a0.00_broad_NIRCam_super_total.fits'
NH201_2 = 'NH_201_2_i0.00_a0.00_nodust_i0.00_a0.00_broad_NIRCam_super_total.fits'
NH201_3 = 'NH_201_3_i0.00_a0.00_nodust_i0.00_a0.00_broad_NIRCam_super_total.fits'
NH201_4 = 'NH_201_4_i0.00_a0.00_nodust_i0.00_a0.00_broad_NIRCam_super_total.fits'
#z =  2.45

NH134_1 = 'NH_134_1_i0.00_a0.00_nodust_i0.00_a0.00_broad_NIRCam_super_total.fits'
NH134_2 = 'NH_134_2_i0.00_a0.00_nodust_i0.00_a0.00_broad_NIRCam_super_total.fits'
NH134_3 = 'NH_134_3_i0.00_a0.00_nodust_i0.00_a0.00_broad_NIRCam_super_total.fits'
NH134_11 = 'NH_134_11_i0.00_a0.00_nodust_i0.00_a0.00_broad_NIRCam_super_total.fits'
NH134_14 = 'NH_134_14_i0.00_a0.00_nodust_i0.00_a0.00_broad_NIRCam_super_total.fits'
#z =  3.45


# In[3]:


def pick_rgb_filters(redshift):
        """
        Pick appropriate filters for RGB imaging based on the redshift, ensuring that R > G > B in terms of wavelength.

        :param redshift: The redshift of the galaxy.
        :return: A tuple (r_filter, g_filter, b_filter) with the chosen filters.
        """

        # Define JWST NIRCam and MIRI filters with wavelengths in microns
        jwst_filters = {
            'F090W': 0.9,
            'F115W': 1.15,
            'F150W': 1.50,
            'F200W': 2.00,
            'F277W': 2.77,
            'F356W': 3.56,
            'F444W': 4.44,
            'F560W': 5.60,
            'F770W': 7.70,
            'F1000W': 10.00,
            'F1280W': 12.80,
            'F1500W': 15.00,
            'F1800W': 18.00,
            'F2100W': 21.00
        }

        # Approximate rest-frame wavelengths for U, B, V bands in microns
        rest_frame_optical = {
            'U': 0.36,  # Near-UV
            'B': 0.44,  # Blue
            'V': 0.55   # Green-yellow
        }

        #print(f"Redshift: {redshift}")
        #print("JWST Filters (sorted by wavelength):", sorted(jwst_filters.items(), key=lambda x: x[1]))

        # Function to find the filter closest to a desired rest-frame wavelength
        def find_closest_filter(target_wavelength, exclude_filters=[]):
            sorted_filters = sorted(jwst_filters.keys(), key=lambda f: jwst_filters[f])  # Sort filters by wavelength
            available_filters = [f for f in sorted_filters if f not in exclude_filters]
            if not available_filters:
                return None
            closest_filter = min(available_filters, key=lambda f: abs(jwst_filters[f] - target_wavelength))
            return closest_filter

        # Calculate the observed wavelength based on redshift for rest-frame U, B, V
        observed_u = rest_frame_optical['U'] * (1 + redshift)
        observed_b = rest_frame_optical['B'] * (1 + redshift)
        observed_v = rest_frame_optical['V'] * (1 + redshift)

        #print(f"Observed U: {observed_u}, B: {observed_b}, V: {observed_v}")

        # Initialize an empty list to keep track of selected filters
        selected_filters = []

        # Pick filters that correspond to these observed wavelengths
        r_filter = find_closest_filter(observed_v, exclude_filters=selected_filters)
        selected_filters.append(r_filter)
        g_filter = find_closest_filter(observed_b, exclude_filters=selected_filters)
        selected_filters.append(g_filter)
        b_filter = find_closest_filter(observed_u, exclude_filters=selected_filters)
        selected_filters.append(b_filter)

        #print(f"Initial Filter Selection - R: {r_filter}, G: {g_filter}, B: {b_filter}")

        # Ensure the filters are ordered by wavelength (R > G > B)
        def enforce_wavelength_order(r, g, b):
            filters_with_wavelengths = [(r, jwst_filters[r]), (g, jwst_filters[g]), (b, jwst_filters[b])]
            sorted_filters = sorted(filters_with_wavelengths, key=lambda x: x[1], reverse=True)
            # Return the filters in correct wavelength order: R (longest), G, B (shortest)
            return sorted_filters[0][0], sorted_filters[1][0], sorted_filters[2][0]

        # Apply wavelength order enforcement
        r_filter, g_filter, b_filter = enforce_wavelength_order(r_filter, g_filter, b_filter)

        #print(f"Final Filter Selection - R: {r_filter}, G: {g_filter}, B: {b_filter}")

        # Return the final (r, g, b) filter combination
        return r_filter, g_filter, b_filter

def rgb_img_make(*args, z, path=''):
    if IPython.get_ipython().parent_header.get('msg_type') != 'execute_request':
        plt.close('all')
    filts = pick_rgb_filters(z)
    superf = ['F090W', 'F115W', 'F150W', 'F200W', 'F277W', 'F356W', 'F444W']
    for j in args:
        if path:
            j = path + j
        imags = []
        for i in range(len(superf)):
            if superf[i] in filts:
                with fits.open(j) as hdul:
                    imags.append(hdul[0].data[i])
        stretch = 0.5
        Q = 10
        rgb_image = make_lupton_rgb(imags[0], imags[1], imags[2], stretch=stretch, Q=Q)
        plt.imshow(rgb_image)
        plt.title(j[5:19])
        plt.show()

def rgb_img_make_statmorph(imgstack, labels, z, scale=0.01):
    if IPython.get_ipython().parent_header.get('msg_type') != 'execute_request':
        plt.close('all')
    filts = pick_rgb_filters(z)
    superf = ['F090W', 'F115W', 'F150W', 'F200W', 'F277W', 'F356W', 'F444W']
    rgb_imgs, rgb_imgs_re = [], []
    tmp = 0
    
    for j in imgstack:
        imags = []
        for i in range(len(superf)):
            if superf[i] in filts:
                imags.append(j[i])
        stretch = 0.5
        Q = 10
        #print(imags)
        #plt.imshow(imags[0][1].data[0])
        #showme3(imags[0], None)
        rgb_image = make_lupton_rgb(imags[0][0], imags[1][0], imags[2][0], stretch=stretch, Q=Q)
        rgb_image_re = make_lupton_rgb(imags[0][1], imags[1][1], imags[2][1], stretch=stretch, Q=Q)
        rgb_imgs.append(rgb_image)
        rgb_imgs_re.append(rgb_image_re)
        
        fig, ax = plt.subplots(1, 2, figsize=(9, 6))
        ax[0].imshow(rgb_image_re, origin='lower')
        ax[1].imshow(rgb_image, origin='lower')
        plt.suptitle('JWST NIRCam RGB Images')
        fig.supxlabel(labels[tmp])
        ax[0].set_ylabel('Raw RGB')
        ax[1].set_ylabel('STATMORPH RGB')
        
        scalebar_pixels = kpc_to_pixels(2, z, arcsec_per_pixel=scale)
        x_pos = rgb_image.shape[1] - 10 - scalebar_pixels
        y_pos = 10
        ax[0].hlines(y=y_pos, xmin=x_pos, xmax=x_pos + scalebar_pixels, color='white', linewidth=2)
        ax[0].text(x_pos + scalebar_pixels / 2, y_pos + 5, '2 kpc', color='white',
            ha='center', va='bottom', fontsize=8)
        ax[1].hlines(y=y_pos, xmin=x_pos, xmax=x_pos + scalebar_pixels, color='white', linewidth=2)
        ax[1].text(x_pos + scalebar_pixels / 2, y_pos + 5, '2 kpc', color='white',
            ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.show();
        tmp+=1
    
    return rgb_imgs, rgb_imgs_re


# In[4]:


def stack_wrapper(filename, verbose=True, verbosef=False, showimg=[False,False], exstr='', z=None, scale=0.01):
    try:
        assert(statLibTestFunc() == 0)
    except NameError:
        import import_ipynb
        import statmorph_lib
    
    longf = ['F277W', 'F356W', 'F444W']
    longwaves = [2.770,3.560,4.440]
    shortf = ['F090W', 'F115W', 'F150W', 'F200W']
    shortwaves = [.900,1.150,1.500,2.000]
    superf = ['F090W', 'F115W', 'F150W', 'F200W', 'F277W', 'F356W', 'F444W']
    superwaves = [.900,1.150,1.500,2.000,2.770,3.560,4.440]
    
    zplongfilters=[26.768, 26.974, 27.173]
    zpshortfilters=[25.395, 25.883, 26.276, 26.518]
    zpsuperfilters=[25.395, 25.883, 26.276, 26.518, 26.768, 26.974, 27.173]

    with fits.open(filename) as hdul:
        hdu = hdul[0]
        header = hdu.header
        naxis3 = header['NAXIS3']
        
    if (naxis3 == 3):
        obs_waves, zeropoint = longwaves, zplongfilters
        f = longf
        #print("Image Type: Long")
    elif (naxis3 == 4):
        obs_waves, zeropoint = shortwaves, zpshortfilters
        f = shortf
        #print("Image Type: Short")
    elif (naxis3 == 7):
        obs_waves, zeropoint = superwaves, zpsuperfilters
        f = superf
        #print("Image Type: Super")
    else:
        print('No filter type specified.')
        return 1
    #print(f'Filter image stack is {f}')
    #print()
    
    frames = []
    imgs = []
    for i in range(len(f)):
        frame, img = statMain(filename, wave=i, rmBG=False, verb=verbosef, plotted=[False, False], exstr=exstr)
        if verbosef:
            print(frame)
        frames.append(frame[0]) #Will need to update this when multiple models are used
        imgs.append(img)

    mags, indices, radius, pas, ars = [],[],[],[],[]
    #t=0
    for i in frames:
    #    if verbose:
    #        print(f'Filter {f[t]} ({t})')
    #        printnicely(i)
    #    t+=1
        mags.append(i.sersic_amplitude)
        indices.append(i.sersic_n)
        #print(i.sersic_rhalf) #in px
        if not z == None:
            rad_kpc = pixels_to_kpc(i.sersic_rhalf, z, pixel_scale=scale)
        else:
            raise ValueError("z cannot be None")
        radius.append(rad_kpc)
        pas.append(i.sersic_theta)
        ars.append(i.sersic_ellip)
        if verbosef:
            print('flag =', i.flag)
            print('flag_sersic =', i.flag_sersic)
    #print(mags)
    #print('Integrated Fluxes:', fluxes) #In MJy
    #print('Half Light Radii:', radius) #In Pix
    #print('Sersic Indicies:', indices)
    #print('Position Angles:', pas) #In Pix
    #print('Axis Ratios:', ars)
    np_sizes = np.array(radius)      # half-light radii at obs_waves
    #np_indices = np.array(indices) # not used here, but could be
    np_pas = np.array(pas)

    if verbose:
        print(f'{"Filt":<6}{"Wavel":<10}{"Amp":<10}{"Index":<10}{"HL Rad":<10}{"Ell":<10}{"Pos Ang":<10}')
        for i in range(len(obs_waves)):
            if not i==0 and (showimg[0] or showimg[1]):
                print(f'{"Filt":<6}{"Wavel":<10}{"Amp":<10}{"Index":<10}{"HL Rad":<10}{"Ell":<10}{"Pos Ang":<10}')
            print(f'{i:<6}{obs_waves[i]:<10.3f}{mags[i]:<10.3f}{indices[i]:<10.3f}{radius[i]:<10.3f}{ars[i]:<10.3f}{pas[i]:<10.3f}')
            if showimg[0]:
                print()
                print(f'{filename[5:19]}, filter {i}')
                plotImage(imgs[i][1], imgs[i][0])
            if showimg[1]:
                print()
                print(f'{filename[5:19]}, filter {i}')
                plotMorph(frames[i])
            

    if showimg[0] and not verbose:
        for i in range(len(imgs)):
            print(f'{filename[5:19]}, filter {i}')
            plotImage(imgs[i][1], imgs[i][0])
    if showimg[1] and not verbose:
        for i in range(len(imgs)):
            print(f'{filename[5:19]}, filter {i}')
            plotMorph(frames[i])
    
    # A model galaxy template (rest-frame)
    #rest_waves = np.linspace(.300, 2.000, 100)  # in microns
    #template_fluxes = np.exp(-((rest_waves - 1.000)/.200)**2)  # simple Gaussian SED
    
    # Base size (e.g., size at z=0)
    #base_size = 1.0
    
    #z_est, ti = estimate_redshift(obs_waves, np_fluxes, np_sizes, np_pas)
    #print(f"Estimated redshift: z = {z_est}") #:.2f

    return frames, imgs, mags, obs_waves, radius, indices, pas, ars, f

def main_stack(*args, path=None, verbose=False, verbosef=False, showimg=[False,False], plotted=False, exstr='', z=None, scale=0.01, col3=True):  
    if IPython.get_ipython().parent_header.get('msg_type') != 'execute_request':
        plt.close('all')
    filts = pick_rgb_filters(z)
    framestack, imgstack, ampstack, zs, radstack = [], [], [], [], []
    i_s,r_s,a_s,p_s=[],[],[],[]
    t=0
    if args:
        print('Number of file(s):', len(args))
        for i in args:
            if path:
                i = path + i
            if verbose or showimg[0] or showimg[1]:
                print()
                print()
                print('For FITS file:', args[t][:14], f'({t})')
                print()
            t+=1
            frames, imgs, amps, obs_waves, radius, i0, p0, a0, f = stack_wrapper(i, verbose, verbosef, showimg, exstr=exstr, z=z, scale=scale)
            
            framestack.append(frames)
            imgstack.append(imgs)
            ampstack.append(amps)
            radstack.append(radius)
            
            if col3:
                i0_s,r0_s,a0_s,p0_s=[],[],[],[]
                for i in range(len(radius)):
                    if f[i] in filts:
                        #print(f[i], filts, radius[i])
                        i0_s.append(i0[i])
                        a0_s.append(a0[i])
                        r0_s.append(radius[i])
                        p0_s.append(p0[i])
            else:
                i0_s=i0
                a0_s=a0
                r0_s=radius
                p0_s=p0
            i_s.append(np.average(i0_s))
            a_s.append(np.average(a0_s))
            r_s.append(np.average(r0_s))
            p_s.append(np.average(p0_s))
    else:
        print('No (argument/s) filename/s provided!')
        return 1

    if plotted:
        fig, ax = plt.subplots(1, 3, figsize=(9, 4))
        ax[0].grid()
        ax[1].grid()
        ax[2].grid()
        for i in range(len(ampstack)):
            ax[0].plot(obs_waves, ampstack[i], label=args[i][:14])
            ax[1].plot(obs_waves, ampstack[i]/np.max(ampstack[i]), label=args[i][:14])
            ax[2].plot(obs_waves, radstack[i], label=args[i][:14])
        plt.suptitle('JWST NIRCam Filter Wavelength vs STATMORPH Estimated Data')
        fig.supxlabel('Filter Wavelength (Microns)')
        ax[0].set_ylabel('Sersic Amplitude')
        ax[1].set_ylabel('Sersic Amplitude (Normalized per Image)')
        ax[2].set_ylabel('Half Light Radius (kpc)')
        plt.legend()
        plt.tight_layout()
        plt.show();

    labels=[]
    for i in range(len(ampstack)):
        labels.append(args[i][:14])

    if showimg[0] or showimg[1] or verbose:
        if col3:
            print('3-Color Filter Averages')
        else:
            print('Filter Averages')
        print(f'{"Chosen Img":<15}{"Index":>10}{"HL Rad":>10}{"Ell":>10}{"Pos Ang":>10}')
        for i in range(len(labels)):
            print(f'{labels[i]:<15}{i_s[i]:>10.3f}{r_s[i]:>10.3f}{a_s[i]:>10.3f}{p_s[i]:>10.3f}')

    #print("Sersic Indicies (ave) by image:",np.round(i_s,3))
    #print("Half Light Radii (ave) by image:",np.round(r_s,3))
    #print("Axis Ratios (ave) by image:",np.round(a_s,3))
    #print("Position Angles (ave) by image:",np.round(p_s,3))
    print()
    #print('Average Redshift: z =', np.average(zs))
    
    return framestack, imgstack, ampstack, radstack, labels


# z1 = 2.45
# x1, y1, f1, r1, l1 = main_stack(NH201_1, NH201_2, NH201_3, NH201_4, path='fits/', plotted=True, verbose=True, showimg=[True,False], z=z1)

# #rgb_img_make(NH201_1, NH201_2, NH201_3, NH201_4, z=z, path='fits/')
# rgb_1 = rgb_img_make_statmorph(y1, l1, z=z1)

# z2 = 3.10
# x2, y2, f2, r2, l2 = main_stack(NH153_1, NH153_2, NH153_3, NH153_4, NH153_14, path='fits/', plotted=True, verbose=True, z=z2)

# #rgb_img_make(NH201_1, NH201_2, NH201_3, NH201_4, z=z, path='fits/')
# rgb_2 = rgb_img_make_statmorph(y2, l2, z=z2)

# z3 = 3.45
# x3, y3, f3, r3, l3 = main_stack(NH134_1, NH134_2, NH134_3, NH134_11, NH134_14, path='fits/', plotted=True, verbose=True, showimg=[True,False], z=z3)

# rgb_3 = rgb_img_make_statmorph(y3, l3, z=z3)

# In[ ]:




