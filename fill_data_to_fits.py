import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

#remove some data from a fits
fits_file='spec-1678-53433-0001(2).fits'

with fits.open(fits_file) as f:
    header=f[0].header
    data=f[1].data

#sky bg from the some data in numpy array
sky=data["sky"]


# Create a PrimaryHDU object (this can be empty if you want to store data only in extensions)
primary_hdu = fits.PrimaryHDU()

# Create an ImageHDU object with the NumPy array as the data
hdu = fits.ImageHDU(data=sky)

# Create an HDU list with the PrimaryHDU and the ImageHDU
hdul = fits.HDUList([primary_hdu, hdu])

# Write the HDU list to a FITS file
hdul.writeto('input.fits', overwrite=True)
