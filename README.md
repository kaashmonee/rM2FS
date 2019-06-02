# rM2FS
This projects intends to develop a better reduction routine for the M2FS telescope located in Las Campanas, Chile. The current methods of reduction including IRAF are not ideal and we would ultimately like to come up with a better alternative.

## Getting Started
At this point in time we are attempting to develop a way to fit a continuum to the different orders of the spectrograph. The `continuum.py` file is executable.

## Prerequisites
This project is written in Python 3.6.7.

## Installing
It is recommended to use a virtual environment. The required libraries are outlined in the `requirements.txt.` Run `pip install -r requirements.txt`.

## Running
As mentioned the `continuum.py` runs and fits continuums for all the spectra in the fits_files/r0760_stitched.fits file. An optional --export command line argument has been 
added which will save all the spectra to an output text file.

## Authors
Skanda Kaashyap, Carnegie Mellon University
