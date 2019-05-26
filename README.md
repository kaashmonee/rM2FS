# rM2FS
This projects intends to develop a better reduction routine for the M2FS telescope located in Las Campanas, Chile. The current methods of reduction including IRAF are not ideal and we would ultimately like to come up with a better alternative.

## Getting Started
At this point in time we are attempting to develop a way to fit a continuum to the different orders of the spectrograph. The `continuum.py` file is executable.

## Prerequisites
This project is written in Python 3.6.7.

## Installing
It is recommended to use a virtual environment. The required libraries are outlined in the `requirements.txt.` Run `pip install -r requirements.txt`.

## Running
As mentioned the `continuum.py` file contains all current progress and when complete, should run and create a list of orders and their continuum fits. Run with `python continuum.py`.

## Authors
Skanda Kaashyap, Carnegie Mellon University
