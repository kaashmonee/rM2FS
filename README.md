# rM2FS
This project aims to develop a robust reduction pipeline for reducing the images taken by the M2FS telescope in Las Campanas, Chile.

## Getting Started
The current codebase fits a continuum to each spectrum in the image while attempting to remove the overlapping regions. The `main.py`
file is executable and is the program's main point of entry. Please run `python main.py` to get started.

## Prerequisites
This project is written in Python 3.6.7.

## Installing
It is recommended to use a virtual environment. The required libraries are outlined in the `requirements.txt.` Run `pip install -r requirements.txt`.

## Running
As mentioned the `main.py` is the primary entry point. Below is the project workflow:

To run, run `python main.py` to run on the default file specified in `main.py`. 

It also takes the following command line arguments: `-l` and `-s`. 
`python main.py -l` will loop through every file in the directory specified in the `directory` variable in `main.py.`
By default, the program will save a pickled version of the fitted file and an svg image. Upon a subsequent run, it will look for the saved pickled file and attempt to re-plot. If the `-s` flag is enabled, then it will also save upon re-plotting and overwrite the original `.svg` file.

## Security Warning
The current way to save files is via the Python pickling tool. This, however, poses a substantial security vulnerability as cited in: https://nvd.nist.gov/vuln/detail/CVE-2019-12760. This is not a concern if it accepts trusted pickle files, but should be of serious concern if this code is used over a network to accept unknown pickle files, which could lead to arbitrary code execution.

## Authors
Skanda Kaashyap, Carnegie Mellon University
