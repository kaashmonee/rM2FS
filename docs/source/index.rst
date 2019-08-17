.. rM2FS documentation master file, created by
   sphinx-quickstart on Sat Aug 17 11:56:07 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

rM2FS
=================================
This project aims to develop a robust reduction pipeline for reducing the images taken by the M2FS telescope in Las Campanas, Chile.

Getting Started
================
The current codebase fits a continuum to each spectrum in the image while 
attempting to remove the overlapping regions. Before running, please run
    
.. codeblock:: bash
    cp config_templ.py config.py

Then populate the fields in `config.py` as desired. After this is done, run 
`python main.py`. The output should be saved in the directories desired in 
`config.py`.

Prerequisites
=============
This project is written in Python 3.6.7.

Installing
==========
It is recommended to use a virtual environment. The required libraries are 
specified in the `requirements.txt` file. Run `pip install -r requirements.txt`.

Running
=======
The `main.py` file is the primary entrypoint for the program. `main.py` also 
can take 2 command line arguments.

`python main.py -l` will run the code on all the files specified in the 
`directory` variable in the `config.py` file that you have created.

`python main.py -s` is for cases where you would like to run on an already saved 
file. The default behavior is not to save upon replotting but this flag will 
resave upon replotting. This is to account for cases where the plotting code has
been modified but the data models have been kept the same.

Security Warning
================
The current way to save files is via the Python pickling tool. This, however, poses a substantial security vulnerability as cited in: https://nvd.nist.gov/vuln/detail/CVE-2019-12760. This is not a concern if it accepts trusted pickle files, but should be of serious concern if this code is used over a network to accept unknown pickle files, which could lead to arbitrary code execution.

Authors
=======
Skanda Kaashyap, Carnegie Mellon University




.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
