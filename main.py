import argparse
from fitsfile import FitsFile
import util

def main():
    # Doing brief cmd line parsing.
    parser = argparse.ArgumentParser(description="Calculate continuum fits.") 
    parser.add_argument("--export", help="--export <outputfile>")
    parser.add_argument("-l", 
       help="use this flag to loop through all fits files", action="store_true")
    args = parser.parse_args()

    # We need to write a function that will automatically perform these routines
    # so that we can determine for which functions this code does/does not work.
    # For each one, we should identify the assertion that failed and see what we
    # can change so that it does work.

    directory = "fits_files/"
    default_path = directory + "r0760_stitched.fits"

    if args.l is not False:
        for fits_path in os.listdir(directory):
            fits_file = FitsFile(directory+fits_path)
            util.perform_fits(fits_file)
    else:
        fits_file = FitsFile(default_path)
        util.perform_fits(fits_file)
        util.save(fits_file)




    if args.export is not None:
        file_name = args.export
        export_spectra(file_name, spectra) # exports the spectrum to a txt.



if __name__ == "__main__":
    main()
