import argparse
from fitsfile import FitsFile
import util
import os
import gaussfit
import config

def main():
    # Doing brief cmd line parsing.
    parser = argparse.ArgumentParser(description="Calculate continuum fits.") 
    parser.add_argument("-l", 
       help="use this flag to loop through all fits files", action="store_true")
    parser.add_argument("-s", help="save", action="store_true")
    args = parser.parse_args()

    # We need to write a function that will automatically perform these routines
    # so that we can determine for which functions this code does/does not work.
    # For each one, we should identify the assertion that failed and see what we
    # can change so that it does work.

    if args.l is not False:
        for num, fits_path in enumerate(os.listdir(config.directory)):
            try:
                fits_file = FitsFile(config.directory+fits_path)

                num_files = len(os.listdir(config.directory))
                print("Fitting %s (%d/%d)" % (fits_file.get_file_name(), num+1, num_files))

                util.perform_fits(fits_file)
            except Exception as e:
                print("exception: ", e)
                print(fits_file.get_file_name() + " failed")
    else:
        # Check if the file already exists in the directory first
        pickled_fits_file = None
        save = args.s

        try:
            with open(config.pickle_fp) as f:
                print("Found pickled file in fitted_files/. Plotting spectra...")
                fits_file = util.load(config.pickle_fp)
                fits_file.plot_spectra(show=True, save=save)
                fits_file.plot_spectra_brightness()
        except FileNotFoundError as e:
            error_message = "Pickled file not found in fitted_files/ directory. Fitting the image..."
            print(error_message)

            fits_file = FitsFile(config.default_path) 
            util.perform_fits(fits_file)

        print("\nDone")


if __name__ == "__main__":
    main()

