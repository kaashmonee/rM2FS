import argparse
from fitsfile import FitsFile
import util
import os
import gaussfit

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

    directory = "fits_batch_2/"
    fn = "r0136_stitched.fits"
    # fn = "b2311_stitched.fits"
    # directory = "fits_batch_1/"
    # fn = "r0760_stitched.fits"

    default_path = directory + fn

    if args.l is not False:
        for num, fits_path in enumerate(os.listdir(directory)):
            try:
                fits_file = FitsFile(directory+fits_path)

                num_files = len(os.listdir(directory))
                print("Fitting %s (%d/%d)" % (fits_file.get_file_name(), num+1, num_files))

                util.perform_fits(fits_file)
            except Exception as e:
                print("exception: ", e)
                print(fits_file.get_file_name() + " failed")
    else:
        # Check if the file already exists in the directory first
        pickle_directory = "fitted_files/"
        pickle_fp = pickle_directory + fn + ".pkl"
        pickled_fits_file = None
        save = True if args.s is not False else False

        try:
            with open(pickle_fp) as f:
                print("Found pickled file in fitted_files/. Plotting spectra...")
                fits_file = util.load(pickle_fp)
                fits_file.plot_spectra(show=True, save=save)
        except FileNotFoundError as e:
            error_message = "Pickled file not found in fitted_files/ directory. Fitting the image..."
            print(error_message)

            fits_file = FitsFile(default_path) 
            util.perform_fits(fits_file)

        print("\nDone")


if __name__ == "__main__":
    main()

