import unittest
import numpy as np
from spectrum import Spectrum
import util

class SpectrumUnitTest(unittest.TestCase):
    """
    Unit tests for the Spectrum class.
    """

    # def test_narrow(self):
    #     """
    #     Unit testing the narrow function.
    #     """
    #     xvals = np.array([1, 2, 3, 4, 5])
    #     yvals = np.array([2, 2.5, 4, 8, 9])
    #     self.test_spectrum = Spectrum(xvals, yvals)
    #     print(self.test_spectrum.xvalues)
    #     print(self.test_spectrum.yvalues)
    #     self.assertTrue(np.array_equal(self.test_spectrum.xvalues, np.array([1, 2, 5])))
    #     self.assertTrue(np.array_equal(self.test_spectrum.yvalues, np.array([2, 2.5, 9])))

    
    def test_sigma_clipping_no_failures(self):
        """
        Unit testing the sigma cipping function.
        """
        # Generate random data
        xvalues = np.arange(0, 100)
        yvalues = np.random.normal(100, 3, 100)

        new_xvals, new_yvals = util.sigma_clip(xvalues, yvalues)

        ## Testing success
        self.assertTrue(new_xvals.all() == xvalues.all()
                        and new_yvals.all() == yvalues.all())




    def test_sigma_clipping_failures1(self):
        """
        Tests sigma clipping failure on elements that are in the middle of an
        array.
        """
        xvalues = np.arange(0, 100)
        yvalues = np.random.normal(100, 3, 100)
        yvalues[4] = 10**5
        yvalues[55] = -40
        yvalues[80] = -90

        new_xvals, new_yvals = util.sigma_clip(xvalues, yvalues)

        xvalues = np.delete(xvalues, [4, 55, 80])
        yvalues = np.delete(yvalues, [4, 55, 80])

        # print("yvalues:", yvalues)
        # print("new_yvals:", new_yvals)

        self.assertTrue(np.array_equal(xvalues, new_xvals))
        self.assertTrue(np.array_equal(yvalues, new_yvals))

    def test_sigma_clipping_failures2(self):
        """
        Tests sigma clipping function on failing the first and last elements.
        """
        xvalues = np.arange(0, 100)
        yvalues = np.random.normal(100, 3, 100)
        yvalues[0] = 10**5
        yvalues[len(yvalues)-1] = -100

        new_xvals, new_yvals = util.sigma_clip(xvalues, yvalues)

        xvalues = np.delete(xvalues, [0,len(xvalues)-1])
        yvalues = np.delete(yvalues, [0,len(yvalues)-1])

        # print("yvals:", yvalues)
        # print("new_yvals:", new_yvals)

        self.assertTrue(len(xvalues) == len(new_xvals))
        self.assertTrue(len(yvalues) == len(new_yvals))
        self.assertTrue(np.array_equal(xvalues, new_xvals))
        self.assertTrue(np.array_equal(yvalues, new_yvals))


if __name__ == "__main__":
    unittest.main()

