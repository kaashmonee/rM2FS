import unittest
import numpy as np
from spectrum import Spectrum
import util
import sys

class SpectrumUnitTest(unittest.TestCase):
    """
    Unit tests for the Spectrum class.
    """

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

        self.assertTrue(len(xvalues) == len(new_xvals))
        self.assertTrue(len(yvalues) == len(new_yvals))
        self.assertTrue(np.array_equal(xvalues, new_xvals))
        self.assertTrue(np.array_equal(yvalues, new_yvals))

    # def test_sigma_clipping_failures3(self):
    #     """
    #     This ensures that the alert message is displayed if over 10% of 
    #     the pixels are removed.
    #     """
    #     xvalues = np.arange(10)
    #     yvalues = np.random.normal(10, 3, 10)
    #     yvalues[3] = -5
    #     yvalues[5] = 100

    #     new_xvals, new_yvals = util.sigma_clip(xvalues, yvalues)

    #     xvalues = np.delete(xvalues, [3, 5])
    #     yvalues = np.delete(yvalues, [3, 5])

    #     message = "Over 10% of pixels have been rejected in the sigma_clip routine."

    #     expected = """
    #     ==========================
    #     Warning: %s
    #     ==========================
    #     """
    #     captured = sys.__stdout__
    #     print(sys.__stdout__)
    #     print(expected % message)

    #     self.assertEqual(expected, captured)


if __name__ == "__main__":
    unittest.main()

