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


        print("sigma_clipping no failures passed...")

    
    def test_sortxy(self):
        """
        Unit testing the sortxy function.
        """
        xvalues = list(range(100, 0, -1))
        yvalues = list(range(100, 0, -1))
        sorted_xvalues, sorted_yvalues = util.sortxy(xvalues, yvalues)

        self.assertTrue(list(sorted_xvalues) == list(range(1, 101)))
        self.assertTrue(list(sorted_yvalues) == list(range(1, 101)))

        yvalues = list(range(500, 400, -1))
        sortedx, sortedy = util.sortxy(xvalues, yvalues)

        self.assertTrue(list(sortedx) == list(range(1, 101)))
        self.assertTrue(list(sortedy) == list(range(401, 501)))

        print("sortxy() passed...")



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

        print("sigma_clipping_failures1 passed...")

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

        print("sigma_clipping_failures2 passed...")

    
    def test_min_ind_range(self):
        test_array1 = [1, 3, 4, 5, 9, 0, 3, 4, -1, 30]
        start, end = 2, 5
        min_ind = util.min_ind_range(test_array1, start, end)
        self.assertTrue(min_ind == 2)

        start, end = 0, len(test_array1)//2
        min_ind = util.min_ind_range(test_array1, start, end)
        self.assertTrue(min_ind == 0)

        start, end = len(test_array1)//2, len(test_array1)
        min_ind = util.min_ind_range(test_array1, start, end)
        self.assertTrue(min_ind == 8)

        print("min_ind_range() tests passed...")



if __name__ == "__main__":
    unittest.main()

