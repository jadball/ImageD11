import unittest
import numpy as np
import ImageD11.stress


class test_full_3x3_to_vector(unittest.TestCase):
    def test_default_strain(self):
        input_tensor = np.array([[11, 12, 13],
                                 [21, 22, 23],
                                 [31, 32, 33]])

        output_format = 'default'
        is_strain = True
        desired_output_vector = np.array([11, 22, 33, 23, 13, 12])
        output_vector = ImageD11.stress.full_3x3_to_vector(input_tensor, output_format, is_strain)
        self.assertTrue(np.array_equal(desired_output_vector, output_vector))
