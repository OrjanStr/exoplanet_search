# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,"..")
from data_processing import process_data
import unittest
import numpy as np

class Test_data_processing(unittest.TestCase):

    def test_shapes(self):
        proc = process_data()

        under_shape = proc.x_train_under.shape
        under_target_shape = (437,3197)

        shrink_shape = proc.x_train_shrink.shape
        shrink_target_shape = (800,3197)

        over_shape = proc.x_train_over.shape
        over_target_shape = (10100,3197)


        # test if shape is correct
        self.assertEqual(under_shape,under_target_shape)
        self.assertEqual(shrink_shape,shrink_target_shape)
        self.assertEqual(over_shape,over_target_shape)


if __name__ == '__main__':
    unittest.main()
