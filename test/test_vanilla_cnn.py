import unittest

from core.model.cnn import VanillaCNN
from core.load.load_data import dirtyMNISTDataset

class TestVanillaCNN(unittest.TestCase):
    DATA_PATH = "../data/dirty_mnist/"
    def testdirtyMNISTDataset(self):
        