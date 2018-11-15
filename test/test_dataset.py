from unittest import TestCase

from dataset import Dataset
import torch


class TestDataset(TestCase):

    def setUp(self):
        super().setUp()
        path_to_data_dir = '../data'
        self._train_dataset = Dataset(path_to_data_dir, mode=Dataset.Mode.TRAIN)
        self._test_dataset = Dataset(path_to_data_dir, mode=Dataset.Mode.TEST)

    def test___len__(self):
        self.assertEqual(len(self._train_dataset), 50000)
        self.assertEqual(len(self._test_dataset), 10000)

    def test___getitem__(self):
        image, label = self._train_dataset[0]
        self.assertEqual(image.shape, torch.Size((3, 32, 32)))
        self.assertEqual(label.item(), 6)

        image, label = self._test_dataset[100]
        self.assertEqual(image.shape, torch.Size((3, 32, 32)))
        self.assertEqual(label.item(), 4)
