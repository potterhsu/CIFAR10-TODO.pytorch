import torch
from unittest import TestCase

from model import Model


class TestModel(TestCase):

    def setUp(self):
        super().setUp()
        self._model = Model()

    def test___init__(self):
        self.assertIsNotNone(self._model)

    def test_forward(self):
        images = torch.randn(8, 3, 32, 32)
        logits = self._model.train().forward(images)
        self.assertEqual(logits.shape, torch.Size((8, 10)))

    def test_loss(self):
        logits = torch.randn(8, 10)
        labels = torch.ones(8, dtype=torch.long)
        loss = self._model.loss(logits, labels)
        self.assertEqual(loss.shape, torch.Size())
