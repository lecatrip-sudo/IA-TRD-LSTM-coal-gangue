import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

import core


class TestCore(unittest.TestCase):
    def test_gold_metrics(self):
        m = core.metrics(np.array([1., 2., 3.]), np.array([2., 2., 1.]))
        self.assertAlmostEqual(m['MAE_K'], 1.)
        self.assertAlmostEqual(m['RMSE_K'], np.sqrt(5 / 3))
        self.assertAlmostEqual(m['MaxAE_K'], 2.)
        self.assertAlmostEqual(m['R2'], -1.5)

    def test_scaler_train_only(self):
        scaler = core.Scaler().fit(np.array([[1.], [3.]]), ['train_a', 'train_b'])
        np.testing.assert_allclose(scaler.transform([[5.]]), [[2.]])
        self.assertEqual(scaler.fit_ids, ['train_a', 'train_b'])
        np.testing.assert_allclose(scaler.inverse(scaler.transform([[9.]])), [[9.]])

    def test_scaler_rejects_nonfinite(self):
        with self.assertRaises(ValueError):
            core.Scaler().fit([[np.nan]], ['a'])

    def test_metric_permutation_property(self):
        rng = np.random.default_rng(91)
        for n in range(2, 30):
            y, p = rng.normal(size=n), rng.normal(size=n)
            order = rng.permutation(n)
            a, b = core.metrics(y, p), core.metrics(y[order], p[order])
            for key in ('MAE_K', 'RMSE_K', 'MaxAE_K', 'R2'):
                self.assertAlmostEqual(a[key], b[key], places=12)

    def test_metric_translation_metamorphic(self):
        y, p = np.array([1., 4., 9.]), np.array([2., 8., 5.])
        a, b = core.metrics(y, p), core.metrics(y + 273., p + 273.)
        for key in ('MAE_K', 'RMSE_K', 'MaxAE_K', 'R2'):
            self.assertAlmostEqual(a[key], b[key])

    def test_split_excludes_held_case(self):
        ids = core.training_ids(['a', 'b', 'c'], 'b')
        self.assertEqual(ids, ['a/24h', 'a/48h', 'a/72h', 'c/24h', 'c/48h', 'c/72h'])
        self.assertFalse(any('96h' in x or x.startswith('b/') for x in ids))

    def test_model_seed_forward_save(self):
        core.seed_all(42)
        a = core.Model()
        a.eval()
        x, t = torch.zeros(4, 3), torch.zeros(4, 2, 1)
        pa = a(x, t)
        core.seed_all(42)
        b = core.Model()
        b.eval()
        np.testing.assert_allclose(pa.detach(), b(x, t).detach(), atol=0., rtol=0.)
        self.assertEqual(pa.shape, (4, 1))
        self.assertTrue(torch.isfinite(pa).all())
        self.assertNotIn('alpha', a.state_dict())
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'state.pt'
            torch.save(a.state_dict(), p)
            b.load_state_dict(torch.load(p, weights_only=True))
            np.testing.assert_allclose(pa.detach(), b(x, t).detach(), atol=0., rtol=0.)

    def test_loss_gold_ablation(self):
        pred = torch.tensor([[2.], [4.]], requires_grad=True)
        target, last = torch.tensor([[1.], [3.]]), torch.tensor([[0.], [2.]])
        self.assertAlmostEqual(core.loss_value(pred, target, last, 0.).item(), 1.)
        self.assertAlmostEqual(core.loss_value(pred, target, last, .02).item(), 1.04, places=6)
        core.loss_value(pred, target, last, .02).backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_coordinate_balanced_no_cancellation(self):
        coords = np.array([[0., 0., 0.], [0., 0., 0.], [1., 0., 0.]])
        y, p = np.array([1., 3., 6.]), np.array([2., 2., 5.])
        m = core.coordinate_metrics(coords, y, p)
        self.assertEqual(m['n_coordinate_groups'], 2)
        self.assertAlmostEqual(m['MAE_K'], 1.)

    def test_aligned_loader(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'a.csv'
            p.write_text('X,Y,Z,temperature\n1,2,3,300\n1,2,3,301\n', encoding='utf-8')
            xyz, t = core.load_csv(p)
            self.assertEqual(xyz.shape, (2, 3))
            np.testing.assert_allclose(t, [[300.], [301.]])


if __name__ == '__main__':
    unittest.main(verbosity=2)
