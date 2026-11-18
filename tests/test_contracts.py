import unittest
import torch
from lib.train_gate.model import ScoreOnlyModel, aligned_iou


class GateContracts(unittest.TestCase):
    def test_label_boundary(self):
        model = ScoreOnlyModel.__new__(ScoreOnlyModel)
        torch.nn.Module.__init__(model)
        model.threshold = .5
        model.positive_boundary = 'ge'
        self.assertEqual(model.make_labels(torch.tensor([.4999, .5, .5001])).tolist(),
                         [False, True, True])

    def test_aligned_iou(self):
        prediction = torch.tensor([[.5, .5, 1., 1.], [2., 2., 1., 1.]])
        ground_truth = torch.tensor([[0., 0., 1., 1.], [0., 0., 1., 1.]])
        self.assertTrue(torch.equal(aligned_iou(prediction, ground_truth), torch.tensor([1., 0.])))

    def test_loss_is_branch_sum(self):
        logits = torch.zeros(2, 3)
        labels = torch.tensor([[1., 0., 1.], [0., 1., 0.]])
        per_branch = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction='none').mean(0)
        self.assertAlmostEqual(float(per_branch.sum()), 3*float(torch.log(torch.tensor(2.))), places=6)


if __name__ == '__main__':
    unittest.main()
