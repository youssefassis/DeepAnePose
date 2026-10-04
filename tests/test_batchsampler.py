from deepanepose.training.batch_sampler import BalancedBatchSampler


def test_batches_hold_integer_indices_without_negative_patches():
    patches = [{"dir": "P0001", "status": True}] * 4 + [{"dir": "P0001", "status": False}]
    sampler = BalancedBatchSampler(patches, batch_size=2, percentage_neg_patches=15)  # rounds to 0 negatives

    indices = [i for batch in sampler for i in batch]

    assert sorted(indices) == [0, 1, 2, 3] and all(isinstance(i, int) for i in indices)
