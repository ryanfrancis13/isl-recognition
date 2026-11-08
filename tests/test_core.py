import numpy as np
from isl.core import FEAT_DIM, FRAME_DIM, SEQ_LEN, frame_features
from isl.train import augment, clip_to_windows


def test_hand_invariance_and_missing_zeros():
    rng = np.random.default_rng(0)
    pose = rng.uniform(0.2, 0.8, (33, 3)).astype(np.float32)
    hand = rng.uniform(0.2, 0.8, (21, 3)).astype(np.float32)

    f1 = frame_features(pose, None, hand)
    f2 = frame_features(pose, None, hand * 3.0 + np.array([5.0, -2.0, 1.0], dtype=np.float32))

    assert f1.shape == (FRAME_DIM,)
    np.testing.assert_allclose(f1[:63], f2[:63], atol=1e-5)
    assert np.all(f1[63:126] == 0.0)


def test_window_slicing_and_masked_augment():
    stream = np.zeros((65, FRAME_DIM), dtype=np.float32)
    stream[:, :63] = 1.0
    wins = clip_to_windows(stream)

    assert len(wins) == 8
    assert wins[0].shape == (SEQ_LEN, FEAT_DIM)

    aug = augment(wins[0])
    assert aug.shape == (SEQ_LEN, FEAT_DIM)
    assert np.all(aug[:, 63:126] == 0.0)
