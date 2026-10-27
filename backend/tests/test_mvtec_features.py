"""Unit tests for preprocessing and score path. Uses stub NN, never an untrusted .joblib."""
import numpy as np
from PIL import Image
from unittest.mock import patch
from app.inference.anomaly.mvtec import patch_features, predict_mvtec


def test_exact_descriptor_dimensions():
    x = np.full((160, 160, 3), 120, np.uint8)
    f = patch_features(x)
    assert f.shape == (256, 14)
    assert np.isfinite(f).all()


def test_score_top_five_percent():
    class FakeNN:
        def kneighbors(self, f, return_distance=True):
            d = np.arange(len(f), dtype=np.float64)
            return d[:, None], np.zeros((len(f),1), dtype=int)
    img = Image.new('RGB', (128,128), 'gray')
    with patch('app.inference.anomaly.mvtec.load_bank', return_value=(FakeNN(),np.zeros(14),np.ones(14),128,16)):
        result = predict_mvtec(img,'bottle',with_overlay=False)
    assert result['heatmap_grid'] == [16,16]
    assert result['anomaly_score_raw'] == round(np.arange(243,256).mean(),6)
    assert result['decision'] == 'not_assessed'
