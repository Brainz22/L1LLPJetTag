import numpy as np
import pytest

from L1LLPJetTagger.core import add
from L1LLPJetTagger import delta_phi


def test_add():
    assert add(2, 3) == 5


def test_delta_phi():
    expected1 = 4 - 2 * np.pi
    assert delta_phi(5, 1) == pytest.approx(expected1)
    expected2 = -4 + 2 * np.pi
    assert delta_phi(1, 5) == pytest.approx(expected2)
