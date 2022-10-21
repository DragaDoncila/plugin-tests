from plugin_tests import ThresholdSegmentationWidget , segment_by_threshold, Threshold
import numpy as np
import pytest
from napari.layers import Image

@pytest.fixture
def im_layer():
    return Image(np.random.random((5, 100, 100)), name="im")

def test_histogram_checkbox_toggle(qtbot, make_napari_viewer):
    viewer = make_napari_viewer()
    widg = ThresholdSegmentationWidget(viewer)
    qtbot.addWidget(widg)

    assert widg.threshold_combo.currentText() == "isodata"
    assert widg.hist_checkbox.isEnabled()

    widg.hist_checkbox.setChecked(True)
    widg.threshold_combo.setCurrentText("li")

    assert not widg.hist_checkbox.isEnabled()
    assert not widg.hist_checkbox.isChecked()


@pytest.mark.parametrize('threshold_func', [Threshold.isodata.value, Threshold.otsu.value, Threshold.yen.value])
def test_segment_with_hist_excludes_values(threshold_func, qtbot, make_napari_viewer):
    viewer = make_napari_viewer()
    widg = ThresholdSegmentationWidget(viewer)
    qtbot.addWidget(widg)

    # this should exclude the 0s and 100s as lower and upper 5%, and leave
    # a threshold of 1.5 (rather than ~50)
    zeroes = np.zeros(5)
    ones = np.ones(45)
    twos = np.empty(45)
    twos.fill(2)
    bigs = np.empty(5)
    bigs.fill(100)
    im = np.concatenate([zeroes, ones, twos, bigs]).reshape((10, 10))
    im_layer = Image(im)

    widg._segment_with_hist(im_layer, threshold_func)

    added_labels = viewer.layers[0].data
    assert added_labels.shape == (10, 10)
    # threshold will be 1.5, so only the 2s and 100s will be labelled
    assert list(np.unique(added_labels)) == [0, 1]
    # we only want 1s in labels layer where original image had 2s or 100s
    twos_labelled = np.where(im >= 2, 1, 0)
    assert np.all(twos_labelled == added_labels)

def test_segment_widg_returns_layer(im_layer):
    widg = segment_by_threshold()

    retval = widg(im_layer, Threshold.triangle)
    assert isinstance(retval[0], np.ndarray)
    assert retval[1]["name"] == "im_triangle"
    assert retval[2] == "labels"
