from enum import Enum
from functools import partial

import dask.array as da
import numpy as np
from magicgui import magic_factory
from napari.layers import Image, Labels
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from skimage.filters import (
    threshold_isodata,
    threshold_li,
    threshold_otsu,
    threshold_triangle,
    threshold_yen,
)
from skimage.measure import label
class Threshold(Enum):
    # plain functions can't be Enum members, so we wrap these in partial
    # this doesn't change their behaviour
    isodata = partial(threshold_isodata)
    li = partial(threshold_li)
    otsu = partial(threshold_otsu)
    triangle = partial(threshold_triangle)
    yen = partial(threshold_yen)


HIST_THRESHOLDS = [Threshold.isodata.name, Threshold.otsu.name, Threshold.yen.name]


class ThresholdSegmentationWidget(QWidget):

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())

        # make combo box for selecting image layer
        self.im_layer_combo = self.add_image_combo_box()

        # if the user adds or removes layers we want to update the combo box
        self.viewer.layers.events.inserted.connect(self._reset_layer_options)
        self.viewer.layers.events.removed.connect(self._reset_layer_options)

        # make combo box for selecting threshold
        self.threshold_combo = self.add_threshold_combo_box()

        # make checkbox for using histogram
        self.hist_checkbox = self.add_histogram_checkbox()
        self.threshold_combo.currentTextChanged.connect(self._toggle_histogram_checkbox)

        # make button for segmentation
        self.segment_btn = QPushButton("Segment")
        self.segment_btn.clicked.connect(self._segment_image)

        self.layout().addWidget(self.segment_btn)
        self.layout().addStretch()

    def add_image_combo_box(self):
        """Add combo box for selecting image layer

        Returns
        -------
        QComboBox
            Combo box with image layers as items
        """

        # make a new row to put label and combo box in
        combo_row = QWidget()
        combo_row.setLayout(QHBoxLayout())
        # we don't want margins so it looks all tidy
        combo_row.layout().setContentsMargins(0, 0, 0, 0)

        new_combo_label = QLabel("Image Layer")
        combo_row.layout().addWidget(new_combo_label)

        new_layer_combo = QComboBox(self)
        # only adding image layers
        new_layer_combo.addItems(
            [layer.name for layer in self.viewer.layers if isinstance(layer, Image)]
        )
        combo_row.layout().addWidget(new_layer_combo)

        self.layout().addWidget(combo_row)

        # returning the combo box so we know which is which when we click run
        return new_layer_combo

    def add_threshold_combo_box(self):
        """Add combo box for selecting threshold method

        Returns
        -------
        QComboBox
            Combo box with Threshold values as items
        """

        # make a new row to put label and combo box in
        combo_row = QWidget()
        combo_row.setLayout(QHBoxLayout())
        # we don't want margins so it looks all tidy
        combo_row.layout().setContentsMargins(0, 0, 0, 0)

        new_combo_label = QLabel("Threshold")
        combo_row.layout().addWidget(new_combo_label)

        threshold_combo = QComboBox(self)
        threshold_combo.addItems([threshold.name for threshold in Threshold])
        combo_row.layout().addWidget(threshold_combo)

        self.layout().addWidget(combo_row)

        return threshold_combo

    def add_histogram_checkbox(self):
        checkbox_row = QWidget()
        checkbox_row.setLayout(QHBoxLayout())
        checkbox_row.setContentsMargins(0, 0, 0, 0)

        checkbox = QCheckBox()
        checkbox.setText("Use 90% histogram")
        checkbox_row.layout().addWidget(checkbox)

        self.layout().addWidget(checkbox_row)

        return checkbox

    def _reset_layer_options(self, event):
        """Clear existing combo boxes and repopulate

        Parameters
        ----------
        event : event
            Clear existing combo box items and query viewer for all image layers
        """
        self.im_layer_combo.clear()
        self.im_layer_combo.addItems(
            [layer.name for layer in self.viewer.layers if isinstance(layer, Image)]
        )

    def _toggle_histogram_checkbox(self, current_threshold):
        if current_threshold not in HIST_THRESHOLDS:
            self.hist_checkbox.setChecked(False)
            self.hist_checkbox.setDisabled(True)
            self.hist_checkbox.setStyleSheet("color: #808080;")
        else:
            self.hist_checkbox.setEnabled(True)
            self.hist_checkbox.setStyleSheet("color: #D3D3D3;")

    def _segment_image(self, event):
        im = self.viewer.layers[self.im_layer_combo.currentText()]
        if isinstance(im.data, da.Array):
            im.data = im.data.compute()
        threshold_func = Threshold[self.threshold_combo.currentText()].value
        if self.hist_checkbox.isChecked():
            self._segment_with_hist(im, threshold_func)
        else:
            self._segment_with_im(im, threshold_func)

    def _segment_with_im(self, im, threshold_func):
        im_data = im.data
        threshold_val = threshold_func(im_data)
        self._add_segment_image(im, threshold_val)

    def _segment_with_hist(self, im, threshold_func):
        im_data = im.data
        lower, upper = tuple(np.quantile(im_data, [0.05, 0.95]))
        counts, bin_edges = np.histogram(im_data, range=(lower, upper))
        bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
        threshold_val = threshold_func(hist=(counts, bin_centers))
        self._add_segment_image(im, threshold_val)

    def _add_segment_image(self, im, threshold_val):
        im_data = im.data
        binarised_im = im_data > threshold_val
        seg_labels = label(binarised_im)

        seg_layer = Labels(
            seg_labels, name=f"{im.name}_{self.threshold_combo.currentText()}"
        )
        self.viewer.add_layer(seg_layer)


@magic_factory
def segment_by_threshold(
    img_layer: "napari.layers.Image", threshold: Threshold
) -> "napari.types.LayerDataTuple":
    """Returns segmented labels layer given an image layer and threshold function.

    Magicgui widget providing access to five scikit-image threshold functions
    and layer selection using a combo box. Layer is segmented based on threshold choice.

    Returns
    -------
    napari.types.LayerDataTuple
        tuple of (data, meta, 'labels') for consumption by napari
    """
    if isinstance(img_layer.data, da.Array):
        img_layer.data = img_layer.data.compute()
    # need to use threshold.value to get the function from the enum member
    threshold_val = threshold.value(img_layer.data)
    binarised_im = img_layer.data > threshold_val
    seg_labels = label(binarised_im)

    seg_layer = (seg_labels, {"name": f"{img_layer.name}_{threshold.name}"}, "labels")

    return seg_layer
