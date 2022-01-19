try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._reader import napari_get_reader #noqa
from ._widget import ThresholdSegmentationWidget, Threshold, segment_by_threshold #noqa
