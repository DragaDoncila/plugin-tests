from typing import List
import numpy as np
from plugin_tests import napari_get_reader
import pytest

@pytest.fixture
def write_im_to_file(tmp_path):

    def write_func(filename):
        my_test_file = str(tmp_path / filename)
        original_data = np.random.rand(20, 20)
        np.save(my_test_file, original_data)

        return my_test_file, original_data

    return write_func

def test_get_reader_returns_callable(tmp_path):
    my_test_file = str(tmp_path / "myfile.npy")
    original_data = np.random.rand(20, 20)
    np.save(my_test_file, original_data)

    reader = napari_get_reader(my_test_file)
    assert callable(reader)

def test_get_reader_pass():
    """Calling get_reader with non-numpy file path returns None"""
    reader = napari_get_reader("fake.file")
    assert reader is None

def test_get_reader_path_list(write_im_to_file):
    """Calling get_reader on list of numpy files returns callable"""
    pth1, _ = write_im_to_file("myfile1.npy")
    pth2, _ = write_im_to_file("myfile2.npy")

    reader = napari_get_reader([pth1, pth2])
    assert callable(reader)

def test_reader_round_trip(write_im_to_file):
    my_test_file, original_data = write_im_to_file("myfile.npy")

    reader = napari_get_reader(my_test_file)
    assert callable(reader)

    layer_data_list = reader(my_test_file)
    assert isinstance(layer_data_list, List) and len(layer_data_list) > 0

    layer_data_tuple = layer_data_list[0]
    layer_data = layer_data_tuple[0]
    np.testing.assert_allclose(layer_data, original_data)
