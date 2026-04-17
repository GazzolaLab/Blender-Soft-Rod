from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from elastica_blender.converter.npz2blend import (
    confirm_pyelastica_npz_structure,
    construct_blender_file,
)


def mock_npz_file(keys_data):
    """Utility function to create a mock .npz file-like object."""
    with BytesIO() as f:
        np.savez(f, **keys_data)
        f.seek(0)
        return f.read()


class TestConfirmPyelasticaNPZStructure:
    def test_confirm_structure_with_tags(self):
        # Create a mock .npz file with correct structure
        keys_data = {
            "time": np.array([0]),
            "rod1_position_history": np.array([0]),
            "rod1_radius_history": np.array([0]),
            "rod2_position_history": np.array([1]),
            "rod2_radius_history": np.array([1]),
        }
        path = BytesIO(mock_npz_file(keys_data))
        path.seek(0)

        # Test should pass without raising an exception
        confirm_pyelastica_npz_structure(path, ["rod1", "rod2"])

    def test_confirm_structure_without_tags(self):
        # Create a mock .npz file with correct structure
        keys_data = {
            "time": np.array([0]),
            "position_history": np.array([0]),
            "radius_history": np.array([0]),
        }
        path = BytesIO(mock_npz_file(keys_data))
        path.seek(0)

        # Test should pass without raising an exception
        confirm_pyelastica_npz_structure(path, None)

    def test_incorrect_key_structure_raises_key_error(self):
        # Create a mock .npz file with incorrect keys
        keys_data = {
            "time": np.array([0]),
            "rod1_history": np.array([0]),  # Incorrect key
            "radius_history": np.array([0]),
        }
        path = BytesIO(mock_npz_file(keys_data))
        path.seek(0)

        # Test should raise KeyError
        with pytest.raises(KeyError):
            confirm_pyelastica_npz_structure(path, ["rod1"])


# Test for constructing blender file from npz data


@pytest.fixture
def data_setup_without_tag(tmp_path_factory):
    directory = tmp_path_factory.mktemp("data1")
    data_path = directory / "data.npz"
    output_path = directory / "output.blend"

    time_array = np.array([0, 1, 2, 3])
    position_history = np.random.rand(1, 4, 3, 5)
    radius_history = np.random.rand(1, 4, 4)
    npz_data = {
        "time": time_array,
        "position_history": position_history,
        "radius_history": radius_history,
    }
    np.savez(data_path, **npz_data)

    return data_path, output_path, None


@pytest.fixture
def data_setup_with_tag(tmp_path_factory):
    directory = tmp_path_factory.mktemp("data2")
    data_path = directory / "data.npz"
    output_path = directory / "output.blend"

    time_array = np.array([0, 1, 2, 3])
    position_history_rod1 = np.random.rand(1, 4, 3, 5)
    radius_history_rod1 = np.random.rand(1, 4, 4)
    position_history_rod2 = np.random.rand(1, 4, 3, 5)
    radius_history_rod2 = np.random.rand(1, 4, 4)
    npz_data = {
        "time": time_array,
        "rod1_position_history": position_history_rod1,
        "rod1_radius_history": radius_history_rod1,
        "rod2_position_history": position_history_rod2,
        "rod2_radius_history": radius_history_rod2,
    }
    np.savez(data_path, **npz_data)

    return data_path, output_path, ["rod1", "rod2"]


class TestConstructBlenderFile:

    @pytest.fixture
    def data_setup_cases(self, data_setup_without_tag, data_setup_with_tag):
        return [data_setup_without_tag, data_setup_with_tag]

    def test_construct_blender_file_run(self, mocker, data_setup_cases):
        import bsr

        for data_setup in data_setup_cases:
            data_path, output_path, tags = data_setup
            rods_mock = mocker.Mock()
            mocker.patch("bsr.create_rod_collection", return_value=rods_mock)
            mocker.patch("bsr.save")

            construct_blender_file(data_path, output_path, tags)

            call_count = 1 if tags is None else len(tags)

            assert bsr.create_rod_collection.call_count == call_count
            assert (
                rods_mock.update_states.call_count == call_count * 4
            )  # 4 frames

            bsr.save.assert_called_once_with(output_path)

    def test_construct_blender_file_creation(
        self, mocker, data_setup_without_tag
    ):
        """Check file creation"""
        data_path, output_path, tags = data_setup_without_tag
        mocker.patch("bsr.create_rod_collection", return_value=mocker.Mock())
        construct_blender_file(data_path, output_path, tags)
        assert output_path.exists()
