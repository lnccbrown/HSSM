from unittest.mock import patch, MagicMock

import pytest
import onnx

from hssm.distribution_utils import download_hf, load_onnx_model
from hssm.distribution_utils.onnx_utils.model import REPO_ID


def test_download_hf():
    with patch(
        "hssm.distribution_utils.onnx_utils.model.hf_hub_download"
    ) as mock_download:
        mock_download.return_value = "/tmp/fake_model.onnx"
        result = download_hf("some_model.onnx")
        mock_download.assert_called_once_with(
            repo_id=REPO_ID, filename="some_model.onnx"
        )
        assert result == "/tmp/fake_model.onnx"


def test_load_onnx_model_with_modelproto():
    fake_model = MagicMock(spec=onnx.ModelProto)
    assert load_onnx_model(fake_model) is fake_model


def test_load_onnx_model_with_local_path(tmp_path):
    fake_onnx_path = tmp_path / "model.onnx"
    fake_onnx_path.write_bytes(b"fake")
    with patch("onnx.load") as mock_load:
        mock_load.return_value = "loaded_model"
        result = load_onnx_model(str(fake_onnx_path))
        mock_load.assert_called_once_with(fake_onnx_path)
        assert result == "loaded_model"


def test_load_onnx_model_with_hf_download():
    with (
        patch(
            "hssm.distribution_utils.onnx_utils.model.hf_hub_download"
        ) as mock_download,
        patch("onnx.load") as mock_load,
    ):
        mock_download.return_value = "/tmp/fake_model.onnx"
        mock_load.return_value = "loaded_model"
        result = load_onnx_model("remote_model.onnx")
        mock_download.assert_called_once_with(
            repo_id=REPO_ID, filename="remote_model.onnx"
        )
        mock_load.assert_called_once_with("/tmp/fake_model.onnx")
        assert result == "loaded_model"


def test_load_onnx_model_invalid_type():
    with pytest.raises(ValueError):
        load_onnx_model(12345)
