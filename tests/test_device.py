"""Unit tests for the DeviceResolver utility class."""

import unittest
from unittest.mock import MagicMock, patch

from src.finetuner.core.device import DeviceResolver


class TestDeviceResolver(unittest.TestCase):
    """Verifies that DeviceResolver resolves and configures PyTorch devices correctly."""

    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    def test_get_optimal_device_cuda(self, mock_mps, mock_cuda) -> None:
        """Verifies that CUDA takes priority when available."""
        mock_cuda.return_value = True
        mock_mps.return_value = False
        self.assertEqual(DeviceResolver.get_optimal_device(), "cuda")

    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    def test_get_optimal_device_mps(self, mock_mps, mock_cuda) -> None:
        """Verifies that MPS is resolved if CUDA is missing but MPS is available."""
        mock_cuda.return_value = False
        mock_mps.return_value = True
        self.assertEqual(DeviceResolver.get_optimal_device(), "mps")

    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    def test_get_optimal_device_cpu(self, mock_mps, mock_cuda) -> None:
        """Verifies that CPU is the fallback device when CUDA and MPS are missing."""
        mock_cuda.return_value = False
        mock_mps.return_value = False
        self.assertEqual(DeviceResolver.get_optimal_device(), "cpu")

    def test_setup_trainer_device_distributed(self) -> None:
        """Verifies device setup under distributed training configuration."""
        mock_trainer = MagicMock()
        mock_trainer.config.local_rank = 0
        mock_trainer.config.fp16 = True
        mock_trainer.config.bf16 = False

        with patch("torch.cuda.set_device") as mock_set_device:
            DeviceResolver.setup_trainer_device(mock_trainer)
            mock_set_device.assert_called_once_with(0)
            self.assertEqual(mock_trainer.device.type, "cuda")
            self.assertTrue(mock_trainer.is_distributed)
