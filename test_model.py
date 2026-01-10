import argparse
import os
import time
from abc import ABC, abstractmethod
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

from nets import Model

class StereoModel(ABC):
    """
    Abstract base class for stereo vision models.
    """

    def __init__(self, model_path: str, input_width: int, input_height: int):
        """
        Initialize base stereo model.

        Args:
            model_path: Path to the model file
            input_width: Model input width
            input_height: Model input height
        """
        self.model_path = model_path
        self.input_width = input_width
        self.input_height = input_height

    @abstractmethod
    def _load_model(self):
        """Load the model. Must be implemented by subclasses."""
        pass

    def _preprocess_images(self, left_img: np.ndarray, right_img: np.ndarray) -> tuple:
        """
        Internal preprocessing function for stereo image pairs.

        Args:
            left_img: Left stereo image (H, W, C)
            right_img: Right stereo image (H, W, C)

        Returns:
            tuple: (left_batch, right_batch) as numpy arrays with shape (1, C, H, W)
        """
        dst_left = cv2.resize(left_img, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        dst_right = cv2.resize(right_img, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)

        dst_left = dst_left.transpose(2, 0, 1)
        dst_right = dst_right.transpose(2, 0, 1)
        dst_left = np.ascontiguousarray(dst_left[None, :, :, :]).astype("float32")
        dst_right = np.ascontiguousarray(dst_right[None, :, :, :]).astype("float32")

        return dst_left, dst_right

    def preprocess(self, left_img: np.ndarray, right_img: np.ndarray) -> tuple:
        """
        Preprocess stereo image pairs.

        Args:
            left_img: Left stereo image (H, W, C)
            right_img: Right stereo image (H, W, C)

        Returns:
            tuple: Preprocessed images (format depends on subclass)
        """
        return self._preprocess_images(left_img, right_img)

    @abstractmethod
    def inference(self, left_img: Union[np.ndarray, torch.Tensor],
                  right_img: Union[np.ndarray, torch.Tensor],
                  preprocess: bool = True) -> np.ndarray:
        """
        Run inference on stereo image pair.

        Args:
            left_img: Left stereo image (H, W, C) or preprocessed tensor/array
            right_img: Right stereo image (H, W, C) or preprocessed tensor/array
            preprocess: Whether to preprocess images (True) or use as-is (False)

        Returns:
            Disparity map as numpy array (H, W)
        """
        pass

    def get_input_shape(self) -> tuple:
        """
        Get model input shape.

        Returns:
            tuple: (width, height)
        """
        return (self.input_width, self.input_height)


class PytorchModel(StereoModel):
    def __init__(self,
                 model_path,
                 device='cuda',
                 max_disp=256,
                 mixed_precision=False,
                 input_width=480,
                 input_height=480,
                 n_iter=10):
        super().__init__(model_path, input_width, input_height)
        self.device = device
        self.max_disp = max_disp
        self.mixed_precision = mixed_precision
        self.n_iter = n_iter

        self.model = self._load_model()

    def _load_model(self):
        model = Model(max_disp=self.max_disp, mixed_precision=self.mixed_precision, test_mode=True)
        state_dict = torch.load(self.model_path, map_location=self.device)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()
        print(f"Pytorch model params: {sum(p.numel() for p in model.parameters())}")
        return model

    def preprocess(self, left_img: np.ndarray, right_img: np.ndarray) -> tuple:
        """
        Preprocess stereo image pairs for PyTorch model.

        Args:
            left_img: Left stereo image (H, W, C)
            right_img: Right stereo image (H, W, C)

        Returns:
            tuple: (left_batch, right_batch) as torch.Tensor on self.device
        """
        left_np, right_np = self._preprocess_images(left_img, right_img)
        batch_left = torch.tensor(left_np).to(self.device)
        batch_right = torch.tensor(right_np).to(self.device)
        return batch_left, batch_right

    def inference(self, left_img: Union[np.ndarray, torch.Tensor],
                  right_img: Union[np.ndarray, torch.Tensor],
                  preprocess: bool = True) -> np.ndarray:
        """
        Run inference on stereo image pair.

        Args:
            left_img: Left stereo image (H, W, C) or preprocessed tensor (B, C, H, W)
            right_img: Right stereo image (H, W, C) or preprocessed tensor (B, C, H, W)
            preprocess: Whether to preprocess images (True) or use as-is (False)

        Returns:
            Disparity map as numpy array (H, W)
        """
        if preprocess:
            left_batch, right_batch = self.preprocess(left_img, right_img)
        else:
            # from dataset loader, already preprocessed
            left_batch = left_img.contiguous().float().to(self.device)
            right_batch = right_img.contiguous().float().to(self.device)

        with torch.inference_mode():
            pred_flow = self.model(left_batch, right_batch, iters=self.n_iter, flow_init=None)
        pred_disp = torch.squeeze(pred_flow[:, 0, :, :])

        return pred_disp.cpu().detach().numpy()


class ONNXModel(StereoModel):
    def __init__(self, model_path: str, provider: str = 'cuda', thread_num: int = None):
        super().__init__(model_path, input_width=0, input_height=0)
        self.provider = provider.lower()
        assert self.provider in ['cpu', 'cuda', 'tensorrt'], "Provider must be 'cpu', 'cuda', or 'tensorrt'"
        self.session = None
        self.input_names = None
        self.output_names = None
        self.thread_num = thread_num
        self._load_model()
        self._get_input_output_info()

    def _get_provider_config(self):
        # TensorRT configuration
        # @note: tensorrt lib must be installed
        # if TensorRT is not available, rollback to CUDA provider
        trt_cache_path = './trt_cache'
        if self.provider == 'tensorrt':
            os.makedirs(trt_cache_path, exist_ok=True)
        tensorrt_config = {
            'providers': ['TensorrtExecutionProvider', 'CUDAExecutionProvider'],
            'provider_options': [
                {
                    'trt_fp16_enable': True,
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': trt_cache_path,
                    'trt_layer_norm_fp32_fallback': True,
                    'trt_timing_cache_enable': True
                },
                {}  # CUDAExecutionProvider options
            ]
        }

        # CUDA configuration
        # @note: onnxruntime-gpu must be installed
        cuda_config = {
            'providers': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
            'provider_options': [{}, {}]
        }

        # CPU configuration
        cpu_config = {
            'providers': ['CPUExecutionProvider'],
            'provider_options': [{}]
        }

        configs = {
            'tensorrt': tensorrt_config,
            'cuda': cuda_config,
            'cpu': cpu_config
        }

        return configs.get(self.provider, cpu_config)

    def _load_model(self):
        try:
            config = self._get_provider_config()
            sess_options = ort.SessionOptions()
            # sess_options.log_severity_level = 0
            if self.thread_num is not None:
                # Thread configuration affects CPU provider performance significantly.
                # intra_op_num_threads: threads within a single operator (e.g., matrix multiplication)
                # inter_op_num_threads: threads across different operators (parallel execution)
                print(f"Setting thread number to {self.thread_num}")
                sess_options.intra_op_num_threads = self.thread_num
                sess_options.inter_op_num_threads = self.thread_num
            self.session = ort.InferenceSession(
                self.model_path,
                providers=config['providers'],
                provider_options=config['provider_options'],
                sess_options=sess_options
            )
            print(f'ONNX model loaded with {self.provider} provider.')
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")

    def _get_input_output_info(self):
        """Extract input/output information from ONNX model."""
        if self.session is None:
            raise RuntimeError("Model session is not initialized.")

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        self.input_names = [input.name for input in inputs]
        self.output_names = [output.name for output in outputs]

        if len(inputs[0].shape) != 4:
            raise ValueError("Expected input tensor to have 4 dimensions (N, C, H, W)")
        self.input_height = inputs[0].shape[2]
        self.input_width = inputs[0].shape[3]

        print(f"Model input names: {self.input_names}")
        print(f"Model output names: {self.output_names}")
        print(f"Model input size: {self.input_width}x{self.input_height}")

    def preprocess(self, left_img: np.ndarray, right_img: np.ndarray) -> tuple:
        """
        Preprocess stereo image pairs for ONNX model.

        Args:
            left_img: Left stereo image (H, W, C)
            right_img: Right stereo image (H, W, C)

        Returns:
            tuple: (left_batch, right_batch) as numpy arrays with shape (1, C, H, W)
        """
        return super().preprocess(left_img, right_img)

    def _tensor_to_numpy(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return np.ascontiguousarray(tensor.cpu().numpy()).astype("float32")
        return np.ascontiguousarray(tensor).astype("float32")

    def inference(self, left_img: Union[np.ndarray, torch.Tensor],
                  right_img: Union[np.ndarray, torch.Tensor],
                  preprocess: bool = True) -> np.ndarray:
        """
        Run inference on stereo image pair.

        Args:
            left_img: Left stereo image (H, W, C) or preprocessed tensor/array
            right_img: Right stereo image (H, W, C) or preprocessed tensor/array
            preprocess: Whether to preprocess images (True) or use as-is (False)

        Returns:
            Disparity map as numpy array (H, W)
        """
        if self.session is None:
            raise RuntimeError("Model session is not initialized.")

        if preprocess:
            left_batch, right_batch = self.preprocess(left_img, right_img)
        else:
            # from dataset loader, already preprocessed
            left_batch = self._tensor_to_numpy(left_img)
            right_batch = self._tensor_to_numpy(right_img)

        pred_disp = self.session.run(
            [self.output_names[0]], {self.input_names[0]: left_batch, self.input_names[1]: right_batch})[0]
        return np.squeeze(pred_disp[:, 0, :, :])

    def set_provider(self, provider: str):
        """Change ONNX Runtime provider and reload model."""
        self.provider = provider.lower()
        self._load_model()
        self._get_input_output_info()

    def get_provider(self) -> str:
        """Get current ONNX Runtime provider."""
        return self.provider


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Test model")
    parser.add_argument("--model_path", default="models/crestereo_eth3d.pth",
                        help="Path to trained model file, .pth or .onnx")
    parser.add_argument("--max_disp", type=int, default=256, help="Maximum disparity value")
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Use mixed precision for PyTorch model inference")
    parser.add_argument("--left_img", default=None, help="Path to left stereo image")
    parser.add_argument("--right_img", default=None, help="Path to right stereo image")
    parser.add_argument("--img_width", type=int, default=None, help="Image width of model input")
    parser.add_argument("--img_height", type=int, default=None, help="Image height of model input")
    parser.add_argument("--device", default="cuda",
                        help="Device to run pytorch model, cpu or cuda. "
                             "Or Provider for ONNX Runtime: cpu, cuda, tensorrt")
    parser.add_argument("--thread_num", type=int, default=None,
                        help="Number of cpu threads for ONNX Runtime")
    parser.add_argument("--inference_num", type=int, default=1, help="Number of inference iterations")
    args = parser.parse_args()
    print("test model path: ", args.model_path)
    print("device: ", args.device)

    if args.left_img is None or args.right_img is None:
        from imread_from_url import imread_from_url
        left_img = imread_from_url("https://raw.githubusercontent.com/megvii-research/CREStereo/master/img/test/left.png")
        right_img = imread_from_url("https://raw.githubusercontent.com/megvii-research/CREStereo/master/img/test/right.png")
    else:
        left_img = cv2.imread(args.left_img, cv2.IMREAD_COLOR)
        right_img = cv2.imread(args.right_img, cv2.IMREAD_COLOR)
    assert left_img is not None, "input left image is empty"
    assert right_img is not None, "input right image is empty"

    in_h, in_w = left_img.shape[:2]

    if args.model_path.endswith('.onnx'):
        model = ONNXModel(args.model_path, provider=args.device, thread_num=args.thread_num)
        eval_w, eval_h = model.get_input_shape()
    elif args.model_path.endswith('.pth'):
        if args.img_width is not None and args.img_height is not None:
          eval_h, eval_w = int(args.img_height), int(args.img_width)
        else:
          eval_h, eval_w = (in_h, in_w)
        assert eval_h%8 == 0, "input height should be divisible by 8"
        assert eval_w%8 == 0, "input width should be divisible by 8"

        model = PytorchModel(args.model_path, device=args.device, max_disp=args.max_disp,
                             mixed_precision=args.mixed_precision, input_width=eval_w,
                             input_height=eval_h, n_iter=10)
    else:
        raise RuntimeError("Unsupported model file format!")

    print("Start inference...")
    avg_inference_time = 0
    for i in range(args.inference_num):
        start_time = time.time()
        pred = model.inference(left_img, right_img, preprocess=True)
        end_time = time.time() - start_time
        print(f"iter {i} inference time: {end_time:.4f} s")
        if i > 0:
            # first inference is longer due to model initialization for cuda/tensorrt providers
            avg_inference_time += end_time
    if args.inference_num > 1:
        avg_inference_time /= args.inference_num - 1
        print(f"average stable inference time: {avg_inference_time:.4f} s")

    t = float(in_w) / float(eval_w)
    disp = cv2.resize(pred, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t

    disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    disp_vis = disp_vis.astype("uint8")
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)

    combined_img = np.hstack((left_img, disp_vis))
    # cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    # cv2.imshow("output", combined_img)
    # cv2.waitKey(0)
    cv2.imwrite("output_combined.png", combined_img)
    cv2.imwrite("output_disp_vis.png", disp_vis)
    print("Output images saved!")
