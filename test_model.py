import torch
import torch.nn.functional as F
import numpy as np
import cv2
from imread_from_url import imread_from_url
import argparse
import time
import onnxruntime as ort

import matplotlib.pyplot as plt

from nets import Model

class PytorchModel:
    def __init__(self, 
                 model_path, 
                 device='cuda', 
                 max_disp=256, 
                 mixed_precision=False, 
                 input_width=480, 
                 input_height=480, 
                 n_iter=10):
        self.model_path = model_path
        self.device = device
        self.max_disp = max_disp
        self.mixed_precision = mixed_precision
        self.input_width = input_width
        self.input_height = input_height
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
    
    def preprocess(self, left_img, right_img):
        dst_left = cv2.resize(left_img, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        dst_right = cv2.resize(right_img, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        dst_left = dst_left.transpose(2, 0, 1)
        dst_right = dst_right.transpose(2, 0, 1)
        dst_left = np.ascontiguousarray(dst_left[None, :, :, :])
        dst_right = np.ascontiguousarray(dst_right[None, :, :, :])

        batch_left = torch.tensor(dst_left.astype("float32")).to(self.device)
        batch_right = torch.tensor(dst_right.astype("float32")).to(self.device)
        return batch_left, batch_right

    def inference(self, left_img, right_img, preprocess=True):
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


class ONNXModel:
    def __init__(self, model_path: str, provider: str = 'cuda'):
        self.model_path = model_path
        self.provider = provider.lower()
        assert self.provider in ['cpu', 'cuda', 'tensorrt'], "Provider must be 'cpu', 'cuda', or 'tensorrt'"
        self.session = None
        self.input_names = None
        self.output_names = None
        self.input_height = None
        self.input_width = None
      
        self._load_model()
        self._get_input_output_info()

    def _load_model(self):
        try:
            if self.provider == 'tensorrt':
                # @note: tensorrt lib must be installed
                providers = [
                    ('TensorrtExecutionProvider', {
                        'trt_engine_cache_enable': True,
                        'trt_engine_cache_path': './trt_cache'
                    }),
                    'CUDAExecutionProvider'
                ]
            elif self.provider == 'cuda':
                # @note: onnxruntime-gpu must be installed
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:  # cpu
                providers = ['CPUExecutionProvider']
          
            self.session = ort.InferenceSession(
                self.model_path, providers=providers
            )
            print(f'ONNX model loaded with {self.provider} provider.')
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")
        
    def _get_input_output_info(self):
        if self.session is None:
            raise RuntimeError("Model session is not initialized.")
        
        intputs = self.session.get_inputs()
        outputs = self.session.get_outputs()    
        self.input_names = [input.name for input in intputs]
        self.output_names = [output.name for output in outputs]

        if len(intputs[0].shape) != 4:
            raise ValueError("Expected input tensor to have 4 dimensions (N, C, H, W)")
        self.input_height = intputs[0].shape[2]
        self.input_width = intputs[0].shape[3]

        print(f"Model input names: {self.input_names}")
        print(f"Model output names: {self.output_names}")
        print(f"Model input size: {self.input_width}x{self.input_height}")

    def preprocess(self, left_img: np.ndarray, right_img: np.ndarray) -> tuple:
        # Resize images to model input size
        dst_left = cv2.resize(
            left_img, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        dst_right = cv2.resize(
            right_img, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)

        dst_left = dst_left.transpose(2, 0, 1)
        dst_right = dst_right.transpose(2, 0, 1)
        dst_left = np.ascontiguousarray(dst_left[None, :, :, :]).astype("float32")
        dst_right = np.ascontiguousarray(dst_right[None, :, :, :]).astype("float32")

        return dst_left, dst_right
    
    def inference(self, left_img: np.ndarray, right_img: np.ndarray, preprocess=True) -> np.ndarray:
        if self.session is None:
            raise RuntimeError("Model session is not initialized.")
        
        if preprocess:
          left_batch, right_batch = self.preprocess(left_img, right_img)
        else:
          # from dataset loader, already preprocessed
          left_batch =  np.ascontiguousarray(left_img.cpu().numpy()).astype("float32")
          right_batch = np.ascontiguousarray(right_img.cpu().numpy()).astype("float32")
          
        pred_disp = self.session.run(
            [self.output_names[0]], {self.input_names[0]: left_batch, self.input_names[1]: right_batch})[0]
        return np.squeeze(pred_disp[:, 0, :, :])
    
    def set_provider(self, provider: str):
        self.provider = provider.lower()
        self._load_model()
        self._get_input_output_info()
    
    def get_input_shape(self) -> tuple:
        return (self.input_width, self.input_height)
    
    def get_provider(self) -> str:
        return self.provider


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Test model")
    parser.add_argument("--model_path", default="models/crestereo_eth3d.pth", help="Path to trained model file, .pth or .onnx")
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--max_disp", default=256)
    parser.add_argument("--left_img", default=None)
    parser.add_argument("--right_img", default=None)
    parser.add_argument("--img_width", default=None, help="Image witdth of model input")
    parser.add_argument("--img_height", default=None, help="Image height of model input")
    parser.add_argument("--device", default="cuda", help="Device to run pytorch model, cpu or cuda. " \
                                                    "Or Provider for ONNX Runtime: cpu, cuda, tensorrt")
    args = parser.parse_args()
    print("test model path: ", args.model_path)
    print("device: ", args.device)
    
    if args.left_img is None or args.right_img is None:
        left_img = imread_from_url("https://raw.githubusercontent.com/megvii-research/CREStereo/master/img/test/left.png")
        right_img = imread_from_url("https://raw.githubusercontent.com/megvii-research/CREStereo/master/img/test/right.png")
    else:
        left_img = cv2.imread(args.left_img, cv2.IMREAD_COLOR)
        right_img = cv2.imread(args.right_img, cv2.IMREAD_COLOR)
    assert left_img is not None, "input left image is empty"
    assert right_img is not None, "input right image is empty"
    
    in_h, in_w = left_img.shape[:2]

    if args.img_width is not None and args.img_height is not None:
      eval_h, eval_w = int(args.img_height), int(args.img_width)
    else:
      eval_h, eval_w = (in_h, in_w)

    assert eval_h%8 == 0, "input height should be divisible by 8"
    assert eval_w%8 == 0, "input width should be divisible by 8"

    if args.model_path.endswith('.onnx'):
        model = ONNXModel(args.model_path, provider=args.device)
    else:
        model = PytorchModel(args.model_path, device=args.device, max_disp=args.max_disp,
                             mixed_precision=args.mixed_precision, input_width=eval_w, 
                             input_height=eval_h, n_iter=10)

    start_time = time.time()
    pred = model.inference(left_img, right_img, preprocess=True)
    end_time = time.time()
    print(f"inference time: {end_time - start_time} seconds")
    
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
