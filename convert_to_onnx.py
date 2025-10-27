import torch
import torch.nn.functional as F
import numpy as np
import argparse

from nets import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert model to ONNX format")
    parser.add_argument("--output_path", type=str, default="./crestereo.onnx", help="Path to the output ONNX file")
    parser.add_argument("--model_path", type=str, default="models/crestereo_eth3d.pth", help="Path to the model file")
    parser.add_argument("--input_height", type=int, default=480, help="Input image height")
    parser.add_argument("--input_width", type=int, default=640, help="Input image width")
    parser.add_argument("--opset_version", type=int, default=12, help="ONNX opset version")
    args = parser.parse_args()

    model_path = args.model_path
    assert args.input_height % 32 == 0 and args.input_width % 32 == 0, "Input dimensions must be multiples of 32"
    assert args.output_path.endswith(".onnx"), "Output path must end with .onnx"

    model = Model(max_disp=256, mixed_precision=False, test_mode=True)
    state_dict = torch.load(model_path)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    t1 = torch.rand(1, 3, args.input_height, args.input_width)
    t2 = torch.rand(1, 3, args.input_height, args.input_width)
    # flow_init = torch.rand(1, 2, args.input_height//2, args.input_width//2)

    # # Export the model
    # torch.onnx.export(model,               
    #                   (t1, t2, flow_init),
    #                   "crestereo.onnx",   # where to save the model (can be a file or file-like object)
    #                   export_params=True,        # store the trained parameter weights inside the model file
    #                   opset_version=12,          # the ONNX version to export the model to
    #                   do_constant_folding=True,  # whether to execute constant folding for optimization
    #                   input_names = ['left', 'right','flow_init'],   # the model's input names
    #                   output_names = ['disp'])

    # Export the model without init_flow (it takes a lot of time)
    # !! Does not work prior to pytorch 1.12 (confirmed working on pytorch 2.0.0)
    # Ref: https://github.com/pytorch/pytorch/pull/73760
    torch.onnx.export(model,               
                      (t1, t2),
                      args.output_path,   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=args.opset_version,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['left', 'right'],   # the model's input names
                      output_names = ['disp'])
