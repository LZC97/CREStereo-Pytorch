import torch
import torch.nn.functional as F
import numpy as np
import argparse

from nets import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert model to ONNX format")
    parser.add_argument("--model_path", type=str, default="models/crestereo_eth3d.pth", help="Path to the model file")
    parser.add_argument("--output_path", type=str, default="./crestereo.onnx", help="Path to the output ONNX file")
    parser.add_argument("--input_height", type=int, default=480, help="Input image height")
    parser.add_argument("--input_width", type=int, default=640, help="Input image width")
    parser.add_argument("--opset_version", type=int, default=12, help="ONNX opset version")
    parser.add_argument("--optimize", action='store_true', help="Whether to optimize the ONNX model")
    parser.add_argument("--fp16", action='store_true', help="Whether to export the model in FP16 precision")
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
    print(f"Model exported to {args.output_path}")

    pre_model_path = args.output_path
    if args.optimize:
        # @note: Netron cannot visualize the default model graph
        from onnxruntime.transformers import optimizer
        optimized_model_path = args.output_path.replace('.onnx', '_optimized.onnx')
        optimized_model = optimizer.optimize_model(input=pre_model_path)
        optimized_model.save_model_to_file(optimized_model_path)
        print(f"Optimized model saved to {optimized_model_path}")
        pre_model_path = optimized_model_path

    if args.fp16:
        from onnx import load_model, save_model, checker
        from onnxconverter_common.float16 import convert_float_to_float16
        model_fp32 = load_model(pre_model_path)
        model_fp16 = convert_float_to_float16(model_fp32, keep_io_types=True)
        fp16_model_path = pre_model_path.replace('.onnx', '_fp16.onnx')
        save_model(model_fp16, fp16_model_path)
        print(f"FP16 model saved to {fp16_model_path}")
        # check fp16 model
        try:
            checker.check_model(model_fp16)
            print("FP16 model check passed.")
        except Exception as e:
            print("FP16 model check failed:", e)
