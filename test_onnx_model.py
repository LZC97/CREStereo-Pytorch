import numpy as np
import cv2
import argparse
import time

import onnxruntime

# Ref: https://github.com/megvii-research/CREStereo/blob/master/test.py
def inference(left, right, model):
    # Get onnx model layer names (see convert_to_onnx.py for what these are)
    input1_name = model.get_inputs()[0].name
    input2_name = model.get_inputs()[1].name
    output_name = model.get_outputs()[0].name

    model_h, model_w = model.get_inputs()[0].shape[2:4]

    # Decimate the image to half the original size for flow estimation network
    imgL = cv2.resize(
        left, (model_w, model_h), interpolation=cv2.INTER_LINEAR)
    imgR = cv2.resize(
        right, (model_w, model_h), interpolation=cv2.INTER_LINEAR)

    # Reshape inputs to match what is expected
    imgL = imgL.transpose(2, 0, 1)
    imgR = imgR.transpose(2, 0, 1)
    imgL = np.ascontiguousarray(imgL[None, :, :, :]).astype("float32")
    imgR = np.ascontiguousarray(imgR[None, :, :, :]).astype("float32")

    print("Model Forwarding...")
    pred_flow = model.run(
        [output_name], {input1_name: imgL, input2_name: imgR})[0]

    return np.squeeze(pred_flow[:, 0, :, :])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CREStereo ONNX model inference")
    parser.add_argument("--model_path", type=str, default="models/crestereo.onnx", help="Path to the ONNX model")
    parser.add_argument("--left_image", type=str, default="left.png", help="Path to the left image")
    parser.add_argument("--right_image", type=str, default="right.png", help="Path to the right image")
    args = parser.parse_args()

    print("Model path: ", args.model_path)
    print("Left image path: ", args.left_image)
    print("Right image path: ", args.right_image)

    left_img = cv2.imread(args.left_image, cv2.IMREAD_COLOR)
    right_img = cv2.imread(args.right_image, cv2.IMREAD_COLOR)
    assert left_img is not None, "Failed to load left image"
    assert right_img is not None, "Failed to load right image"

    in_h, in_w = left_img.shape[:2]

    model = onnxruntime.InferenceSession(args.model_path)

    start = time.time()
    pred_disp = inference(left_img, right_img, model)
    end = time.time()
    print(f"Inference time: {end - start} seconds")

    eval_h, eval_w = pred_disp.shape

    t = float(in_w) / float(eval_w)
    disp = cv2.resize(pred_disp, (in_w, in_h),
                      interpolation=cv2.INTER_LINEAR) * t
    disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    disp_vis = disp_vis.astype("uint8")
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)

    combined_img = np.hstack((left_img, disp_vis))
    cv2.imwrite("output_combined.jpg", combined_img)
    cv2.imwrite("output_disp_vis.jpg", disp_vis)
