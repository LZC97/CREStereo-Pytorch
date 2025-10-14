import torch
import torch.nn.functional as F
import numpy as np
import cv2
from imread_from_url import imread_from_url
import argparse
import time

import matplotlib.pyplot as plt

from nets import Model

def load_model(model_path, device, max_disp=256, mixed_precision=False):
	model = Model(max_disp=max_disp, mixed_precision=mixed_precision, test_mode=True)
	state_dict = torch.load(model_path, map_location=device)
	if "state_dict" in state_dict:
		state_dict = state_dict["state_dict"]
	model.load_state_dict(state_dict, strict=True)
	model.to(device)
	model.eval()
	return model

#Ref: https://github.com/megvii-research/CREStereo/blob/master/test.py
def inference(left, right, model, device, n_iter=20):
	print("Model Forwarding...")
	imgL = left.transpose(2, 0, 1)
	imgR = right.transpose(2, 0, 1)
	imgL = np.ascontiguousarray(imgL[None, :, :, :])
	imgR = np.ascontiguousarray(imgR[None, :, :, :])

	imgL = torch.tensor(imgL.astype("float32")).to(device)
	imgR = torch.tensor(imgR.astype("float32")).to(device)

	# imgL_dw2 = F.interpolate(
	# 	imgL,
	# 	size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
	# 	mode="bilinear",
	# 	align_corners=True,
	# )
	# imgR_dw2 = F.interpolate(
	# 	imgR,
	# 	size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
	# 	mode="bilinear",
	# 	align_corners=True,
	# )
	# print(imgR_dw2.shape)
	with torch.inference_mode():
		# pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=n_iter, flow_init=None)
		# pred_flow = model(imgL, imgR, iters=n_iter, flow_init=pred_flow_dw2)
		
		pred_flow = model(imgL, imgR, iters=n_iter, flow_init=None)
	pred_disp = torch.squeeze(pred_flow[:, 0, :, :]).cpu().detach().numpy()

	return pred_disp

if __name__ == '__main__':
	parser = argparse.ArgumentParser("Test model")
	parser.add_argument("--model_path", default="models/crestereo_eth3d.pth")
	parser.add_argument("--mixed_precision", action="store_true")
	parser.add_argument("--max_disp", default=256)
	parser.add_argument("--left_img", default=None)
	parser.add_argument("--right_img", default=None)
	parser.add_argument("--img_width", default=None, help="Image witdth of model input")
	parser.add_argument("--img_height", default=None, help="Image height of model input")
	parser.add_argument("--device", default="cuda", help="Device to run model, cpu or cuda.")
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
	
	imgL = cv2.resize(left_img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
	imgR = cv2.resize(right_img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)

	model = load_model(args.model_path, args.device, max_disp=args.max_disp, mixed_precision=args.mixed_precision)
	print("params: ", sum(p.numel() for p in model.parameters()))

	start_time = time.time()
	pred = inference(imgL, imgR, model, args.device, n_iter=20)
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
