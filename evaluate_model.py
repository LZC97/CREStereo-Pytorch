import argparse
import numpy as np
import tqdm
import torch
import time
from nets import Model
from torch.utils.data import DataLoader
from dataset import DataSetWrapper, MixedDataset
from test_model import PytorchModel, ONNXModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluate model")
    parser.add_argument("--model_path", required=True, type=str, help="Path to trained model, .pth or .onnx")
    parser.add_argument("--data_path", required=True, type=str, help="Path to dataset")
    parser.add_argument("--dataset_name", default="eth3d", type=str, help="Dataset name")
    parser.add_argument("--image_height", default=384, type=int, help="Image height for evaluation")
    parser.add_argument("--image_width", default=512, type=int, help="Image width for evaluation")
    parser.add_argument("--max_disp", default=256, type=int, help="Max disparity for evaluation")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision for inference")
    parser.add_argument("--num_workers", default=1, type=int, help="Number of workers for data loading")
    parser.add_argument("--n_iter", default=10, type=int, help="Number of iterations for model inference")
    parser.add_argument("--device", default="cuda", type=str, help="Device to run Pytorch model, cpu or cuda. " \
                                                            "Or Provider for ONNX Runtime: cpu, cuda, tensorrt")
    args = parser.parse_args()

    print("Model path:", args.model_path)
    print("Dataset path:", args.data_path)

    if args.model_path.endswith('.onnx'):
        model = ONNXModel(args.model_path, provider=args.device)
        args.image_width, args.image_height = model.get_input_shape()
    else:
        model = PytorchModel(args.model_path, device=args.device, max_disp=args.max_disp,
                            mixed_precision=args.mixed_precision, n_iter=args.n_iter)

    use_mixed_dataset = True
    if use_mixed_dataset:
        dataset_roots = {
            "ETH3D": args.data_path,
        }
        dataset = MixedDataset(dataset_roots=dataset_roots,
            image_height=args.image_height,
            image_width=args.image_width,
            max_disp=args.max_disp,
            train_mode=False)
    else:
        dataset = DataSetWrapper(dataset_name=args.dataset_name,
            data_dir=args.data_path,
            image_height=args.image_height,
            image_width=args.image_width,
            max_disp=args.max_disp,
            train_mode=False)

    if dataset is None:
        print("Failed to load dataset.")
        exit(-1)

    total_samples = len(dataset)
    if total_samples == 0:
        print("Dataset is empty.")
        exit(-1)
    print(f"Dataset size: {total_samples}")

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                             drop_last=False, pin_memory=True)

    epe = []
    bad_px_thresholds = [0.5, 1.0, 2.0, 3.0]
    bad_px_errors = {threshold: [] for threshold in bad_px_thresholds}

    start_time = time.time()

    with torch.no_grad():
        for data in tqdm.tqdm(data_loader, desc="Validating"):
            left_img = data["left"].to(args.device)
            right_img = data["right"].to(args.device)
            disp_gt = data["disparity"].to(args.device)
            valid_mask = data["mask"].to(args.device)

            disp_pred = model.inference(left_img, right_img, preprocess=False)
            disp_pred = torch.from_numpy(disp_pred).to(args.device)

            error = torch.abs(disp_pred - disp_gt) * valid_mask
            valid_pixels = valid_mask.sum()
            epe_sample = error.sum() / valid_pixels
            epe.append(epe_sample.item())

            for threshold in bad_px_errors.keys():
                bad_pixels = ((error > threshold) * valid_mask).sum()
                bad_px_errors[threshold].append((bad_pixels / valid_pixels).item())

    end_time = time.time()

    print("Evaluation Results:")
    print(f"Average EPE: {np.mean(epe):.4f} px")
    for threshold, errors in bad_px_errors.items():
        print(f"Bad Pixel Ratio (> {threshold} px): {np.mean(errors) * 100:.2f}%")
    print(f"Average Inference Time: {(end_time - start_time) / total_samples:.3f} s")
