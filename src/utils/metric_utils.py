import os
import json
import shutil
import functools
from PIL import Image

import torch
import numpy as np
from skimage.metrics import structural_similarity
from cleanfid import fid
from einops import reduce, rearrange

from src.utils.typing import *
import warnings
# Suppress warnings for LPIPS loss loading
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated since 0.13")
warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum.*")

@torch.no_grad()
def compute_psnr(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, "batch"]:
    """
    Compute Peak Signal-to-Noise Ratio between ground truth and predicted images.
    
    Args:
        ground_truth: Images with shape [batch, channel, height, width], values in [0, 1]
        predicted: Images with shape [batch, channel, height, width], values in [0, 1]
        
    Returns:
        PSNR values for each image in the batch
    """
    ground_truth = torch.clamp(ground_truth, 0, 1)
    predicted = torch.clamp(predicted, 0, 1)
    mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
    return -10 * torch.log10(mse) 



@functools.lru_cache(maxsize=None)
def get_lpips_model(net_type="vgg", device="cuda"):
    from lpips import LPIPS
    return LPIPS(net=net_type).to(device)

@torch.no_grad()
def compute_lpips(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
    normalize: bool = True,
) -> Float[Tensor, "batch"]:
    """
    Compute Learned Perceptual Image Patch Similarity between images.
    
    Args:
        ground_truth: Images with shape [batch, channel, height, width]
        predicted: Images with shape [batch, channel, height, width]
        The value range is [0, 1] when we have set the normalize flag to True.
        It will be [-1, 1] when the normalize flag is set to False.
    Returns:
        LPIPS values for each image in the batch (lower is better)
    """

    _lpips_fn = get_lpips_model(device=predicted.device)
    batch_size = 50  # Process in batches to save memory
    values = [
        _lpips_fn(
            ground_truth[i : i + batch_size],
            predicted[i : i + batch_size],
            normalize=normalize,
        )
        for i in range(0, ground_truth.shape[0], batch_size)
    ]
    return torch.cat(values, dim=0).squeeze()



@torch.no_grad()
def compute_ssim(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    """
    Compute Structural Similarity Index between images.
    
    Args:
        ground_truth: Images with shape [batch, channel, height, width], values in [0, 1]
        predicted: Images with shape [batch, channel, height, width], values in [0, 1]
        
    Returns:
        SSIM values for each image in the batch (higher is better)
    """
    ssim_values= []
    
    for gt, pred in zip(ground_truth, predicted):
        # Move to CPU and convert to numpy
        gt_np = gt.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        
        # Calculate SSIM
        ssim = structural_similarity(
            gt_np,
            pred_np,
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        ssim_values.append(ssim)
    
    # Convert back to tensor on the same device as input
    return torch.tensor(ssim_values, dtype=predicted.dtype, device=predicted.device)

def compute_fid(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    """
    Compute Fréchet Inception Distance (FID) between two sets of images.
    Args:
        ground_truth: Images with shape [batch, channel, height, width], values in [0, 1]
        predicted: Images with shape [batch, channel, height, width], values in [0, 1]
    Returns:
        FID value (lower is better)
    """
    ground_truth = ground_truth.permute(0, 2, 3, 1)  # Convert to [batch, height, width, channel]
    predicted = predicted.permute(0, 2, 3, 1)  # Convert to [batch, height, width, channel]
    
    # Convert to numpy arrays
    gt_np = ground_truth.detach().cpu().numpy() * 255
    pred_np = predicted.detach().cpu().numpy() * 255
    
    # Save images to temporary files
    gt_temp = [Image.fromarray(gt.astype(np.uint8)).convert("RGB") for gt in gt_np]
    pred_temp = [Image.fromarray(pred.astype(np.uint8)).convert("RGB") for pred in pred_np]
    gt_tmp_dir = os.path.join(os.getcwd(), "tmp_fid_gt")
    pred_tmp_dir = os.path.join(os.getcwd(), "tmp_fid_pred")
    if os.path.exists(gt_tmp_dir):
        shutil.rmtree(gt_tmp_dir)
    if os.path.exists(pred_tmp_dir):
        shutil.rmtree(pred_tmp_dir)
    os.makedirs(gt_tmp_dir, exist_ok=True)
    os.makedirs(pred_tmp_dir, exist_ok=True)
    gt_paths = [os.path.join(gt_tmp_dir, f"gt_{i}.png") for i in range(len(gt_temp))]
    pred_paths = [os.path.join(pred_tmp_dir, f"pred_{i}.png") for i in range(len(pred_temp))]
    for img, path in zip(gt_temp, gt_paths):
        img.save(path)
    for img, path in zip(pred_temp, pred_paths):
        img.save(path)
    # Calculate FID
    fid_score = fid.compute_fid(pred_tmp_dir, gt_tmp_dir, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # Convert back to tensor
    return torch.tensor(fid_score, dtype=torch.float32, device=predicted.device)
    
    
def save_metrics(target, prediction, view_indices, scene_name, out_dir: str = "./"):
    """
    Compute and save evaluation metrics (PSNR, LPIPS, SSIM) for a given scene.
    Args:
        target: Ground truth images with shape [batch, channel, height, width]
        prediction: Predicted images with shape [batch, channel, height, width]
        view_indices: Indices of the views to evaluate
        scene_name: Name of the scene for identification
        out_dir: Directory to save the metrics JSON file
    Returns:
        metrics: Dictionary containing computed metrics
    """
    target = target.to(torch.float32)
    prediction = prediction.to(torch.float32)
    
    psnr_values = compute_psnr(target, prediction)
    lpips_values = compute_lpips(target, prediction)
    ssim_values = compute_ssim(target, prediction)
    # fid_values = compute_fid(target, prediction)

    metrics = {
        "summary": {
            "scene_name": scene_name,
            "psnr": float(psnr_values.mean()),
            "lpips": float(lpips_values.mean()),
            "ssim": float(ssim_values.mean()),
            # "fid": float(fid_values.mean())
        },
        "per_view": []
    }
    
    for i, view_idx in enumerate(view_indices):
        metrics["per_view"].append({
            "view": f"frame_{int(view_idx)}", "psnr": float(psnr_values[i]), "lpips": float(lpips_values[i]), "ssim": float(ssim_values[i])
        })
        
    # Save metrics to a single JSON file
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics
        
# @torch.no_grad()
# def export_results(
#     result: edict,
#     out_dir: str, 
#     compute_metrics: bool = False
# ):
#     """
#     Save results including images and optional metrics and videos.
    
#     Args:
#         result: EasyDict containing input, target, and rendered images, and optionally video frames
#         out_dir: Directory to save the evaluation results
#         compute_metrics: Whether to compute and save metrics
#     """
#     os.makedirs(out_dir, exist_ok=True)
    
#     input_data, target_data = result.input, result.target
    
#     for batch_idx in range(input_data.image.size(0)):
#         uid = input_data.index[batch_idx, 0, -1].item()
#         scene_name = input_data.scene_name[batch_idx]
#         sample_dir = os.path.join(out_dir, f"{uid:06d}")
#         os.makedirs(sample_dir, exist_ok=True)
        
#         # Get target view indices
#         target_indices = target_data.index[batch_idx, :, 0].cpu().numpy()
        
#         # Save images
#         _save_images(result, batch_idx, sample_dir)
        
#         # Compute and save metrics if requested
#         if compute_metrics:
#             save_metrics(
#                 target_data.image[batch_idx],
#                 result.render[batch_idx],
#                 target_indices,
#                 sample_dir,
#                 scene_name
#             )
        
#         # Save video if available
#         if hasattr(result, "video_rendering"):
#             _save_video(result.video_rendering[batch_idx], sample_dir)

def summarize_evaluation(evaluation_folder):
    # Find and sort all valid subfolders
    subfolders = sorted(
        [
            os.path.join(evaluation_folder, dirname)
            for dirname in os.listdir(evaluation_folder)
            if os.path.isdir(os.path.join(evaluation_folder, dirname))
        ],
        key=lambda x: int(os.path.basename(x)) if os.path.basename(x).isdigit() else os.path.basename(x)
    )

    metrics = {}
    valid_subfolders = []
    
    for subfolder in subfolders:
        json_path = os.path.join(subfolder, "metrics.json")
        if not os.path.exists(json_path):
            print(f"!!! Metrics file not found in {subfolder}, skipping...")
            continue
            
        valid_subfolders.append(subfolder)
        
        with open(json_path, "r") as f:
            try:
                data = json.load(f)
                # Extract summary metrics
                for metric_name, metric_value in data["summary"].items():
                    if metric_name == "scene_name":
                        continue
                    metrics.setdefault(metric_name, []).append(metric_value)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading metrics from {json_path}: {e}")

    if not valid_subfolders:
        print(f"No valid metrics files found in {evaluation_folder}")
        return

    csv_file = os.path.join(evaluation_folder, "summary.csv")
    with open(csv_file, "w") as f:
        header = ["Index"] + list(metrics.keys())
        f.write(",".join(header) + "\n")
        
        for i, subfolder in enumerate(valid_subfolders):
            basename = os.path.basename(subfolder)
            values = [str(metric_values[i]) for metric_values in metrics.values()]
            f.write(f"{basename},{','.join(values)}\n")
        
        f.write("\n")
        
        averages = [str(sum(values) / len(values)) for values in metrics.values()]
        f.write(f"average,{','.join(averages)}\n")
    
    print(f"Summary written to {csv_file}")
    print(f"Average: {','.join(averages)}")

    # export average metrics to a text file
    with open(os.path.join(evaluation_folder, "average_metrics.txt"), "w") as f:
        f.write(f"Average: {','.join(averages)}\n")