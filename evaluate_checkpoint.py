"""Evaluate a trained checkpoint."""
import os, sys, argparse, time, json
import torch
import numpy as np
from train_compare_models import build_model, evaluate, DATASET_CONFIGS, build_dataloader
from models.text_aware_model import TextDetector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="rdd")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--res", type=int, default=384)
    parser.add_argument("--output", type=str, default="./experiment_results_rdd/full_200ep/")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    m, model_type = build_model(args.model, device, model_variant="v2")
    m = m.to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model" in ckpt:
        m.load_state_dict(ckpt["model"])
        ep = ckpt.get("epoch", "?")
        print(f"Loaded checkpoint epoch {ep}")
    else:
        m.load_state_dict(ckpt, strict=False)
        ep = "?"

    dataset_cfg = DATASET_CONFIGS[args.dataset]
    val_loader = build_dataloader(dataset_cfg, "test", False, args.batch_size, args.res)

    text_detector = TextDetector(method="opencv")
    metrics = evaluate(m, val_loader, text_detector, device, model_type)

    print(f"PSNR={metrics['psnr']:.2f} SSIM={metrics['ssim']:.4f} RMSE={metrics['rmse']:.2f} TextPSNR={metrics['text_psnr']:.2f}")

    result = {
        "checkpoint": args.checkpoint,
        "model": args.model,
        "epoch": ep,
        "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_path = os.path.join(args.output, f"eval_textaware_epoch{ep}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
