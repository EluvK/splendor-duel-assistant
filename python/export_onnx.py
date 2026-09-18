"""将 PyTorch 模型权重 (.pt) 导出为轻量标准 ONNX 模型 (.onnx) 并提取版本元数据."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import torch

# 保证能正确导入 splendor_ai
sys.path.insert(0, str(Path(__file__).parent))
from splendor_ai.net import SplendorNet


def export_checkpoint_to_onnx(
    ckpt_path: str = "checkpoints/best.pt",
    onnx_path: str = "checkpoints/best.onnx",
    meta_path: str = "checkpoints/best.json",
    message: str = None,
):
    src = Path(ckpt_path)
    if not src.exists():
        raise FileNotFoundError(f"Checkpoint 文件不存在: {src.resolve()}")

    print(f"==> 正在读取模型权重: {src} ...")
    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    net = SplendorNet()
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        net.load_state_dict(ckpt["model_state"])
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        net.load_state_dict(ckpt["state_dict"])
    elif isinstance(ckpt, dict):
        net.load_state_dict(ckpt)
    else:
        raise ValueError(f"无法识别的 checkpoint 格式: {type(ckpt)}")

    print("==> 正在执行 torch.onnx.export (opset=17) ...")
    onnx_bytes = net.export_onnx_bytes()

    out_onnx = Path(onnx_path)
    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    out_onnx.write_bytes(onnx_bytes)

    # 极简元数据：用户自定义消息/版本号 + 导出时间 + 短哈希
    sha256 = hashlib.sha256(onnx_bytes).hexdigest()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    tag = message if message else f"{datetime.now().strftime('%m%d')}版本"

    model_info = {
        "message": tag,
        "export_time": now_str,
        "model_hash": sha256[:8],
    }

    if meta_path:
        out_meta = Path(meta_path)
        out_meta.parent.mkdir(parents=True, exist_ok=True)
        out_meta.write_text(json.dumps(model_info, indent=2, ensure_ascii=False), encoding="utf-8")

    size_mb = round(len(onnx_bytes) / (1024 * 1024), 2)
    print("\n✅ ONNX 导出成功！")
    print(f"  • 输出模型: {out_onnx} ({size_mb} MB)")
    if meta_path:
        print(f"  • 元数据:   {out_meta}")
    print(f"  • 版本描述: {model_info['message']}")
    print(f"  • 导出时间: {model_info['export_time']}")
    print(f"  • 模型指纹: #{model_info['model_hash']}\n")

    return model_info


def main():
    parser = argparse.ArgumentParser(description="导出 PyTorch 模型为 ONNX")
    parser.add_argument("--input", "-i", default="checkpoints/best.pt", help="输入的 .pt 路径")
    parser.add_argument("--output", "-o", default="checkpoints/best.onnx", help="输出的 .onnx 路径")
    parser.add_argument("--meta", "-m", default="checkpoints/best.json", help="输出的元数据 .json 路径")
    parser.add_argument("--message", "-msg", default=None, help="自定义版本描述 (例如: '0918版本')")
    args = parser.parse_args()

    export_checkpoint_to_onnx(args.input, args.output, args.meta, args.message)


if __name__ == "__main__":
    main()
