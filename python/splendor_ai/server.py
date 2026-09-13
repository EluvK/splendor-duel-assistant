"""Lightweight HTTP Inference Server for SplendorNet."""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import sys
from typing import Optional
import torch
import torch.nn.functional as F

# 确保包路径可用
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from splendor_ai.net import SplendorNet


class ModelInferenceService:
    def __init__(self, checkpoint_path: str, device_name: str = "auto") -> None:
        if device_name == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_name)

        self.checkpoint_path = checkpoint_path
        self.net = SplendorNet().to(self.device)
        self.meta = {}
        self.epoch = 0

        self._load_checkpoint()
        self.net.eval()

    def _load_checkpoint(self) -> None:
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        print(f"📦 Loading SplendorNet from: {self.checkpoint_path} on {self.device}...")
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt["model_state"])
        self.epoch = ckpt.get("epoch", 0)
        self.meta = ckpt.get("meta", {})
        print(f"✅ Checkpoint loaded successfully! Epoch: {self.epoch}")

    @torch.no_grad()
    def predict(self, obs_list: list, mask_list: list, temperature: float = 1.0) -> dict:
        obs_t = torch.tensor(obs_list, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.tensor(mask_list, dtype=torch.bool, device=self.device).unsqueeze(0)

        logits, value = self.net(obs_t)
        masked_logits = self.net.mask_logits(logits, mask_t)

        if temperature <= 0.01:
            best_id = int(torch.argmax(masked_logits, dim=-1).item())
            probs = F.softmax(masked_logits, dim=-1)[0]
        else:
            probs = F.softmax(masked_logits / temperature, dim=-1)[0]
            best_id = int(torch.argmax(probs).item())

        value_scalar = float(value[0, 0].item())

        # 候选动作排行
        legal_indices = torch.where(mask_t[0])[0]
        if len(legal_indices) > 0:
            legal_probs = probs[legal_indices]
            sorted_indices = torch.argsort(legal_probs, descending=True)

            top_actions = []
            for idx in sorted_indices[:8]:
                act_id = int(legal_indices[idx].item())
                top_actions.append({
                    "action_id": act_id,
                    "prob": float(probs[act_id].item()),
                    "logit": float(logits[0, act_id].item()),
                })
        else:
            top_actions = []

        return {
            "best_action_id": best_id,
            "value": value_scalar,
            "top_actions": top_actions,
            "epoch": self.epoch,
        }


def make_handler(service: ModelInferenceService):
    class InferenceHTTPHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send_json(self, status: int, data: dict):
            body = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self._send_json(200, {"status": "ok"})

        def do_GET(self):
            if self.path == "/health":
                self._send_json(200, {
                    "status": "ok",
                    "checkpoint": service.checkpoint_path,
                    "epoch": service.epoch,
                    "device": str(service.device),
                })
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path == "/predict":
                content_length = int(self.headers.get("Content-Length", 0))
                if content_length <= 0:
                    self._send_json(400, {"error": "empty body"})
                    return

                try:
                    body = self.rfile.read(content_length)
                    data = json.loads(body.decode("utf-8"))
                    obs = data.get("obs")
                    mask = data.get("mask")
                    temp = float(data.get("temperature", 1.0))

                    if not obs or not mask or len(obs) != 725 or len(mask) != 256:
                        self._send_json(400, {
                            "error": f"Invalid obs length ({len(obs) if obs else 0}/725) or mask length ({len(mask) if mask else 0}/256)"
                        })
                        return

                    result = service.predict(obs, mask, temperature=temp)
                    self._send_json(200, result)
                except Exception as e:
                    self._send_json(500, {"error": str(e)})
            else:
                self._send_json(404, {"error": "unknown endpoint"})

        def log_message(self, format, *args):
            # 保持静默控制台，减少高频日志打扰
            pass

    return InferenceHTTPHandler


def main():
    parser = argparse.ArgumentParser(description="SplendorNet Inference Server")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="Path to checkpoint")
    parser.add_argument("--port", type=int, default=8088, help="Listening port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Listening host")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, auto)")
    args = parser.parse_args()

    # 处理相对路径
    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        candidate_paths = [
            ckpt_path,
            os.path.join(WORKSPACE_ROOT, ckpt_path),
            os.path.join(PROJECT_ROOT, ckpt_path),
        ]
        for p in candidate_paths:
            if os.path.exists(p):
                ckpt_path = p
                break

    try:
        service = ModelInferenceService(ckpt_path, device_name=args.device)
    except Exception as e:
        print(f"❌ Failed to initialize model service: {e}")
        sys.exit(1)

    server_address = (args.host, args.port)
    httpd = HTTPServer(server_address, make_handler(service))
    print("==================================================")
    print("🧠 SplendorNet 神经网络在线推理服务已启动！")
    print(f"🌐 监听地址: http://{args.host}:{args.port}")
    print(f"📁 权重路径: {ckpt_path} (Epoch: {service.epoch})")
    print(f"⚙️ 运行硬件: {service.device}")
    print("==================================================")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()


if __name__ == "__main__":
    main()
