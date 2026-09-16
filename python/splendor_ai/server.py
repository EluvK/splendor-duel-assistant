"""Lightweight HTTP Inference Server for SplendorNet."""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import sys
import threading
import time
from typing import Optional
import torch
import torch.nn.functional as F

# 确保包路径可用
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from splendor_ai.env import SplendorDuelEnv
from splendor_ai.net import SplendorNet


class ModelInferenceService:
    def __init__(self, checkpoint_path: str, device_name: str = "auto") -> None:
        if device_name == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_name)

        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self.net = SplendorNet().to(self.device)
        self.meta = {}
        self.epoch = 0
        self.last_mtime = 0.0
        self._onnx_bytes: Optional[bytes] = None
        self._lock = threading.Lock()

        self._load_checkpoint(is_reload=False)
        self.net.eval()
        self._start_fs_watcher(interval=1.5)

    def _load_checkpoint(self, is_reload: bool = False) -> bool:
        """加载或重载权重文件，支持热重载时的写入防抖和异常保护"""
        if not os.path.exists(self.checkpoint_path):
            if not is_reload:
                raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")
            return False

        try:
            current_mtime = os.path.getmtime(self.checkpoint_path)
            file_size = os.path.getsize(self.checkpoint_path)

            # 文件小于 1KB 极可能是写入刚开始，跳过等待完整写入
            if file_size < 1024:
                return False

            if not is_reload:
                print(f"📦 Loading SplendorNet from: {self.checkpoint_path} on {self.device}...")

            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

            with self._lock:
                self.net.load_state_dict(ckpt["model_state"])
                self.epoch = ckpt.get("epoch", 0)
                self.meta = ckpt.get("meta", {})
                self.last_mtime = current_mtime
                self.net.eval()
                self._onnx_bytes = None

            if is_reload:
                print(
                    f"🔄 [Auto-Reload] 检测到权重文件已更新，已成功热重载模型！"
                    f"最新 Epoch: {self.epoch} (体积: {file_size / 1024:.1f}KB)"
                )
            else:
                print(f"✅ Checkpoint loaded successfully! Epoch: {self.epoch}")
            return True
        except Exception as e:
            if is_reload:
                # 热重载期间可能遇到文件写入尚未完结，仅记录警告并保留现有模型可用
                print(f"⏳ 正在写入新权重中或重载受阻: {e}，将在下次变动时自动重试...")
            else:
                raise e
            return False

    def check_and_reload(self) -> bool:
        """检查文件修改时间并在发现新权重时热重载"""
        if not os.path.exists(self.checkpoint_path):
            return False
        try:
            current_mtime = os.path.getmtime(self.checkpoint_path)
            if current_mtime > self.last_mtime:
                return self._load_checkpoint(is_reload=True)
        except OSError:
            pass
        return False

    def _start_fs_watcher(self, interval: float = 1.5) -> None:
        """后台轮询线程，自动感知 best.pt 更新并实时热加载"""
        def _watch():
            while True:
                time.sleep(interval)
                try:
                    self.check_and_reload()
                except Exception as e:
                    pass

        t = threading.Thread(target=_watch, daemon=True, name="ModelFsWatcher")
        t.start()

    def get_onnx_bytes(self) -> bytes:
        """获取当前模型的 ONNX 二进制字节流，带线程安全缓存"""
        self.check_and_reload()
        with self._lock:
            if self._onnx_bytes is None:
                try:
                    # 优先在 CPU 上导出以保证跨平台稳定性
                    cpu_net = SplendorNet().to("cpu")
                    cpu_net.load_state_dict(self.net.state_dict())
                    cpu_net.eval()
                    self._onnx_bytes = cpu_net.export_onnx_bytes()
                except Exception:
                    self._onnx_bytes = self.net.export_onnx_bytes()
            return self._onnx_bytes

    @torch.no_grad()
    def predict(self, obs_list: list, mask_list: list, temperature: float = 1.0) -> dict:
        # 推理前顺便做一次快速检查
        self.check_and_reload()

        obs_t = torch.tensor(obs_list, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.tensor(mask_list, dtype=torch.bool, device=self.device).unsqueeze(0)

        with self._lock:
            logits, value, _turns, _reasons = self.net(obs_t)
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

            epoch_snapshot = self.epoch

        return {
            "best_action_id": best_id,
            "value": value_scalar,
            "top_actions": top_actions,
            "epoch": epoch_snapshot,
        }


def monitor_parent_process(parent_pid: int) -> None:
    """监听父进程存活状态，当父进程终止时自动自杀，彻底防止孤儿进程常驻后台"""
    if parent_pid <= 0:
        return

    def _watch():
        print(f"🛡️ 父进程监视守护已就绪，正在监听父进程 PID: {parent_pid}")
        if sys.platform == "win32":
            import ctypes
            SYNCHRONIZE = 0x00100000
            handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, parent_pid)
            if handle:
                # 阻塞等待父进程句柄变为 signaled（退出），零 CPU 开销、毫秒级响应
                ctypes.windll.kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)
                ctypes.windll.kernel32.CloseHandle(handle)
                print(f"\n🛑 检测到父进程 (PID: {parent_pid}) 已终止，Python 推理微服务退出...")
                os._exit(0)
            else:
                # 无法获取句柄，说明父进程已不在，直接退出
                print(f"\n🛑 未能获取父进程 (PID: {parent_pid}) 句柄，可能已退出，直接终止...")
                os._exit(0)
        else:
            while True:
                time.sleep(1.0)
                try:
                    os.kill(parent_pid, 0)
                except OSError:
                    print(f"\n🛑 检测到父进程 (PID: {parent_pid}) 已终止，Python 推理微服务退出...")
                    os._exit(0)

    t = threading.Thread(target=_watch, daemon=True, name="ParentProcessWatcher")
    t.start()


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
                    "mtime": service.last_mtime,
                })
            elif self.path == "/reload":
                reloaded = service.check_and_reload()
                self._send_json(200, {
                    "status": "ok",
                    "reloaded": reloaded,
                    "epoch": service.epoch,
                    "checkpoint": service.checkpoint_path,
                    "mtime": service.last_mtime,
                })
            elif self.path == "/onnx":
                try:
                    onnx_bytes = service.get_onnx_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(onnx_bytes)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(onnx_bytes)
                except Exception as e:
                    self._send_json(500, {"error": f"Failed to export onnx: {e}"})
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path == "/reload":
                reloaded = service.check_and_reload()
                self._send_json(200, {
                    "status": "ok",
                    "reloaded": reloaded,
                    "epoch": service.epoch,
                    "checkpoint": service.checkpoint_path,
                    "mtime": service.last_mtime,
                })
            elif self.path == "/predict":
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

                    if not obs or not mask or len(obs) != SplendorDuelEnv.OBS_SIZE or len(mask) != SplendorDuelEnv.ACTION_SIZE:
                        self._send_json(400, {
                            "error": f"Invalid obs length ({len(obs) if obs else 0}/{SplendorDuelEnv.OBS_SIZE}) or mask length ({len(mask) if mask else 0}/{SplendorDuelEnv.ACTION_SIZE})"
                        })
                        return

                    result = service.predict(obs, mask, temperature=temp)
                    self._send_json(200, result)
                except Exception as e:
                    self._send_json(500, {"error": str(e)})
            else:
                self._send_json(404, {"error": "unknown endpoint"})

        def log_message(self, format, *args):
            # 保持静默控制台，减少高频预测日志打扰
            pass

    return InferenceHTTPHandler


def main():
    parser = argparse.ArgumentParser(description="SplendorNet Inference Server")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="Path to checkpoint")
    parser.add_argument("--port", type=int, default=8088, help="Listening port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Listening host")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, auto)")
    parser.add_argument("--parent-pid", type=int, default=0, help="Parent process PID to monitor for auto exit")
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

    # 启动父进程守护监听
    if args.parent_pid > 0:
        monitor_parent_process(args.parent_pid)

    server_address = (args.host, args.port)
    httpd = HTTPServer(server_address, make_handler(service))
    print("==================================================")
    print("🧠 SplendorNet 神经网络在线推理服务已启动！")
    print(f"🌐 监听地址: http://{args.host}:{args.port}")
    print(f"📁 权重路径: {ckpt_path} (Epoch: {service.epoch})")
    print(f"⚙️ 运行硬件: {service.device}")
    if args.parent_pid > 0:
        print(f"🔗 绑定父进程: PID {args.parent_pid} (父进程退出将自动销毁)")
    print("🔄 自动热重载: 已开启 (文件变更时自动加载新权重)")
    print("==================================================")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()


if __name__ == "__main__":
    main()
