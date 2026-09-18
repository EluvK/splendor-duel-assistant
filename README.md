# 璀璨宝石：对决 (Splendor Duel) - AI 对战系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![GitHub Pages](https://img.shields.io/badge/Play%20Online-GitHub%20Pages-blue?logo=github)](https://eluvk.github.io/splendor-duel-assistant/index.html) [![Rust](https://img.shields.io/badge/Engine-Rust%202024-orange?logo=rust)](engine/) [![Python](https://img.shields.io/badge/AI-PyTorch%20%7C%20ONNX-green?logo=pytorch)](python/) [![WebAssembly](https://img.shields.io/badge/WASM-Supported-purple?logo=webassembly)](engine-wasm/)

本项目是专为双人经典桌游**《璀璨宝石：对决》(Splendor Duel)** 打造的 AI 策略研究与人机交互对战平台。

结合了**高性能 Rust 游戏规则引擎**、**类 AlphaZero 的深度强化学习体系**以及**纯前端无后端的 Web 现代化交互界面**，你既可以在浏览器中直接挑战深度学习 AI，也可以在本地进行深度复盘和策略研究。

---

## 🎮 立即游玩 (Play Online)

无需安装任何环境、无需下载任何依赖，直接在浏览器中与 AI 展开对决：

👉 **[点击这里立即开始在线对战 (GitHub Pages)](https://eluvk.github.io/splendor-duel-assistant/index.html)**

> 💡 **完全单机运行**：游戏规则计算由编译为 **WebAssembly (WASM)** 的 Rust 引擎在浏览器本地完成，AI 决策由 **ONNX Runtime Web** 直接在前端推理，完全不消耗服务器算力，离线畅玩，~~所以也完全不支持联机~~。

---

## 🛠 开发者进阶

本项目使用 Rust 作为核心逻辑与高性能模拟层，Python 负责深度强化学习训练，类 AlphaZero 强化学习网络。

---

## 📚 详细文档

想要深入了解技术实现与算法细节？请参阅 `docs` 目录下的完整技术文档：

- 📖 [游戏详细规则说明](docs/splendor-duel-rules.md)
- ⚙️ [Rust 引擎架构设计](docs/engine-architecture.md)
- 🧠 [AI 网络与强化学习架构](docs/ai-architecture.md)
- 🔢 [状态与动作编码协议](docs/action-encoding.md)
- 🎓 [AI 训练与自博弈实战指南](docs/training_guide.md)
- 🌐 [Web 可视化与对战交互架构](docs/web-architecture.md)
- 🚀 [研发路线图](docs/roadmap.md)

---

## 📄 开源协议

本项目采用 [MIT License](LICENSE) 协议开源。欢迎交流学习、提出 Issue 与 PR！
