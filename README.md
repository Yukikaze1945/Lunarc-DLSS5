**简体中文** · [English](README.en.md) · [日本語](README.ja.md) · [Русский](README.ru.md)

# Lunarc — DLSS 5 视频与图片增强

<p>
  <img src="docs/assets/lunarc-logo.svg" width="32" height="32" alt="Lunarc" />
  &nbsp; <strong>LUNARC</strong>
</p>

![Lunarc — 视频增强与插帧](docs/assets/lunarc-banner.png)

**Windows x64** &nbsp; · &nbsp; **GPL-3.0-or-later**

Lunarc 使用 DLSS 5（DLSS5 / DLSSNR）神经渲染增强视频与图片，支持 DLAA、RTX VSR、NVIDIA VFG 补帧、TrueHDR SDR 转 HDR，以及独立图片放大与导出。

[下载安装版](https://github.com/Yukikaze1945/Lunarc-DLSS5/releases/tag/v1.4.0) &nbsp; / &nbsp; [功能手册](docs/user-guide.md) &nbsp; / &nbsp; [官方网站](https://lunarc-workbench.defaddg-9590.chatgpt.site/) &nbsp; / &nbsp; [安装说明](docs/installing.md) &nbsp; / &nbsp; [源码运行](docs/getting-started.md)

本文介绍当前源码；安装版功能请以对应发行说明为准。

## 01 / 界面

![视频预览、渲染顺序与画质设置](docs/assets/lunarc-workbench.png)

<sub>截图中的视频为演示素材。</sub>

| 画面处理 | 视频对比 | 批量导出 |
| :--- | :--- | :--- |
| 拖动卡片调整 DLAA、DLSSNR、RTX VSR 和 VFG 的处理顺序。 | 查看原图、处理结果和分屏预览，对比导出前后的视频。 | 设置编码、画质和保存位置，管理导出任务。 |

## 02 / 功能

| 功能 | 说明 |
| :--- | :--- |
| DLSSNR | 单层、双层和三层处理，各层参数可单独调整；支持顺序处理和实验性多尺度数据一致性融合。 |
| DLAA / RTX VSR | DLAA 模型预设；RTX VSR 质量档位，按倍率或指定分辨率输出。 |
| 光流 | NVIDIA NVOF、AMD FidelityFX Optical Flow 和 RAFT-Large。RAFT 分析尺寸可选长边 512、720、1080。 |
| NVIDIA VFG | 按固定倍率或目标帧率插帧，支持镜头切换检测。仅用于导出；高倍率为实验性选项，最高 16×。 |
| 色彩 | SDR 8-bit、SDR 10-bit 和 HDR10。HDR10 为实验性功能，支持保留静态 HDR 元数据。 |
| TrueHDR | SDR 转 HDR10，可调整对比度、饱和度、中灰亮度和目标峰值，支持 AI 去色带；需要外部 NVIDIA 运行库。 |
| 工作台 | 预设选择与保存、参数差异查看、可折叠卡片、区间导出和预渲染缓存。已完成且参数匹配的缓存区间显示绿条。 |
| 素材与封装 | 拖入视频或图片；视频导出可保留音轨、字幕、章节、附件与元数据，具体取决于输出容器。图片支持独立处理和 PNG、JPEG、WebP、AVIF、HEIC/HEIF、Ultra HDR、JPEG XL 等格式导出。 |
| 运行库管理 | 下载、导入和切换 DLSSNR 版本，检查文件摘要、数字签名和版本。 |

## 03 / 安装

1. 在 [Releases](https://github.com/Yukikaze1945/Lunarc-DLSS5/releases/tag/v1.4.0) 下载并运行 `Lunarc-1.4.0-x64-Setup.exe`。
2. 首次启动按提示安装基础运行环境和视频处理组件。
3. 在 **运行与诊断 → DLSSNR 运行库** 中下载或导入运行库，校验后自动选用。

RAFT 光流需要另外安装 PyTorch/CUDA 和模型，可以在 **软件设置 → 组件管理** 中安装。不使用 RAFT 时无需下载这些文件。更新软件会保留已安装且版本匹配的组件。

### 系统要求与限制

- Windows x64、支持所选功能的 NVIDIA 显卡及驱动。
- DLSSNR、DLAA、RTX VSR 和 VFG 需要对应的 NVIDIA 运行库，安装包不包含这些文件。
- 首次安装组件需要联网；RAFT 下载量为数 GB，安装后需重启软件。
- Windows 工作台使用原生 GPU 视频预览。启用 Windows HDR 时通过 FP16/scRGB 显示 HDR；未启用时映射为 SDR。文件对比页仍使用 SDR 映射。
- 处理速度取决于显卡、素材分辨率和所选设置。已测试的环境与功能见 [测试记录](docs/public-release.md)。

## 04 / 源码运行

```powershell
git clone https://github.com/Yukikaze1945/Lunarc-DLSS5.git
cd Lunarc-DLSS5
```

按 [源码运行指南](docs/getting-started.md) 安装依赖、编译原生库，然后运行根目录的 **`run_gui.bat`**。

## 05 / 文档

| 文档 | 内容 |
| :--- | :--- |
| [使用手册](docs/user-guide.md) | 渲染链、参数、图片、导出与问题排查；提供四种语言。 |
| [语言与翻译维护](docs/localization.md) | 界面语言、覆盖范围与翻译维护。 |
| [安装说明](docs/installing.md) | 安装、组件下载和更新。 |
| [DLSSNR 运行库](docs/dlssnr-runtime.md) | 下载、本地导入和版本管理。 |
| [DLAA / VSR / VFG 配置](docs/native-runtime.md) | NVIDIA 运行库与 SDK 路径。 |
| [AMD 光流](docs/AMD_FLOW_DEPENDENCIES.md) | 编译依赖与 SDK 版本。 |
| [软件更新](docs/app-updates.md) | 检查和下载新版安装程序。 |
| [多尺度融合](docs/multiscale-solver.md) | 求解方法、实测耗时与限制。 |
| [GPU 预览](docs/native-hdr-preview.md) | SDR/HDR 显示与中央滑块对比。 |
| [预设与区间预览](docs/workbench-presets.md) | 预设管理、区间导出和缓存绿条。 |
| [MCP 自动化](docs/mcp.md) | 预设、自定义参数、渲染提交及完成/错误结果。 |
| [构建安装包](packaging/README.md) | 打包方法与依赖。 |
| [测试记录](docs/public-release.md) | 测试环境、结果和未覆盖的项目。 |

## 06 / 许可证

Lunarc 使用 **GPL-3.0-or-later**，详见 [LICENSE](LICENSE)。第三方组件适用各自的许可证，见 [第三方说明](THIRD_PARTY_NOTICES.md)。
