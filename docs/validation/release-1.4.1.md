# 1.4.1 发布检查

日期：2026-11-06。Windows x64，Electron 42.2.0。

## 安装包

- 文件：`Lunarc-1.4.1-x64-Setup.exe`。
- 大小：149,732,729 字节（约 142.8 MiB）。
- SHA-256：`2f797b0061d0122b2563079aa752365afcd85278bf145504496297cc717d050d`。
- Authenticode：未签名。
- 更新附件：安装程序、`INSTALL.md`、`INSTALLER.json`、`SHA256SUMS.txt`。

## 验证

- PR #2 已合并，修复原生预览 DLL 的打包遗漏和环境页面的翻译。
- TypeScript、Vite、Electron、NSIS 构建通过。
- 打包目录包含 `gpu_native_presenter.dll`，与项目原生 DLL 哈希一致；测试不再将该 DLL 排除。
- 打包的 Python、前端、原生 DLL 与源码目录逐项匹配；组件清单与许可证检查通过，未包含 NVIDIA 模型或完整 Python/AI 环境。
- 使用全新临时配置启动打包版，首次使用界面显示正常；组件缺失提示正确，更新器报告当前版本 1.4.1。
- 26 项 Python、12 项打包与翻译 Node 检查，以及 9 项更新器检查通过。
- 更新器继续使用 GitHub 发布元数据、`INSTALLER.json` 和 `SHA256SUMS.txt` 核对下载文件；安装需退出软件后运行 Setup。

## 范围

本轮没有执行 GPU 渲染或在 issue 报告者的机器上复现；没有覆盖安装本机软件或重新下载运行组件。打包、库加载与 CPU 测试不能替代对方更新后的实际预览验证。
