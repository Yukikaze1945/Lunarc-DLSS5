# 媒体格式与实际编码

扩展名只用于选择导入或导出路径，不能单独证明文件的编码、位深或色彩标记。视频和部分图片会检查实际流信息；不符合当前支持条件的素材会被拒绝，不会静默降位深。

## 视频

| 用途 | 扩展名／容器 | 实际编码与限制 |
| --- | --- | --- |
| 导入 | MP4、MKV、MOV、AVI、WMV、FLV、WebM、M4V、MPG／MPEG、TS／MTS／M2TS 等 | 按实际视频流探测编码、像素格式、位深和色彩标记；扩展名不保证特定编码。ProRes 422 10-bit 输入需实际为 `codec_name=prores`、`yuv422p10le`。 |
| H.264 SDR 8-bit | MKV、MP4、MOV | H.264 NVENC；输出为 8-bit SDR。 |
| HEVC SDR 8-bit | MKV、MP4、MOV | HEVC NVENC；输出为 8-bit SDR。 |
| HEVC SDR 10-bit 或 HDR | MKV | HEVC NVENC；SDR 10-bit 与 HDR 使用 10-bit 像素格式，HDR 另写入 PQ／BT.2020 色彩标记。 |
| AV1 SDR 8-bit | MKV、MP4、MOV、WebM | AV1 NVENC；输出为 8-bit SDR。 |
| AV1 SDR 10-bit 或 HDR | MKV | AV1 NVENC；SDR 10-bit 与 HDR 使用 10-bit 像素格式，HDR 另写入 PQ／BT.2020 色彩标记。 |
| ProRes 422 HQ 导出 | MOV（`.mov`） | `prores_ks`，HQ profile 3，`yuv422p10le`。GPU 完成渲染后传回主机，由 CPU 软件编码；不是 NVENC。 |

ProRes 支持 BT.709 SDR 和 BT.2020／PQ HDR 图像及色彩声明；当前 ProRes 写入器不写 mastering display／MaxCLL 静态元数据。MOV 中的音轨和字幕仍需符合容器支持的编码。

## 图片

| 用途 | 扩展名／格式 | 实际编码与输入限制 |
| --- | --- | --- |
| SDR 输入 | PNG、JPG／JPEG、BMP、WebP、TIF／TIFF | 当前图片入口按 8-bit SDR 处理；不支持的高位深会拒绝。 |
| AVIF 输入 | `.avif` | 支持 8-bit SDR；HDR 需完整 BT.2020／PQ 标记及受支持位深。仅看扩展名不能判断 SDR／HDR。 |
| JPEG XL 输入 | `.jxl` | 仅接受 8-bit RGB、sRGB primaries 和 sRGB transfer 的 SDR。 |
| HEIC／HEIF 输入 | `.heic`、`.heif` | 仅接受 8-bit；高位深或 HDR 输入拒绝。HEIC／HEIF 名称及扩展名不保证文件位深。 |
| JPEG XR 输入 | `.jxr`、`.wdp`、`.hdp` | 支持 8-bit SDR 及 scRGB fixed／float HDR；16／32-bit 整数输入拒绝。 |
| SDR 输出 | PNG、JPEG、WebP、AVIF、HEIC、HEIF、JXL | 由格式按钮选择；JPEG 为 `.jpg`，HEIC 为 `.heic`，HEIF 为 `.heif`。HEIC 和 HEIF 使用同一 HEVC-in-HEIF 编码器。 |
| HDR 输出 | AVIF、HEIC、HEIF、UHDR JPEG、JXL | AVIF／HEIC／HEIF 写入 10-bit BT.2020／PQ；JXL 写入 16-bit RGB PQ；UHDR JPEG 使用 SDR 基图与 HDR 增益图（`.jpg`）。 |

图片的格式支持和位深限制详见[图片加载与导出](image-export.md)。
