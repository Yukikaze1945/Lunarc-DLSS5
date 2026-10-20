# Lunarc · User guide

[简体中文](../user-guide.md) · [English](../en/user-guide.md) · [日本語](../ja/user-guide.md) · [Русский](../ru/user-guide.md)

Based on current source. The released installer may not include newer features. Availability depends on GPU, driver and runtimes.

The interface follows your system language. To override it, open Preferences → Language.

<a id="quickstart"></a>
## First steps

Lunarc processes videos and images locally on Windows. Install the runtime first, then add the effects you need. You do not have to download every model.

1. Download the installer from [GitHub Releases](https://github.com/Yukikaze1945/Lunarc-DLSS5/releases), install it and launch Lunarc.
2. Follow the first-run tutorial to install the **base runtime** and **video processing components**.
3. For DLSSNR, open **Runtime & diagnostics → DLSSNR runtime**, download or import a library, then select it for use. DLAA, VSR, VFG and TrueHDR each need the corresponding NVIDIA components.
4. Click **Import**, or drag a local video or image into the workbench.
5. Enable one effect first and compare the original with the result. Add other stages after checking it.
6. For video, open **Export & queue**. For images, use **Export image** at the top right.

The installed application does not require a separate Python, Node.js or CUDA Toolkit installation. RAFT and dedicated image models need the optional AI model runtime. Skip it if you do not use those models.

<a id="pipeline"></a>
## Processing order and switches

The chain on the right of the workbench runs **from top to bottom**. Each stage receives the preceding stage’s output. Changing the order can change the result, processing time and memory requirements.

| Control | Action |
| --- | --- |
| Left drag handle | Reorder a stage. Text fields and parameter sliders do not start a card drag. |
| Up / down arrows | Move a stage by one position. |
| Eye button | Temporarily disable the stage while keeping its settings and position. Click again to restore it. |
| ＋ / − | Expand or collapse the parameters. Collapsing does not disable the effect. |
| Remove | Remove the stage from the current chain. |

DLAA, DLSSNR, RTX VSR, TrueHDR and NVIDIA VFG all use this ordering system. You can add VSR more than once; each card keeps its own mode, quality, strength and output dimensions.

Models placed after VFG also process generated frames, so moving the same models before or after frame generation can change the workload. No single order is right for every source.

[Try the chain ordering demo ↗](https://lunarc-workbench.defaddg-9590.chatgpt.site/en/#render-order)

<a id="nr"></a>
## DLSSNR and multiple layers

DLSSNR is the neural reconstruction stage. Adjust the model’s own parameters first, then decide whether to use additional residual, color and temporal controls.

| Parameter | Meaning |
| --- | --- |
| NR strength | Controls the strength of the model’s corrections. Higher values mean stronger changes, not necessarily a better result. |
| Local tone strength | Controls local tone correction inside the model. It is not global exposure or HDR display tone mapping. |
| Local structure strength | Controls corrections to texture and local structure. |
| Skin structure strength | Controls skin structure correction independently of local structure. |
| Automatic mask | Uses the model’s automatic mask. Manual region protection is a separate control. |
| NR interface correction | Requests correction of text and interface regions. It cannot guarantee recognition of every subtitle. |
| Image style | Default, Natural and Cinematic. Compare the same frame to judge the actual change. |

### One, two or three layers

With **sequential processing**, layer two processes layer one’s output, and layer three processes layer two’s. Use the per-layer parameter control to configure each layer independently; layer one’s settings are not shared settings for the entire chain.

**Multiscale data-consistency fusion** is a separate experimental route. A solver combines results from different scales; this is not the same as processing one layer after another. Layer sizes, weights and solver settings affect the result and resource use. Layer count alone does not indicate quality.

### Working resolution and export size

Working scale changes the dimensions used by the processing stage. Reducing it is not simply shrinking the preview window. The resolution on the export page determines the final file size in pixels; check these two settings separately.

<a id="dlaa"></a>
## DLAA models

Select a model preset on the DLAA card. The current interface offers **Default, J, K, L, M** and the legacy compatibility options **E / F**.

- Default: lets the runtime choose.
- K: the project’s suggested starting point for DLAA comparisons.
- J, L, M: compare fine lines, text and moving edges on your source.
- E, F: legacy hints that may be substituted by the current runtime.

These are model hints, not encoding profiles or VSR quality levels. The driver or runtime may override them. Switching models can reload the session, making the first update slower.

<a id="vsr"></a>
## RTX VSR

Processing mode, quality, effect strength and output size are independent settings. Choose a mode first, then check which controls it supports.

| Mode | Purpose and output |
| --- | --- |
| Standard super resolution | General video upscaling by a multiplier or target resolution. |
| High-bitrate super resolution | Super resolution for sources with relatively little compression damage. |
| Streaming super resolution | Requires a supported GPU and runtime; the current mode requires RTX 30 series or newer. |
| Denoise | Keeps the input size. Excessive strength may smooth away detail. |
| Deblur | Keeps the input size. It can also amplify noise. |
| Bicubic scaling | Conventional resizing without an AI model. |

### Quality and strength

Quality levels are **Low, Medium, High and Very high**. Streaming mode offers only **Medium and Very high**. Keep the runtime’s default effect strength, or set a custom value from 0–100%. The speed, memory use and effect of higher quality levels depend on the source and hardware.

### Output dimensions

**By multiplier** offers 2×, 4× and Custom. A single super-resolution stage currently supports a factor greater than 1 and no more than 4, preserves aspect ratio and aligns output dimensions to even values. **By resolution** offers presets or custom dimensions, subject to the engine’s aspect-ratio and per-axis scale limits.

Multiple VSR cards scale sequentially. If export is set to Source resolution, the enlarged chain result is resized back before saving. Choose Processing-chain output to keep the chain’s final dimensions.

<a id="truehdr"></a>
## TrueHDR and color

TrueHDR converts SDR to HDR10 and requires VFX SDK Core plus the TrueHDR effect module. Its card provides **Contrast, Saturation, Middle gray, Target peak brightness** and an **AI debanding** switch.

| Parameter | Range |
| --- | --- |
| Contrast / Saturation | 0–200; default 100. |
| Middle gray | 10–100; default 50. |
| Target peak | 400–2000 nit; default 650 nit. |
| AI debanding | Enabled by default; can be disabled independently. |

HDR10 sources can also enter TrueHDR. At that stage, the source is first mapped to SDR, then converted into a new HDR image. This reinterprets brightness and color; it is **not a lossless round trip of the original HDR**.

The TrueHDR interface accepts RGBA8 SDR, so even a 10-bit SDR source does not make this effect an end-to-end lossless 10-bit path. HDR10 processing is experimental. HLG, HDR10+ and Dolby Vision are outside the support described here.

### Why can the preview look different?

The Windows desktop app supports native FP16/scRGB preview for video and TrueHDR stills. It can present HDR when both the display and Windows HDR are enabled; otherwise it maps HDR to SDR. File comparison, ordinary image exports and this website use SDR tone mapping and do not represent an HDR display’s actual peak brightness.

<a id="flow"></a>
## Optical-flow guidance

Optical flow guides motion in the reconstruction stage. It is separate from NVIDIA VFG, which uses its own frame-generation process. You do not need to calculate RAFT flow specifically for VFG.

| Engine | Settings |
| --- | --- |
| NVIDIA NVOF | Uses NVIDIA’s optical-flow interface. The quality choices reflect the presets exposed by the current engine. |
| AMD FidelityFX Optical Flow | Performance and Quality use half-resolution and full-resolution analysis. This does not reduce the main image’s dimensions. |
| RAFT-Large | Analysis long-edge limits of 512 / 720 / 1080 with independently adjustable iterations. Requires optional PyTorch/CUDA and model weights. |

RAFT’s Speed, Balanced and Quality defaults are **512 / 4 iterations, 720 / 6 iterations and 1080 / 12 iterations**. Explicit settings below the preset take priority. Analysis size affects motion estimation, not the main processing resolution. Larger sizes and more iterations usually require more computation; inspect actual moving regions to judge the result.

<a id="quality"></a>
## Image quality and temporal stability

Post-processing controls adjust the **model’s correction relative to the original**. They are not global grading of the whole image and should not be confused with NR strength.

| Control | Meaning |
| --- | --- |
| Overall / residual strength | Scales the correction between the model output and the original, separately from the model’s internal NR strength. |
| Lightness / chroma strength | Controls lightness and chroma corrections independently. Chroma 0 retains the original chroma; 1 retains the model’s chroma change. |
| Low / high frequency strength | Controls broad changes and fine texture corrections respectively. |
| Hue / shadow / highlight protection | Limits corrections in the corresponding regions or directions. Zero disables that protection. |
| Soft compression | Reduces excessive corrections while preserving their positive or negative direction. |
| Region protection | Reduces corrections inside a rectangle or ellipse, with optional feathering. Independent of the automatic model mask. |

### Temporal stability and scene detection

Static low-frequency accumulation mixes historical residuals only where the source is locally stable. It does not perform optical-flow reprojection. Seeking, changing media or scene changes reset the history. Scene detection on its own does not retain full-resolution temporal history.

Debug views output residuals or protection weights as the image and may also affect GPU export. Turn off debug display before the final export.

<a id="preview"></a>

Importing media displays the source without loading effect models. Scrubbing also shows source frames; HDR source thumbnails use SDR display mapping. Configure the chain, then click **Start rendering** above the preview. Choose **Real-time rendering** or **Pre-render** for video. Pre-render offers 5/10 frames or 0.5/1/2/3/5 seconds from the current frame, clamped at the end. Use **Review result** or the green cache marker to inspect it in Video comparison. This does not change the export selection, and parameter edits do not automatically re-render cached clips. Stopping real-time rendering waits for the current frame and releases preview VRAM. Changing media, reopening the app or recovering the backend requires another manual start.
## Preview and range selection

The workbench offers Original, Enhanced, central comparison and side-by-side views. The central divider redraws retained GPU images without rerunning the model. Changing model parameters may still require a new processing pass.

### Render a short section first

1. Load a video and enable **Export selection** below the timeline.
2. Drag either boundary or enter start and end frames. Both boundaries can position the displayed frame.
3. Use **Pre-render selection** to produce a genuinely encoded clip and inspect the final result.
4. A **green bar** marks a completed range that still matches the current source and settings. Click it or View result to open comparison playback.

Changing the source, rendering or encoding settings invalidates the old green marker. Pre-rendering uses the same GPU export queue and pauses real-time preview. Disable Export selection to return to whole-file export.

Frame numbers start at 1 and the end frame is included. Frame-based selection requires constant frame rate and continuous timestamps; unsupported variable-frame-rate or missing-frame inputs are rejected. Temporal history resets at the clip’s beginning, so boundary frames may differ from a continuous render starting at the beginning of the full source.

### Reading the status bar

- **GPU memory** is total card usage, including other applications, not Lunarc-only usage.
- **Render FPS** is calculated from the latest processed frame. Initialization, seeking and caching affect it.
- **Display FPS** is actual presentation speed, not the source’s declared frame rate.
- **Stage latency** is not a set of additive GPU timestamps. Treat cache reuse, fusion, flow and display measurements separately.

<a id="presets"></a>
## Preset management

The preset selector and **Save preset** button sit above Processing order on the right of the workbench. Adjust parameters, then name and save them. Saving the same name replaces that preset.

Preset management lists saved presets. Expand one to see changes from the defaults, the old and saved values, and their differences. NR strength changing from 1.00 to 0.70 is shown as −0.30 and −30%.

- When the default is zero, only the absolute difference is shown.
- Switches, styles and stage order show option changes rather than meaningless percentages.
- Changes to the second and third layers are recorded separately.
- Editing after applying a preset changes the selector to Current settings.
- JSON import and export support backup and migration.

[Try the preset differences demo ↗](https://lunarc-workbench.defaddg-9590.chatgpt.site/en/#preset-demo)

<a id="images"></a>
## Images and upscaling models

Import PNG, JPEG, BMP, WebP, TIFF, AVIF, HEIC / HEIF and JXR (JPEG XR) images. Ordinary images use 8-bit SDR; fully tagged BT.2020/PQ AVIF and scRGB HDR JXR (fixed/float) use the linear FP16 master. 16/32-bit integer JXR and other high-bit-depth SDR input are explicitly rejected; animated images use only the first frame.

### Single-frame rendering, memory and cached results

Importing an image shows the original first. Click **Render one frame** to run the effects. Setting changes do not render automatically unless you enable **Realtime rendering**. Disable it to return to manual rendering. The current work completes before the latest request starts; model chains do not run concurrently.

- **Lower VRAM (one stage at a time)** is the default. Each card stores its complete intermediate pixels in RAM and releases its GPU resources before the next card opens. SDR uses RGBA8 and HDR uses linear FP16, without preview-image compression. Transfers and model reloads add time.
- **Fast (keep the full chain loaded)** retains models and processing resources for repeated adjustments, at higher VRAM cost.
- Release boundaries are chain cards. Internal multilayer NR processing stays unchanged. No automatic tiling or precision reduction is used; an individual stage may still exceed available VRAM.

Only the **current and previous** full-resolution results are kept in RAM. The third successful render evicts the oldest; closing the backend clears them. Select **Comparison reference → Previous result** to compare them without inference. Different sizes are fitted to the current result for display; each result retains its SDR/HDR interpretation and export dimensions.

### Inspecting an image

- **Alt + mouse wheel**: zoom around the pointer, from 25% to 800%.
- **Left drag**: pan. When enlarged, the ordinary wheel scrolls vertically.
- **Double-click**: return to Fit.

The view position does not affect export dimensions. Zooming, panning and moving the comparison divider do not rerun the model.

### Dedicated upscaling models

| Model | Current integration |
| --- | --- |
| IllustrationJaNai V3 | 4× DAT2, Denoise and Detail variants. |
| AnimeSharp V4 | 2× RCAN standard variant. |

These models need the AI runtime, image-model component and separate weights. Official-source downloads and local imports are supported. Full-image FP32 inference is the default. Optional tiling changes context and edge results and is not silently enabled after running out of memory. These models currently accept SDR images only, so place them before TrueHDR.

### Saving an image

Click **Export image**. The default **Rendered result (format conversion only)** saves the full-resolution cache without running models again. Choose the current or previous result from the same source. After changing settings, render again or explicitly allow exporting the saved result without later changes. **Render again and export** remains available. PNG, JPEG, WebP, AVIF and HEIC availability depends on installed encoders.

Choose **Output color**:

- **SDR · sRGB**: PNG, JPEG, WebP, AVIF or HEIC at 8-bit SDR. TrueHDR results are tone-mapped to SDR. PNG supports lossless output and transparency; JPEG fills transparent areas with white. WebP quality 100 uses lossless encoding.
- **HDR · 10-bit PQ**: enable TrueHDR in the processing chain, then choose AVIF or HEIC. The processed FP16/scRGB image is converted directly to BT.2020/PQ and saved at 10-bit, full range, 4:4:4, without SDR display tone mapping. AVIF requires the video component's 10-bit libaom-av1 encoder and AVIF muxer; HEIC requires pillow-heif/libheif. The controls reflect installed components.

- **UHDR JPEG**: writes `.jpg` with a Display-P3 SDR base and HDR gain map. Ordinary JPEG viewers display the base; compatible viewers reconstruct HDR. No transparency. Available only for HDR results.
- **JXL**: 8-bit sRGB for SDR, or 16-bit BT.2020/PQ RGB for HDR, with transparency. Quality 100 losslessly encodes the supplied pixels; earlier gamut conversion and quantization still apply.

Codecs are isolated in the application runtime; missing components disable the formats. 8-bit sRGB JXL can be imported again. HDR JXL input is not connected yet; UHDR JPEG import reads the SDR base. [Codec and color details](../image-codecs.md).

Quality 100 does not imply lossless HDR RGB: gamut limits and 10-bit quantization still apply. Negative BT.2020 components and components above 10000 nits are clipped and counted in the save report; exposure is not normalized. Unmeasured mastering-display metadata is not invented.

Source transparency can be retained when the encoder supports it. Cached results use an SDR-mapped thumbnail; a fresh render may still show the original thumbnail after saving. Check the saved file in an HDR-capable viewer. **Re-importing HDR stills is not yet supported.** The thumbnail does not represent the saved HDR image.

Still images do not calculate inter-frame flow or temporal stability, do not run VFG and do not enter the video batch-export queue.

<a id="framegen"></a>
## NVIDIA VFG frame generation

VFG is a reorderable stage that can be disabled independently. It runs for **export**. Real-time preview skips it and retains the source frame rate. Pre-render a selection to inspect final motion.

| Setting | Use |
| --- | --- |
| NVIDIA VFG (Fixed) | Outputs 2×–16× the source frame rate. Multipliers 5×–16× are experimental. |
| NVIDIA VFG (Dynamic) | Enter a frame rate higher than the source, such as 60, 120 or 60000/1001. Intermediate timestamps are calculated automatically. |
| Quality | Low, Medium or High: VFG model modes, not RAFT or NVOF flow quality. |
| Scene-change detection | Avoids blending across a cut. Disabling it does not disable scene detection in other stages. |

The final source frame is explicitly held where no future frame exists. Dynamic mode uses a uniform output time grid; end-duration rounding is less than one output frame. Nonuniform source timestamps are rejected.

Requires a supported NVIDIA GPU, VFX Core and VideoFrameGeneration module. Frame generation does not repair baked-in duplicate frames or every motion artifact. Inspect the actual output.

<a id="export"></a>
## Video export and queue

### Final resolution

| Option | Final result |
| --- | --- |
| Processing-chain output | Uses the entire chain’s output dimensions. An 8K VSR chain is saved at its 8K dimensions. |
| Source resolution | Uses the source dimensions, resizing an enlarged chain result back before saving. |
| Specified resolution | Uses a preset or custom width and height. |
| Multiplier | Calculates ×2, ×0.5 or a custom factor from the input dimensions. |

The **export resizing algorithm** appears and runs only when the final and chain dimensions differ. Nearest neighbor, bilinear, bicubic and Lanczos 2 run after the complete chain, including generated frames.

### 10-bit and HDR encoding

The native P010 input currently supports these combinations in **MKV**:

| Codec | Input / color | Profile |
| --- | --- | --- |
| HEVC | SDR 10-bit / BT.709 | Main10 |
| HEVC | HDR10 / BT.2020, PQ | Main10 |
| AV1 | SDR 10-bit / BT.709 | Main, 10-bit |
| AV1 | HDR10 / BT.2020, PQ | Main, 10-bit |

AV1 10-bit uses Main, not HEVC’s Main10 name. Controls restrict choices according to input color, codec and runtime. Unsupported hardware is rejected; do not expect a silent switch to 8-bit or another encoder.

Audio, subtitles, chapters, metadata and attachments are retained according to export choices and container support. Source static HDR metadata is retained only when applicable. After TrueHDR creates a new image, old mastering information is not presented as measurements of the new output.

### Managing the queue

1. Set the output directory, filename and encoding options, then click **Add to queue**.
2. Start the queue to view progress, FPS and estimated remaining time.
3. Stop a running job before deleting it. Pending, completed, cancelled and failed jobs can be removed.
4. On failure, open the diagnostic report and keep the error and settings. Correct the issue before submitting again rather than repeatedly retrying the same invalid request.

<a id="runtime"></a>
## Runtimes, memory and updates

### Install components as needed

Start with the base runtime and video components. Native flow, DLSSNR and GPU decoding/encoding do not require PyTorch. RAFT and dedicated image models need the AI runtime and their own weights.

DLAA checks the local NGX cache by default. VSR, VFG and TrueHDR look for the relevant VFX directories. If a component is missing or incompatible, configure its actual directory in Components, Runtime & diagnostics or advanced settings. 10-bit / HDR effects need runtimes that support those interfaces.

### DLSSNR versions

The manager supports local DLL / ZIP imports, trusted directories, version downloads and an independent cache. It checks SHA-256, the Windows signature, publisher, architecture, product and version. Old preview sessions close before a version switch. An ordinary DLSS DLL is not a DLSSNR library.

Original and known community builds are identified. Other third-party DLLs can also be imported and selected automatically, with a one-time unidentified-DLL notice per file and no confirmation. File format and hash checks remain. New runtimes offer A/B models, with independent choices for layers 2 and 3. Switching closes the old pipeline before rebuilding; existing presets are preserved.

Libraries retain their own licenses. Community version catalogs are not NVIDIA’s official releases, and a valid signature does not guarantee compatibility with every GPU, resolution or HDR combination. An active export prevents switching or removing its runtime.

### Release preview memory

**Release preview memory** pauses playback, waits for the current frame and unloads preview resources. Settings and disk model files stay intact. Editing does not reload them automatically; click **Resume preview** to load them again.

When memory is tight and you switch models, dimensions, sources or chains, the old preview is fully released first. The protection does not reduce precision or resolution, or silently enable tiling. SDK internal peaks cannot be fully predicted, so out-of-memory errors remain possible. This button cannot release export resources while exporting.

### Updates

Open **About Lunarc → Software update** to check and download an update, then exit the application and run the new installer. Components and models are usually stored separately from the application. Check the runtime diagnostics again after updating.

<a id="troubleshooting"></a>
## Common questions and limitations

### Why are image-upscaling cards missing after importing an image?

Check that you loaded an image rather than a video, and check the installed version. The model entry, runtime, component and weights are separate requirements. A model name does not mean its weights are already installed. Newer features may appear in the source build first.

### Why is a VSR export still at the original resolution?

Check whether export resolution is Source resolution. Choose Processing-chain output to keep the chain size. A specified export size adds a final resize before saving.

### Why does TrueHDR preview look washed out or too bright?

Identify whether you are viewing native HDR in the workbench or an SDR-mapped comparison or web image, then check Windows HDR. An ordinary SDR image cannot directly show HDR peak brightness.

### Why does enabling VFG not change preview FPS?

Real-time preview skips VFG. Pre-render a selection or export a short clip to inspect frame generation. The status bar’s render speed is not the final file’s frame rate.

### What if switching models runs out of GPU memory?

Release preview memory first, switch models, then resume. Reduce other GPU workloads and check the actual upscaled dimensions and layer count. Lowering precision or resolution, or enabling tiling, is your choice based on the speed and quality trade-off.

### Can repeated preview clicks recover a driver-reset GPU?

After a device reset, old resources are unreliable. Wait for backend recovery. If it still fails, fully exit both the workbench and backend, then restart. Save the failure report; repeated errors from the same lost device are not necessarily separate faults.

### Which features are experimental?

HDR10, multiscale data-consistency fusion and VFG 5×–16× include experimental implementations. HLG, HDR10+, Dolby Vision and unsupported codec/container combinations are outside this scope. There is no claim of real-time 4K 120 fps processing on arbitrary hardware.

See the [project documentation](https://github.com/Yukikaze1945/Lunarc-DLSS5#05--文档) for build and validation records. Website demos show interface behavior only; they do not run models or modify local application settings.
