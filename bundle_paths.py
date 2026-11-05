"""Where this program's own binaries live, in either layout it runs from.

A PyInstaller ONEDIR bundle has two roots and they are not the same directory. The
launcher sits in ``<dist>/DLSS5Tool/``; everything the spec collects sits in
``<dist>/DLSS5Tool/_internal/``, which is what ``sys._MEIPASS`` names. A file the
user is expected to supply or replace - an external torch model tree - belongs
beside the launcher, where they can see it. A file the program loads itself
belongs in the contents directory, because that is where the bundle put it.

Every loader used to pick its own expression for "the native directory":
``os.path.dirname(__file__)``, ``os.path.dirname(sys.executable)``, or
``sys._MEIPASS``. In the source tree those are all the same path, which is why the
disagreement stayed invisible until a packaged build was actually run: in a frozen
tree they name two different directories, and a GUI could load
``native/dlss_gpu_host.dll`` from a location no other module would ever search.

Resolve through here instead. The manifest below is also what
``DLSS5Tool_standalone.spec`` collects, so the spec, the loaders, and the packaging
tests cannot drift apart from each other.

In the source tree both roots are this directory, so a plain ``python gui.py`` run
resolves exactly what it resolved before.
"""
import glob
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))

# Proprietary DLSSNR is acquired by the user and validated in an independent cache.
CONTENTS_BINARIES = ()

# Loaded out of native/. Listed in dependency order so the reason an entry is here is
# readable: dlss_gpu_host.dll imports nvof_d3d12_interop.dll at load time, and so do
# gpu_lanczos_bridge, gpu_residual_bridge and nvenc_bridge - which is why that one has
# to ship in the same directory as its users rather than at the bundle root, where
# PyInstaller would otherwise flatten a copy of it.
NATIVE_BINARIES = (
    "nvof_d3d12_interop.dll",   # D3D12/CUDA interop for the host and three bridges
    "dlss_gpu_host.dll",        # the project DLSS host: what the GUI opens by default
    "nvof_bridge.dll",          # CPU-path NVOF optical flow
    "nvdec_bridge.dll",         # GPU transfer: decode
    "nv12_rgba_bridge.dll",     # GPU transfer: colour convert
    "gpu_lanczos_bridge.dll",   # GPU transfer: scale
    "gpu_residual_bridge.dll",  # GPU transfer: Magpie residual
    "gpu_low_residual_bridge.dll",  # Multi-scale: signed layer observations
    "gpu_multiscale_fusion_bridge.dll",  # Multi-scale: GPU consistency solve
    "gpu_post_control_bridge.dll",  # Independent post-fusion residual controls
    "gpu_temporal_bridge.dll",  # Scene observation and optional residual memory
    "nvenc_bridge.dll",         # GPU transfer: encode
    "gpu_hdr_scrgb_bridge.dll",  # HDR10: P010 -> scRGB FP16 -> P010 colour bridge
    "gpu_preview_readback.dll", # Explicit display-only readback
    "gpu_preview_display.dll",  # GPU display resampling before viewport readback
    "gpu_native_presenter.dll", # Workbench FP16/scRGB native preview window
    "gpu_export_resize.dll",    # Final export resize; RGBA8 or signed FP16
    "gpu_rgba_exchange.dll",    # GPU resources for native output effects
    "rtx_vsr_bridge.dll",       # Adapter; licensed NVIDIA runtime stays external
    "fidelityfx_flow_bridge.dll", # MIT FidelityFX Optical Flow
    "residual_bridge.dll",      # CPU residual backend
)

# External commands the GUI shells out to. A bundle ships the installation the build
# machine actually ran, which is why these are resolved rather than hardcoded: see
# resolve_sources().
TOOL_BINARIES = ("ffmpeg.exe", "ffprobe.exe")

# NVIDIA ships these with the driver, and they have to stay that way. A copy inside the
# bundle would be found before the driver's own, and a driver that is newer than the
# copy cannot be matched by anything in the bundle: the bridges load them by bare name
# so the loader resolves them against the installed driver.
NEVER_BUNDLE = (
    "nvngx_dlssnr.dll",
    "nvcuda.dll", "nvofapi64.dll", "nvencodeapi64.dll", "nvencodeapi.dll", "nvml.dll",
    "d3d12.dll", "d3dcompiler_47.dll", "dxgi.dll",
)

_dll_directories = {}


def contents_dir():
    """The directory a spec destination of ``'.'`` collects into."""
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(sys.executable)))
    return _ROOT


def app_dir():
    """The directory beside the launcher, where a user may put their own files."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return _ROOT


def native_dir():
    return os.path.join(contents_dir(), "native")


def native_dll(name):
    """Absolute path of a bridge/host DLL, wherever this layout put ``native/``."""
    return os.path.join(native_dir(), name)


def collected(name):
    """Absolute path of a file the spec collected at the contents-directory root."""
    return os.path.join(contents_dir(), name)


def missing_binary(path, purpose):
    """The error for a binary this layout does not have.

    Naming the directory is not enough: the two layouts are fixed by different things,
    so the message has to say which one fell short. A source tree is rebuilt from the
    matching script under native/; a packaged tree needs the entry in the spec, and the
    spec now refuses to build without it, so a bundle missing one was built from an
    older tree.
    """
    name = os.path.basename(path)
    directory = os.path.dirname(os.path.abspath(path))
    if getattr(sys, "frozen", False):
        fix = (f"{name} is not in this bundle. Rebuild it from the current source tree "
               f"and spec; this build cannot {purpose} without it.")
    else:
        fix = (f"{name} has not been built; rebuild it from native/ "
               f"(native/CMakeLists.txt or the matching native/build_*.bat) so this "
               f"source tree can {purpose}.")
    return FileNotFoundError(
        f"{fix} Searched: {directory} "
        f"({'frozen' if getattr(sys, 'frozen', False) else 'source tree'} layout).")


def require_native(name, purpose):
    path = native_dll(name)
    if not os.path.isfile(path):
        raise missing_binary(path, purpose)
    return path


def require_collected(name, purpose):
    path = collected(name)
    if not os.path.isfile(path):
        raise missing_binary(path, purpose)
    return path


def add_dll_directory(directory):
    """Put ``directory`` in the loader's search path and keep it there.

    Only adds a directory the process has not been given before: the same path handed
    to os.add_dll_directory twice gets two handles, and each handle on its own keeps
    the directory registered, so discarding one afterwards changes nothing. The handle
    os.add_dll_directory returns un-registers the directory when it is garbage
    collected, so a caller that discards it has written a search path that works until
    the next collection. This module holds the references for the life of the process.
    """
    if not directory or not os.path.isdir(directory):
        return None
    resolved = os.path.abspath(directory)
    if resolved in _dll_directories:
        return resolved
    try:
        _dll_directories[resolved] = os.add_dll_directory(directory)
    except (AttributeError, OSError):
        pass
    return resolved


def resolve_sources(root=None, ffmpeg_dir=None):
    """The ``(source, destination)`` pairs a PyInstaller spec has to collect.

    Lives here, beside the names, so the spec ships what the loaders look for instead
    of restating the list somewhere a fix cannot reach. Sources are resolved against
    ``root`` (the spec directory) so a build from another working directory collects
    the same files, and an absent one raises here rather than producing a bundle that
    fails at a frame the build machine never ran.

    ``ffmpeg_dir`` overrides where the FFmpeg installation is; otherwise it is the one
    this machine actually runs, found on PATH. Which installation that is cannot be
    recorded in the spec as a literal path from whoever's machine it was written on:
    the bundle has to ship the build machine's own.
    """
    root = os.path.abspath(root or _ROOT)
    pairs = [(os.path.join(root, name), ".") for name in CONTENTS_BINARIES]
    pairs += [(os.path.join(root, "native", name), "native") for name in NATIVE_BINARIES]
    tools = ffmpeg_dir or _find_ffmpeg_dir()
    if not tools:
        raise FileNotFoundError(
            "No ffmpeg.exe/ffprobe.exe found to bundle. The GUI shells out to both for "
            "capture and export. Install FFmpeg and put it on PATH, or point "
            "DLSS5_FFMPEG_DIR at the directory that holds them, then build again.")
    for name in TOOL_BINARIES:
        source = os.path.join(tools, name)
        if not os.path.isfile(source):
            raise FileNotFoundError(
                f"{source} ({name}) does not exist, so it cannot be bundled; "
                f"{tools} is the installation this build resolved. Set "
                f"DLSS5_FFMPEG_DIR to an install that has both tools.")
        pairs.append((source, "."))
    for source, _ in pairs:
        if not os.path.isfile(source):
            raise missing_binary(source, "packaged build")
    return pairs


def _find_ffmpeg_dir():
    """The FFmpeg installation a bundle should ship: the one this machine runs.

    A directory only qualifies if it holds both tools, because the GUI shells out to
    each. Preference goes to one that also carries its own shared libraries: PATH often
    answers with a launcher shim first (scoop keeps one), and a shim copied into a
    bundle resolves nothing on the machine that receives it.
    """
    explicit = os.environ.get("DLSS5_FFMPEG_DIR")
    if explicit:
        return os.path.abspath(explicit)
    candidates = [os.path.abspath(directory) for directory in
                  os.environ.get("PATH", "").split(os.pathsep) if directory
                  and all(os.path.isfile(os.path.join(directory, name))
                          for name in TOOL_BINARIES)]
    if not candidates:
        return None
    with_libraries = [d for d in candidates if glob.glob(os.path.join(d, "avcodec-*.dll"))]
    return (with_libraries or candidates)[0]


# Small installer carries the bridge import closure separately from CLI FFmpeg.
add_dll_directory(os.path.join(os.path.dirname(os.path.abspath(__file__)), "native-deps"))
