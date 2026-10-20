/* Experimental single-layer HDR10 GPU Transfer bridge.
 *
 * This translation unit is deliberately self-contained: it imports its own
 * D3D12 resources into CUDA instead of reusing nvof_d3d12_interop, because that
 * interop describes a row as width*4 bytes and an HDR plane's row is twice
 * that. Keeping the HDR geometry here means the SDR chain's shared code cannot
 * change behaviour while this stage is still experimental.
 *
 * Everything is half precision or better until the last stage, which is the
 * 10-bit code the encoder is required to accept. No sample of main colour is
 * ever 8-bit, and no pixel ever crosses to the host: the planes are shared
 * D3D12 resources whose CUDA addresses are handed to the Feature 18 host, the
 * flow engine and the encoder where they are.
 *
 * What does cross is scalar. Each executed pass brings its own counter run back
 * to host memory so the report getters can answer from it (see run_pass), and
 * that transfer is permitted by the same rule that forbids the picture transfers:
 * the contract is about images, not about diagnostics. It is also the one piece
 * of host traffic no counter in GpuHdrTransferCounters measures, which is why a
 * transfer report must not read that struct as a complete account of the bytes
 * that crossed. */
#define GPU_HDR_SCRGB_EXPORTS
#include "gpu_hdr_scrgb_bridge.h"
#include "include/cuda.h"

#include <windows.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

constexpr uint32_t kCounterSlots = 64u;

thread_local char last_error[512] = "";

void fail(const char* step, long code) {
    std::snprintf(last_error, sizeof(last_error), "%s: 0x%08lx", step,
                  static_cast<unsigned long>(code));
}
void say(const char* text) {
    std::snprintf(last_error, sizeof(last_error), "%s", text);
}
void error_reset() { last_error[0] = 0; }

uint32_t align256(uint32_t bytes) { return (bytes + 255u) & ~255u; }

/* --- The colour contract, derived rather than copied ---------------------
 *
 * Physical units: linear scRGB whose 1.0 is the 80 cd/m^2 diffuse white of
 * IEC 61966-2-1. ST 2084 is an absolute-encoding transfer function whose
 * normalised 1.0 is 10000 cd/m^2, so one relative PQ step is 125 scRGB units
 * and both directions use exactly that number.
 *
 * The primary conversions come out of the published chromaticities, not out of
 * a table someone typed: each gamut's RGB->XYZ is built by scaling its
 * primaries' unit-luminance XYZ until the three sum to the shared D65 white,
 * and the conversion is the destination's inverse times the source. That
 * reproduces BT.709's own published matrix, which is the check that the
 * construction is the one the standard documents, and applying it to Rec.2020
 * yields the Y row that Rec.2020 publishes as Kr, Kg, Kb - so the same
 * chromaticities serve the primary conversion and the non-constant-luminance
 * matrix. */
struct Mat3 { double m[9]; };

Mat3 mat_multiply(const Mat3& a, const Mat3& b) {
    Mat3 out{};
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col) {
            double sum = 0.0;
            for (int k = 0; k < 3; ++k) sum += a.m[row * 3 + k] * b.m[k * 3 + col];
            out.m[row * 3 + col] = sum;
        }
    return out;
}

bool mat_inverse(const Mat3& in, Mat3* out) {
    const double& a = in.m[0]; const double& b = in.m[1]; const double& c = in.m[2];
    const double& d = in.m[3]; const double& e = in.m[4]; const double& f = in.m[5];
    const double& g = in.m[6]; const double& h = in.m[7]; const double& i = in.m[8];
    const double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if (!std::isfinite(det) || std::fabs(det) < 1e-12) return false;
    out->m[0] = (e * i - f * h) / det;
    out->m[1] = (c * h - b * i) / det;
    out->m[2] = (b * f - c * e) / det;
    out->m[3] = (f * g - d * i) / det;
    out->m[4] = (a * i - c * g) / det;
    out->m[5] = (c * d - a * f) / det;
    out->m[6] = (d * h - e * g) / det;
    out->m[7] = (b * g - a * h) / det;
    out->m[8] = (a * e - b * d) / det;
    return true;
}

/* RGB -> XYZ for one set of chromaticities and a shared white point. Each
 * primary has unit luminance, so its XYZ is (x/y, 1, z/y) with y == 1, and the
 * three unknown scales S solve A * S = (xw, 1, zw) where A holds the primaries'
 * x, Y and z. Cramer on that system is the whole derivation; the result is the
 * matrix whose rows are the scaled primary XYZ columns. */
Mat3 rgb_to_xyz(const double primaries[6], const double white[2]) {
    const double xw = white[0] / white[1];
    const double zw = (1.0 - white[0] - white[1]) / white[1];
    double a[9];
    for (int k = 0; k < 3; ++k) {
        a[k] = primaries[k * 2] / primaries[k * 2 + 1];           /* x / y */
        a[3 + k] = 1.0;                                           /* Y      */
        a[6 + k] = (1.0 - primaries[k * 2] - primaries[k * 2 + 1]) /
                   primaries[k * 2 + 1];                          /* z / y */
    }
    const double det = a[0] * (a[4] * a[8] - a[5] * a[7]) -
                       a[1] * (a[3] * a[8] - a[5] * a[6]) +
                       a[2] * (a[3] * a[7] - a[4] * a[6]);
    const double rhs[3] = {xw, 1.0, zw};
    Mat3 out{};
    for (int k = 0; k < 3; ++k) {
        double c[9];
        std::memcpy(c, a, sizeof(c));
        c[k] = rhs[0];
        c[3 + k] = rhs[1];
        c[6 + k] = rhs[2];
        const double det_k = c[0] * (c[4] * c[8] - c[5] * c[7]) -
                             c[1] * (c[3] * c[8] - c[5] * c[6]) +
                             c[2] * (c[3] * c[7] - c[4] * c[6]);
        const double scale = det_k / det;
        for (int row = 0; row < 3; ++row) out.m[row * 3 + k] = a[row * 3 + k] * scale;
    }
    return out;
}

/* The D65 white of ITU-R BT.709 and BT.2020 (both standards share it), and the
 * two sets of published primaries. */
const double kD65[2] = {0.3127, 0.3290};
const double kBt709Primaries[6] = {0.640, 0.330, 0.300, 0.600, 0.150, 0.060};
const double kBt2020Primaries[6] = {0.708, 0.292, 0.170, 0.795, 0.131, 0.046};

/* Filled at process start by derive_contract(); never typed by hand. */
Mat3 g_to_scrgb{{}};
Mat3 g_to_bt2020{{}};
double g_kr = 0.2627, g_kg = 0.6780, g_kb = 0.0593;

/* ST 2084's own constants. */
constexpr double kPqM1 = 2610.0 / 16384.0;
constexpr double kPqM2 = (2523.0 / 4096.0) * 128.0;
constexpr double kPqC1 = 3424.0 / 4096.0;
constexpr double kPqC2 = (2413.0 / 4096.0) * 32.0;
constexpr double kPqC3 = (2392.0 / 4096.0) * 32.0;
constexpr double kReferenceWhiteNits = 80.0;   /* IEC 61966-2-1 diffuse white */
constexpr double kPqPeakNits = 10000.0;        /* ST 2084's normalised 1.0 */
/* One relative PQ step in this bridge's scRGB units: the top of ST 2084 is 125
 * times the diffuse white, so a PQ code of 1.0 is scRGB 125.0. */
constexpr double kScrgbPerRelative = kPqPeakNits / kReferenceWhiteNits;

/* The bounded display proxy of native/hdr_proxy_math.py. Mirrored exactly: the
 * GPU and the CPU reference must not be two opinions of one curve. */
constexpr double kProxyKnee = 0.75;
constexpr double kProxySpan = 0.25;
constexpr double kProxyK = 5.770780;
constexpr double kPaperFloor = 0.05;
constexpr double kMaxStops = 1.0;
constexpr double kHalfUlpAtOne = 1.0 / 2048.0;
constexpr double kMinSlopePerStop = kHalfUlpAtOne;
constexpr double kSlopeStopStep = 1.0 / 64.0;

/* Relative whites at which the proxy pins, and the top of the band in which the
 * quantized code still advances measurably. The first is closed-form; the
 * second is measured on the half grid by the stage-A enumeration and recorded
 * here as the number the job's support domain is derived from. Both are
 * invariants of the proxy, independent of paperWhite, which is why the gate
 * compares the GPU's own counts against them rather than trusting a claim. */
constexpr double kProxySaturationFallback = 1.808731372940256;
constexpr double kProxyActionableFallback = 1.4091796875;

double proxy_saturation_relative() {
    /* The code that destroys magnitude is the half rounding of the *encoded*
     * value: at the RNE midpoint 1 - 2**-12 everything above it becomes the
     * identical code 1.0. Decoding that boundary back through sRGB gives the
     * linear ceiling the shoulder runs out of headroom against. */
    const double boundary = 1.0 - kHalfUlpAtOne / 2.0;
    const double ceiling = 1.0 - std::pow((boundary + 0.055) / 1.055, 2.4);
    if (!(ceiling > 0.0)) return kProxySaturationFallback;
    return kProxyKnee + std::log(kProxySpan / ceiling) / kProxyK;
}

/* --- CUDA driver entry points, loaded rather than linked ----------------- */

struct ExternalMemoryDesc {
    int type;
    union { int fd; struct { void* handle; const void* name; } win32; } handle;
    unsigned long long size;
    unsigned int flags;
    unsigned int reserved[16];
};
struct ExternalBufferDesc {
    unsigned long long offset;
    unsigned long long size;
    unsigned int flags;
    unsigned int reserved[16];
};
using ExternalMemory = void*;

using CtxPush = CUresult(CUDAAPI*)(CUcontext);
using CtxPop = CUresult(CUDAAPI*)(CUcontext*);
using CtxDevice = CUresult(CUDAAPI*)(CUdevice*);
using DeviceLuid = CUresult(CUDAAPI*)(char*, unsigned int*, CUdevice);
using ImportMemory = CUresult(CUDAAPI*)(ExternalMemory*, const ExternalMemoryDesc*);
using MapBuffer = CUresult(CUDAAPI*)(CUdeviceptr*, ExternalMemory,
                                     const ExternalBufferDesc*);
using DestroyMemory = CUresult(CUDAAPI*)(ExternalMemory);
using MemFree = CUresult(CUDAAPI*)(CUdeviceptr);
using Memcpy2D = CUresult(CUDAAPI*)(const CUDA_MEMCPY2D*);
using MemcpyDtoH = CUresult(CUDAAPI*)(void*, CUdeviceptr, size_t);

struct Cuda {
    HMODULE module = nullptr;
    CtxPush push = nullptr;
    CtxPop pop = nullptr;
    CtxDevice ctx_device = nullptr;
    DeviceLuid device_luid = nullptr;
    ImportMemory import_memory = nullptr;
    MapBuffer map_buffer = nullptr;
    DestroyMemory destroy_memory = nullptr;
    MemFree mem_free = nullptr;
    Memcpy2D memcpy_2d = nullptr;
    MemcpyDtoH memcpy_dto_h = nullptr;
};

template<class T> bool symbol(HMODULE module, const char* name, T* out) {
    *out = reinterpret_cast<T>(GetProcAddress(module, name));
    return *out != nullptr;
}

bool load_cuda(Cuda* c) {
    c->module = LoadLibraryW(L"nvcuda.dll");
    if (!c->module) { fail("LoadLibrary(nvcuda)", GetLastError()); return false; }
    return symbol(c->module, "cuCtxPushCurrent_v2", &c->push) &&
        symbol(c->module, "cuCtxPopCurrent_v2", &c->pop) &&
        symbol(c->module, "cuCtxGetDevice", &c->ctx_device) &&
        symbol(c->module, "cuDeviceGetLuid", &c->device_luid) &&
        symbol(c->module, "cuImportExternalMemory", &c->import_memory) &&
        symbol(c->module, "cuExternalMemoryGetMappedBuffer", &c->map_buffer) &&
        symbol(c->module, "cuDestroyExternalMemory", &c->destroy_memory) &&
        symbol(c->module, "cuMemFree_v2", &c->mem_free) &&
        symbol(c->module, "cuMemcpy2D_v2", &c->memcpy_2d) &&
        symbol(c->module, "cuMemcpyDtoH_v2", &c->memcpy_dto_h);
}

/* Pushes the bridge's context for the lifetime of the scope. Every CUDA call in
 * this file goes through one, so a caller thread that had some other context
 * current does not silently make the first call fail. */
struct ContextScope {
    Cuda* cuda;
    CUcontext context;
    bool ok = false;
    CUcontext previous = nullptr;
    ContextScope(Cuda* c, CUcontext ctx) : cuda(c), context(ctx) {
        if (!c || !c->push || !context) return;
        ok = c->push(context) == CUDA_SUCCESS;
        if (!ok) fail("cuCtxPushCurrent(hdr)", -1);
    }
    ~ContextScope() { if (ok) cuda->pop(&previous); }
    ContextScope(const ContextScope&) = delete;
    ContextScope& operator=(const ContextScope&) = delete;
};

/* --- Planes -------------------------------------------------------------- */

enum PlaneId {
    kPlaneSourceY = 0, kPlaneSourceUV, kPlaneMaster, kPlaneWorking,
    kPlaneProxy, kPlaneGuidance, kPlaneNeural, kPlaneResidual,
    kPlaneComposited, kPlaneEncodeY, kPlaneEncodeUV, kPlaneScratch,
    kPlaneTaps, kPlaneCounters, kPlaneInitPattern, kPlaneCounterReadback,
    kPlaneCount
};

struct Plane {
    ComPtr<ID3D12Resource> resource;
    HANDLE shared_handle = nullptr;
    ExternalMemory external_memory = nullptr;
    uint64_t mapped = 0;
    /* The CPU address of a host-visible plane; null for every device plane. */
    uint8_t* host = nullptr;
    uint32_t pitch = 0;
    uint32_t bytes_per_row = 0;
    uint32_t rows = 0;
    /* The state the resource is left in, so barriers are only recorded when the
     * state actually changes rather than on every binding. */
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
    /* Descriptor slots, filled in when the heap is laid out. */
    uint32_t srv_slot = 0, uav_slot = 0;
};

/* One tap row is GPU_HDR_MAX_TAPS weights followed by the same number of
 * source indices, which is exactly the TapRow the shaders declare. */
constexpr uint32_t kTapStride = GPU_HDR_MAX_TAPS * 2u * sizeof(uint32_t);

/* --- Passes -------------------------------------------------------------- */

enum PassId {
    kPassDecode = 0, kPassVertical, kPassHorizontal, kPassProxy,
    kPassResidual, kPassComposite, kPassEncode, kPassCount
};

/* One descriptor slot of one pass: which plane, which view of it, and for the
 * structured views the element window. The order of a pass's SRV run is the
 * order of its t registers, and likewise for its UAV run and u, because a
 * descriptor table binds consecutive slots to consecutive registers. */
enum ViewKind { kRawSrv = 0, kRawUav, kCountersUav, kScratchSrv, kScratchUav,
                kTapsSrv };
struct SlotSpec { uint32_t plane; uint32_t kind; uint32_t first, rows; };
struct PassBinding {
    SlotSpec srv[4];
    uint32_t srv_count;
    SlotSpec uav[3];
    uint32_t uav_count;
    uint32_t const_count;
    const char* entry;
};

/* Counter slots. Absolute, shared by all passes through one typed UAV, because
 * the reports are assembled on the host after the frame and a pass must never
 * have to know another pass's layout. */
enum CounterSlot {
    kSrcNonfinite = 0, kSrcNegative, kSrcAboveCeiling, kSrcMinKey, kSrcMaxKey,
    kSrcPixels,
    kProxyNegative = 10, kProxyPinned, kProxyDead, kProxyAboveActionable,
    kProxyAboveSaturation, kProxyMinKey, kProxyMaxKey, kProxyPixels,
    kRecApplied = 20, kRecCap, kRecRejectedPinned, kRecRejectedDead,
    kRecRejectedNegative, kRecNonfinite, kRecMaxStopsKey, kRecMaxResidualKey,
    kCompOutside = 30, kCompMinKey, kCompMaxKey, kCompNonfinite,
    kEncClampedHigh = 40, kEncClampedLow, kEncOutOfGamut, kEncNonfinite,
    kEncMaxLinearKey, kEncMinCodeY, kEncMaxCodeY, kEncMinCodeUV, kEncMaxCodeUV
};

uint32_t counter_init_value(uint32_t slot) {
    if (slot == kSrcMinKey || slot == kProxyMinKey || slot == kCompMinKey ||
        slot == kEncMinCodeY || slot == kEncMinCodeUV) return 0xFFFFFFFFu;
    return 0u;
}

/* --- HLSL ---------------------------------------------------------------- */

const char* matrix_block(const Mat3& m, const char* name) {
    static thread_local char text[640];
    std::snprintf(text, sizeof(text),
        "static const float3 %s0 = float3(%.9f, %.9f, %.9f);\n"
        "static const float3 %s1 = float3(%.9f, %.9f, %.9f);\n"
        "static const float3 %s2 = float3(%.9f, %.9f, %.9f);\n"
        "float3 %s(float3 v) { return float3(dot(%s0, v), dot(%s1, v), "
        "dot(%s2, v)); }\n",
        name, m.m[0], m.m[1], m.m[2], name, m.m[3], m.m[4], m.m[5], name,
        m.m[6], m.m[7], m.m[8], name, name, name, name);
    return text;
}

const char* numeric_header = R"hlsl(
#define TAPS 24u
struct TapRow { float W[TAPS]; int I[TAPS]; };
)hlsl";

/* float -> binary16, round to nearest even, in integer arithmetic.
 *
 * This replaces the compiler's f32tof16, and it has to: on this driver that
 * intrinsic rounds toward zero, so every plane the bridge writes was biased
 * down by up to one half step. A decoded master of 12.4354887 landed on
 * 12.4296875 where the reference module's binary16 puts it on 12.4375, and a
 * proxy code of 0.99999994 - the value the shoulder reaches at the ceiling -
 * landed on 1 - 2**-11 instead of 1.0, which is why no channel ever registered
 * as pinned. The grid the audited CPU reference works on is round-to-nearest,
 * so the store has to do the same arithmetic to get the same number.
 * scratch/hdr_half_round_probe.py checks this against numpy's own cast at every
 * approach point of the half grid. */
const char* half_round_source = R"hlsl(
uint shr_rne(uint value, uint shift) {
    if (shift == 0u) return value;
    if (shift >= 32u) return 0u;
    const uint kept = value >> shift;
    const uint dropped = value & ((1u << shift) - 1u);
    const uint half = 1u << (shift - 1u);
    const bool up = dropped > half ||
        (dropped == half && (kept & 1u) != 0u);
    return kept + (up ? 1u : 0u);
}

uint f32_to_f16_bits(float value) {
    const uint bits = asuint(value);
    const uint sign = (bits >> 16) & 0x8000u;
    const uint rest = bits & 0x7FFFFFFFu;
    if (rest >= 0x7F800000u)                 /* infinity or NaN */
        return sign | (rest > 0x7F800000u ? 0x7E00u : 0x7C00u);
    const uint f_exp = rest >> 23;            /* biased float32 exponent */
    if (f_exp == 0u) return sign;            /* float32 subnormal: under the grid */
    const int h_exp = int(f_exp) - 112;       /* biased binary16 exponent */
    if (h_exp >= 31) return sign | 0x7C00u;  /* overflow to infinity */
    if (h_exp <= 0) {
        /* Subnormal half: the implicit leading one goes back before the shift,
         * and the rounding may carry the value up to the smallest half normal,
         * which is exponent 1 with a zero mantissa. */
        const uint kept = shr_rne((rest & 0x7FFFFFu) | 0x800000u,
                                  13u + uint(1 - h_exp));
        return (kept >= 0x400u) ? (sign | 0x0400u) : (sign | kept);
    }
    const uint kept = shr_rne(rest & 0x7FFFFFu, 13u);
    if (kept > 0x3FFu) return sign | (uint(h_exp + 1) << 10);   /* carried up */
    return sign | (uint(h_exp) << 10) | kept;
}
)hlsl";

/* log2 and exp2 that do not go through the compiler's transcendentals.
 *
 * HLSL pow() lowers to the hardware lg.approx/ex2.approx pair, and on the driver
 * this bridge runs on that answer is off by up to 4e-4 relative. That is the
 * same size as the binary16 step the proxy code is judged on, so whole channels
 * land one code away from the CPU reference, and at the top of the curve it is
 * larger than the last sliver of headroom between the shoulder and the plateau:
 * pow(1, y) comes back below 1 and the proxy never reports a pinned channel at
 * all. Both series here are the textbook ones with the range reduction done in
 * exact bit arithmetic, and each truncates three orders of magnitude below the
 * float32 step, which is what makes the curve reproducible against the host
 * mirror rather than merely close to it. Positive normal inputs only; the call
 * sites below screen their own zeros. */
const char* transcendental_source = R"hlsl(
static const float HDR_LOG2_E = 1.4426950408889634f;
static const float HDR_LN2 = 0.6931471805599453f;

float hdr_exp2(float x) {
    if (x != x) return x;
    /* Below 2**-127 the answer is subnormal, and every consumer here works on a
     * quantity that float32 has already rounded away: the shoulder's 1 - e is
     * 1.0f for any e at or under 2**-25. Above the top the curve saturates
     * rather than inflicting an infinity on the plane. */
    if (x < -127.0f) return 0.0f;
    if (x > 127.0f) return 3.4e+38f;
    const float whole = round(x);
    const int biased = int(whole) + 127;
    const float t = (x - whole) * HDR_LN2;           /* |t| <= ln2/2 */
    /* e^t through the t**8 term; the first dropped term is 2e-10 relative. */
    float p = 2.4801587e-5f;
    p = p * t + 1.9841270e-4f;
    p = p * t + 1.3888889e-3f;
    p = p * t + 8.3333333e-3f;
    p = p * t + 4.1666667e-2f;
    p = p * t + 1.6666667e-1f;
    p = p * t + 5.0e-1f;
    p = p * t + 1.0f;
    p = p * t + 1.0f;
    return asfloat(uint(biased) << 23) * p;
}

float hdr_log2(float x) {
    const uint bits = asuint(x);
    int exponent = int(bits >> 23) - 127;
    float m = asfloat((bits & 0x007FFFFFu) | 0x3F800000u);   /* [1, 2) */
    if (m > 1.41421354f) { m *= 0.5f; ++exponent; }
    /* m is now in [2^-0.5, 2^0.5], so log(m) = 2 * atanh(z) with
     * z = (m - 1) / (m + 1) bounded by 0.1716, where the odd series in z
     * reaches 1e-16 by its tenth term. */
    const float z = (m - 1.0f) / (m + 1.0f);
    const float z2 = z * z;
    float p = 1.0f / 19.0f;
    p = p * z2 + 1.0f / 17.0f;
    p = p * z2 + 1.0f / 15.0f;
    p = p * z2 + 1.0f / 13.0f;
    p = p * z2 + 1.0f / 11.0f;
    p = p * z2 + 1.0f / 9.0f;
    p = p * z2 + 1.0f / 7.0f;
    p = p * z2 + 1.0f / 5.0f;
    p = p * z2 + 1.0f / 3.0f;
    p = p * z2 + 1.0f;
    return float(exponent) + 2.0f * z * p * HDR_LOG2_E;
}

float hdr_pow(float x, float y) {
    if (x <= 0.0f) return 0.0f;   /* y is positive at every call site */
    return hdr_exp2(y * hdr_log2(x));
}
)hlsl";

/* Every number the shaders use for the colour contract is printed here out of
 * the derived doubles, so there is no second copy of a standard to drift from
 * the first. pq_eotf returns the value normalised to 10000 cd/m^2; the scRGB
 * plane multiplies by RELATIVE_TO_SCRGB to get 1.0 == 80 cd/m^2, and the
 * encoder divides by the same number. */
std::string colour_header() {
    const double cr_gain = 2.0 * (1.0 - g_kr);
    const double cb_gain = 2.0 * (1.0 - g_kb);
    const double cg_cr = 2.0 * g_kr * (1.0 - g_kr) / g_kg;
    const double cg_cb = 2.0 * g_kb * (1.0 - g_kb) / g_kg;
    char line[768];
    std::snprintf(line, sizeof(line),
        "static const float PQ_M1 = %.9ff;\n"
        "static const float PQ_M2 = %.9ff;\n"
        "static const float PQ_C1 = %.9ff;\n"
        "static const float PQ_C2 = %.9ff;\n"
        "static const float PQ_C3 = %.9ff;\n"
        "static const float RELATIVE_TO_SCRGB = %.9ff;\n"
        "/* ITU-R BT.2020 non-constant-luminance. Kr, Kg and Kb are the Y row of\n"
        " * the RGB->XYZ matrix the published chromaticities produce; the\n"
        " * inverse below is derived from them rather than typed. */\n"
        "static const float KR = %.9ff;\n"
        "static const float KG = %.9ff;\n"
        "static const float KB = %.9ff;\n"
        "static const float CR_GAIN = %.9ff;\n"
        "static const float CB_GAIN = %.9ff;\n"
        "static const float CG_CR = %.9ff;\n"
        "static const float CG_CB = %.9ff;\n",
        kPqM1, kPqM2, kPqC1, kPqC2, kPqC3, kScrgbPerRelative,
        g_kr, g_kg, g_kb, cr_gain, cb_gain, cg_cr, cg_cb);
    std::string text = line;
    text += half_round_source;
    text += transcendental_source;
    text += "float pq_eotf(float p) {\n"
            "    float np = hdr_pow(max(p, 0.0f), 1.0f / PQ_M2);\n"
            "    return hdr_pow(max(np - PQ_C1, 0.0f) / (PQ_C2 - PQ_C3 * np),\n"
            "               1.0f / PQ_M1);\n"
            "}\n"
            "float pq_oetf(float e) {\n"
            "    float em = hdr_pow(max(e, 0.0f), PQ_M1);\n"
            "    return hdr_pow((PQ_C1 + PQ_C2 * em) / (1.0f + PQ_C3 * em),"
            " PQ_M2);\n"
            "}\n";
    text += matrix_block(g_to_scrgb, "convert2020To709");
    text += matrix_block(g_to_bt2020, "convert709To2020");
    return text;
}

const char* sample_header = R"hlsl(
/* AV_PIX_FMT_P010LE stores a 10-bit code left-aligned in a 16-bit
 * little-endian sample: two codes per 32-bit word, the low half first. The
 * shift that was a float shift in an earlier draft is now an explicit divide,
 * which is also what makes the code's low six bits irrelevant on input. */
void unpack_p010(uint word, out float a, out float b) {
    a = float(word & 0xffffu) / 64.0f;
    b = float(word >> 16) / 64.0f;
}
uint pack_p010(float a, float b) {
    uint ia = uint(clamp(round(a), 0.0, 1023.0));
    uint ib = uint(clamp(round(b), 0.0, 1023.0));
    return (ia << 6) | (ib << 22);
}
/* Limited range: luma 64..940, colour difference 512 +- 448. */
float y_from_code(float code) { return (code - 64.0f) / 876.0f; }
float c_from_code(float code) { return (code - 512.0f) / 896.0f; }
float code_from_y(float y) { return round(y * 876.0f) + 64.0f; }
float code_from_c(float c) { return round(c * 896.0f) + 512.0f; }
/* binary16 in and out of a 32-bit word. Unpacking is exact and free, so it uses
 * the compiler's f16tof32; packing goes through f32_to_f16_bits above because
 * the compiler's f32tof16 rounds toward zero on this driver and would bias
 * every plane the bridge writes down by up to one half step. */
float2 unpack_half2(uint word) {
    return float2(f16tof32(word & 0xFFFFu), f16tof32(word >> 16));
}
uint pack_half2(float2 value) {
    return f32_to_f16_bits(value.x) | (f32_to_f16_bits(value.y) << 16);
}
float4 load16(ByteAddressBuffer plane, uint row, uint column, uint pitch) {
    uint2 word = plane.Load2(row * pitch + column * 8u);
    float2 low = unpack_half2(word.x);
    float2 high = unpack_half2(word.y);
    return float4(low.x, low.y, high.x, high.y);
}
void store16(RWByteAddressBuffer plane, uint row, uint column, uint pitch,
             float4 value) {
    plane.Store2(row * pitch + column * 8u,
                 uint2(pack_half2(value.xy), pack_half2(value.zw)));
}
/* Round-trips a float through the half grid with the same RNE the store does,
 * so a value the plane holds and a value the arithmetic compares are one
 * number rather than two that nearly agree. */
float half_round(float value) {
    return unpack_half2(pack_half2(float2(value, 0.0f))).x;
}
/* A component the half grid cannot hold, or a NaN, fails both comparisons;
 * finite values inside +-65504 pass them. */
bool unrepresentable(float value) {
    return !(value <= 65504.0f && value >= -65504.0f);
}
/* An order-preserving float to uint map, so the extremes of a plane can be
 * found with the only atomics a typed buffer offers. The host decodes these
 * keys with unsortable_to_bits(), so this is exactly its inverse and has to
 * stay the inverse: a non-negative float flips only the sign bit (the key
 * lands at or above 0x80000000), a negative float complements every bit (the
 * key lands below it). Over the finite floats the map is strictly increasing,
 * both signed zeros sit at the join (-0 -> 0x7FFFFFFF, +0 -> 0x80000000), and
 * the same bit rule extends the order to -inf lowest and +inf highest, with a
 * NaN of matching sign just beyond each. A non-finite sample never reaches
 * these counters on a frame that passes: decode counts non-finite samples and
 * the caller refuses the job, so the total order above is what the atomics see
 * for real data. The previous form, `bits ^ ((bits >> 31) | 0x80000000u)`,
 * complemented only the sign and low bits of a negative, not every bit, so it
 * reversed the order among negatives and did not roundtrip through the host
 * inverse; that mis-ordered minimum is what reported a phantom source_min. */
uint sortable(float value) {
    uint bits = asuint(value);
    return bits ^ ((bits >> 31) ? 0xFFFFFFFFu : 0x80000000u);
}
)hlsl";

/* The bounded proxy, in one place, so the code Feature 18 receives and the
 * slope the reconstruction divides by are the same curve. */
const char* proxy_functions = R"hlsl(
#ifdef SDR10_NR
// SDR NR receives the original BT.709 RGB code values, stored in FP16.
// This is not the bounded HDR display proxy and has no shoulder/dead zone.
float sdr_nr_encode(float x) {
    return x < 0.018f ? 4.5f*x : 1.099f*pow(x,0.45f)-0.099f;
}
float sdr_nr_decode(float x) {
    return x < 0.081f ? x/4.5f : pow((x+0.099f)/1.099f,1.0f/0.45f);
}
#endif
/* finite_nonnegative of the reference module: a negative or non-finite scene
 * value codes as zero. The master is never touched by this; only its proxy. */
float proxy_relative_input(float value, float paperWhite) {
    float magnitude = (value >= 0.0f && value < 1e30f) ? value : 0.0f;
    return magnitude / paperWhite;
}
float proxy_encode(float x) {
    /* exp(t) as 2^(t * log2 e): the same series the host mirror's std::exp
     * answers to within a float32 step, and under 2**-127 it is a zero that
     * float32 would have rounded away in `1.0f - e` anyway, which is exactly
     * where the shoulder's plateau has to become one. */
    const float t = -PROXY_K * max(x - PROXY_KNEE, 0.0f);
    float soft = PROXY_KNEE + PROXY_SPAN * (1.0f - hdr_exp2(t * HDR_LOG2_E));
    float bounded = clamp(x > PROXY_KNEE ? soft : x, 0.0f, 1.0f);
    float coded = bounded <= 0.0031308f ? 12.92f * bounded
        : 1.055f * hdr_pow(bounded, 1.0f / 2.4f) - 0.055f;
    return half_round(coded);
}
float proxy_code(float value, float paperWhite) {
#ifdef SDR10_NR
    return half_round(sdr_nr_encode(value));
#else
    return proxy_encode(proxy_relative_input(value, paperWhite));
#endif
}
/* The measured quantized advance per stop, exactly as the reference module
 * measures it: the perturbation is applied to the scene value and both arms go
 * through the whole proxy, which is the bug the corrected reference fixed.
 * A channel already at the ceiling reports zero, because there the difference
 * is the cliff face and not a signal. */
float proxy_slope(float value, float paperWhite) {
    float base = proxy_code(value, paperWhite);
    if (base == 1.0f) return 0.0f;
    float high = proxy_code(value * SLOPE_STOP_FACTOR, paperWhite);
    float low = proxy_code(value / SLOPE_STOP_FACTOR, paperWhite);
    return (high - low) / (2.0f * SLOPE_STEP);
}
)hlsl";

/* The proxy's own constants, printed from the same doubles the CPU reference
 * module carries, plus the slope floor that decides a dead zone. */
std::string proxy_header() {
    char line[320];
    std::snprintf(line, sizeof(line),
        "static const float PROXY_KNEE = %.9ff;\n"
        "static const float PROXY_SPAN = %.9ff;\n"
        "static const float PROXY_K = %.9ff;\n"
        "static const float SLOPE_STEP = %.9ff;\n"
        "static const float SLOPE_STOP_FACTOR = %.9ff;\n"
        "static const float MIN_SLOPE_PER_STOP = %.9ff;\n",
        kProxyKnee, kProxySpan, kProxyK, kSlopeStopStep,
        std::exp2(kSlopeStopStep), kMinSlopePerStop);
    return std::string(line) + proxy_functions;
}

/* P010 -> linear scRGB FP16. PQ is decoded here, before any stage downstream
 * sees a sample. The chroma of the 4:2:0 source is replicated to the four
 * pixels of its box, which is the exact inverse of the box mean the encoder
 * stage applies. */
const char* decode_body = R"hlsl(
ByteAddressBuffer SourceY : register(t0);
ByteAddressBuffer SourceUV : register(t1);
RWByteAddressBuffer Destination : register(u0);
RWBuffer<uint> Counters : register(u1);
cbuffer Frame : register(b0) {
    uint Width, Height, SourceYPitch, SourceUVPitch;
    uint DestinationPitch, InputFullRange, Unused1, Unused2;
    float Ceiling;
};
[numthreads(8, 8, 1)]
void main_decode(uint3 id : SV_DispatchThreadID) {
    if (id.x >= Width || id.y >= Height) return;
    uint yWord = SourceY.Load(id.y * SourceYPitch + (id.x >> 1) * 4u);
    float ys0, ys1;
    unpack_p010(yWord, ys0, ys1);
    float ySample = (id.x & 1u) ? ys1 : ys0;
    #ifdef P210_INPUT
    uint cWord = SourceUV.Load(id.y * SourceUVPitch + (id.x >> 1) * 4u);
#else
    uint cWord = SourceUV.Load((id.y >> 1) * SourceUVPitch + (id.x >> 1) * 4u);
#endif
    float cbSample, crSample;
    unpack_p010(cWord, cbSample, crSample);
    // AVColorRange full-range uses 1023 for both scales and chroma centre 512.
    // The limited-range branch keeps its original arithmetic exactly.
    float Y = InputFullRange ? ySample / 1023.0f : y_from_code(ySample);
    float Cb = InputFullRange ? (cbSample - 512.0f) / 1023.0f : c_from_code(cbSample);
    float Cr = InputFullRange ? (crSample - 512.0f) / 1023.0f : c_from_code(crSample);
    precise float3 encoded = float3(Y + CR_GAIN * Cr,
        Y - CG_CB * Cb - CG_CR * Cr, Y + CB_GAIN * Cb);
    precise float3 linear2020 = float3(pq_eotf(encoded.r), pq_eotf(encoded.g),
        pq_eotf(encoded.b));
    precise float3 scrgb = convert2020To709(linear2020) * RELATIVE_TO_SCRGB;
    store16(Destination, id.y, id.x, DestinationPitch, float4(scrgb, 1.0));
    InterlockedAdd(Counters[5], 1u);
    uint bad = uint(unrepresentable(scrgb.r)) + uint(unrepresentable(scrgb.g)) +
        uint(unrepresentable(scrgb.b));
    uint negatives = uint(scrgb.r < 0.0f) + uint(scrgb.g < 0.0f) +
        uint(scrgb.b < 0.0f);
    uint above = uint(scrgb.r > Ceiling) + uint(scrgb.g > Ceiling) +
        uint(scrgb.b > Ceiling);
    InterlockedAdd(Counters[0], bad);
    InterlockedAdd(Counters[1], negatives);
    InterlockedAdd(Counters[2], above);
    float lowest = min(min(scrgb.r, scrgb.g), scrgb.b);
    float highest = max(max(scrgb.r, scrgb.g), scrgb.b);
    InterlockedMin(Counters[3], sortable(lowest));
    InterlockedMax(Counters[4], sortable(highest));
}
)hlsl";

/* Limited-range BT.709 SDR10 P010 -> relative linear Rec.709. The negative
 * branch of the OETF inverse is the linear extension, preserving signed
 * out-of-gamut values in the FP16 master for later stages. */
const char* sdr10_decode_body = R"hlsl(
ByteAddressBuffer SourceY : register(t0);
ByteAddressBuffer SourceUV : register(t1);
RWByteAddressBuffer Destination : register(u0);
RWBuffer<uint> Counters : register(u1);
cbuffer Frame : register(b0) {
    uint Width, Height, SourceYPitch, SourceUVPitch;
    uint DestinationPitch, Unused0, Unused1, Unused2;
    float Ceiling;
};
float bt709_eotf(float e) {
    return e < 0.081f ? e / 4.5f : pow((e + 0.099f) / 1.099f, 1.0f / 0.45f);
}
[numthreads(8, 8, 1)]
void main_decode(uint3 id : SV_DispatchThreadID) {
    if (id.x >= Width || id.y >= Height) return;
    uint yWord = SourceY.Load(id.y * SourceYPitch + (id.x >> 1) * 4u);
    float ys0, ys1;
    unpack_p010(yWord, ys0, ys1);
    float yCode = (id.x & 1u) ? ys1 : ys0;
    #ifdef P210_INPUT
    uint cWord = SourceUV.Load(id.y * SourceUVPitch + (id.x >> 1) * 4u);
#else
    uint cWord = SourceUV.Load((id.y >> 1) * SourceUVPitch + (id.x >> 1) * 4u);
#endif
    float cbCode, crCode;
    unpack_p010(cWord, cbCode, crCode);
    float Y = y_from_code(yCode);
    float Cb = c_from_code(cbCode), Cr = c_from_code(crCode);
    const float kr = 0.2126f, kb = 0.0722f, kg = 0.7152f;
    precise float3 encoded = float3(Y + (2.0f * (1.0f - kr)) * Cr,
        (Y - kr * (Y + 2.0f * (1.0f - kr) * Cr) -
         kb * (Y + 2.0f * (1.0f - kb) * Cb)) / kg,
        Y + (2.0f * (1.0f - kb)) * Cb);
    precise float3 linear709 = float3(bt709_eotf(encoded.r),
        bt709_eotf(encoded.g), bt709_eotf(encoded.b));
    store16(Destination, id.y, id.x, DestinationPitch, float4(linear709, 1.0f));
    InterlockedAdd(Counters[5], 1u);
    uint bad = uint(unrepresentable(linear709.r)) + uint(unrepresentable(linear709.g)) +
        uint(unrepresentable(linear709.b));
    InterlockedAdd(Counters[0], bad);
    InterlockedAdd(Counters[1], uint(linear709.r < 0.0f) + uint(linear709.g < 0.0f) +
        uint(linear709.b < 0.0f));
    InterlockedAdd(Counters[2], uint(linear709.r > Ceiling) + uint(linear709.g > Ceiling) +
        uint(linear709.b > Ceiling));
    InterlockedMin(Counters[3], sortable(min(min(linear709.r, linear709.g), linear709.b)));
    InterlockedMax(Counters[4], sortable(max(max(linear709.r, linear709.g), linear709.b)));
}
)hlsl";

/* The two separable passes of the plan-driven Lanczos-2 downsample, in the same
 * order and accumulation as the SDR bridge: vertical into a full-precision
 * float32 scratch, horizontal into the working plane. The scratch is float32
 * and not half so the second axis does not re-round what the first already
 * rounded; the reference sums in float32 as well. */
const char* vertical_body = R"hlsl(
ByteAddressBuffer Source : register(t0);
StructuredBuffer<TapRow> Plan : register(t1);
RWStructuredBuffer<float4> Scratch : register(u0);
cbuffer Frame : register(b0) {
    uint FullWidth, FullHeight, LowHeight, SourcePitch;
    uint Taps, Unused0, Unused1, Unused2;
};
[numthreads(16, 8, 1)]
void main_vertical(uint3 id : SV_DispatchThreadID) {
    if (id.x >= FullWidth || id.y >= LowHeight) return;
    TapRow row = Plan[id.y];
    precise float4 acc = float4(0.0, 0.0, 0.0, 0.0);
    for (uint t = 0; t < Taps; ++t) {
        int sy = clamp(row.I[t], 0, int(FullHeight) - 1);
        precise float4 term = row.W[t] *
            load16(Source, uint(sy), id.x, SourcePitch);
        acc = acc + term;
    }
    Scratch[id.y * FullWidth + id.x] = acc;
}
)hlsl";

const char* horizontal_body = R"hlsl(
StructuredBuffer<float4> Scratch : register(t0);
StructuredBuffer<TapRow> Plan : register(t1);
RWByteAddressBuffer Working : register(u0);
RWBuffer<uint> Counters : register(u1);
cbuffer Frame : register(b0) {
    uint LowWidth, LowHeight, FullWidth, WorkingPitch;
    uint Taps, Unused0, Unused1, Unused2;
};
[numthreads(16, 8, 1)]
void main_horizontal(uint3 id : SV_DispatchThreadID) {
    if (id.x >= LowWidth || id.y >= LowHeight) return;
    TapRow row = Plan[id.x];
    precise float4 acc = float4(0.0, 0.0, 0.0, 0.0);
    for (uint t = 0; t < Taps; ++t) {
        int sx = clamp(row.I[t], 0, int(FullWidth) - 1);
        precise float4 term = row.W[t] * Scratch[id.y * FullWidth + uint(sx)];
        acc = acc + term;
    }
    /* Signed, unclamped: this is the pristine working master the residual is
     * measured against, and a resampler with negative lobes is expected to
     * leave the interval its input was in. */
    store16(Working, id.y, id.x, WorkingPitch, acc);
}
)hlsl";

/* The bounded proxy plane Feature 18 receives, and the motion-only guidance
 * surface, from the same working master in one pass. */
const char* proxy_body = R"hlsl(
ByteAddressBuffer Working : register(t0);
RWByteAddressBuffer Proxy : register(u0);
RWByteAddressBuffer Guidance : register(u1);
RWBuffer<uint> Counters : register(u2);
cbuffer Frame : register(b0) {
    uint LowWidth, LowHeight, WorkingPitch, ProxyPitch;
    uint GuidancePitch, Unused0, Unused1, Unused2;
    float PaperWhite, ActionableRelative, SaturationRelative;
};
[numthreads(8, 8, 1)]
void main_proxy(uint3 id : SV_DispatchThreadID) {
    if (id.x >= LowWidth || id.y >= LowHeight) return;
    float4 master = load16(Working, id.y, id.x, WorkingPitch);
    float4 codes = float4(proxy_code(master.r, PaperWhite),
                          proxy_code(master.g, PaperWhite),
                          proxy_code(master.b, PaperWhite), 1.0);
    store16(Proxy, id.y, id.x, ProxyPitch, codes);
    /* Guidance is a separate motion-only encoding; it never replaces the FP16
     * proxy or master. */
    float luma = 0.2126f * master.r + 0.7152f * master.g + 0.0722f * master.b;
#ifdef SDR10_GUIDANCE
    float code = luma < 0.018f ? 4.5f * max(luma, 0.0f)
        : 1.099f * pow(max(luma, 0.0f), 0.45f) - 0.099f;
    float normalised = saturate(code);
#else
    float code = pq_oetf(max(luma, 0.0f) / RELATIVE_TO_SCRGB);
    float ceiling_code = pq_oetf(ActionableRelative * PaperWhite /
        RELATIVE_TO_SCRGB);
    float normalised = saturate(code / max(ceiling_code, 1e-7f));
#endif
    uint value = uint(round(normalised * 255.0f));
    uint packed_guidance = value | (value << 8u) | (value << 16u) | 0xff000000u;
    Guidance.Store(id.y * GuidancePitch + id.x * 4u, packed_guidance);
    InterlockedAdd(Counters[17], 1u);
    float negatives = uint(master.r < 0.0f) + uint(master.g < 0.0f) +
        uint(master.b < 0.0f);
    float pinned = uint(codes.r == 1.0f) + uint(codes.g == 1.0f) +
        uint(codes.b == 1.0f);
    float relative_r = max(master.r, 0.0f) / PaperWhite;
    float relative_g = max(master.g, 0.0f) / PaperWhite;
    float relative_b = max(master.b, 0.0f) / PaperWhite;
    float above_actionable = uint(relative_r > ActionableRelative) +
        uint(relative_g > ActionableRelative) +
        uint(relative_b > ActionableRelative);
    float above_saturation = uint(relative_r > SaturationRelative) +
        uint(relative_g > SaturationRelative) +
        uint(relative_b > SaturationRelative);
    float dead = uint(master.r >= 0.0f && codes.r < 1.0f &&
                      abs(proxy_slope(master.r, PaperWhite)) <
                      MIN_SLOPE_PER_STOP) +
        uint(master.g >= 0.0f && codes.g < 1.0f &&
             abs(proxy_slope(master.g, PaperWhite)) <
             MIN_SLOPE_PER_STOP) +
        uint(master.b >= 0.0f && codes.b < 1.0f &&
             abs(proxy_slope(master.b, PaperWhite)) <
             MIN_SLOPE_PER_STOP);
#ifdef SDR10_NR
    pinned = 0.0f;
    dead = 0.0f;
#endif
    InterlockedAdd(Counters[10], uint(negatives));
    InterlockedAdd(Counters[11], uint(pinned));
    InterlockedAdd(Counters[12], uint(dead));
    InterlockedAdd(Counters[13], uint(above_actionable));
    InterlockedAdd(Counters[14], uint(above_saturation));
    float lowest = min(min(codes.r, codes.g), codes.b);
    float highest = max(max(codes.r, codes.g), codes.b);
    InterlockedMin(Counters[15], sortable(lowest));
    InterlockedMax(Counters[16], sortable(highest));
}
)hlsl";

/* The residual, in the proxy domain against the pristine working master. This
 * is audited local_linear_stops ported to the device: the reconstruction never
 * inverts the proxy, the master supplies the magnitude and the network supplies
 * only a signed change measured in the proxy domain. */
const char* residual_body = R"hlsl(
ByteAddressBuffer Neural : register(t0);
ByteAddressBuffer ProxyIn : register(t1);
ByteAddressBuffer Working : register(t2);
RWByteAddressBuffer Residual : register(u0);
RWBuffer<uint> Counters : register(u1);
cbuffer Frame : register(b0) {
    uint LowWidth, LowHeight, NeuralPitch, ProxyPitch;
    uint WorkingPitch, ResidualPitch, PreserveDetail, Unused1;
    float PaperWhite, MaxStops, MinSlope;
};
/* One channel of the audited reconstruction: the proxy-domain change divided
 * by the measured local slope, in stops, clipped to the stated budget, with the
 * reason a channel gets nothing written to `kind`. */
float reconstruct_channel(float value, float delta, float slope,
                          float paper_white, out uint kind) {
    kind = 0u;                                       /* applied */
    if (delta == 0.0f) return 0.0f;                  /* identity: no move */
    if (value < 0.0f) { kind = 4u; return 0.0f; }    /* negative master */
    if (proxy_code(value, paper_white) == 1.0f) { kind = 2u; return 0.0f; }
    if (abs(slope) < MinSlope) { kind = 3u; return 0.0f; }
    float raw = delta / slope;
    if (unrepresentable(raw)) { kind = 5u; return 0.0f; }
    if (abs(raw) > MaxStops) {
        kind = 1u;
        return raw > 0.0f ? MaxStops : -MaxStops;
    }
    return raw;
}
float3 reconstruct3(float3 master, float3 input_code, float3 neural_code,
                    out uint3 disposition, out float max_stops,
                    out float max_residual) {
#ifdef SDR10_NR
    // Convert the output back through the source transfer function exactly.
    // In particular, never divide by a quantized per-channel slope in shadows.
    float3 decoded = float3(sdr_nr_decode(neural_code.r),sdr_nr_decode(neural_code.g),sdr_nr_decode(neural_code.b));
    float3 input_linear = float3(sdr_nr_decode(input_code.r),sdr_nr_decode(input_code.g),sdr_nr_decode(input_code.b));
    float3 delta = decoded-input_linear;
    disposition = uint3(0,0,0);
    max_stops = 0.0f;
    max_residual = max(abs(delta.r),max(abs(delta.g),abs(delta.b)));
    return PreserveDetail != 0 ? delta : decoded;
#else
    /* 0 applied, 1 gain cap, 2 pinned, 3 dead zone, 4 negative, 5 nonfinite */
    precise float3 delta = clamp(neural_code, 0.0f, 1.0f) -
        clamp(input_code, 0.0f, 1.0f);
    precise float3 slope = float3(proxy_slope(master.r, PaperWhite),
        proxy_slope(master.g, PaperWhite), proxy_slope(master.b, PaperWhite));
    uint kind_r, kind_g, kind_b;
    precise float3 stops = float3(
        reconstruct_channel(master.r, delta.r, slope.r, PaperWhite, kind_r),
        reconstruct_channel(master.g, delta.g, slope.g, PaperWhite, kind_g),
        reconstruct_channel(master.b, delta.b, slope.b, PaperWhite, kind_b));
    disposition = uint3(kind_r, kind_g, kind_b);
    max_stops = max(max(abs(stops.r), abs(stops.g)), abs(stops.b));
    precise float3 reconstructed = half3_stops(master, stops);
    max_residual = max(max(abs(reconstructed.r - master.r),
            abs(reconstructed.g - master.g)),
        abs(reconstructed.b - master.b));
    return reconstructed - master;
#endif
}
)hlsl";

const char* residual_entry = R"hlsl(
[numthreads(8, 8, 1)]
void main_residual(uint3 id : SV_DispatchThreadID) {
    if (id.x >= LowWidth || id.y >= LowHeight) return;
    float4 neural = load16(Neural, id.y, id.x, NeuralPitch);
    float4 input_code = load16(ProxyIn, id.y, id.x, ProxyPitch);
    float4 master = load16(Working, id.y, id.x, WorkingPitch);
    uint3 disposition;
    float stops = 0.0f;
    float residual = 0.0f;
    float3 delta = reconstruct3(master.rgb, input_code.rgb, neural.rgb,
        disposition, stops, residual);
    if (unrepresentable(delta.r) || unrepresentable(delta.g) ||
        unrepresentable(delta.b)) {
        delta = float3(0.0, 0.0, 0.0);
        InterlockedAdd(Counters[25], 3u);
    }
    store16(Residual, id.y, id.x, ResidualPitch, float4(delta, 0.0));
    InterlockedAdd(Counters[20],
        uint(disposition[0] == 0u) + uint(disposition[1] == 0u) +
        uint(disposition[2] == 0u));
    InterlockedAdd(Counters[21],
        uint(disposition[0] == 1u) + uint(disposition[1] == 1u) +
        uint(disposition[2] == 1u));
    InterlockedAdd(Counters[22],
        uint(disposition[0] == 2u) + uint(disposition[1] == 2u) +
        uint(disposition[2] == 2u));
    InterlockedAdd(Counters[23],
        uint(disposition[0] == 3u) + uint(disposition[1] == 3u) +
        uint(disposition[2] == 3u));
    InterlockedAdd(Counters[24],
        uint(disposition[0] == 4u) + uint(disposition[1] == 4u) +
        uint(disposition[2] == 4u));
    InterlockedMax(Counters[26], sortable(stops));
    InterlockedMax(Counters[27], sortable(residual));
}
)hlsl";

/* Catmull-Rom upscale of the signed residual, scaled by strength, composited on
 * the pristine full-resolution master. strength == 0 returns the master
 * exactly, and a zero residual does too, which is the identity gate. */
const char* composite_body = R"hlsl(
ByteAddressBuffer Residual : register(t0);
ByteAddressBuffer Master : register(t1);
StructuredBuffer<TapRow> PlanV : register(t2);
StructuredBuffer<TapRow> PlanH : register(t3);
RWByteAddressBuffer Destination : register(u0);
RWBuffer<uint> Counters : register(u1);
cbuffer Frame : register(b0) {
    uint FullWidth, FullHeight, LowWidth, LowHeight;
    uint ResidualPitch, MasterPitch, DestinationPitch, Taps;
    float Strength, Ceiling;
    uint PreserveDetail;
};
[numthreads(8, 8, 1)]
void main_composite(uint3 id : SV_DispatchThreadID) {
    if (id.x >= FullWidth || id.y >= FullHeight) return;
    TapRow vrow = PlanV[id.y];
    precise float3 up = float3(0.0, 0.0, 0.0);
    for (uint j = 0; j < Taps; ++j) {
        int sy = clamp(vrow.I[j], 0, int(LowHeight) - 1);
        precise float3 horizontal = float3(0.0, 0.0, 0.0);
        TapRow hrow;
        for (uint i = 0; i < Taps; ++i) {
            hrow = PlanH[id.x];
            int sx = clamp(hrow.I[i], 0, int(LowWidth) - 1);
            precise float3 sample = load16(Residual, uint(sy), uint(sx),
                ResidualPitch).rgb;
            precise float3 term = hrow.W[i] * sample;
            horizontal = horizontal + term;
        }
        precise float3 rowTerm = vrow.W[j] * horizontal;
        up = up + rowTerm;
    }
    precise float3 residual = up * Strength;
    precise float3 master = load16(Master, id.y, id.x, MasterPitch).rgb;
    precise float3 composite = PreserveDetail != 0 ? master + residual : lerp(master,up,Strength);
    /* The composite is not clamped. The master is signed, and the negatives a
     * gamut conversion produces are part of the picture: flattening them here
     * would silently alter the source the job was asked to correct, and the
     * encoder is the stage that has to clamp, because that is where the code
     * grid runs out of interval - and it counts what it did. A value the half
     * grid cannot hold falls back to the pristine master rather than to black. */
    if (unrepresentable(composite.r) || unrepresentable(composite.g) ||
        unrepresentable(composite.b)) {
        composite = master;
        InterlockedAdd(Counters[33], 3u);
    }
    /* Counted, not corrected: how far the corrected picture sits outside the
     * domain the job declared. */
    if (composite.r < 0.0f || composite.r > Ceiling)
        InterlockedAdd(Counters[30], 1u);
    if (composite.g < 0.0f || composite.g > Ceiling)
        InterlockedAdd(Counters[30], 1u);
    if (composite.b < 0.0f || composite.b > Ceiling)
        InterlockedAdd(Counters[30], 1u);
    store16(Destination, id.y, id.x, DestinationPitch,
        float4(composite, 1.0));
    float lowest = min(min(composite.r, composite.g), composite.b);
    float highest = max(max(composite.r, composite.g), composite.b);
    InterlockedMin(Counters[31], sortable(lowest));
    InterlockedMax(Counters[32], sortable(highest));
}
)hlsl";

/* linear scRGB -> PQ-coded BT.2020 limited-range P010. The chroma is the box
 * mean of the four pixels' *linear* colour, encoded once, which is the exact
 * inverse of the replication in decode. */
const char* encode_body = R"hlsl(
ByteAddressBuffer Source : register(t0);
RWByteAddressBuffer DestinationY : register(u0);
RWByteAddressBuffer DestinationUV : register(u1);
RWBuffer<uint> Counters : register(u2);
cbuffer Frame : register(b0) {
    uint Width, Height, SourcePitch, DestinationYPitch;
    uint DestinationUVPitch, Unused0, Unused1, Unused2;
    float Ceiling;
};
float3 encode_relative(float3 scrgb) {
    /* Gamut conversion first, then the absolute-to-relative step: the interval
     * PQ encodes is 0..10000 cd/m^2, which is 0..125 in this plane's units. */
    precise float3 relative = convert709To2020(scrgb) / RELATIVE_TO_SCRGB;
    float negatives = uint(relative.r < 0.0f) + uint(relative.g < 0.0f) +
        uint(relative.b < 0.0f);
    float high = uint(relative.r > Ceiling / RELATIVE_TO_SCRGB) +
        uint(relative.g > Ceiling / RELATIVE_TO_SCRGB) +
        uint(relative.b > Ceiling / RELATIVE_TO_SCRGB);
    float low = uint(relative.r < 0.0f) + uint(relative.g < 0.0f) +
        uint(relative.b < 0.0f);
    float bad = uint(unrepresentable(relative.r)) +
        uint(unrepresentable(relative.g)) +
        uint(unrepresentable(relative.b));
    InterlockedAdd(Counters[42], negatives);
    InterlockedAdd(Counters[40], high);
    InterlockedAdd(Counters[41], low);
    InterlockedAdd(Counters[43], bad);
    float limit = Ceiling / RELATIVE_TO_SCRGB;
    return float3(pq_oetf(clamp(relative.r, 0.0f, limit)),
                  pq_oetf(clamp(relative.g, 0.0f, limit)),
                  pq_oetf(clamp(relative.b, 0.0f, limit)));
}
void luma_code_for(float3 scrgb, uint row, uint column, uint pitch,
                   RWByteAddressBuffer DestinationY) {
    float3 encoded = encode_relative(scrgb);
    float y = KR * encoded.r + KG * encoded.g + KB * encoded.b;
    float code = code_from_y(clamp(y, 0.0f, 1.0f));
    uint word = DestinationY.Load(row * pitch + (column >> 1) * 4u);
    if (column & 1u) word = (word & 0xffffu) | (pack_p010(0.0f, code) & 0xffff0000u);
    else word = (word & 0xffff0000u) | (pack_p010(code, 0.0f) & 0xffffu);
    DestinationY.Store(row * pitch + (column >> 1) * 4u, word);
    InterlockedMin(Counters[45], uint(code));
    InterlockedMax(Counters[46], uint(code));
}
[numthreads(8, 8, 1)]
void main_encode(uint3 id : SV_DispatchThreadID) {
    if (id.x >= (Width >> 1) || id.y >= (P210_OUTPUT ? Height : (Height >> 1))) return;
    uint baseX = id.x * 2u, baseY = P210_OUTPUT ? id.y : id.y * 2u;
    precise float3 total = float3(0.0, 0.0, 0.0);
    for (uint dy = 0u; dy < (P210_OUTPUT ? 1u : 2u); ++dy)
        for (uint dx = 0u; dx < 2u; ++dx) {
            float3 scrgb = load16(Source, baseY + dy, baseX + dx,
                SourcePitch).rgb;
            total = total + scrgb;
            luma_code_for(scrgb, baseY + dy, baseX + dx, DestinationYPitch,
                DestinationY);
            InterlockedMax(Counters[44], sortable(
                max(max(scrgb.r, scrgb.g), scrgb.b)));
        }
    /* Colour difference is quarter resolution and one encoded mean serves the
     * block; the 4:2:0 UV plane interleaves U and V, so the block owns one
     * whole 32-bit word. */
    precise float3 mean = total * (P210_OUTPUT ? 0.5 : 0.25);
    float3 encodedMean = encode_relative(mean);
    precise float yMean = KR * encodedMean.r + KG * encodedMean.g +
        KB * encodedMean.b;
    float cb = (encodedMean.b - yMean) / (2.0f * (1.0f - KB));
    float cr = (encodedMean.r - yMean) / (2.0f * (1.0f - KR));
    float cbCode = code_from_c(clamp(cb, -0.5f, 0.5f));
    float crCode = code_from_c(clamp(cr, -0.5f, 0.5f));
    DestinationUV.Store(id.y * DestinationUVPitch + id.x * 4u,
                        pack_p010(cbCode, crCode));
    InterlockedMin(Counters[47], uint(cbCode));
    InterlockedMax(Counters[48], uint(crCode));
}
)hlsl";

/* Relative linear Rec.709 -> limited-range BT.709 OETF P010. Clipping occurs
 * only here, at the output quantizer; no transfer or gamut conversion is
 * applied to the signed FP16 master before this point. */
const char* sdr10_encode_body = R"hlsl(
ByteAddressBuffer Source : register(t0);
RWByteAddressBuffer DestinationY : register(u0);
RWByteAddressBuffer DestinationUV : register(u1);
RWBuffer<uint> Counters : register(u2);
cbuffer Frame : register(b0) {
    uint Width, Height, SourcePitch, DestinationYPitch;
    uint DestinationUVPitch, Unused0, Unused1, Unused2;
    float Ceiling;
};
float bt709_oetf(float l) {
    return l < 0.018f ? 4.5f * l : 1.099f * pow(l, 0.45f) - 0.099f;
}
float3 quantizer_linear(float3 linearRgb) {
    return float3(bt709_oetf(clamp(linearRgb.r, 0.0f, Ceiling)),
                  bt709_oetf(clamp(linearRgb.g, 0.0f, Ceiling)),
                  bt709_oetf(clamp(linearRgb.b, 0.0f, Ceiling)));
}
float3 encode_pixel(float3 linearRgb) {
    InterlockedAdd(Counters[42], uint(linearRgb.r < 0.0f) + uint(linearRgb.g < 0.0f) +
        uint(linearRgb.b < 0.0f));
    InterlockedAdd(Counters[40], uint(linearRgb.r > Ceiling) + uint(linearRgb.g > Ceiling) +
        uint(linearRgb.b > Ceiling));
    InterlockedAdd(Counters[41], uint(linearRgb.r < 0.0f) + uint(linearRgb.g < 0.0f) +
        uint(linearRgb.b < 0.0f));
    InterlockedAdd(Counters[43], uint(unrepresentable(linearRgb.r)) +
        uint(unrepresentable(linearRgb.g)) + uint(unrepresentable(linearRgb.b)));
    return quantizer_linear(linearRgb);
}
void luma_code_for(float3 linearRgb, uint row, uint column, uint pitch,
                   RWByteAddressBuffer Destination) {
    float3 encoded = encode_pixel(linearRgb);
    float y = 0.2126f * encoded.r + 0.7152f * encoded.g + 0.0722f * encoded.b;
    float code = code_from_y(clamp(y, 0.0f, 1.0f));
    uint word = Destination.Load(row * pitch + (column >> 1) * 4u);
    if (column & 1u) word = (word & 0xffffu) | (pack_p010(0.0f, code) & 0xffff0000u);
    else word = (word & 0xffff0000u) | (pack_p010(code, 0.0f) & 0xffffu);
    Destination.Store(row * pitch + (column >> 1) * 4u, word);
    InterlockedMin(Counters[45], uint(code));
    InterlockedMax(Counters[46], uint(code));
}
[numthreads(8, 8, 1)]
void main_encode(uint3 id : SV_DispatchThreadID) {
    if (id.x >= (Width >> 1) || id.y >= (P210_OUTPUT ? Height : (Height >> 1))) return;
    uint baseX = id.x * 2u, baseY = P210_OUTPUT ? id.y : id.y * 2u;
    precise float3 encoded_total = float3(0.0f, 0.0f, 0.0f);
    for (uint dy = 0u; dy < (P210_OUTPUT ? 1u : 2u); ++dy)
        for (uint dx = 0u; dx < 2u; ++dx) {
            float3 linearRgb = load16(Source, baseY + dy, baseX + dx, SourcePitch).rgb;
            encoded_total += quantizer_linear(linearRgb);
            luma_code_for(linearRgb, baseY + dy, baseX + dx, DestinationYPitch,
                DestinationY);
            InterlockedMax(Counters[44], sortable(max(max(linearRgb.r, linearRgb.g), linearRgb.b)));
        }
    /* 4:2:0 chroma is the mean of encoded RGB samples. OETF is nonlinear, so
     * averaging linear RGB before encoding would shift saturated blocks. Pixel
     * clipping counters are intentionally charged only by luma_code_for. */
    float3 encoded = encoded_total * (P210_OUTPUT ? 0.5f : 0.25f);
    float y = 0.2126f * encoded.r + 0.7152f * encoded.g + 0.0722f * encoded.b;
    float cb = (encoded.b - y) / (2.0f * (1.0f - 0.0722f));
    float cr = (encoded.r - y) / (2.0f * (1.0f - 0.2126f));
    float cbCode = code_from_c(clamp(cb, -0.5f, 0.5f));
    float crCode = code_from_c(clamp(cr, -0.5f, 0.5f));
    DestinationUV.Store(id.y * DestinationUVPitch + id.x * 4u,
        pack_p010(cbCode, crCode));
    InterlockedMin(Counters[47], uint(cbCode));
    InterlockedMax(Counters[48], uint(crCode));
}
)hlsl";

/* --- Bridge -------------------------------------------------------------- */

struct Bridge {
    Cuda cuda;
    CUcontext context = nullptr;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    HANDLE fence_event = nullptr;
    uint64_t fence_value = 0;
    uint32_t full_w = 0, full_h = 0, low_w = 0, low_h = 0;
    uint32_t down_taps = 0, rec_taps = 0;
    double paper_white = 1.0;
    double actionable_relative = kProxyActionableFallback;
    double saturation_relative = kProxySaturationFallback;
    double support_ceiling = 0.0;
    double clip_ceiling = 0.0;
    bool sdr10_mode = false;
    uint32_t purpose = GPU_HDR_PURPOSE_FULL;
    bool pass_enabled[kPassCount] = {};
    Plane planes[kPlaneCount];
    ComPtr<ID3D12DescriptorHeap> heap;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pso[kPassCount];
    ComPtr<ID3DBlob> code[kPassCount];
    uint32_t srv_step = 0;
    /* Slots reserved for one pass: its SRV run then its UAV run, laid out as
     * the root signature's two ranges require. */
    uint32_t max_srv = 0, max_uav = 0, pass_slots = 0;
    D3D12_CPU_DESCRIPTOR_HANDLE heap_cpu_start{};
    D3D12_GPU_DESCRIPTOR_HANDLE heap_gpu_start{};
    PassBinding bindings[kPassCount];
    /* Where each tap run lives in the shared tap buffer, as element windows. */
    uint32_t tap_first[4] = {0, 0, 0, 0};
    uint32_t tap_rows[4] = {0, 0, 0, 0};
    uint32_t* counter_readback = nullptr;
    /* The last counter snapshot of every pass, so a stage's report survives
     * until the next time that stage runs rather than being overwritten by the
     * stage after it. */
    uint32_t stage_counters[kPassCount][kCounterSlots] = {};
    GpuHdrTransferCounters counters_state{sizeof(GpuHdrTransferCounters)};
    /* The multiscale foundation's own attribution, in a struct of its own so the
     * layout above never changes. */
    GpuHdrMultiscaleCounters multiscale{sizeof(GpuHdrMultiscaleCounters)};
    /* Which master the working planes were derived from, and which generation
     * of the working plane the proxy actually read. The second one is the whole
     * point: a proxy's provenance is the working contents it sampled, not the
     * master that happens to be current while it runs, so an import that lands
     * between a proxy and the downscale that supersedes it cannot leave a proxy
     * that reads as fresh. The new observation entry refuses a stale
     * derivation, because a residual measured against a proxy of some earlier
     * frame is a number that looks valid. The existing entries never read
     * these, so the single-layer order stays as unchecked as it always was. */
    uint64_t master_epoch = 0;
    uint64_t working_epoch = 0;
    uint64_t working_generation = 0;
    uint64_t proxy_generation = 0;
    bool failed = false;
    ~Bridge();
};

/* The reference module's `master * 2**stops` in half precision, kept as a named
 * shader helper so the residual pass and any later arm apply the same gain. The
 * hand-rolled exp2 is what makes that gain land on the same half as the CPU
 * reference's: the hardware approximation is a thousandth of a code away, which
 * is enough to move a value sitting near a rounding boundary. */
const char* half_stops_helper = R"hlsl(
float3 half3_stops(float3 master, float3 stops) {
    return float3(half_round(master.r * hdr_exp2(stops.x)),
                  half_round(master.g * hdr_exp2(stops.y)),
                  half_round(master.b * hdr_exp2(stops.z)));
}
)hlsl";

const char* concat(const std::vector<const char*>& pieces) {
    static thread_local std::vector<char> text;
    text.clear();
    for (const char* piece : pieces) {
        if (!piece) continue;
        text.insert(text.end(), piece, piece + std::strlen(piece));
    }
    text.push_back('\0');
    return text.data();
}

void transition(ID3D12GraphicsCommandList* list, Plane& plane,
                D3D12_RESOURCE_STATES after) {
    if (plane.state == after || !plane.resource) return;
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = plane.resource.Get();
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = plane.state;
    b.Transition.StateAfter = after;
    list->ResourceBarrier(1, &b);
    plane.state = after;
}

bool wait_gpu(Bridge* b, const char* what) {
    if (b->fence->GetCompletedValue() >= b->fence_value) return true;
    HRESULT hr = b->fence->SetEventOnCompletion(b->fence_value, b->fence_event);
    if (FAILED(hr)) { fail(what, hr); return false; }
    if (WaitForSingleObject(b->fence_event, 60000) != WAIT_OBJECT_0) {
        fail("HDR fence timeout", ERROR_TIMEOUT); return false;
    }
    return true;
}

/* The exact inverse of the HLSL sortable(): a key at or above 0x80000000 came
 * from a non-negative float whose sign bit the map set, so clearing it returns
 * the original bits; a key below it came from a negative float the map
 * complemented, so complementing it back returns the original bits. */
uint32_t unsortable_to_bits(uint32_t key) {
    return key >= 0x80000000u ? (key ^ 0x80000000u) : ~key;
}

float read_sortable(uint32_t key) {
    const uint32_t bits = unsortable_to_bits(key);
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

}  // namespace

/* --- Plan and pipeline construction ------------------------------------- */

namespace {

HRESULT create_plane(Bridge* b, PlaneId id, uint32_t bytes_per_row,
                     uint32_t rows, bool allow_uav, bool share_to_cuda) {
    Plane& p = b->planes[id];
    p.bytes_per_row = bytes_per_row;
    p.rows = rows;
    /* Every row pitch is a multiple of 256, which is what both a placed
     * footprint and a CUDA pitched copy require, and what lets a raw view count
     * the buffer in 4-byte words without a remainder. */
    p.pitch = align256(bytes_per_row);
    const uint64_t size = uint64_t(p.pitch) * rows;
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (allow_uav) desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    /* A shared resource has to be created in COMMON, whatever its usage flags,
     * and the barriers below move it to the state each pass needs. A raw view
     * requires the unordered-access flag even when the plane is only ever read,
     * which is why every device plane carries it. */
    HRESULT hr = b->device->CreateCommittedResource(
        &heap, share_to_cuda ? D3D12_HEAP_FLAG_SHARED : D3D12_HEAP_FLAG_NONE,
        &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
        IID_PPV_ARGS(p.resource.GetAddressOf()));
    if (FAILED(hr)) { fail("create HDR plane", hr); return hr; }
    p.state = D3D12_RESOURCE_STATE_COMMON;
    if (!share_to_cuda) return S_OK;
    hr = b->device->CreateSharedHandle(p.resource.Get(), nullptr, GENERIC_ALL,
                                       nullptr, &p.shared_handle);
    if (FAILED(hr)) { fail("share HDR plane", hr); return hr; }
    const auto allocation =
        b->device->GetResourceAllocationInfo(0, 1, &desc);
    ContextScope scope(&b->cuda, b->context);
    if (!scope.ok) return E_FAIL;
    ExternalMemoryDesc memory_desc{};
    memory_desc.type = 5;   /* CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE */
    memory_desc.handle.win32.handle = p.shared_handle;
    memory_desc.size = allocation.SizeInBytes;
    memory_desc.flags = 1;  /* CUDA_EXTERNAL_MEMORY_DEDICATED */
    CUresult cu = b->cuda.import_memory(&p.external_memory, &memory_desc);
    if (cu != CUDA_SUCCESS) { fail("cuImportExternalMemory(HDR plane)", cu);
        return E_FAIL; }
    ExternalBufferDesc buffer_desc{};
    buffer_desc.size = allocation.SizeInBytes;
    cu = b->cuda.map_buffer(&p.mapped, p.external_memory, &buffer_desc);
    if (cu != CUDA_SUCCESS) { fail("cuExternalMemoryGetMappedBuffer(HDR)", cu);
        return E_FAIL; }
    return S_OK;
}

/* Releases what the bridge owns, in the order CUDA wants it: the mapped device
 * addresses first, then the imported allocations, then the NT handles, then the
 * COM objects. The borrowed device and context are never released. */
HRESULT create_host_visible(Bridge* b, PlaneId id, uint64_t size,
                            D3D12_HEAP_TYPE type, bool allow_uav);

Bridge::~Bridge() {
    if (context && cuda.module && cuda.push && cuda.pop) {
        CUcontext previous = nullptr;
        if (cuda.push(context) == CUDA_SUCCESS) {
            for (Plane& p : planes) {
                if (p.mapped && cuda.mem_free) cuda.mem_free(CUdeviceptr(p.mapped));
                if (p.external_memory && cuda.destroy_memory)
                    cuda.destroy_memory(p.external_memory);
                p.mapped = 0;
                p.external_memory = nullptr;
            }
            cuda.pop(&previous);
        }
    }
    for (Plane& p : planes) {
        if (p.shared_handle) CloseHandle(p.shared_handle);
        p.shared_handle = nullptr;
    }
    if (fence_event) CloseHandle(fence_event);
    if (cuda.module) FreeLibrary(cuda.module);
}

/* The four tap runs, in the order the descriptors expect, and the buffer that
 * holds them. */
void plan_taps(Bridge* b, const GpuHdrTapTable& downscale,
               const GpuHdrTapTable& reconstruct) {
    const uint32_t down_v = downscale.vertical_dst_len;
    const uint32_t down_h = downscale.horizontal_dst_len;
    const uint32_t rec_v = b->purpose == GPU_HDR_PURPOSE_FULL ? reconstruct.vertical_dst_len : 0;
    const uint32_t rec_h = b->purpose == GPU_HDR_PURPOSE_FULL ? reconstruct.horizontal_dst_len : 0;
    b->tap_rows[0] = down_v;
    b->tap_rows[1] = down_h;
    b->tap_rows[2] = rec_v;
    b->tap_rows[3] = rec_h;
    b->tap_first[0] = 0;
    b->tap_first[1] = down_v;
    b->tap_first[2] = down_v + down_h;
    b->tap_first[3] = down_v + down_h + rec_v;
    const uint32_t rows = down_v + down_h + rec_v + rec_h;
    if (FAILED(create_plane(b, kPlaneTaps, rows * kTapStride, 1, false,
                            false)))
        b->failed = true;
}

void layout_planes(Bridge* b, const GpuHdrTapTable& downscale,
                   const GpuHdrTapTable& reconstruct) {
    const uint32_t full = b->full_w, full_h = b->full_h;
    const uint32_t low = b->low_w, low_h = b->low_h;
    struct Request { PlaneId id; uint32_t bytes_per_row, rows; };
    /* P010 rows are 2 bytes per sample, an scRGB row 8, the guidance surface 4
     * and the float32 scratch one contiguous block of full x low float4s. */
    const Request requests[] = {
        {kPlaneSourceY, full * 2u, full_h},
        {kPlaneSourceUV, full * 2u, (b->purpose == GPU_HDR_PURPOSE_DECODE422 ? full_h : full_h / 2u)},
        {kPlaneMaster, full * 8u, full_h},
        {kPlaneWorking, low * 8u, low_h},
        {kPlaneProxy, low * 8u, low_h},
        {kPlaneGuidance, low * 4u, low_h},
        {kPlaneNeural, low * 8u, low_h},
        {kPlaneResidual, low * 8u, low_h},
        {kPlaneComposited, full * 8u, full_h},
        {kPlaneEncodeY, full * 2u, full_h},
        {kPlaneEncodeUV, full * 2u, (b->purpose == GPU_HDR_PURPOSE_ENCODE422 ? full_h : full_h / 2u)},
        {kPlaneScratch, full * low_h * 16u, 1u},
        {kPlaneCounters, kCounterSlots * 4u, 1u},
    };
    for (const Request& r : requests) {
        const bool needed = b->purpose == GPU_HDR_PURPOSE_FULL || r.id == kPlaneCounters ||
            r.id == kPlaneMaster ||
            ((b->purpose == GPU_HDR_PURPOSE_DECODE || b->purpose == GPU_HDR_PURPOSE_DECODE422) &&
                (r.id == kPlaneSourceY || r.id == kPlaneSourceUV)) ||
            (b->purpose == GPU_HDR_PURPOSE_GUIDANCE &&
                (r.id == kPlaneWorking || r.id == kPlaneProxy ||
                 r.id == kPlaneGuidance || r.id == kPlaneScratch)) ||
            ((b->purpose == GPU_HDR_PURPOSE_ENCODE || b->purpose == GPU_HDR_PURPOSE_ENCODE422) &&
                (r.id == kPlaneComposited || r.id == kPlaneEncodeY || r.id == kPlaneEncodeUV));
        if (!needed) continue;
        if (FAILED(create_plane(b, r.id, r.bytes_per_row, r.rows, true, true))) {
            b->failed = true;
            return;
        }
    }
    if (b->purpose == GPU_HDR_PURPOSE_FULL || b->purpose == GPU_HDR_PURPOSE_GUIDANCE)
        plan_taps(b, downscale, reconstruct);
    if (b->failed) return;
    if (FAILED(create_host_visible(b, kPlaneInitPattern,
            kCounterSlots * sizeof(uint32_t), D3D12_HEAP_TYPE_UPLOAD, false)) ||
        FAILED(create_host_visible(b, kPlaneCounterReadback,
            kCounterSlots * sizeof(uint32_t), D3D12_HEAP_TYPE_READBACK, false))) {
        b->failed = true;
        return;
    }
    /* The seed the counters start every pass from: the min keys begin at the
     * top of the sortable range so the first pixel decides them. */
    uint32_t* seed = reinterpret_cast<uint32_t*>(
        b->planes[kPlaneInitPattern].host);
    for (uint32_t slot = 0; slot < kCounterSlots; ++slot)
        seed[slot] = counter_init_value(slot);
    b->counter_readback = reinterpret_cast<uint32_t*>(
        b->planes[kPlaneCounterReadback].host);
}

/* A readback-heap buffer is not shareable and never needs a CUDA mapping, so it
 * does not go through create_plane. */
HRESULT create_host_visible(Bridge* b, PlaneId id, uint64_t size,
                            D3D12_HEAP_TYPE type, bool allow_uav) {
    Plane& p = b->planes[id];
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = type;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (allow_uav) desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    const D3D12_RESOURCE_STATES initial =
        type == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ
        : (allow_uav ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                     : D3D12_RESOURCE_STATE_COPY_DEST);
    HRESULT hr = b->device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE,
        &desc, initial, nullptr, IID_PPV_ARGS(p.resource.GetAddressOf()));
    if (FAILED(hr)) { fail("create HDR host-visible plane", hr); return hr; }
    p.state = initial;
    p.rows = 1;
    p.bytes_per_row = uint32_t(size);
    p.pitch = uint32_t(size);
    void* host = nullptr;
    hr = p.resource->Map(0, nullptr, &host);
    if (FAILED(hr)) { fail("map HDR host-visible plane", hr); return hr; }
    p.host = static_cast<uint8_t*>(host);
    return S_OK;
}

void raw_srv(Bridge* b, Plane& p, D3D12_CPU_DESCRIPTOR_HANDLE at) {
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_R32_TYPELESS;
    srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Buffer.NumElements = (p.pitch * p.rows) / 4;
    srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    b->device->CreateShaderResourceView(p.resource.Get(), &srv, at);
}

void raw_uav(Bridge* b, Plane& p, D3D12_CPU_DESCRIPTOR_HANDLE at) {
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = DXGI_FORMAT_R32_TYPELESS;
    uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uav.Buffer.NumElements = (p.pitch * p.rows) / 4;
    uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    b->device->CreateUnorderedAccessView(p.resource.Get(), nullptr, &uav, at);
}

void typed_uav(Bridge* b, Plane& p, D3D12_CPU_DESCRIPTOR_HANDLE at) {
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = DXGI_FORMAT_R32_UINT;
    uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uav.Buffer.FirstElement = 0;
    uav.Buffer.NumElements = p.pitch / 4;
    uav.Buffer.StructureByteStride = 0;
    uav.Buffer.CounterOffsetInBytes = 0;
    uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
    b->device->CreateUnorderedAccessView(p.resource.Get(), nullptr, &uav, at);
}

void structured_srv(Bridge* b, Plane& p, uint32_t first, uint32_t rows,
                    uint32_t stride, D3D12_CPU_DESCRIPTOR_HANDLE at) {
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_UNKNOWN;
    srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Buffer.FirstElement = first;
    srv.Buffer.NumElements = rows;
    srv.Buffer.StructureByteStride = stride;
    b->device->CreateShaderResourceView(p.resource.Get(), &srv, at);
}

void structured_uav(Bridge* b, Plane& p, uint32_t elements, uint32_t stride,
                    D3D12_CPU_DESCRIPTOR_HANDLE at) {
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = DXGI_FORMAT_UNKNOWN;
    uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uav.Buffer.FirstElement = 0;
    uav.Buffer.NumElements = elements;
    uav.Buffer.StructureByteStride = stride;
    uav.Buffer.CounterOffsetInBytes = 0;
    uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
    b->device->CreateUnorderedAccessView(p.resource.Get(), nullptr, &uav, at);
}

/* The fixed register order of every pass. Plane views are all created once at
 * open: no descriptor is created per frame, because nothing in the chain borrows
 * a resource that has to be read by a shader (the Feature 18 output is copied
 * into this bridge's own plane first, which also means the bridge never
 * transitions a borrowed resource). */
void layout_passes(Bridge* b) {
    const uint32_t down_v = b->tap_first[0], down_h = b->tap_first[1];
    const uint32_t rec_v = b->tap_first[2], rec_h = b->tap_first[3];
    PassBinding* d = b->bindings;
    /* decode: t0 borrowed source luma, t1 borrowed source colour difference;
     * u0 the full-resolution master, u1 counters. */
    d[kPassDecode] = {{{kPlaneSourceY, kRawSrv, 0, 0},
                       {kPlaneSourceUV, kRawSrv, 0, 0}}, 2,
                      {{kPlaneMaster, kRawUav, 0, 0},
                       {kPlaneCounters, kCountersUav, 0, 0}}, 2, 9,
                      "main_decode"};
    /* vertical: master and the downscale's vertical rows; writes float32
     * scratch. */
    d[kPassVertical] = {{{kPlaneMaster, kRawSrv, 0, 0},
                         {kPlaneTaps, kTapsSrv, down_v, b->low_h}}, 2,
                        {{kPlaneScratch, kScratchUav, 0, 0}}, 1, 8,
                        "main_vertical"};
    d[kPassHorizontal] = {{{kPlaneScratch, kScratchSrv, 0, 0},
                           {kPlaneTaps, kTapsSrv, down_h, b->low_w}}, 2,
                          {{kPlaneWorking, kRawUav, 0, 0},
                           {kPlaneCounters, kCountersUav, 0, 0}}, 2, 8,
                          "main_horizontal"};
    d[kPassProxy] = {{{kPlaneWorking, kRawSrv, 0, 0}}, 1,
                     {{kPlaneProxy, kRawUav, 0, 0},
                      {kPlaneGuidance, kRawUav, 0, 0},
                      {kPlaneCounters, kCountersUav, 0, 0}}, 3, 11,
                     "main_proxy"};
    d[kPassResidual] = {{{kPlaneNeural, kRawSrv, 0, 0},
                         {kPlaneProxy, kRawSrv, 0, 0},
                         {kPlaneWorking, kRawSrv, 0, 0}}, 3,
                        {{kPlaneResidual, kRawUav, 0, 0},
                         {kPlaneCounters, kCountersUav, 0, 0}}, 2, 11,
                        "main_residual"};
    d[kPassComposite] = {{{kPlaneResidual, kRawSrv, 0, 0},
                          {kPlaneMaster, kRawSrv, 0, 0},
                          {kPlaneTaps, kTapsSrv, rec_v, b->full_h},
                          {kPlaneTaps, kTapsSrv, rec_h, b->full_w}}, 4,
                         {{kPlaneComposited, kRawUav, 0, 0},
                          {kPlaneCounters, kCountersUav, 0, 0}}, 2, 11,
                         "main_composite"};
    d[kPassEncode] = {{{kPlaneComposited, kRawSrv, 0, 0}}, 1,
                      {{kPlaneEncodeY, kRawUav, 0, 0},
                       {kPlaneEncodeUV, kRawUav, 0, 0},
                       {kPlaneCounters, kCountersUav, 0, 0}}, 3, 9,
                      "main_encode"};
    uint32_t max_srv = 0, max_uav = 0;
    for (uint32_t p = 0; p < kPassCount; ++p) {
        max_srv = max(max_srv, d[p].srv_count);
        max_uav = max(max_uav, d[p].uav_count);
    }
    b->max_srv = max_srv;
    b->max_uav = max_uav;
    b->pass_slots = max_srv + max_uav;
}

/* One view of one plane, of the kind the pass's register expects. */
void make_view(Bridge* b, const SlotSpec& spec, D3D12_CPU_DESCRIPTOR_HANDLE at) {
    Plane& p = b->planes[spec.plane];
    switch (spec.kind) {
        case kRawSrv: raw_srv(b, p, at); break;
        case kRawUav: raw_uav(b, p, at); break;
        case kCountersUav: typed_uav(b, p, at); break;
        case kScratchSrv:
            structured_srv(b, p, 0, b->full_w * b->low_h, 16, at); break;
        case kScratchUav:
            structured_uav(b, p, b->full_w * b->low_h, 16, at); break;
        case kTapsSrv:
            structured_srv(b, p, spec.first, spec.rows, kTapStride, at); break;
    }
}

/* The CPU and GPU handles of slot `slot` of pass `pass`. */
D3D12_CPU_DESCRIPTOR_HANDLE cpu_slot(Bridge* b, uint32_t pass, uint32_t slot) {
    D3D12_CPU_DESCRIPTOR_HANDLE h = b->heap_cpu_start;
    h.ptr += uint64_t(pass * b->pass_slots + slot) * b->srv_step;
    return h;
}
D3D12_GPU_DESCRIPTOR_HANDLE gpu_slot(Bridge* b, uint32_t pass, uint32_t slot) {
    D3D12_GPU_DESCRIPTOR_HANDLE h = b->heap_gpu_start;
    h.ptr += uint64_t(pass * b->pass_slots + slot) * b->srv_step;
    return h;
}

HRESULT make_heap(Bridge* b) {
    D3D12_DESCRIPTOR_HEAP_DESC heap{};
    heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap.NumDescriptors = kPassCount * b->pass_slots;
    heap.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    HRESULT hr = b->device->CreateDescriptorHeap(&heap,
        IID_PPV_ARGS(b->heap.GetAddressOf()));
    if (FAILED(hr)) { fail("create HDR descriptor heap", hr); return hr; }
    b->srv_step = b->device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    b->heap_cpu_start = b->heap->GetCPUDescriptorHandleForHeapStart();
    b->heap_gpu_start = b->heap->GetGPUDescriptorHandleForHeapStart();
    /* Every slot gets a valid descriptor before the real ones overwrite it, so
     * a table slot a pass does not use is still a descriptor and not
     * uninitialised heap memory. */
    for (uint32_t pass = 0; pass < kPassCount; ++pass) {
        for (uint32_t slot = 0; slot < b->max_srv; ++slot)
            raw_srv(b, b->planes[kPlaneCounters], cpu_slot(b, pass, slot));
        for (uint32_t slot = 0; slot < b->max_uav; ++slot)
            typed_uav(b, b->planes[kPlaneCounters],
                      cpu_slot(b, pass, b->max_srv + slot));
    }
    for (uint32_t pass = 0; pass < kPassCount; ++pass) {
        if (!b->pass_enabled[pass]) continue;
        const PassBinding& d = b->bindings[pass];
        for (uint32_t i = 0; i < d.srv_count; ++i)
            make_view(b, d.srv[i], cpu_slot(b, pass, i));
        for (uint32_t i = 0; i < d.uav_count; ++i)
            make_view(b, d.uav[i], cpu_slot(b, pass, b->max_srv + i));
    }
    return S_OK;
}

}  // namespace

namespace {

bool build_pipeline(Bridge* b, PassId pass, const char* source,
                    const char* entry) {
    UINT flags = D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_IEEE_STRICTNESS |
                 D3DCOMPILE_OPTIMIZATION_LEVEL3;
    ComPtr<ID3DBlob> errors;
    HRESULT hr = D3DCompile(source, std::strlen(source), "gpu_hdr", nullptr,
        nullptr, entry, "cs_5_0", flags, 0, b->code[pass].GetAddressOf(),
        errors.GetAddressOf());
    if (FAILED(hr)) {
        const char* detail = errors ? static_cast<const char*>(
            errors->GetBufferPointer()) : "(no detail)";
        std::snprintf(last_error, sizeof(last_error), "compile %s: %s", entry,
                      detail);
        return false;
    }
    D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
    pso.pRootSignature = b->root.Get();
    pso.CS = {b->code[pass]->GetBufferPointer(), b->code[pass]->GetBufferSize()};
    hr = b->device->CreateComputePipelineState(&pso,
        IID_PPV_ARGS(b->pso[pass].GetAddressOf()));
    if (FAILED(hr)) { fail("CreateComputePipelineState(HDR)", hr); return false; }
    return true;
}

/* One root signature for every pass: one descriptor table holding the pass's
 * SRV run followed by its UAV run, and 32 root constants. The table layout is
 * identical across passes, so a pass differs from its neighbour only in which
 * slots the table starts at and how many values the shader reads. */
HRESULT make_root_signature(Bridge* b) {
    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = b->max_srv;
    ranges[0].BaseShaderRegister = 0;
    ranges[0].RegisterSpace = 0;
    ranges[0].OffsetInDescriptorsFromTableStart = 0;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = b->max_uav;
    ranges[1].BaseShaderRegister = 0;
    ranges[1].RegisterSpace = 0;
    ranges[1].OffsetInDescriptorsFromTableStart =
        D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
    D3D12_ROOT_PARAMETER params[2]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable.NumDescriptorRanges = 2;
    params[0].DescriptorTable.pDescriptorRanges = ranges;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[1].Constants.ShaderRegister = 0;
    params[1].Constants.RegisterSpace = 0;
    params[1].Constants.Num32BitValues = 32;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC desc{};
    desc.NumParameters = 2;
    desc.pParameters = params;
    desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    ComPtr<ID3DBlob> blob, errors;
    HRESULT hr = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1,
        blob.GetAddressOf(), errors.GetAddressOf());
    if (FAILED(hr)) {
        if (errors) say(static_cast<const char*>(errors->GetBufferPointer()));
        fail("serialize HDR root signature", hr);
        return hr;
    }
    hr = b->device->CreateRootSignature(0, blob->GetBufferPointer(),
        blob->GetBufferSize(), IID_PPV_ARGS(b->root.GetAddressOf()));
    if (FAILED(hr)) fail("CreateRootSignature(HDR)", hr);
    return hr;
}

/* Derives the matrices and the luminance coefficients once per process. Every
 * value the shaders print comes from here, so a wrong constant can only be a
 * wrong derivation, not a mistyped one. */
void derive_contract() {
    static bool done = false;
    if (done) return;
    const Mat3 xyz_of_709 = rgb_to_xyz(kBt709Primaries, kD65);
    const Mat3 xyz_of_2020 = rgb_to_xyz(kBt2020Primaries, kD65);
    Mat3 back_to_709{};
    if (!mat_inverse(xyz_of_709, &back_to_709)) { say("matrix inverse failed");
        return; }
    g_to_scrgb = mat_multiply(back_to_709, xyz_of_2020);
    Mat3 back_to_2020{};
    if (!mat_inverse(xyz_of_2020, &back_to_2020)) { say("matrix inverse failed");
        return; }
    g_to_bt2020 = mat_multiply(back_to_2020, xyz_of_709);
    /* The Y row of Rec.2020's own RGB->XYZ is the non-constant-luminance
     * matrix's Kr, Kg, Kb. Row 1, not row 2: row 2 is the Z row. */
    g_kr = xyz_of_2020.m[3];
    g_kg = xyz_of_2020.m[4];
    g_kb = xyz_of_2020.m[5];
    done = true;
}

/* Assembles one pass's shader: the derived contract, the shared sample helpers,
 * the proxy curve, then the pass body. */
std::string pass_source(PassId pass, bool sdr10_mode, bool p210_mode) {
    std::string text = colour_header();
    text += numeric_header;
    text += sample_header;
    if (p210_mode && pass == kPassDecode) text += "#define P210_INPUT 1\n";
    if (pass == kPassEncode) text += p210_mode ? "#define P210_OUTPUT 1\n" : "#define P210_OUTPUT 0\n";
    if (sdr10_mode && (pass == kPassProxy || pass == kPassResidual)) text += "#define SDR10_NR 1\n";
    if (pass == kPassProxy || pass == kPassResidual) text += proxy_header();
    if (pass == kPassResidual) text += half_stops_helper;
    switch (pass) {
        case kPassDecode: text += sdr10_mode ? sdr10_decode_body : decode_body; break;
        case kPassVertical: text += vertical_body; break;
        case kPassHorizontal: text += horizontal_body; break;
        case kPassProxy:
            if (sdr10_mode) text += "#define SDR10_GUIDANCE 1\n";
            text += proxy_body;
            break;
        case kPassResidual:
            text += residual_body;
            text += residual_entry;
            break;
        case kPassComposite: text += composite_body; break;
        case kPassEncode: text += sdr10_mode ? sdr10_encode_body : encode_body; break;
        case kPassCount: break;
    }
    return text;
}

/* The tap runs, in the order the four views expect: downscale vertical,
 * downscale horizontal, reconstruction vertical, reconstruction horizontal. */
uint32_t tap_run_rows(const GpuHdrTapTable& plan, bool horizontal) {
    return horizontal ? plan.horizontal_dst_len : plan.vertical_dst_len;
}

/* The two runs' rows and widths in the shared tap buffer; plan_taps has already
 * put them in the bridge, and the descriptors are built from the same numbers. */
HRESULT upload_taps(Bridge* b, const GpuHdrTapTable& downscale,
                    const GpuHdrTapTable& reconstruct) {
    const uint32_t rows = b->tap_rows[0] + b->tap_rows[1] + b->tap_rows[2] +
        b->tap_rows[3];
    const uint64_t bytes = uint64_t(rows) * kTapStride;
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> staging;
    HRESULT hr = b->device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE,
        &desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(staging.GetAddressOf()));
    if (FAILED(hr)) { fail("create HDR tap staging", hr); return hr; }
    uint8_t* mapped = nullptr;
    hr = staging->Map(0, nullptr, reinterpret_cast<void**>(&mapped));
    if (FAILED(hr)) { fail("map HDR tap staging", hr); return hr; }
    std::memset(mapped, 0, size_t(bytes));
    /* A row shorter than the stride is padded with weight 0.0f at index 0, the
     * same padding the CPU reference's plan carries, so the shader sums the same
     * number of terms as the reference and the extra ones contribute nothing. */
    for (int run_index = 0; run_index < 4; ++run_index) {
        const bool horizontal = run_index == 1 || run_index == 3;
        const GpuHdrTapTable& plan = horizontal ?
            (run_index == 1 ? downscale : reconstruct)
            : (run_index == 0 ? downscale : reconstruct);
        const uint32_t taps = plan.taps;
        const int32_t* indices = horizontal ? plan.horizontal_indices
            : plan.vertical_indices;
        const float* weights = horizontal ? plan.horizontal_weights
            : plan.vertical_weights;
        const uint32_t count = b->tap_rows[run_index];
        for (uint32_t row = 0; row < count; ++row) {
            uint8_t* target = mapped + uint64_t(b->tap_first[run_index] + row) *
                kTapStride;
            float* w = reinterpret_cast<float*>(target);
            int32_t* ix = reinterpret_cast<int32_t*>(target +
                GPU_HDR_MAX_TAPS * sizeof(uint32_t));
            for (uint32_t t = 0; t < GPU_HDR_MAX_TAPS; ++t) {
                const bool real = t < taps;
                w[t] = real ? weights[size_t(row) * taps + t] : 0.0f;
                ix[t] = real ? indices[size_t(row) * taps + t] : 0;
            }
        }
    }
    staging->Unmap(0, nullptr);
    HRESULT hr2 = b->allocator->Reset();
    if (SUCCEEDED(hr2))
        hr2 = b->list->Reset(b->allocator.Get(), nullptr);
    if (FAILED(hr2)) { fail("reset HDR tap list", hr2); return hr2; }
    transition(b->list.Get(), b->planes[kPlaneTaps],
               D3D12_RESOURCE_STATE_COPY_DEST);
    b->list->CopyBufferRegion(b->planes[kPlaneTaps].resource.Get(), 0,
                              staging.Get(), 0, bytes);
    transition(b->list.Get(), b->planes[kPlaneTaps],
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    hr = b->list->Close();
    if (FAILED(hr)) { fail("close HDR tap list", hr); return hr; }
    ID3D12CommandList* commands[] = {b->list.Get()};
    b->queue->ExecuteCommandLists(1, commands);
    hr = b->queue->Signal(b->fence.Get(), ++b->fence_value);
    if (FAILED(hr)) { fail("signal HDR tap upload", hr); return hr; }
    if (!wait_gpu(b, "tap upload")) return E_FAIL;
    return S_OK;
}

/* --- Measured proxy invariants -------------------------------------------
 *
 * The two relative whites the job's support domain is derived from are measured
 * on the half grid here rather than taken on faith from the stage-A record, so
 * a change to the proxy constants moves them and the report says so. The scan
 * is a few thousand scalar evaluations of the same curve the shader runs, at
 * open, on the host, over no pixels. */

/* The nearest half to `value`, round-to-nearest-even, by way of the float bits:
 * the same grid the store lands on. */
double cpu_half_round(double value) {
    const float wide = float(value);
    uint32_t bits = 0;
    std::memcpy(&bits, &wide, sizeof(bits));
    const uint32_t sign = bits & 0x80000000u;
    const uint32_t rest = bits & 0x7FFFFFFFu;
    uint32_t half = 0;
    if (rest >= 0x38800000u) {            /* normal, and inside half range */
        uint32_t exponent = (rest >> 23) - 127 + 15;
        uint32_t mantissa = rest & 0x7FFFFFu;
        /* binary16 keeps 10 of the 23 mantissa bits, so 13 are discarded and
         * the round-to-nearest-even decision is made on those 13: guard bit 12,
         * round bit 11, sticky everything below. */
        const uint32_t dropped = mantissa & 0x1FFFu;
        mantissa >>= 13;
        const uint32_t guard = 0x1000u;
        if (dropped > guard || (dropped == guard && (mantissa & 1u))) ++mantissa;
        if (mantissa > 0x3FFu) { mantissa = 0; ++exponent; }
        half = (exponent << 10) | mantissa;
    } else {                              /* subnormal or zero */
        /* Below the smallest half normal the grid is the integer grid of
         * value / 2**-24, so scale and round there rather than shifting bits. */
        half = uint32_t(std::nearbyint(double(wide) * 16777216.0));
    }
    const uint16_t out = uint16_t(half | (sign >> 16));
    /* Decode the 16-bit pattern back to double. */
    const uint32_t e = (out >> 10) & 0x1Fu;
    const uint32_t m = out & 0x3FFu;
    double value_out;
    if (e == 0) value_out = std::ldexp(double(m), -24);
    else if (e == 0x1F) value_out = m ? std::numeric_limits<double>::quiet_NaN()
                                     : std::numeric_limits<double>::infinity();
    else value_out = std::ldexp(double(m | 0x400u), int(e) - 25);
    return (out & 0x8000u) ? -value_out : value_out;
}

/* The proxy curve exactly as the reference module computes it: single
 * precision, the same operation order, and the same half rounding of the
 * *encoded* value. A double mirror would enumerate the plateaus of a different
 * function from the one the module documents and the shader runs, and the dead
 * levels it reports would not be the ones the GPU actually lands on. */
float cpu_proxy_code(float relative) {
    const float knee = float(kProxyKnee);
    const float span = float(kProxySpan);
    const float steepness = float(kProxyK);
    float bounded;
    if (relative > knee) {
        const float soft =
            knee + span * (1.0f - std::exp(-steepness * (relative - knee)));
        bounded = soft > 1.0f ? 1.0f : (soft < 0.0f ? 0.0f : soft);
    } else {
        bounded = relative < 0.0f ? 0.0f : relative;
    }
    const float coded = bounded <= 0.0031308f
        ? 12.92f * bounded
        : 1.055f * std::pow(bounded, 1.0f / 2.4f) - 0.055f;
    return float(cpu_half_round(double(coded)));
}

/* The reference module's slope: the perturbation applied to the scene value,
 * both arms through the whole curve, divided by the width of the step. */
float cpu_code_slope_per_stop(float scene) {
    const float base = cpu_proxy_code(scene);
    if (base == 1.0f) return 0.0f;
    const float factor = float(std::exp2(kSlopeStopStep));
    const float high = cpu_proxy_code(scene * factor);
    const float low = cpu_proxy_code(scene / factor);
    return (high - low) / float(2.0 * kSlopeStopStep);
}

/* Walks the half grid in [1, 2) and returns the first value, downwards or
 * upwards, whose answer satisfies `predicate`, or 0.0 when none does. Both
 * invariants live in that band, and the grid step there is exactly 2**-10. */
template<bool (*predicate)(double)>
double scan_half_grid_down() {
    for (uint32_t code = 0x3FFFu; code >= 0x3C00u; --code) {
        const uint16_t pattern = uint16_t(code);
        const uint32_t e = (pattern >> 10) & 0x1Fu;
        const uint32_t m = pattern & 0x3FFu;
        const double value = std::ldexp(double(m | 0x400u), int(e) - 25);
        if (predicate(value)) return value;
        if (code == 0x3C00u) break;
    }
    return 0.0;
}

template<bool (*predicate)(double)>
double scan_half_grid_up() {
    for (uint32_t code = 0x3C00u; code <= 0x3FFFu; ++code) {
        const uint16_t pattern = uint16_t(code);
        const uint32_t e = (pattern >> 10) & 0x1Fu;
        const uint32_t m = pattern & 0x3FFu;
        const double value = std::ldexp(double(m | 0x400u), int(e) - 25);
        if (predicate(value)) return value;
    }
    return 0.0;
}

bool slope_is_measurable(double relative) {
    return std::fabs(cpu_code_slope_per_stop(float(relative))) >=
        float(kMinSlopePerStop);
}
bool slope_collapses(double relative) { return !slope_is_measurable(relative); }
bool code_is_not_pinned(double relative) {
    return cpu_proxy_code(float(relative)) < 1.0f;
}

/* The first relative white at or above paper white whose quantized code stops
 * advancing under the +-1/64-stop probe. The proxy has no analytic form for
 * this: it is a rounding artifact of the half code, so it is enumerated rather
 * than derived, and the band the reconstruction can read is everything below
 * it. Above the knee the dead ground is not one contiguous run, which is why
 * the scan is ascending and not a search for the last measurable level. */
double first_dead_relative() { return scan_half_grid_up<slope_collapses>(); }

/* The largest half value in the band whose code is still below 1.0, which
 * brackets the closed-form saturation point one grid step from below. */
double largest_unpinned_relative() {
    return scan_half_grid_down<code_is_not_pinned>();
}

/* --- One pass ------------------------------------------------------------ */

/* Root constants for one dispatch, packed in the cbuffer's own order. */
struct Constants {
    uint32_t value[32] = {};
    uint32_t count = 0;
    void put(uint32_t v) { if (count < 32) value[count++] = v; }
    void put(float v) {
        uint32_t bits = 0;
        std::memcpy(&bits, &v, sizeof(bits));
        put(bits);
    }
};

/* Every plane a pass binds, moved to the state that binding requires. Buffers
 * read by a shader are in NON_PIXEL_SHADER_RESOURCE and written ones in
 * UNORDERED_ACCESS, which is the pair the working SDR bridge uses. */
void prepare_states(ID3D12GraphicsCommandList* list, Bridge* b, PassId pass) {
    const PassBinding& d = b->bindings[pass];
    for (uint32_t i = 0; i < d.srv_count; ++i)
        transition(list, b->planes[d.srv[i].plane],
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    for (uint32_t i = 0; i < d.uav_count; ++i)
        transition(list, b->planes[d.uav[i].plane],
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}

/* Runs one pass to completion and keeps its counter snapshot. The command list
 * is closed and waited on inside the call, so a plane this pass wrote is
 * readable by the CUDA side the moment the call returns. */
bool run_pass(Bridge* b, PassId pass, const Constants& constants,
              uint32_t groups_x, uint32_t groups_y) {
    if (!b->pass_enabled[pass]) {
        say("shader pass is not available for this HDR bridge purpose"); return false;
    }
    const PassBinding& d = b->bindings[pass];
    if (constants.count != d.const_count) {
        std::snprintf(last_error, sizeof(last_error),
                      "%s takes %u constants, given %u", d.entry,
                      d.const_count, constants.count);
        return false;
    }
    HRESULT hr = b->allocator->Reset();
    if (SUCCEEDED(hr)) hr = b->list->Reset(b->allocator.Get(), nullptr);
    if (FAILED(hr)) { fail("reset HDR command list", hr); return false; }
    auto* list = b->list.Get();
    /* Every pass of this bridge is compute, so the signature is bound as one.
     * Binding it as a graphics signature and then setting the table through the
     * compute API faults the process inside the runtime, on the first dispatch
     * and nowhere else. */
    list->SetComputeRootSignature(b->root.Get());
    ID3D12DescriptorHeap* heaps[] = {b->heap.Get()};
    list->SetDescriptorHeaps(1, heaps);
    /* The pass's own counter run, from the seed pattern, so a report is the
     * measurement of this call and not of every call before it. */
    transition(list, b->planes[kPlaneCounters], D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyBufferRegion(b->planes[kPlaneCounters].resource.Get(), 0,
        b->planes[kPlaneInitPattern].resource.Get(), 0,
        kCounterSlots * sizeof(uint32_t));
    prepare_states(list, b, pass);
    list->SetPipelineState(b->pso[pass].Get());
    list->SetComputeRootDescriptorTable(0, gpu_slot(b, pass, 0));
    list->SetComputeRoot32BitConstants(1, d.const_count, constants.value, 0);
    list->Dispatch(groups_x, groups_y, 1);
    transition(list, b->planes[kPlaneCounters],
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    /* The one host boundary a production pass crosses, and it is scalar:
     * kCounterSlots * sizeof(uint32_t) = 256 bytes of counters per executed
     * dispatch, copied into a CPU-visible readback buffer and then memcpy'd out
     * below, because every report getter answers out of that host copy. Nothing
     * in GpuHdrTransferCounters counts these bytes; report_readback_bytes counts
     * the image planes the diagnostic readback entries copy, not this. The pass
     * that crossed it is counted in GpuHdrMultiscaleCounters.pass_dispatches at
     * the end of this function, so a caller can state the boundary as
     * pass_dispatches * 256 instead of calling it zero. */
    list->CopyBufferRegion(b->planes[kPlaneCounterReadback].resource.Get(), 0,
        b->planes[kPlaneCounters].resource.Get(), 0,
        kCounterSlots * sizeof(uint32_t));
    hr = list->Close();
    if (FAILED(hr)) { fail("close HDR command list", hr); return false; }
    ID3D12CommandList* commands[] = {list};
    b->queue->ExecuteCommandLists(1, commands);
    hr = b->queue->Signal(b->fence.Get(), ++b->fence_value);
    if (FAILED(hr)) { fail("signal HDR pass", hr); return false; }
    if (!wait_gpu(b, d.entry)) return false;
    /* A readback-heap mapping only shows the GPU's own writes once the range
     * has been unmapped and mapped again; left mapped, WDDM can hand back the
     * page as the CPU last saw it, and a pass that ran reports zeros. */
    ID3D12Resource* readback = b->planes[kPlaneCounterReadback].resource.Get();
    readback->Unmap(0, nullptr);
    void* host = nullptr;
    hr = readback->Map(0, nullptr, &host);
    if (FAILED(hr)) { fail("map HDR counter readback", hr); return false; }
    b->counter_readback = reinterpret_cast<uint32_t*>(host);
    std::memcpy(b->stage_counters[pass], b->counter_readback,
                sizeof(b->stage_counters[pass]));
    /* One dispatch ran to completion and its 256-byte counter run is now in
     * hand, so this is the point that a scalar counter readback has been
     * crossed. It is counted here, in the one path every shader pass takes,
     * rather than by whichever entry happens to dispatch: decode, downscale's
     * two axes, proxy, the residual, the composite and the encode all arrive
     * through this function, and an entry that dispatches none of them -
     * gpu_hdr_import_master and gpu_hdr_set_composite, which are copies - adds
     * nothing. A pass that failed above this line is not counted, so after a
     * failure pass_dispatches is the number of completed passes and its 256-byte
     * product is what this bridge measured, not what it submitted. */
    b->multiscale.pass_dispatches += 1;
    return true;
}

uint32_t counter(const Bridge* b, PassId pass, uint32_t slot) {
    return b->stage_counters[pass][slot];
}
double counter_float(const Bridge* b, PassId pass, uint32_t slot) {
    return double(read_sortable(counter(b, pass, slot)));
}

/* Copies one device plane out of a borrowed CUDA surface, row by row, so no
 * pixel of the job ever travels through host memory. */
bool copy_device_to_device(Bridge* b, uint64_t source, uint32_t source_pitch,
                           Plane& destination, uint32_t bytes_per_row,
                           uint32_t rows, const char* what) {
    if (!destination.resource) { say("destination plane is not allocated for this bridge purpose"); return false; }
    if (source_pitch % 16 || destination.pitch % 16) {
        std::snprintf(last_error, sizeof(last_error),
                      "%s pitch is not a multiple of 16", what);
        return false;
    }
    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.srcDevice = CUdeviceptr(source);
    copy.srcPitch = source_pitch;
    copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.dstDevice = CUdeviceptr(destination.mapped);
    copy.dstPitch = destination.pitch;
    copy.WidthInBytes = bytes_per_row;
    copy.Height = rows;
    const CUresult rc = b->cuda.memcpy_2d(&copy);
    if (rc != CUDA_SUCCESS) { fail("cuMemcpy2D(HDR plane)", rc); return false; }
    return true;
}

bool read_plane_to_host(Bridge* b, const Plane& plane, uint8_t* destination,
                        size_t row_stride, const char* what) {
    if (!plane.resource) { say("readback plane is not allocated for this bridge purpose"); return false; }
    if (!destination) { say("HDR readback destination is null"); return false; }
    if (row_stride < plane.bytes_per_row) {
        std::snprintf(last_error, sizeof(last_error),
                      "%s readback row stride %zu is under the plane's %u bytes",
                      what, row_stride, plane.bytes_per_row);
        return false;
    }
    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.srcDevice = CUdeviceptr(plane.mapped);
    copy.srcPitch = plane.pitch;
    copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    copy.dstHost = destination;
    copy.dstPitch = row_stride;
    copy.WidthInBytes = plane.bytes_per_row;
    copy.Height = plane.rows;
    const CUresult rc = b->cuda.memcpy_2d(&copy);
    if (rc != CUDA_SUCCESS) { fail("cuMemcpy2D(HDR readback)", rc); return false; }
    /* Image bytes, counted as image bytes: this is the counter that says a
     * diagnostic run copied planes to the host, and a production run that binds
     * none of the gpu_hdr_readback_* entries leaves it at zero. */
    b->counters_state.report_readback_bytes += plane.pitch * plane.rows;
    return true;
}

uint32_t groups(uint32_t extent, uint32_t threads) {
    return (extent + threads - 1u) / threads;
}

/* FFmpeg's AVColorRange / AVColorPrimaries / AVColorTransferCharacteristic /
 * AVColorSpace values, named here so the decode entry can refuse a domain it
 * has no maths for without dragging the whole AV headers into a GPU bridge. */
constexpr int32_t kRangeLimited = 1;
constexpr int32_t kRangeFull = 2;
constexpr int32_t kPrimariesBt2020 = 9;
constexpr int32_t kTransferPq = 16;
// AVCOL_SPC_BT2020_NCL is 9 in libavutil/pixfmt.h; 10 is AVCOL_SPC_BT2020_CL,
// a constant-luminance space this bridge has no maths for.
constexpr int32_t kSpaceBt2020Ncl = 9;
constexpr int32_t kPrimariesBt709 = 1;
constexpr int32_t kTransferBt709 = 1;
constexpr int32_t kSpaceBt709 = 1;

}  // namespace

/* --- Exported ABI -------------------------------------------------------- */

struct GpuHdrBridge {};

extern "C" GPU_HDR_API uint32_t gpu_hdr_abi_version(void) {
    return GPU_HDR_ABI_VERSION;
}

extern "C" GPU_HDR_API uint32_t gpu_sdr10_abi_version(void) {
    return GPU_HDR_ABI_VERSION;
}

extern "C" GPU_HDR_API const char* gpu_hdr_last_error(void) {
    return last_error;
}

extern "C" GPU_HDR_API void gpu_hdr_fill_contract(GpuHdrContract* out) {
    if (!out || out->struct_size != sizeof(GpuHdrContract)) {
        say("GpuHdrContract struct_size is wrong");
        return;
    }
    derive_contract();
    std::memset(out, 0, sizeof(*out));
    out->struct_size = sizeof(GpuHdrContract);
    out->scrgb_per_nit = 1.0 / kReferenceWhiteNits;
    out->nits_per_scrgb = kReferenceWhiteNits;
    for (int i = 0; i < 9; ++i) {
        out->to_scrgb[i] = g_to_scrgb.m[i];
        out->to_bt2020[i] = g_to_bt2020.m[i];
    }
    out->pq_m1 = kPqM1; out->pq_m2 = kPqM2;
    out->pq_c1 = kPqC1; out->pq_c2 = kPqC2; out->pq_c3 = kPqC3;
    out->kr = g_kr; out->kg = g_kg; out->kb = g_kb;
    out->proxy_knee = kProxyKnee;
    out->proxy_shoulder_span = kProxySpan;
    out->proxy_shoulder_k = kProxyK;
    out->proxy_saturation_relative = proxy_saturation_relative();
    out->proxy_actionable_relative = first_dead_relative();
    if (out->proxy_actionable_relative <= 0.0)
        out->proxy_actionable_relative = kProxyActionableFallback;
    out->max_stops = kMaxStops;
    out->min_slope_per_stop = kMinSlopePerStop;
}

extern "C" GPU_HDR_API void gpu_sdr10_contract(GpuHdrContract* out) {
    if (!out || out->struct_size != sizeof(GpuHdrContract)) {
        say("GpuHdrContract struct_size is wrong");
        return;
    }
    std::memset(out, 0, sizeof(*out));
    out->struct_size = sizeof(GpuHdrContract);
    /* SDR10 values are relative linear Rec.709, with no nits/scRGB mapping. */
    out->scrgb_per_nit = 0.0;
    out->nits_per_scrgb = 0.0;
    for (int i = 0; i < 9; ++i) {
        out->to_scrgb[i] = (i % 4 == 0) ? 1.0 : 0.0;
        out->to_bt2020[i] = (i % 4 == 0) ? 1.0 : 0.0;
    }
    out->kr = 0.2126; out->kg = 0.7152; out->kb = 0.0722;
    out->proxy_knee = kProxyKnee;
    out->proxy_shoulder_span = kProxySpan;
    out->proxy_shoulder_k = kProxyK;
    out->proxy_saturation_relative = proxy_saturation_relative();
    out->proxy_actionable_relative = first_dead_relative();
    if (out->proxy_actionable_relative <= 0.0)
        out->proxy_actionable_relative = kProxyActionableFallback;
    out->max_stops = kMaxStops;
    out->min_slope_per_stop = kMinSlopePerStop;
}

extern "C" GPU_HDR_API double gpu_hdr_paper_white(const GpuHdrBridge* bridge) {
    const Bridge* b = reinterpret_cast<const Bridge*>(bridge);
    return b ? b->paper_white : 0.0;
}

static int32_t gpu_hdr_open_impl(
    void* d3d12_device, void* cuda_context, uint32_t full_width,
    uint32_t full_height, uint32_t working_width, uint32_t working_height,
    const GpuHdrTapTable* downscale_plan, const GpuHdrTapTable* reconstruct_plan,
    double paper_white_scrgb, GpuHdrBridge** out_bridge, bool sdr10_mode,
    uint32_t purpose = GPU_HDR_PURPOSE_FULL) {
    error_reset();
    if (out_bridge) *out_bridge = nullptr;
    if (purpose > GPU_HDR_PURPOSE_ENCODE422) { say("unknown HDR bridge purpose"); return -1; }
    if (sdr10_mode && paper_white_scrgb != 1.0) {
        say("SDR10 reference white is fixed at 1.0 relative-linear unit");
        return -1;
    }
    if (out_bridge) *out_bridge = nullptr;
    if (!d3d12_device || !cuda_context || !out_bridge) {
        say("gpu_hdr_open needs a device, a context and an output slot");
        return -1;
    }
    derive_contract();
    /* Both extents are multiples of 8: the 4:2:0 box means need even rows, and
     * a row of P010 or FP16 has to be a multiple of 16 bytes for a pitched
     * CUDA copy. */
    if (!full_width || !full_height || !working_width || !working_height ||
        full_width % 8 || full_height % 8 || working_width % 8 ||
        working_height % 8 || full_width > 16384u || full_height > 16384u) {
        say("HDR extents must be non-zero multiples of 8 up to 16384");
        return -1;
    }
    if (!std::isfinite(paper_white_scrgb) || paper_white_scrgb < 1.0) {
        say("paperWhite must be a finite scRGB white of at least 1.0");
        return -1;
    }
    const GpuHdrContract contract_probe = [&]() {
        GpuHdrContract c{sizeof(GpuHdrContract)};
        gpu_hdr_fill_contract(&c);
        return c;
    }();
    if (!downscale_plan || !reconstruct_plan) {
        say("gpu_hdr_open needs both tap plans");
        return -1;
    }
    const GpuHdrTapTable* plans[2] = {downscale_plan, reconstruct_plan};
    for (uint32_t i = 0; i < 2; ++i) {
        const GpuHdrTapTable& p = *plans[i];
        const bool down = i == 0;
        if (p.struct_size != sizeof(GpuHdrTapTable) || !p.taps ||
            p.taps > GPU_HDR_MAX_TAPS || !p.vertical_indices ||
            !p.vertical_weights || !p.horizontal_indices ||
            !p.horizontal_weights) {
            std::snprintf(last_error, sizeof(last_error),
                "%s plan is unusable (taps %u, struct_size %u)",
                down ? "downscale" : "reconstruct", p.taps, p.struct_size);
            return -1;
        }
        const uint32_t want_v_src = down ? full_height : working_height;
        const uint32_t want_v_dst = down ? working_height : full_height;
        const uint32_t want_h_src = down ? full_width : working_width;
        const uint32_t want_h_dst = down ? working_width : full_width;
        if (p.vertical_src_len != want_v_src ||
            p.vertical_dst_len != want_v_dst ||
            p.horizontal_src_len != want_h_src ||
            p.horizontal_dst_len != want_h_dst) {
            std::snprintf(last_error, sizeof(last_error),
                "%s plan runs %u->%u x %u->%u, the job needs %u->%u x %u->%u",
                down ? "downscale" : "reconstruct", p.vertical_src_len,
                p.vertical_dst_len, p.horizontal_src_len,
                p.horizontal_dst_len, want_v_src, want_v_dst, want_h_src,
                want_h_dst);
            return -1;
        }
    }

    auto* b = new (std::nothrow) Bridge;
    if (!b) { say("out of memory for the HDR bridge"); return -1; }
    b->device = static_cast<ID3D12Device*>(d3d12_device);
    // ComPtr assignment already holds one reference; its destructor releases it.
    b->context = static_cast<CUcontext>(cuda_context);
    b->full_w = full_width;
    b->full_h = full_height;
    b->low_w = working_width;
    b->low_h = working_height;
    b->down_taps = downscale_plan->taps;
    b->rec_taps = reconstruct_plan->taps;
    b->paper_white = paper_white_scrgb;
    b->sdr10_mode = sdr10_mode;
    b->purpose = purpose;
    for (uint32_t p = 0; p < kPassCount; ++p)
        b->pass_enabled[p] = purpose == GPU_HDR_PURPOSE_FULL ||
            ((purpose == GPU_HDR_PURPOSE_DECODE || purpose == GPU_HDR_PURPOSE_DECODE422) && p == kPassDecode) ||
            (purpose == GPU_HDR_PURPOSE_GUIDANCE &&
                (p == kPassVertical || p == kPassHorizontal || p == kPassProxy)) ||
            ((purpose == GPU_HDR_PURPOSE_ENCODE || purpose == GPU_HDR_PURPOSE_ENCODE422) && p == kPassEncode);
    b->actionable_relative = contract_probe.proxy_actionable_relative > 0.0
        ? contract_probe.proxy_actionable_relative : kProxyActionableFallback;
    b->saturation_relative = contract_probe.proxy_saturation_relative > 0.0
        ? contract_probe.proxy_saturation_relative : kProxySaturationFallback;
    /* The two band limits have to say the same thing about one curve: the dead
     * ground starts above paper white and before the code pins, and the
     * closed-form saturation point has to sit inside the one grid step the
     * enumeration leaves for it. A derived constant is validated here against
     * the half grid rather than trusted. */
    if (!(b->actionable_relative > 1.0 && b->actionable_relative <
          b->saturation_relative && b->saturation_relative < 2.0)) {
        say("the proxy's measurable band is not inside [1, 2) as expected");
        delete b;
        return -1;
    }
    const double unpinned = largest_unpinned_relative();
    const double grid_step = std::ldexp(1.0, -10);
    if (!(unpinned < b->saturation_relative &&
          b->saturation_relative < unpinned + grid_step)) {
        std::snprintf(last_error, sizeof(last_error),
            "saturation point %g is not bracketed by the half grid (%g is the "
            "largest unpinned level)", b->saturation_relative, unpinned);
        delete b;
        return -1;
    }
    /* Whether this paperWhite can actually cover the domain the caller declares
     * is only knowable once the ceiling is declared, so that gate is in decode. */

    if (!load_cuda(&b->cuda)) { delete b; return -1; }
    {
        ContextScope scope(&b->cuda, b->context);
        if (!scope.ok) { delete b; return -1; }
        CUdevice cuda_device = 0;
        char luid[8]{};
        unsigned int node_mask = 0;
        CUresult rc = b->cuda.ctx_device(&cuda_device);
        if (rc == CUDA_SUCCESS)
            rc = b->cuda.device_luid(luid, &node_mask, cuda_device);
        if (rc != CUDA_SUCCESS) {
            fail("cuDeviceGetLuid(HDR)", rc);
            delete b;
            return -1;
        }
        const LUID dx = b->device->GetAdapterLuid();
        if (std::memcmp(luid, &dx, sizeof(luid)) != 0) {
            say("CUDA context and D3D12 device are different adapters");
            delete b;
            return -1;
        }
    }

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    HRESULT hr = b->device->CreateCommandQueue(&queue_desc,
        IID_PPV_ARGS(&b->queue));
    if (FAILED(hr)) { fail("create HDR command queue", hr); delete b; return -1; }
    hr = b->device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_PPV_ARGS(&b->allocator));
    if (FAILED(hr)) { fail("create HDR allocator", hr); delete b; return -1; }
    hr = b->device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
        b->allocator.Get(), nullptr, IID_PPV_ARGS(&b->list));
    if (FAILED(hr)) { fail("create HDR command list", hr); delete b; return -1; }
    hr = b->list->Close();
    if (FAILED(hr)) { fail("close HDR command list", hr); delete b; return -1; }
    hr = b->device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
        IID_PPV_ARGS(&b->fence));
    if (FAILED(hr)) { fail("create HDR fence", hr); delete b; return -1; }
    b->fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!b->fence_event) { fail("CreateEvent(HDR)", GetLastError());
        delete b; return -1; }

    layout_planes(b, *downscale_plan, *reconstruct_plan);
    if (b->failed) { delete b; return -1; }
    layout_passes(b);
    if (FAILED(make_root_signature(b))) { delete b; return -1; }
    if (FAILED(make_heap(b))) { delete b; return -1; }
    for (uint32_t p = 0; p < kPassCount; ++p) {
        if (!b->pass_enabled[p]) continue;
        const std::string source = pass_source(PassId(p), sdr10_mode, purpose == GPU_HDR_PURPOSE_DECODE422 || purpose == GPU_HDR_PURPOSE_ENCODE422);
        if (!build_pipeline(b, PassId(p), source.c_str(),
                            b->bindings[p].entry)) {
            delete b;
            return -1;
        }
    }
    if (b->planes[kPlaneTaps].resource &&
        FAILED(upload_taps(b, *downscale_plan, *reconstruct_plan))) {
        delete b;
        return -1;
    }
    *out_bridge = reinterpret_cast<GpuHdrBridge*>(b);
    return 0;
}

extern "C" GPU_HDR_API int32_t gpu_hdr_open(
    void* d3d12_device, void* cuda_context, uint32_t full_width,
    uint32_t full_height, uint32_t working_width, uint32_t working_height,
    const GpuHdrTapTable* downscale_plan, const GpuHdrTapTable* reconstruct_plan,
    double paper_white_scrgb, GpuHdrBridge** out_bridge) {
    return gpu_hdr_open_impl(d3d12_device, cuda_context, full_width, full_height,
        working_width, working_height, downscale_plan, reconstruct_plan,
        paper_white_scrgb, out_bridge, false);
}

extern "C" GPU_HDR_API int32_t gpu_sdr10_open(
    void* d3d12_device, void* cuda_context, uint32_t full_width,
    uint32_t full_height, uint32_t working_width, uint32_t working_height,
    const GpuHdrTapTable* downscale_plan, const GpuHdrTapTable* reconstruct_plan,
    double paper_white_scrgb, GpuHdrBridge** out_bridge) {
    return gpu_hdr_open_impl(d3d12_device, cuda_context, full_width, full_height,
        working_width, working_height, downscale_plan, reconstruct_plan,
        paper_white_scrgb, out_bridge, true);
}

extern "C" GPU_HDR_API int32_t gpu_hdr_open_purpose(
    void* device, void* context, uint32_t width, uint32_t height,
    uint32_t low_width, uint32_t low_height, const GpuHdrTapTable* down,
    const GpuHdrTapTable* up, double white, uint32_t purpose, GpuHdrBridge** out) {
    return gpu_hdr_open_impl(device, context, width, height, low_width, low_height,
        down, up, white, out, false, purpose);
}
extern "C" GPU_HDR_API int32_t gpu_sdr10_open_purpose(
    void* device, void* context, uint32_t width, uint32_t height,
    uint32_t low_width, uint32_t low_height, const GpuHdrTapTable* down,
    const GpuHdrTapTable* up, double white, uint32_t purpose, GpuHdrBridge** out) {
    return gpu_hdr_open_impl(device, context, width, height, low_width, low_height,
        down, up, white, out, true, purpose);
}
// DEFAULT heap allocations, including padding/alignment; excludes CPU counter heaps.
extern "C" GPU_HDR_API uint64_t gpu_hdr_allocated_device_bytes(GpuHdrBridge* handle) {
    Bridge* b = reinterpret_cast<Bridge*>(handle);
    if (!b) return 0;
    uint64_t bytes = 0;
    for (uint32_t i = 0; i < kPlaneCount; ++i) {
        if (i == kPlaneInitPattern || i == kPlaneCounterReadback || !b->planes[i].resource) continue;
        auto desc = b->planes[i].resource->GetDesc();
        bytes += b->device->GetResourceAllocationInfo(0, 1, &desc).SizeInBytes;
    }
    return bytes;
}

/* The ceiling the composite stage clamps to, which is the larger of what the
 * caller declared the content reaches and what the encode will clip to: a
 * sample inside the declared domain must never be clamped on its way through
 * the reconstruction, because that would be the bridge quietly tone-mapping. */
double composite_ceiling(const Bridge* b) {
    const double support = b->support_ceiling > 0.0 ? b->support_ceiling : 0.0;
    const double clip = b->clip_ceiling > 0.0 ? b->clip_ceiling : 0.0;
    const double top = support > clip ? support : clip;
    return top > 0.0 ? top : kScrgbPerRelative;
}

static int32_t gpu_decode_impl(
    GpuHdrBridge* handle, uint64_t p010_y_ptr, size_t p010_y_pitch,
    uint64_t p010_uv_ptr, size_t p010_uv_pitch, int32_t color_range,
    int32_t color_primaries, int32_t color_transfer, int32_t color_space,
    double support_ceiling_scrgb, uint64_t* out_scrgb_ptr,
    uint32_t* out_scrgb_pitch, bool sdr10_mode) {
    Bridge* b = reinterpret_cast<Bridge*>(handle);
    error_reset();
    if (!b) { say("gpu_hdr_decode on a closed bridge"); return -1; }
    if (!b->pass_enabled[kPassDecode]) { say("decode is unavailable for this bridge purpose"); return -1; }
    if (b->sdr10_mode != sdr10_mode) {
        say(sdr10_mode ? "gpu_sdr10_decode needs an SDR10 bridge" :
                         "gpu_hdr_decode refuses an SDR10 bridge");
        return -1;
    }
    if (!p010_y_ptr || !p010_uv_ptr) {
        say("HDR decode needs both borrowed P010 planes");
        return -1;
    }
    if (!std::isfinite(support_ceiling_scrgb) || support_ceiling_scrgb <= 0.0) {
        say("the declared support ceiling must be a finite positive scRGB value");
        return -1;
    }
    if (sdr10_mode && support_ceiling_scrgb > 2.0) {
        say("SDR10 support ceiling must be at most 2.0 relative-linear units");
        return -1;
    }
    /* The declared domain is the job's claim about its own content, and the
     * proxy's measurable band is [0, paperWhite * actionable). A domain that
     * reaches past the band is not a configuration error: the channels above it
     * arrive pinned or in the dead ground, where the reconstruction keeps the
     * master exactly and says so in its own counters. What *is* a failure is
     * content that lands above the declared ceiling, because then the metadata
     * was a lie and the correction would be applied to a picture the job never
     * said it carried - so the source report counts it and the caller aborts. */
    /* The one domain this stage has maths for. HLG and an
     * untagged source are refused here rather than assumed, because guessing a
     * transfer function is how an HDR picture quietly becomes a different one. */
    const bool color_match = sdr10_mode
        ? (color_range == kRangeLimited && color_primaries == kPrimariesBt709 &&
           color_transfer == kTransferBt709 && color_space == kSpaceBt709)
        : ((color_range == kRangeLimited || color_range == kRangeFull) &&
           color_primaries == kPrimariesBt2020 &&
           color_transfer == kTransferPq && color_space == kSpaceBt2020Ncl);
    if (!color_match) {
        std::snprintf(last_error, sizeof(last_error),
            "%s decode rejected range/primaries/transfer/space "
            "range %d primaries %d transfer %d space %d",
            sdr10_mode ? "SDR10" : "HDR", color_range, color_primaries,
            color_transfer, color_space);
        return -1;
    }
    Plane& dst = b->planes[kPlaneMaster];
    ContextScope scope(&b->cuda, b->context);
    if (!scope.ok) return -1;
    const uint32_t luma_bytes = b->full_w * 2u;
    const uint32_t uv_rows = b->purpose == GPU_HDR_PURPOSE_DECODE422 ? b->full_h : b->full_h / 2u;
    if (!copy_device_to_device(b, p010_y_ptr, uint32_t(p010_y_pitch),
            b->planes[kPlaneSourceY], luma_bytes, b->full_h, "source luma") ||
        !copy_device_to_device(b, p010_uv_ptr, uint32_t(p010_uv_pitch),
            b->planes[kPlaneSourceUV], luma_bytes, uv_rows,
            "source colour difference"))
        return -1;
    b->support_ceiling = support_ceiling_scrgb;
    Constants c;
    c.put(b->full_w); c.put(b->full_h);
    c.put(b->planes[kPlaneSourceY].pitch);
    c.put(b->planes[kPlaneSourceUV].pitch);
    c.put(dst.pitch); c.put(color_range == kRangeFull ? 1u : 0u); c.put(0u); c.put(0u);
    c.put(float(support_ceiling_scrgb));
    if (!run_pass(b, kPassDecode, c, groups(b->full_w, 8u),
                  groups(b->full_h, 8u)))
        return -1;
    b->counters_state.process_count += 1;
    b->counters_state.source_device_copy_bytes +=
        uint64_t(luma_bytes) * b->full_h + uint64_t(luma_bytes) * uv_rows;
    b->counters_state.source_device_write_bytes +=
        uint64_t(b->full_w) * b->full_h * 8u;
    if (counter(b, kPassDecode, kSrcNonfinite) != 0) {
        std::snprintf(last_error, sizeof(last_error),
            "%u decoded components do not fit the half grid; the source is not "
            "the domain this stage declares",
            counter(b, kPassDecode, kSrcNonfinite));
        return -1;
    }
    /* A master this bridge derived its own planes from, and the decode pass is
     * the only thing that measures one: gpu_hdr_get_source_report below reports
     * this frame, which is what master_provenance is for. */
    b->master_epoch += 1;
    b->multiscale.master_provenance = 1u;
    if (out_scrgb_ptr) *out_scrgb_ptr = dst.mapped;
    if (out_scrgb_pitch) *out_scrgb_pitch = dst.pitch;
    return 0;
}

extern "C" GPU_HDR_API int32_t gpu_hdr_decode(
    GpuHdrBridge* bridge, uint64_t p010_y_ptr, size_t p010_y_pitch,
    uint64_t p010_uv_ptr, size_t p010_uv_pitch, int32_t color_range,
    int32_t color_primaries, int32_t color_transfer, int32_t color_space,
    double support_ceiling_scrgb, uint64_t* out_scrgb_ptr,
    uint32_t* out_scrgb_pitch) {
    return gpu_decode_impl(bridge, p010_y_ptr, p010_y_pitch, p010_uv_ptr,
        p010_uv_pitch, color_range, color_primaries, color_transfer, color_space,
        support_ceiling_scrgb, out_scrgb_ptr, out_scrgb_pitch, false);
}

extern "C" GPU_HDR_API int32_t gpu_sdr10_decode(
    GpuHdrBridge* bridge, uint64_t p010_y_ptr, size_t p010_y_pitch,
    uint64_t p010_uv_ptr, size_t p010_uv_pitch, int32_t color_range,
    int32_t color_primaries, int32_t color_transfer, int32_t color_space,
    double support_ceiling_scrgb, uint64_t* out_scrgb_ptr,
    uint32_t* out_scrgb_pitch) {
    return gpu_decode_impl(bridge, p010_y_ptr, p010_y_pitch, p010_uv_ptr,
        p010_uv_pitch, color_range, color_primaries, color_transfer, color_space,
        support_ceiling_scrgb, out_scrgb_ptr, out_scrgb_pitch, true);
}

extern "C" GPU_HDR_API int32_t gpu_hdr_downscale(
    GpuHdrBridge* handle, uint64_t* out_working_ptr,
    uint32_t* out_working_pitch) {
    Bridge* b = reinterpret_cast<Bridge*>(handle);
    error_reset();
    if (!b) { say("gpu_hdr_downscale on a closed bridge"); return -1; }
    Plane& working = b->planes[kPlaneWorking];
    Constants v;
    v.put(b->full_w); v.put(b->full_h); v.put(b->low_h);
    v.put(b->planes[kPlaneMaster].pitch);
    v.put(b->down_taps); v.put(0u); v.put(0u); v.put(0u);
    if (!run_pass(b, kPassVertical, v, groups(b->full_w, 16u),
                  groups(b->low_h, 8u)))
        return -1;
    Constants h;
    h.put(b->low_w); h.put(b->low_h); h.put(b->full_w); h.put(working.pitch);
    h.put(b->down_taps); h.put(0u); h.put(0u); h.put(0u);
    if (!run_pass(b, kPassHorizontal, h, groups(b->low_w, 16u),
                  groups(b->low_h, 8u)))
        return -1;
    b->counters_state.working_device_write_bytes +=
        uint64_t(b->low_w) * b->low_h * 8u;
    b->working_epoch = b->master_epoch;
    /* A new working plane, so no proxy that exists now describes it. The proxy
     * has to run again for gpu_hdr_observe to take an observation. */
    b->working_generation += 1;
    if (out_working_ptr) *out_working_ptr = working.mapped;
    if (out_working_pitch) *out_working_pitch = working.pitch;
    return 0;
}

extern "C" GPU_HDR_API int32_t gpu_hdr_proxy(
    GpuHdrBridge* handle, uint64_t* out_proxy_ptr, uint32_t* out_proxy_pitch,
    uint64_t* out_guidance_ptr, uint32_t* out_guidance_pitch) {
    Bridge* b = reinterpret_cast<Bridge*>(handle);
    error_reset();
    if (!b) { say("gpu_hdr_proxy on a closed bridge"); return -1; }
    Plane& proxy = b->planes[kPlaneProxy];
    Plane& guidance = b->planes[kPlaneGuidance];
    Constants c;
    c.put(b->low_w); c.put(b->low_h);
    c.put(b->planes[kPlaneWorking].pitch); c.put(proxy.pitch);
    c.put(guidance.pitch); c.put(0u); c.put(0u); c.put(0u);
    c.put(float(b->paper_white));
    /* The shader's two band limits are relative to paper white, which is how
     * the contract publishes them and how the dead-zone test reads them; the
     * guidance ceiling is the one place that needs them in scRGB, and the
     * kernel multiplies by PaperWhite there itself. */
    c.put(float(b->actionable_relative));
    c.put(float(b->saturation_relative));
    if (!run_pass(b, kPassProxy, c, groups(b->low_w, 8u), groups(b->low_h, 8u)))
        return -1;
    b->counters_state.working_device_write_bytes +=
        uint64_t(b->low_w) * b->low_h * 8u;
    b->counters_state.guidance_device_write_bytes +=
        uint64_t(b->low_w) * b->low_h * 4u;
    b->proxy_generation = b->working_generation;
    if (out_proxy_ptr) *out_proxy_ptr = proxy.mapped;
    if (out_proxy_pitch) *out_proxy_pitch = proxy.pitch;
    if (out_guidance_ptr) *out_guidance_ptr = guidance.mapped;
    if (out_guidance_pitch) *out_guidance_pitch = guidance.pitch;
    return 0;
}

/* The Feature 18 output, in this bridge's own plane. The borrowed texture is
 * only ever a copy source and is never transitioned, which is the same rule the
 * encoder's staging path follows.
 *
 * One contract object says how a borrowed producer is described and what it owes
 * the bridge, so the two callers of the same copy mechanism cannot drift into
 * checking different things about it. */
struct BorrowedCopy {
    const char* producer;       /* the noun the refusals name */
    const char* extent_name;    /* how the refusals name the extent */
    const char* missing_error;  /* full message when no texture was handed over */
    const char* wait_step;      /* named step for the queue wait */
    const char* copy_step;      /* named step for the completion wait */
    bool require_fence;         /* a null fence is a refusal, not an assumption */
};

bool copy_texture_to_plane(Bridge* b, const BorrowedCopy& contract,
                           void* raw_texture, void* raw_fence,
                           uint64_t fence_value, Plane& destination,
                           uint32_t width, uint32_t height, uint64_t* counted) {
    auto* texture = static_cast<ID3D12Resource*>(raw_texture);
    if (!destination.resource) { say("texture destination is unavailable for this bridge purpose"); return false; }
    if (!texture) { say(contract.missing_error); return false; }
    const D3D12_RESOURCE_DESC desc = texture->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
        desc.Format != DXGI_FORMAT_R16G16B16A16_FLOAT ||
        desc.Width != width || desc.Height != height ||
        desc.DepthOrArraySize != 1 || desc.MipLevels != 1 ||
        desc.SampleDesc.Count != 1) {
        std::snprintf(last_error, sizeof(last_error),
                      "%s is not an RGBA16_FLOAT 2D texture at %s",
                      contract.producer, contract.extent_name);
        return false;
    }
    ComPtr<ID3D12Device> texture_device;
    HRESULT hr = texture->GetDevice(IID_PPV_ARGS(texture_device.GetAddressOf()));
    if (FAILED(hr) || texture_device.Get() != b->device.Get()) {
        std::snprintf(last_error, sizeof(last_error),
                      "%s belongs to another D3D12 device", contract.producer);
        return false;
    }
    auto* fence = static_cast<ID3D12Fence*>(raw_fence);
    if (!fence || !fence_value) {
        if (contract.require_fence) {
            std::snprintf(last_error, sizeof(last_error),
                          "%s needs its producer fence and value: the copy would "
                          "race a producer on another queue", contract.producer);
            return false;
        }
    } else {
        ComPtr<ID3D12Device> fence_device;
        hr = fence->GetDevice(IID_PPV_ARGS(fence_device.GetAddressOf()));
        if (FAILED(hr) || fence_device.Get() != b->device.Get()) {
            say("the producer fence belongs to another D3D12 device");
            return false;
        }
        const HRESULT waited = b->queue->Wait(fence, fence_value);
        if (FAILED(waited)) { fail(contract.wait_step, waited); return false; }
    }
    hr = b->allocator->Reset();
    if (SUCCEEDED(hr)) hr = b->list->Reset(b->allocator.Get(), nullptr);
    if (FAILED(hr)) { fail("reset HDR copy list", hr); return false; }
    ID3D12GraphicsCommandList* list = b->list.Get();
    transition(list, destination, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
    dst.pResource = destination.resource.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint.Offset = 0;
    dst.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    dst.PlacedFootprint.Footprint.Width = width;
    dst.PlacedFootprint.Footprint.Height = height;
    dst.PlacedFootprint.Footprint.Depth = 1;
    dst.PlacedFootprint.Footprint.RowPitch = destination.pitch;
    src.pResource = texture;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.SubresourceIndex = 0;
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    hr = list->Close();
    if (FAILED(hr)) { fail("close HDR copy list", hr); return false; }
    ID3D12CommandList* commands[] = {list};
    b->queue->ExecuteCommandLists(1, commands);
    hr = b->queue->Signal(b->fence.Get(), ++b->fence_value);
    if (FAILED(hr)) { fail("signal HDR copy", hr); return false; }
    if (!wait_gpu(b, contract.copy_step)) return false;
    *counted += uint64_t(width) * height * 8u;
    return true;
}

const BorrowedCopy kFeature18Copy = {
    "Feature 18 output", "the working extent",
    "HDR reconstruction needs the Feature 18 output",
    "queue wait Feature 18 fence", "Feature 18 output copy", false};

bool copy_neural(Bridge* b, void* raw_texture, void* raw_fence,
                 uint64_t fence_value) {
    return copy_texture_to_plane(b, kFeature18Copy, raw_texture, raw_fence,
                                 fence_value, b->planes[kPlaneNeural], b->low_w,
                                 b->low_h,
                                 &b->counters_state.working_device_write_bytes);
}

/* The one residual pass that gpu_hdr_reconstruct and gpu_hdr_observe share: the
 * same three inputs, the same owned residual plane, the same constants and the
 * same dispatch extents, so an observation is the same number the single-layer
 * path computes rather than a second opinion of it. */
bool run_residual_pass(Bridge* b, bool preserve_detail = true) {
    Constants r;
    r.put(b->low_w); r.put(b->low_h);
    r.put(b->planes[kPlaneNeural].pitch);
    r.put(b->planes[kPlaneProxy].pitch);
    r.put(b->planes[kPlaneWorking].pitch);
    r.put(b->planes[kPlaneResidual].pitch);
    r.put(preserve_detail ? 1u : 0u); r.put(0u);
    r.put(float(b->paper_white)); r.put(float(kMaxStops));
    r.put(float(kMinSlopePerStop));
    if (!run_pass(b, kPassResidual, r, groups(b->low_w, 8u),
                  groups(b->low_h, 8u)))
        return false;
    b->counters_state.working_device_write_bytes +=
        uint64_t(b->low_w) * b->low_h * 8u;
    return true;
}

int32_t reconstruct_picture(
    GpuHdrBridge* handle, void* dlss_texture, void* dlss_fence,
    uint64_t dlss_fence_value, double residual_strength,
    uint64_t* out_composited_ptr, uint32_t* out_composited_pitch, bool preserve_detail) {
    Bridge* b = reinterpret_cast<Bridge*>(handle);
    error_reset();
    if (!b) { say("gpu_hdr_reconstruct on a closed bridge"); return -1; }
    if (!std::isfinite(residual_strength) || residual_strength < 0.0 ||
        residual_strength > (b->sdr10_mode ? 1.5 : 1.0)) {
        say("residual strength is outside the bridge's supported range");
        return -1;
    }
    if (!copy_neural(b, dlss_texture, dlss_fence, dlss_fence_value)) return -1;
    if (!run_residual_pass(b, preserve_detail)) return -1;
    Constants c;
    c.put(b->full_w); c.put(b->full_h); c.put(b->low_w); c.put(b->low_h);
    c.put(b->planes[kPlaneResidual].pitch);
    c.put(b->planes[kPlaneMaster].pitch);
    c.put(b->planes[kPlaneComposited].pitch);
    c.put(b->rec_taps);
    c.put(float(residual_strength)); c.put(float(composite_ceiling(b)));
    c.put(preserve_detail ? 1u : 0u);
    if (!run_pass(b, kPassComposite, c, groups(b->full_w, 8u),
                  groups(b->full_h, 8u)))
        return -1;
    b->counters_state.composited_device_write_bytes +=
        uint64_t(b->full_w) * b->full_h * 8u;
    if (out_composited_ptr) *out_composited_ptr = b->planes[kPlaneComposited].mapped;
    if (out_composited_pitch)
        *out_composited_pitch = b->planes[kPlaneComposited].pitch;
    return 0;
}

extern "C" GPU_HDR_API int32_t gpu_hdr_reconstruct(
    GpuHdrBridge* handle, void* texture, void* fence, uint64_t value,
    double strength, uint64_t* pointer, uint32_t* pitch) {
    return reconstruct_picture(handle, texture, fence, value, strength, pointer, pitch, true);
}
extern "C" GPU_HDR_API uint32_t gpu_sdr10_nr_color_abi_version(void) { return 1u; }
extern "C" GPU_HDR_API int32_t gpu_sdr10_reconstruct(
    GpuHdrBridge* handle, void* texture, void* fence, uint64_t value,
    double strength, uint32_t preserve_detail, uint64_t* pointer, uint32_t* pitch) {
    Bridge* b = reinterpret_cast<Bridge*>(handle);
    if (!b || !b->sdr10_mode || preserve_detail > 1u) {
        say("gpu_sdr10_reconstruct needs an SDR10 bridge and a boolean detail mode"); return -1;
    }
    return reconstruct_picture(handle, texture, fence, value, strength, pointer, pitch, preserve_detail != 0);
}

extern "C" GPU_HDR_API int32_t gpu_hdr_encode(
    GpuHdrBridge* handle, double clip_ceiling_scrgb, uint64_t* out_y_ptr,
    uint32_t* out_y_pitch, uint64_t* out_uv_ptr, uint32_t* out_uv_pitch) {
    Bridge* b = reinterpret_cast<Bridge*>(handle);
    error_reset();
    if (!b) { say("gpu_hdr_encode on a closed bridge"); return -1; }
    if (!std::isfinite(clip_ceiling_scrgb) || clip_ceiling_scrgb <= 0.0) {
        say("the encode clip ceiling must be a finite positive scRGB value");
        return -1;
    }
    b->clip_ceiling = clip_ceiling_scrgb;
    Constants c;
    c.put(b->full_w); c.put(b->full_h);
    c.put(b->planes[kPlaneComposited].pitch);
    c.put(b->planes[kPlaneEncodeY].pitch);
    c.put(b->planes[kPlaneEncodeUV].pitch);
    c.put(0u); c.put(0u); c.put(0u);
    c.put(float(clip_ceiling_scrgb));
    if (!run_pass(b, kPassEncode, c, groups(b->full_w / 2u, 8u),
                  groups((b->purpose == GPU_HDR_PURPOSE_ENCODE422 ? b->full_h : b->full_h / 2u), 8u)))
        return -1;
    const uint32_t output_uv_rows = b->purpose == GPU_HDR_PURPOSE_ENCODE422
        ? b->full_h : b->full_h / 2u;
    b->counters_state.encode_device_write_bytes +=
        uint64_t(b->full_w) * b->full_h * 2u +
        uint64_t(b->full_w) * output_uv_rows * 2u;
    if (out_y_ptr) *out_y_ptr = b->planes[kPlaneEncodeY].mapped;
    if (out_y_pitch) *out_y_pitch = b->planes[kPlaneEncodeY].pitch;
    if (out_uv_ptr) *out_uv_ptr = b->planes[kPlaneEncodeUV].mapped;
    if (out_uv_pitch) *out_uv_pitch = b->planes[kPlaneEncodeUV].pitch;
    return 0;
}

/* --- Reports ------------------------------------------------------------- */

extern "C" GPU_HDR_API int32_t gpu_hdr_get_source_report(
    GpuHdrBridge* handle, GpuHdrSourceReport* out_report) {
    Bridge* b = reinterpret_cast<Bridge*>(handle);
    if (!b || !out_report ||
        out_report->struct_size != sizeof(GpuHdrSourceReport)) {
        say("gpu_hdr_get_source_report needs an open bridge and a sized struct");
        return -1;
    }
    const uint32_t pixels = counter(b, kPassDecode, kSrcPixels);
    out_report->support_ceiling_scrgb = b->support_ceiling;
    out_report->paper_white_scrgb = b->paper_white;
    out_report->nonfinite_samples = counter(b, kPassDecode, kSrcNonfinite);
    out_report->negative_components = counter(b, kPassDecode, kSrcNegative);
    out_report->above_ceiling_components =
        counter(b, kPassDecode, kSrcAboveCeiling);
    out_report->min_component = pixels ? counter_float(b, kPassDecode, kSrcMinKey) : 0.0;
    out_report->max_component = pixels ? counter_float(b, kPassDecode, kSrcMaxKey) : 0.0;
    out_report->pixels = pixels;
    return 0;
}

extern "C" GPU_HDR_API int32_t gpu_hdr_get_proxy_report(
    GpuHdrBridge* handle, GpuHdrProxyReport* out_report) {
    Bridge* b = reinterpret_cast<Bridge*>(handle);
    if (!b || !out_report ||
        out_report->struct_size != sizeof(GpuHdrProxyReport)) {
        say("gpu_hdr_get_proxy_report needs an open bridge and a sized struct");
        return -1;
    }
    const uint32_t pixels = counter(b, kPassProxy, kProxyPixels);
    out_report->negative_components = counter(b, kPassProxy, kProxyNegative);
    out_report->pinned_components = counter(b, kPassProxy, kProxyPinned);
    out_report->dead_zone_components = counter(b, kPassProxy, kProxyDead);
    out_report->above_actionable_components =
        counter(b, kPassProxy, kProxyAboveActionable);
    out_report->above_saturation_components =
        counter(b, kPassProxy, kProxyAboveSaturation);
    out_report->min_code = pixels ? counter_float(b, kPassProxy, kProxyMinKey) : 0.0;
    out_report->max_code = pixels ? counter_float(b, kPassProxy, kProxyMaxKey) : 0.0;
    out_report->pixels = pixels;
    return 0;
}

extern "C" GPU_HDR_API int32_t gpu_hdr_get_reconstruct_report(
    GpuHdrBridge* handle, GpuHdrReconstructReport* out_report) {
    Bridge* b = reinterpret_cast<Bridge*>(handle);
    if (!b || !out_report ||
        out_report->struct_size != sizeof(GpuHdrReconstructReport)) {
        say("gpu_hdr_get_reconstruct_report needs an open bridge and a sized "
            "struct");
        return -1;
    }
    const uint32_t pixels = counter(b, kPassProxy, kProxyPixels);
    out_report->applied_components = counter(b, kPassResidual, kRecApplied);
    out_report->gain_cap_components = counter(b, kPassResidual, kRecCap);
    out_report->rejected_pinned_components =
        counter(b, kPassResidual, kRecRejectedPinned);
    out_report->rejected_dead_components =
        counter(b, kPassResidual, kRecRejectedDead);
    out_report->rejected_negative_components =
        counter(b, kPassResidual, kRecRejectedNegative);
    out_report->nonfinite_components = counter(b, kPassResidual, kRecNonfinite);
    out_report->max_abs_stops = pixels
        ? counter_float(b, kPassResidual, kRecMaxStopsKey) : 0.0;
    out_report->max_abs_residual_scrgb = pixels
        ? counter_float(b, kPassResidual, kRecMaxResidualKey) : 0.0;
    out_report->composite_out_of_domain_components =
        counter(b, kPassComposite, kCompOutside);
    out_report->composite_nonfinite_components =
        counter(b, kPassComposite, kCompNonfinite);
    /* The composite pass counts no pixels of its own: it runs over the whole
     * extent the bridge was opened with, and its extremes are only meaningful
     * if that extent is non-zero. */
    const uint32_t full_pixels = b->full_w * b->full_h;
    out_report->composite_min_component = full_pixels
        ? counter_float(b, kPassComposite, kCompMinKey) : 0.0;
    out_report->composite_max_component = full_pixels
        ? counter_float(b, kPassComposite, kCompMaxKey) : 0.0;
    out_report->pixels = pixels;
    return 0;
}

extern "C" GPU_HDR_API int32_t gpu_hdr_get_encode_report(
    GpuHdrBridge* handle, GpuHdrEncodeReport* out_report) {
    Bridge* b = reinterpret_cast<Bridge*>(handle);
    if (!b || !out_report ||
        out_report->struct_size != sizeof(GpuHdrEncodeReport)) {
        say("gpu_hdr_get_encode_report needs an open bridge and a sized struct");
        return -1;
    }
    const uint32_t max_y = counter(b, kPassEncode, kEncMaxCodeY);
    out_report->clip_ceiling_scrgb = b->clip_ceiling;
    out_report->clamped_high_components =
        counter(b, kPassEncode, kEncClampedHigh);
    out_report->clamped_low_components = counter(b, kPassEncode, kEncClampedLow);
    out_report->out_of_gamut_components = counter(b, kPassEncode, kEncOutOfGamut);
    out_report->nonfinite_components = counter(b, kPassEncode, kEncNonfinite);
    out_report->max_linear_component =
        counter_float(b, kPassEncode, kEncMaxLinearKey);
    out_report->max_code_y = double(max_y);
    out_report->min_code_y = max_y ? double(counter(b, kPassEncode, kEncMinCodeY)) : 0.0;
    const uint32_t max_uv = counter(b, kPassEncode, kEncMaxCodeUV);
    out_report->max_code_uv = double(max_uv);
    out_report->min_code_uv = max_uv
        ? double(counter(b, kPassEncode, kEncMinCodeUV)) : 0.0;
    return 0;
}

/* --- Planes, readbacks and counters ------------------------------------ */

namespace {
/* The header's plane numbering, which is the order the chain sees the planes. */
PlaneId plane_of(int32_t index) {
    static const PlaneId order[] = {
        kPlaneSourceY, kPlaneSourceUV, kPlaneMaster, kPlaneWorking, kPlaneProxy,
        kPlaneGuidance, kPlaneResidual, kPlaneComposited, kPlaneEncodeY,
        kPlaneEncodeUV};
    return index >= 0 && index < int32_t(sizeof(order) / sizeof(order[0]))
        ? order[index] : kPlaneCount;
}
}  // namespace

extern "C" GPU_HDR_API uint32_t gpu_hdr_plane_pitch(GpuHdrBridge* handle,
                                                   int32_t plane) {
    Bridge* b = reinterpret_cast<Bridge*>(handle);
    if (!b) return 0;
    const PlaneId id = plane_of(plane);
    return id == kPlaneCount ? 0u : b->planes[id].pitch;
}

extern "C" GPU_HDR_API uint64_t gpu_hdr_plane_bytes(GpuHdrBridge* handle,
                                                   int32_t plane) {
    Bridge* b = reinterpret_cast<Bridge*>(handle);
    if (!b) return 0;
    const PlaneId id = plane_of(plane);
    return id == kPlaneCount ? 0ull
        : uint64_t(b->planes[id].pitch) * b->planes[id].rows;
}

extern "C" GPU_HDR_API int32_t gpu_hdr_readback_source_p010(
    GpuHdrBridge* handle, uint8_t* y, size_t y_row_stride, uint8_t* uv,
    size_t uv_row_stride) {
    Bridge* b = reinterpret_cast<Bridge*>(handle);
    error_reset();
    if (!b) { say("gpu_hdr_readback_source_p010 on a closed bridge"); return -1; }
    if (!read_plane_to_host(b, b->planes[kPlaneSourceY], y, y_row_stride,
        "source luma")) return -1;
    return read_plane_to_host(b, b->planes[kPlaneSourceUV], uv, uv_row_stride,
        "source colour difference") ? 0 : -1;
}

extern "C" GPU_HDR_API int32_t gpu_hdr_readback_encoded_p010(
    GpuHdrBridge* handle, uint8_t* y, size_t y_row_stride, uint8_t* uv,
    size_t uv_row_stride) {
    Bridge* b = reinterpret_cast<Bridge*>(handle);
    error_reset();
    if (!b) { say("gpu_hdr_readback_encoded_p010 on a closed bridge"); return -1; }
    if (!read_plane_to_host(b, b->planes[kPlaneEncodeY], y, y_row_stride,
        "encoded luma")) return -1;
    return read_plane_to_host(b, b->planes[kPlaneEncodeUV], uv, uv_row_stride,
        "encoded colour difference") ? 0 : -1;
}

extern "C" GPU_HDR_API int32_t gpu_hdr_readback_scrgb(GpuHdrBridge* handle,
    int32_t plane, uint8_t* rgba, size_t row_stride) {
    Bridge* b = reinterpret_cast<Bridge*>(handle);
    error_reset();
    if (!b) { say("gpu_hdr_readback_scrgb on a closed bridge"); return -1; }
    const PlaneId id = plane_of(plane);
    if (id == kPlaneCount) { say("no such HDR plane index"); return -1; }
    return read_plane_to_host(b, b->planes[id], rgba, row_stride,
        "scRGB plane") ? 0 : -1;
}

extern "C" GPU_HDR_API int32_t gpu_hdr_get_transfer_counters(
    GpuHdrBridge* handle, GpuHdrTransferCounters* counters) {
    Bridge* b = reinterpret_cast<Bridge*>(handle);
    if (!b || !counters ||
        counters->struct_size != sizeof(GpuHdrTransferCounters)) {
        say("gpu_hdr_get_transfer_counters needs an open bridge and a sized "
            "struct");
        return -1;
    }
    *counters = b->counters_state;
    return 0;
}

/* --- Multiscale foundation ---------------------------------------------- */

extern "C" GPU_HDR_API uint32_t gpu_hdr_multiscale_abi_version(void) {
    return GPU_HDR_MULTISCALE_ABI_VERSION;
}

extern "C" GPU_HDR_API int32_t gpu_hdr_import_master(
    GpuHdrBridge* handle, uint64_t source_ptr, uint32_t source_width,
    uint32_t source_height, uint32_t source_pitch, uint64_t* out_master_ptr,
    uint32_t* out_master_pitch) {
    Bridge* b = reinterpret_cast<Bridge*>(handle);
    error_reset();
    if (!b) { say("gpu_hdr_import_master on a closed bridge"); return -1; }
    if (!source_ptr) {
        say("an imported master needs a source plane to copy from");
        return -1;
    }
    if (source_width != b->full_w || source_height != b->full_h) {
        std::snprintf(last_error, sizeof(last_error),
            "the imported master is %u x %u; this bridge's full extent is "
            "%u x %u", source_width, source_height, b->full_w, b->full_h);
        return -1;
    }
    /* R16G16B16A16_FLOAT is 8 bytes a sample, so a row of the full extent is what
     * the copy has to read; the pitch may pad it but never truncate it, and
     * copy_device_to_device refuses a pitch that is not a multiple of 16, which
     * is what a pitched cuMemcpy2D requires of both sides. */
    const uint32_t row_bytes = b->full_w * 8u;
    if (source_pitch < row_bytes) {
        std::snprintf(last_error, sizeof(last_error),
            "the imported master's pitch %u is under the %u bytes an RGBA16F row "
            "of %u samples carries", source_pitch, row_bytes, b->full_w);
        return -1;
    }
    Plane& master = b->planes[kPlaneMaster];
    ContextScope scope(&b->cuda, b->context);
    if (!scope.ok) return -1;
    /* The copy is the whole entry. The source's samples land in the bridge's own
     * master exactly as they are: signed, unclamped, and with no transfer
     * function, resampler or 8-bit step anywhere in the call. */
    if (!copy_device_to_device(b, source_ptr, source_pitch, master, row_bytes,
                              b->full_h, "imported master"))
        return -1;
    b->master_epoch += 1;
    b->multiscale.master_provenance = 2u;
    b->multiscale.master_import_count += 1;
    b->multiscale.master_import_device_bytes += uint64_t(row_bytes) * b->full_h;
    /* One frame of colour entered this bridge, by import rather than by decode,
     * and the full-resolution FP16 write is the same traffic gpu_hdr_decode's own
     * master write is. The borrowed read is not P010, so source_device_copy_bytes
     * - which is the decoder's borrowed P010 - stays untouched; the volume that
     * crossed is in the multiscale counters. */
    b->counters_state.process_count += 1;
    b->counters_state.source_device_write_bytes +=
        uint64_t(b->full_w) * b->full_h * 8u;
    if (out_master_ptr) *out_master_ptr = master.mapped;
    if (out_master_pitch) *out_master_pitch = master.pitch;
    return 0;
}

extern "C" GPU_HDR_API int32_t gpu_hdr_observe(
    GpuHdrBridge* handle, void* dlss_texture, void* dlss_fence,
    uint64_t dlss_fence_value, uint64_t* out_residual_ptr,
    uint32_t* out_residual_pitch) {
    Bridge* b = reinterpret_cast<Bridge*>(handle);
    error_reset();
    if (!b) { say("gpu_hdr_observe on a closed bridge"); return -1; }
    /* The residual is measured against the pristine working master through the
     * proxy the layer was shown, so the working plane has to be this master's
     * and the proxy has to be the one that read *that* working plane. An
     * observe on a proxy of a superseded working plane would return a signed
     * half number that reads as valid. */
    if (!b->master_epoch) {
        say("gpu_hdr_observe needs a master in this bridge: decode or import one "
            "first");
        return -1;
    }
    if (b->working_epoch != b->master_epoch) {
        say("gpu_hdr_observe needs gpu_hdr_downscale for the master this bridge "
            "now holds");
        return -1;
    }
    if (b->proxy_generation != b->working_generation) {
        say("gpu_hdr_observe needs gpu_hdr_proxy for the working plane this "
            "bridge now holds: the proxy this observation would read belongs to "
            "a superseded working plane");
        return -1;
    }
    if (!copy_neural(b, dlss_texture, dlss_fence, dlss_fence_value)) return -1;
    if (!run_residual_pass(b)) return -1;
    b->multiscale.observe_count += 1;
    b->multiscale.observe_copy_bytes += uint64_t(b->low_w) * b->low_h * 8u;
    b->multiscale.observe_residual_bytes += uint64_t(b->low_w) * b->low_h * 8u;
    /* pass_dispatches is not touched here: the one shader pass this entry runs
     * counts itself inside run_pass, alongside every other pass of this bridge. */
    if (out_residual_ptr) *out_residual_ptr = b->planes[kPlaneResidual].mapped;
    if (out_residual_pitch)
        *out_residual_pitch = b->planes[kPlaneResidual].pitch;
    return 0;
}

const BorrowedCopy kCompositeCopy = {
    "fused composite", "the full extent",
    "gpu_hdr_set_composite needs the fused composite texture",
    "queue wait fused composite fence", "fused composite copy", true};

extern "C" GPU_HDR_API int32_t gpu_hdr_set_composite(
    GpuHdrBridge* handle, void* composite_texture, void* composite_fence,
    uint64_t composite_fence_value, uint64_t* out_composited_ptr,
    uint32_t* out_composited_pitch) {
    Bridge* b = reinterpret_cast<Bridge*>(handle);
    error_reset();
    if (!b) { say("gpu_hdr_set_composite on a closed bridge"); return -1; }
    if (!b->master_epoch) {
        say("gpu_hdr_set_composite is refused: this bridge never took a frame, so "
            "there is no master its composite belongs to");
        return -1;
    }
    /* The same placed-footprint device-to-device copy the Feature 18 output comes
     * in by, one extent up, into the plane gpu_hdr_encode already reads. The
     * producer's values are stored as they arrive; the encode stage stays the only
     * place anything clamps, and this entry adds no tone map, PQ maths, colour
     * claim or encoder of its own. */
    if (!copy_texture_to_plane(b, kCompositeCopy, composite_texture,
                               composite_fence, composite_fence_value,
                               b->planes[kPlaneComposited], b->full_w, b->full_h,
                               &b->counters_state.composited_device_write_bytes))
        return -1;
    b->multiscale.composite_import_count += 1;
    b->multiscale.composite_import_bytes +=
        uint64_t(b->full_w) * b->full_h * 8u;
    if (out_composited_ptr)
        *out_composited_ptr = b->planes[kPlaneComposited].mapped;
    if (out_composited_pitch)
        *out_composited_pitch = b->planes[kPlaneComposited].pitch;
    return 0;
}

extern "C" GPU_HDR_API int32_t gpu_hdr_get_multiscale_counters(
    GpuHdrBridge* handle, GpuHdrMultiscaleCounters* out_counters) {
    Bridge* b = reinterpret_cast<Bridge*>(handle);
    if (!b || !out_counters ||
        out_counters->struct_size != sizeof(GpuHdrMultiscaleCounters)) {
        say("gpu_hdr_get_multiscale_counters needs an open bridge and a sized "
            "struct");
        return -1;
    }
    *out_counters = b->multiscale;
    return 0;
}

extern "C" GPU_HDR_API void gpu_hdr_close(GpuHdrBridge* handle) {
    delete reinterpret_cast<Bridge*>(handle);
}
