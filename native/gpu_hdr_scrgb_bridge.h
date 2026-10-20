#ifndef GPU_HDR_SCRGB_BRIDGE_H
#define GPU_HDR_SCRGB_BRIDGE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#  ifdef GPU_HDR_SCRGB_EXPORTS
#    define GPU_HDR_API __declspec(dllexport)
#  else
#    define GPU_HDR_API __declspec(dllimport)
#  endif
#endif

#define GPU_HDR_ABI_VERSION 1u
typedef struct GpuHdrBridge GpuHdrBridge;
/* Additive SDR NR colour contract. Does not change the HDR proxy ABI. */
GPU_HDR_API uint32_t gpu_sdr10_nr_color_abi_version(void);
GPU_HDR_API int32_t gpu_sdr10_reconstruct(GpuHdrBridge* bridge, void* texture,
    void* fence, uint64_t fence_value, double strength, uint32_t preserve_detail,
    uint64_t* out_ptr, uint32_t* out_pitch);

/* Experimental HDR10 stage. One colour pipeline that never lets a main-colour
 * sample be 8-bit and is only ever reached from the HDR10 chain. It carries
 * PQ-coded BT.2020 non-constant-luminance 10-bit semi-planar surfaces in and
 * out and keeps the working colour volume as linear scRGB in half precision.
 *
 * Physical units, one mapping in both directions:
 *   1.0 of the linear scRGB plane is the 80 cd/m^2 diffuse white of
 *   IEC 61966-2-1 (conventional scRGB). A P010 sample decodes to
 *   nits = 10000 * ST2084_EOTF(code) and then to scRGB = nits / 80, so
 *   scRGB = 125 * pq_eotf(code). Encode divides by 125 before the OETF.
 *   ITU-R BT.2408's 100 cd/m^2 reference is *not* used here; whichever
 *   reference white is chosen only has to be the same one on both sides, and
 *   gpu_hdr_contract() reports the one this build actually applies.
 *
 * Conversions are derived from the standards, not from another implementation:
 * SMPTE ST 2084 for the transfer function, ITU-R BT.2020 for the primaries and
 * the non-constant-luminance matrix (Kr 0.2627, Kb 0.0593), ITU-R BT.709 for
 * the destination primaries, and the Rec.2020 <-> Rec.709 primary conversion
 * computed from the two sets of published chromaticities.
 *
 * Every plane is a D3D12 resource that is also mapped into CUDA, so the
 * consumer of one stage - the Feature 18 host, the optical flow engine, the
 * encoder - reads the producer's result as device memory and no stage of this
 * bridge copies a pixel to or from the host. The readback entry points are
 * explicit correctness instrumentation, are not part of the chain, and are
 * counted separately from the payload. */

/* Bounds the tap row stride a plan may carry. The Lanczos-2 downsample plan
 * the CPU reference builds needs at most GPU_LANCZOS_MAX_TAPS and the
 * Catmull-Rom reconstruction plan exactly 4; a narrower row is padded with
 * weight 0.0f at a valid index rather than dropped. */
#define GPU_HDR_MAX_TAPS 24u

/* GpuHdrTapTable repeats GpuLanczosTapTables field for field on purpose, so one
 * mirror in the Python caller covers all the bridges and a header test fails if
 * any header drifts from it. Which way an axis points is the plan's, not the
 * struct's: the downscale plan's source is the full extent and its destination
 * the working extent, and the reconstruction plan's are the other way round.
 * Every array is [dst_len][taps] row-major and clamp-to-edge is already
 * resolved in the indices. A plan upload is filter coefficients, not a frame:
 * it never moves a GpuHdrTransferCounters field. */
typedef struct GpuHdrTapTable {
    uint32_t struct_size;
    uint32_t taps;
    uint32_t vertical_src_len;
    uint32_t vertical_dst_len;
    const int32_t* vertical_indices;
    const float* vertical_weights;
    uint32_t horizontal_src_len;
    uint32_t horizontal_dst_len;
    const int32_t* horizontal_indices;
    const float* horizontal_weights;
} GpuHdrTapTable;

/* The numbers this build actually applies, so the caller's reference can be
 * compared against them instead of against a copy of the prose. All of them
 * are derived from the standards' own published constants. */
typedef struct GpuHdrContract {
    uint32_t struct_size;
    /* Linear scRGB per cd/m^2 and the inverse: scRGB 1.0 == 80 cd/m^2. */
    double scrgb_per_nit;
    double nits_per_scrgb;
    /* Rec.2020 linear -> Rec.709 (scRGB) linear and the inverse, row-major. */
    double to_scrgb[9];
    double to_bt2020[9];
    /* ST 2084 constants as the standard writes them. */
    double pq_m1, pq_m2, pq_c1, pq_c2, pq_c3;
    /* Non-constant-luminance matrix coefficients. */
    double kr, kg, kb;
    /* The bounded proxy of hdr_proxy_math.py, whose constants are mirrored here
     * so the GPU and the CPU reference cannot drift apart silently. */
    double proxy_knee;
    double proxy_shoulder_span;
    double proxy_shoulder_k;
    /* Relative whites that bound what the proxy can carry. The first is the
     * point above which the encoded half rounds to exactly 1.0, so magnitude is
     * destroyed; it is closed-form and gpu_hdr_open refuses to start unless the
     * enumerated half grid brackets it within one grid step (2**-10) from below.
     * The second is the first level at or above paper white whose measured code
     * slope collapses below min_slope_per_stop - the exclusive top of the band
     * the reconstruction can read a correction in. It has no analytic form and
     * is enumerated on the half grid, and it is invariant across paperWhite. */
    double proxy_saturation_relative;
    double proxy_actionable_relative;
    /* The reconstruction budget, in stops, and the smallest slope that can
     * still carry a correction, in proxy codes per stop. */
    double max_stops;
    double min_slope_per_stop;
} GpuHdrContract;

/* What the source data did against the declared support domain, measured on the
 * frame the bridge last decoded. The master is never altered by any of it: a
 * negative component is a legal gamut-conversion result and only costs the
 * proxy a channel, while a component above the declared ceiling or a non-finite
 * sample is the caller's declaration being wrong and must fail the job rather
 * than be clamped away. */
typedef struct GpuHdrSourceReport {
    uint32_t struct_size;
    double support_ceiling_scrgb;   /* the ceiling this call declared */
    double paper_white_scrgb;       /* the fixed job constant, recorded back */
    uint32_t nonfinite_samples;     /* NaN or infinity in the decoded master */
    uint32_t negative_components;   /* legal, uncorrectable by the proxy */
    uint32_t above_ceiling_components;
    double min_component;
    double max_component;
    uint32_t pixels;
} GpuHdrSourceReport;

/* What the bounded proxy did to the working master. The proxy is the only
 * surface Feature 18 ever sees, so this is the honest statement of what it
 * cannot see: a channel that is negative, pinned at code 1.0, or inside the
 * quantized dead zone carries no correctable signal. */
typedef struct GpuHdrProxyReport {
    uint32_t struct_size;
    uint32_t negative_components;      /* coded as 0.0, master untouched */
    uint32_t pinned_components;        /* code == 1.0: magnitude destroyed */
    uint32_t dead_zone_components;     /* code advances by nothing per stop */
    uint32_t above_actionable_components; /* relative > actionable limit */
    uint32_t above_saturation_components; /* relative > saturation point */
    double min_code;
    double max_code;
    uint32_t pixels;
} GpuHdrProxyReport;

/* What the reconstruction did against the pristine working master. Every
 * channel lands in exactly one disposition, and the four counts sum to
 * pixels * 3. */
typedef struct GpuHdrReconstructReport {
    uint32_t struct_size;
    uint32_t applied_components;
    uint32_t gain_cap_components;      /* applied, clipped to max_stops */
    uint32_t rejected_pinned_components;
    uint32_t rejected_dead_components; /* zero slope below the ceiling */
    uint32_t rejected_negative_components;
    uint32_t nonfinite_components;
    double max_abs_stops;
    double max_abs_residual_scrgb;
    /* The composite is never clamped: it keeps the signed master, so these two
     * are measurements of the corrected picture and not of a correction applied
     * to it. `out_of_domain` counts the channels that sit outside the declared
     * [0, ceiling] interval, which the caller has to decide about; the encoder
     * is the stage that clamps, and it counts what it clamped. */
    uint32_t composite_out_of_domain_components;
    uint32_t composite_nonfinite_components;
    double composite_min_component;
    double composite_max_component;
    uint32_t pixels;
} GpuHdrReconstructReport;

/* What the encode-side quantiser did. The three counts are disjoint questions:
 * was the *input* scRGB outside the BT.2020 interval the PQ code can hold, and
 * did the signal have to be clamped at the top or the bottom. A 2x2 block's
 * chroma mean is judged alongside its four pixels, since the encoder writes a
 * Cb/Cr code for it too. */
typedef struct GpuHdrEncodeReport {
    uint32_t struct_size;
    double clip_ceiling_scrgb;
    uint32_t clamped_high_components;
    uint32_t clamped_low_components;
    uint32_t out_of_gamut_components; /* negative after 709 -> 2020 */
    uint32_t nonfinite_components;
    double max_linear_component;
    double max_code_y;   /* extremes actually written, 0..1023 domain */
    double min_code_y;
    double max_code_uv;
    double min_code_uv;
} GpuHdrEncodeReport;

/* Every field is incremented by the calls that ran, so a zero is a measurement
 * of a boundary that was not crossed rather than a default. The first six are
 * device-to-device traffic: a bridge that is working has all of them rising
 * together.
 *
 * Two classes of traffic cross to the host, and this struct measures exactly one
 * of them:
 *
 *   * the per-pass counter readback. Every executed pass copies its counter run -
 *     kCounterSlots 32-bit slots, so kCounterSlots * sizeof(uint32_t) = 256 bytes
 *     - from the device plane into a CPU-visible readback buffer, and the host
 *     takes a copy of it, because a report struct is answered out of that copy.
 *     This is GPU-to-host traffic on every production run; no field here measures
 *     it, so no consumer may report it as zero. It is scalar: the contract this
 *     bridge is written against forbids image and pixel transfers, not small
 *     diagnostics.
 *   * report_readback_bytes, below, which counts only the explicit plane readback
 *     entries at the end of this header. Those copy whole image planes, so a
 *     non-zero value here says a diagnostic run moved pictures to the host. */
typedef struct GpuHdrTransferCounters {
    uint32_t struct_size;
    uint64_t process_count;
    uint64_t source_device_copy_bytes;    /* P010 borrowed from the decoder */
    uint64_t source_device_write_bytes;   /* full-resolution FP16 scRGB */
    uint64_t working_device_write_bytes;  /* working master, proxy, the copied
                                             Feature 18 input and the residual */
    uint64_t guidance_device_write_bytes; /* the 8-bit motion-only surface */
    uint64_t composited_device_write_bytes; /* full-resolution FP16 out */
    uint64_t encode_device_write_bytes;   /* P010 handed to the encoder */
    uint64_t report_readback_bytes;       /* image-plane bytes copied out by the
                                             gpu_hdr_readback_* entries below; it
                                             does NOT count the per-pass counter
                                             readback, which GpuHdrMultiscale-
                                             Counters.pass_dispatches states
                                             instead */
} GpuHdrTransferCounters;

/* The multiscale foundation's own traffic, in a struct of its own so
 * GpuHdrTransferCounters keeps the layout every existing consumer reads. The
 * same device traffic is added to the fields above as well - an imported master
 * is a full-resolution FP16 write, an observation's residual is working-resident
 * traffic, an imported composite is a full-resolution FP16 write - so a single
 * layer's report and a layer's multiscale report add up the same way; what this
 * struct adds is the attribution per new entry, and nothing here is a defaulted
 * zero standing in for a count the bridge does not keep.
 *
 * Like GpuHdrTransferCounters this struct does not measure the per-pass counter
 * readback in bytes. `pass_dispatches` is the number of shader passes this bridge
 * *completed*, counted in the one function every pass runs through - so it
 * covers decode, both downscale axes, proxy, the residual and composite of
 * reconstruct and observe, and encode, not the new entries alone. A caller that
 * has to state that boundary computes it as
 * pass_dispatches * (kCounterSlots * sizeof(uint32_t)) = pass_dispatches * 256,
 * which is then every scalar counter readback this bridge took on a run that
 * succeeded; after a failed pass it is the completed total and not the submitted
 * one, because a pass that never read its counters back is not counted. The
 * copies the new entries run (imported master, Feature 18 input, composite
 * import) dispatch no shader and so add nothing to it. No field in either struct
 * claims total host traffic is zero. */
typedef struct GpuHdrMultiscaleCounters {
    uint32_t struct_size;
    /* What last wrote this bridge's full-resolution master: 0 nothing yet,
     * 1 gpu_hdr_decode, 2 gpu_hdr_import_master. The source report beside it is
     * produced by the decode pass alone, so this says whether that report
     * describes the master the bridge currently holds. */
    uint32_t master_provenance;
    uint64_t master_import_count;
    /* FP16 scRGB bytes of borrowed master read in, and the same volume written
     * into the bridge's own plane. Equal for a whole-extent copy. */
    uint64_t master_import_device_bytes;
    uint64_t observe_count;
    /* The Feature 18 output this entry copied into its own neural plane, and the
     * signed residual it wrote at the working extent. */
    uint64_t observe_copy_bytes;
    uint64_t observe_residual_bytes;
    uint64_t composite_import_count;
    /* Full-resolution FP16 scRGB copied into the composited plane. */
    uint64_t composite_import_bytes;
    uint64_t pass_dispatches;
} GpuHdrMultiscaleCounters;

GPU_HDR_API uint32_t gpu_hdr_abi_version(void);
GPU_HDR_API void gpu_hdr_fill_contract(GpuHdrContract* out_contract);

/* The borrowed device must be the same D3D12 device the Feature 18 host and the
 * encoder use, because this bridge copies their textures and shares its buffers
 * with them; the borrowed CUDA context must be the one the decoder's surfaces
 * live in, and must not change for the life of the bridge. Both extents are
 * fixed for the whole job, and the two plans are the CPU reference's own tap
 * plans, so the device resamplers are the reference mathematics in half
 * precision rather than a second opinion of it.
 *
 * paper_white_scrgb is the proxy's diffuse white in linear scRGB units, chosen
 * once for the whole job and never adapted per frame. It must be finite and at
 * least 1.0. Channels above paperWhite * proxy_actionable_relative arrive in the
 * proxy's dead ground or pinned, where no correction can be measured; that is a
 * property of the declared domain, not a reason to refuse the job, so those
 * channels keep the master and are counted instead. The failure mode is the
 * other direction: content that lands above the ceiling the job declared from
 * its own metadata is reported per frame, because metadata is a claim and not a
 * measurement.
 *
 * The bridge allocates its planes from the extents and the plans' tap counts at
 * open, so a call later never allocates; a plan wider than GPU_HDR_MAX_TAPS is
 * refused here rather than truncated. */
/* Additive resource-purpose API; old open entries retain full-pipeline ABI 1. */
enum GpuHdrPurpose {
    GPU_HDR_PURPOSE_FULL = 0, GPU_HDR_PURPOSE_GUIDANCE = 1,
    GPU_HDR_PURPOSE_DECODE = 2, GPU_HDR_PURPOSE_ENCODE = 3,
    GPU_HDR_PURPOSE_DECODE422 = 4, GPU_HDR_PURPOSE_ENCODE422 = 5
};
GPU_HDR_API int32_t gpu_hdr_open_purpose(void* device, void* context,
    uint32_t width, uint32_t height, uint32_t low_width, uint32_t low_height,
    const GpuHdrTapTable* down, const GpuHdrTapTable* up, double white,
    uint32_t purpose, GpuHdrBridge** out);
GPU_HDR_API int32_t gpu_sdr10_open_purpose(void* device, void* context,
    uint32_t width, uint32_t height, uint32_t low_width, uint32_t low_height,
    const GpuHdrTapTable* down, const GpuHdrTapTable* up, double white,
    uint32_t purpose, GpuHdrBridge** out);
GPU_HDR_API uint64_t gpu_hdr_allocated_device_bytes(GpuHdrBridge* bridge);

GPU_HDR_API int32_t gpu_hdr_open(void* d3d12_device, void* cuda_context,
    uint32_t full_width, uint32_t full_height, uint32_t working_width,
    uint32_t working_height, const GpuHdrTapTable* downscale_plan,
    const GpuHdrTapTable* reconstruct_plan, double paper_white_scrgb,
    GpuHdrBridge** out_bridge);

/* Additive SDR10 variant. Inputs and outputs are limited-range BT.709 P010 or P210 according to bridge purpose;
 * the FP16 master is relative linear Rec.709 (1.0 == reference white), not
 * scRGB nits. paper_white_scrgb must be exactly 1.0. Shared stage/report/close
 * APIs operate on either bridge mode; gpu_hdr_decode remains HDR-only. */
GPU_HDR_API int32_t gpu_sdr10_open(void* d3d12_device, void* cuda_context,
    uint32_t full_width, uint32_t full_height, uint32_t working_width,
    uint32_t working_height, const GpuHdrTapTable* downscale_plan,
    const GpuHdrTapTable* reconstruct_plan, double paper_white_scrgb,
    GpuHdrBridge** out_bridge);
GPU_HDR_API void gpu_sdr10_contract(GpuHdrContract* out);
GPU_HDR_API uint32_t gpu_sdr10_abi_version(void);

GPU_HDR_API double gpu_hdr_paper_white(const GpuHdrBridge* bridge);

/* P010 -> linear scRGB FP16 at full resolution. The four colour arguments are
 * the source's own AVColorRange, AVColorPrimaries,
 * AVColorTransferCharacteristic and AVColorSpace as the decoder reported them,
 * and this entry accepts limited or full-range BT.2020
 * non-constant-luminance with the ST 2084 transfer: HLG and
 * an untagged source are all refused rather than assumed. PQ is decoded here,
 * before anything downstream sees a sample, so no later stage ever has to know
 * the transfer function. Nothing downloads the surface: the two planes are
 * copied device-to-device from where the decoder put them and read there.
 *
 * The chroma of the 4:2:0 source is lifted to full resolution by 2x2 nearest
 * replication, which is the inverse of the box mean this bridge's own encoder
 * applies, so an encoding made here can be decoded back exactly. */
GPU_HDR_API int32_t gpu_hdr_decode(GpuHdrBridge* bridge, uint64_t p010_y_ptr,
    size_t p010_y_pitch, uint64_t p010_uv_ptr, size_t p010_uv_pitch,
    int32_t color_range, int32_t color_primaries, int32_t color_transfer,
    int32_t color_space, double support_ceiling_scrgb, uint64_t* out_scrgb_ptr,
    uint32_t* out_scrgb_pitch);

/* Same plane and ceiling arguments as gpu_hdr_decode, but accepts only
 * limited-range BT.709 NCL OETF P010 (AV values 1/1/1/1). */
GPU_HDR_API int32_t gpu_sdr10_decode(GpuHdrBridge* bridge, uint64_t p010_y_ptr,
    size_t p010_y_pitch, uint64_t p010_uv_ptr, size_t p010_uv_pitch,
    int32_t color_range, int32_t color_primaries, int32_t color_transfer,
    int32_t color_space, double support_ceiling_scrgb, uint64_t* out_scrgb_ptr,
    uint32_t* out_scrgb_pitch);

/* The pristine working master: the CPU plan's Lanczos-2 filter over the
 * full-resolution scRGB source, in linear scRGB, signed and never clamped.
 * Feature 18 never reads this plane; the residual is measured against it and it
 * is what a rejected correction falls back to. */
GPU_HDR_API int32_t gpu_hdr_downscale(GpuHdrBridge* bridge,
    uint64_t* out_working_ptr, uint32_t* out_working_pitch);

/* The bounded display proxy of the working master, at working resolution, in
 * the domain Feature 18 receives: half codes in [0, 1] after the shoulder and
 * the sRGB encode of hdr_proxy_math.py, with paperWhite the job constant fixed
 * at open. A negative component codes as 0.0 and is counted; the master is not
 * modified to match.
 *
 * The guidance surface is the second output of the same pass: RGBA8 at working
 * resolution, built from the same plane by taking display-referred ST 2084
 * codes back out of the linear values and normalising them to the declared
 * ceiling. It carries no colour volume for the picture - only NVOF ever reads
 * it, and only to estimate motion - which is why it may be 8-bit when the main
 * colour may not. */
GPU_HDR_API int32_t gpu_hdr_proxy(GpuHdrBridge* bridge,
    uint64_t* out_proxy_ptr, uint32_t* out_proxy_pitch,
    uint64_t* out_guidance_ptr, uint32_t* out_guidance_pitch);

/* Feature 18 output -> composited full-resolution scRGB. The audited
 * local_linear_stops reconstruction, ported to the device: per channel the
 * proxy-domain change is delta = neural - input_proxy, the local slope is the
 * measured quantized advance per stop of the proxy at the master's own value,
 * stops = delta / slope clipped to the stated budget, and the reconstructed
 * working HDR value is master * 2**stops. The slope is measured on the device
 * from the master by the same central difference the CPU reference uses, so an
 * identity output (neural == input) yields stops == 0 and returns the master
 * bit-exactly, and a pinned, dead-zone or negative channel is refused rather
 * than divided by a near-zero slope.
 *
 * The residual is reconstructed_working - master, signed half, upsampled by the
 * CPU plan's Catmull-Rom filter, multiplied by residual_strength and added to
 * the pristine full-resolution master. A sample is never reduced to 8 bits on
 * the way; the only clamp is the final one to the declared ceiling, and every
 * clamped sample is counted. residual_strength == 0 returns the full-resolution
 * master exactly.
 *
 * The texture must be R16G16B16A16_FLOAT, made by the same D3D12 device the
 * bridge was opened with, and in COPY_SOURCE state as the Feature 18 host
 * leaves it. */
GPU_HDR_API int32_t gpu_hdr_reconstruct(GpuHdrBridge* bridge,
    void* dlss_texture, void* dlss_fence, uint64_t dlss_fence_value,
    double residual_strength, uint64_t* out_composited_ptr,
    uint32_t* out_composited_pitch);

/* linear scRGB -> PQ-coded BT.2020 limited-range P010 at full resolution: the
 * inverse primary conversion, the ST 2084 OETF per component, the
 * non-constant-luminance matrix and the 64..940 / 512 +-448 quantisation, with
 * 2x2 box means of the *linear* chroma before it is encoded - the exact inverse
 * of the replication in decode. Samples are left-aligned in their 16-bit code
 * units (code << 6), as AV_PIX_FMT_P010LE requires.
 *
 * clip_ceiling_scrgb is the top of the interval the signal is clamped into, in
 * the same linear scRGB units the plane holds, so with 1.0 == 80 cd/m^2 the
 * 10000 cd/m^2 top of ST 2084 is 125.0. It is finite and positive; how much
 * was clamped is reported, not hidden.
 *
 * Both planes are handed back as CUDA device memory for the encoder to take
 * where they are. */
GPU_HDR_API int32_t gpu_hdr_encode(GpuHdrBridge* bridge,
    double clip_ceiling_scrgb, uint64_t* out_y_ptr, uint32_t* out_y_pitch,
    uint64_t* out_uv_ptr, uint32_t* out_uv_pitch);

GPU_HDR_API int32_t gpu_hdr_get_source_report(GpuHdrBridge* bridge,
    GpuHdrSourceReport* out_report);
GPU_HDR_API int32_t gpu_hdr_get_proxy_report(GpuHdrBridge* bridge,
    GpuHdrProxyReport* out_report);
GPU_HDR_API int32_t gpu_hdr_get_reconstruct_report(GpuHdrBridge* bridge,
    GpuHdrReconstructReport* out_report);
GPU_HDR_API int32_t gpu_hdr_get_encode_report(GpuHdrBridge* bridge,
    GpuHdrEncodeReport* out_report);

/* Every out_*_ptr above is the mapped CUDA device address of the bridge's own
 * shared D3D12 plane, and every out_*_pitch is that plane's row pitch in bytes.
 * A downstream consumer - the Feature 18 host, the flow engine, the encoder -
 * takes the pair without any further copy, which is the whole point of the
 * bridge. The addresses belong to the bridge and are valid until its close; the
 * *contents* of a plane are rewritten by the next call of the stage that
 * produces it. */

/* The pitch and the byte size of one of the bridge's own planes, for a caller
 * that has to size a buffer for a readback. plane is 0 the borrowed source
 * luma, 1 the borrowed source colour difference, 2 the full-resolution scRGB
 * master, 3 the working master, 4 the Feature 18 proxy input, 5 the guidance
 * surface, 6 the working residual, 7 the composited full-resolution output,
 * 8 the encoded luma and 9 the encoded colour difference. */
GPU_HDR_API uint32_t gpu_hdr_plane_pitch(GpuHdrBridge* bridge, int32_t plane);
GPU_HDR_API uint64_t gpu_hdr_plane_bytes(GpuHdrBridge* bridge, int32_t plane);

/* Explicit correctness readbacks. None of them is part of the chain, each one
 * says which plane it copies, and each counts its bytes in
 * report_readback_bytes so a transfer report can tell a diagnostic run from a
 * production one: that counter is a measure of image planes that came to the
 * host, and a production run that binds none of these entries reads zero on it.
 * The per-pass counter readback described beside GpuHdrTransferCounters is not
 * these entries and is not in that counter. A readback copies a whole plane
 * *including its row padding*, so row_stride must be at least
 * gpu_hdr_plane_pitch(plane) and the caller reads the useful samples of a row as
 * the first width*bpp bytes of it. An FP16 scRGB plane holds 8 bytes per sample;
 * a P010 plane holds 2 bytes per sample; the guidance surface holds 4. */
GPU_HDR_API int32_t gpu_hdr_readback_source_p010(GpuHdrBridge* bridge,
    uint8_t* y, size_t y_row_stride, uint8_t* uv, size_t uv_row_stride);
GPU_HDR_API int32_t gpu_hdr_readback_encoded_p010(GpuHdrBridge* bridge,
    uint8_t* y, size_t y_row_stride, uint8_t* uv, size_t uv_row_stride);
GPU_HDR_API int32_t gpu_hdr_readback_scrgb(GpuHdrBridge* bridge,
    int32_t plane, uint8_t* rgba, size_t row_stride);

GPU_HDR_API int32_t gpu_hdr_get_transfer_counters(GpuHdrBridge* bridge,
    GpuHdrTransferCounters* counters);

/* --- Multiscale foundation ---------------------------------------------- */

/* The optional capability group, versioned on its own. It does not change
 * GPU_HDR_ABI_VERSION and it does not change any entry above: a bridge built
 * without it still passes every ABI-1 check, and a caller must probe for these
 * symbols by name and read this version, never infer them from the ABI number.
 * gpu_hdr_multiscale_abi_version() reports the version this build carries. */
#define GPU_HDR_MULTISCALE_ABI_VERSION 1u

GPU_HDR_API uint32_t gpu_hdr_multiscale_abi_version(void);

/* A second or third layer's pristine master, taken as a whole from a typed
 * full-resolution linear scRGB FP16 plane the caller already holds - the plane
 * another bridge's gpu_hdr_decode handed back, or any CUDA-resident half plane
 * of the same layout. This entry is the only way an extra layer gets its main
 * colour, and it is a copy and nothing else: no PQ decode, no transfer function,
 * no clip to 1, no 8-bit round trip, no resample and no renormalisation. Every
 * sample lands in this bridge's own master plane exactly as the source holds it,
 * signed and unclamped, and the existing downscale, proxy, residual and encode
 * stages then run on it as they always did, in this layer's own working
 * resources.
 *
 * The source is borrowed and the bridge keeps no reference to it: the copy is
 * issued, executed and waited for inside the call, so the caller may reuse or
 * free the source the moment the call returns.
 *
 * Completion contract, stated rather than assumed: the source must be fully
 * written and device-visible when this call runs. A plane returned by
 * gpu_hdr_decode satisfies that - the decode pass waits its own fence before the
 * call returns - and so does any producer whose call blocks until its GPU work
 * has retired. This entry deliberately takes no fence: a CUDA device pointer has
 * no fence to name, and inventing one would be a claim the bridge cannot keep. A
 * producer that is still running must be made complete by its own caller first.
 *
 * Validated here: the extents equal the bridge's full extent, pitch is at least
 * width * 8 bytes and a multiple of 16 (what a pitched CUDA copy requires), and
 * the source is non-null. Format is not checkable from a bare address, so the
 * caller declares it and the wrapper refuses anything that is not
 * R16G16B16A16_FLOAT before this entry is reached; a source from a different CUDA
 * context fails the copy rather than reading another context's memory. */
GPU_HDR_API int32_t gpu_hdr_import_master(GpuHdrBridge* bridge,
    uint64_t source_ptr, uint32_t source_width, uint32_t source_height,
    uint32_t source_pitch, uint64_t* out_master_ptr,
    uint32_t* out_master_pitch);

/* The observation one layer's Feature 18 output carries: the signed working
 * residual, with no composite. This is gpu_hdr_reconstruct's first half and the
 * same kPassResidual helper - the identical binding, constants, dispatch extents
 * and the identical reconstruction against the pristine working master - so the
 * residual here is bit-for-bit the residual the single-layer path computes.
 *
 * What comes back is the bridge's own working-resolution plane: signed half, RGB
 * in linear scRGB units, holding reconstruct(working_master) - working_master,
 * i.e. the HDR correction the layer measured, not a difference of proxy RGB
 * codes. Alpha is 0.0 in every sample; it is padding, not coverage. The values
 * are not normalised to [0, 1] and a component is legitimately negative.
 *
 * What this entry does not do: no Catmull-Rom upscale, no residual_strength
 * multiply, no composite on the full master, no encode. Upsampling, weighting and
 * compositing are the caller's fused decision; gpu_hdr_reconstruct still does the
 * whole original sequence and keeps its existing order, semantics and counts.
 *
 * The producer rule for the texture is gpu_hdr_reconstruct's: R16G16B16A16_FLOAT,
 * made by this bridge's D3D12 device, at the working extent, in COPY_SOURCE
 * state, with the fence and value that say the layer's own Feature 18 pass has
 * retired. The texture is borrowed, copied into this bridge's own plane and never
 * transitioned.
 *
 * Refused unless the working plane is the current master's own downscale *and*
 * the proxy read that working plane: a residual measured against a proxy of a
 * superseded working frame is a number that looks valid, and a proxy run before
 * a later downscale is exactly that, however fresh the master beside it is.
 * gpu_hdr_downscale retires every proxy that predates the working plane it
 * rewrites, and gpu_hdr_proxy records the working generation it sampled. */
GPU_HDR_API int32_t gpu_hdr_observe(GpuHdrBridge* bridge, void* dlss_texture,
    void* dlss_fence, uint64_t dlss_fence_value, uint64_t* out_residual_ptr,
    uint32_t* out_residual_pitch);

/* The fused full-resolution composite, taken as a whole from a
 * R16G16B16A16_FLOAT texture on this bridge's D3D12 device, at the bridge's full
 * extent, in COPY_SOURCE state. The copy is the placed-footprint device-to-device
 * copy gpu_hdr_reconstruct already uses for the Feature 18 output - same
 * mechanism, same validation, one size up - into the bridge's own composited
 * plane, after which gpu_hdr_encode reads it exactly as it reads the composite
 * the single-layer path produced. No host picture is involved, and this entry adds
 * no tone mapping, no PQ maths, no colour claim and no encoder of its own; the
 * producer's values are stored as they arrive and the existing encode stage is the
 * only place anything clamps.
 *
 * The producer's fence is required here, not optional: the fusion solver runs
 * asynchronously on another queue, so a copy without a wait would race it. Passing
 * a null fence, or a value of 0, is refused rather than assumed complete.
 *
 * This is the D3D12-texture form only. A CUDA-plane variant would be a second API
 * for one consumer, and the solver's output is a texture, so it is not offered.
 *
 * Refused unless a master has been written into this bridge - a composite has no
 * meaning for a bridge that never took a frame. */
GPU_HDR_API int32_t gpu_hdr_set_composite(GpuHdrBridge* bridge,
    void* composite_texture, void* composite_fence,
    uint64_t composite_fence_value, uint64_t* out_composited_ptr,
    uint32_t* out_composited_pitch);

GPU_HDR_API int32_t gpu_hdr_get_multiscale_counters(GpuHdrBridge* bridge,
    GpuHdrMultiscaleCounters* out_counters);

GPU_HDR_API void gpu_hdr_close(GpuHdrBridge* bridge);
GPU_HDR_API const char* gpu_hdr_last_error(void);

#ifdef __cplusplus
}
#endif
#endif
