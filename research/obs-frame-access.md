# How an OBS plugin reads a source's frames cheaply

Research for [#2](https://github.com/kreuzhofer/obs-ai-director/issues/2), part of map [#1](https://github.com/kreuzhofer/obs-ai-director/issues/1).

**Question:** What is the cheapest way for an OBS plugin to get frames from an existing video source (a Camera) for analysis at about 10-15 fps on downscaled images, on Windows and macOS? How does a filter compare with rendering the source from elsewhere, what does the GPU→CPU readback cost, can frames be downscaled on the GPU first, and what changes when the source isn't in the active scene?

Sources were read at these commits (all links below are permalinks):

- obs-studio `7d98beb` (master, Oct 2026)
- obs-backgroundremoval `ac26a4a`
- Advanced Scene Switcher (`WarmUpTill/SceneSwitcher`) `81683d1`

## Recommendation

**Render each Camera's source yourself from a plugin-wide graphics callback. Don't use a per-source filter.**

1. Register one callback, `obs_add_main_render_callback` or `obs_add_tick_callback` plus `obs_enter_graphics`. On each analysis tick (every 4th frame at 60 fps gives about 15 fps), for every Camera:
   - Call `gs_texrender_begin(tr, W_small, H_small)`.
   - Set `gs_ortho(0, base_w, 0, base_h, ...)` and call `obs_source_video_render(source)`.
   - Call `gs_texrender_end`.

   Drawing the full-size source into a small target makes the GPU downscale it during the draw. A 1080p camera comes out at, say, 320×180 or 640×360 BGRA without a separate downscale pass.
2. Call `gs_stage_texture()` into a ring of 2-3 staging surfaces per Camera. Call `gs_stagesurface_map()` on the surface staged on a **previous** tick, never the one you just staged. Copy the bytes out, `unmap`, and hand the copy to the inference thread. This is the pattern OBS uses for its own output, and the pattern the newer obs-backgroundremoval Metal path uses.
3. Hold `obs_source_inc_showing()` on every bound Camera source for as long as the Director is enabled. This keeps cameras that aren't on Program delivering frames (see "Not in the active scene" below). Release it with `obs_source_dec_showing()`.

Why this approach and not a filter:

- A filter's `video_render` only runs when something renders the source: Program, Preview, a projector or Multiview. Cameras the Director might cut *to* are usually not being rendered.
- A filter would also need to be added to every Camera by hand, and its cost scales with however many times the source is drawn per frame.

With the readback deferred by one or more frames and the surface at ≤640×360, the cost is roughly:

- One extra small draw per Camera per analysis tick.
- A 0.23-0.92 MB copy per Camera per tick.

That is negligible next to inference. One exception is the **experimental Metal renderer on macOS**, where `gs_stage_texture` itself blocks; keep the surface small there (see below).

**Fallback/alternative worth knowing:** an *async* video filter (`filter_video`) receives the camera's raw CPU frame (NV12/YUY2/…) with **no** GPU readback at all. It has four drawbacks:

- It only works on async sources. macOS "Capture Card Device" (`macos-avcapture-fast`) and any non-async source are excluded.
- It runs on the graphics thread under the source's async mutex.
- It gets full-resolution YUV, so downscaling and colour conversion happen on the CPU.
- It doesn't see the user's GPU filters, such as crop.

It's a reasonable optimisation for later, not the v1 baseline.

## Evidence

### 1. The two ways to reach a source's pixels

| | Effect/GPU filter (`video_render`) | Render from elsewhere (main render / tick callback) | Async filter (`filter_video`) |
|---|---|---|---|
| Where the pixels are | GPU texture | GPU texture | CPU `obs_source_frame` (native camera format) |
| Runs when | Only when the parent source is drawn | Whenever you choose (once per OBS frame available) | Every OBS tick for async sources (`async_tick`) |
| Works for Cameras not on Program | No, unless also in Preview, a projector or Multiview | Yes (with a `showing` ref, see §4) | Yes, if the source is still producing frames |
| Setup for the user | Add a filter to each Camera | None (plugin binds by source) | Add a filter to each Camera |
| Readback | Needed | Needed | None |
| Source types | All | All | Async only |

Sources for each column:

- **Filters run only when the source is drawn.** In `render_video()`, filters run via `obs_source_render_filters` only as part of `obs_source_video_render(source)` ([obs-source.c L2943-2983](https://github.com/obsproject/obs-studio/blob/7d98bebe1115b12608837b622164ea5fe6d9483a/libobs/obs-source.c#L2943-L2983)). Nothing calls a filter unless something draws its parent.
- **Calling `obs_source_video_render` from your own code is the documented, supported way to draw a source elsewhere.** OBS's own screenshot feature does exactly this from a tick callback ([frontend/utility/ScreenshotObj.cpp L39](https://github.com/obsproject/obs-studio/blob/7d98bebe1115b12608837b622164ea5fe6d9483a/frontend/utility/ScreenshotObj.cpp#L39), [L149-197](https://github.com/obsproject/obs-studio/blob/7d98bebe1115b12608837b622164ea5fe6d9483a/frontend/utility/ScreenshotObj.cpp#L149-L197)), as does Advanced Scene Switcher's video condition ([screenshot-helper.cpp L85-117](https://github.com/WarmUpTill/SceneSwitcher/blob/81683d1945ee8dedbbe0997c1f97f99179654cf5/lib/utils/screenshot-helper.cpp#L85-L117)).
- **Callback options.** `obs_add_main_render_callback` runs every OBS frame inside the graphics context, before the main view is drawn ([reference-core.rst L589-606](https://github.com/obsproject/obs-studio/blob/7d98bebe1115b12608837b622164ea5fe6d9483a/docs/sphinx/reference-core.rst#L589-L606), [obs-video.c L172-215](https://github.com/obsproject/obs-studio/blob/7d98bebe1115b12608837b622164ea5fe6d9483a/libobs/obs-video.c#L172-L215)). Tick callbacks run on the graphics thread before rendering, and you call `obs_enter_graphics()` yourself ([obs-video.c L32-88](https://github.com/obsproject/obs-studio/blob/7d98bebe1115b12608837b622164ea5fe6d9483a/libobs/obs-video.c#L32-L88)). `gs_texrender_begin/end` save and restore the current render target, so drawing inside the main render callback doesn't disturb the main mix.
- **Drawing a source twice per frame is cheap.** For async sources, the CPU→GPU frame upload happens once per frame. `async_tick` sets `async_update_texture`, and the first render consumes it ([obs-source.c L1365-1392](https://github.com/obsproject/obs-studio/blob/7d98bebe1115b12608837b622164ea5fe6d9483a/libobs/obs-source.c#L1365-L1392), [L2551-2570](https://github.com/obsproject/obs-studio/blob/7d98bebe1115b12608837b622164ea5fe6d9483a/libobs/obs-source.c#L2551-L2570)). Standard filters cache their output per frame through `gs_texrender`'s `rendered` flag, which resets once per tick ([texture-render.c L93-95](https://github.com/obsproject/obs-studio/blob/7d98bebe1115b12608837b622164ea5fe6d9483a/libobs/graphics/texture-render.c#L93-L95), [obs-source.c L1414-1416](https://github.com/obsproject/obs-studio/blob/7d98bebe1115b12608837b622164ea5fe6d9483a/libobs/obs-source.c#L1414-L1416)). The extra draw is mostly the small downscaling quad.
- **Async filters get the raw CPU frame.** `filter_video` "is only used with asynchronous video filters" and receives the raw `obs_source_frame` ([reference-sources.rst L303-310](https://github.com/obsproject/obs-studio/blob/7d98bebe1115b12608837b622164ea5fe6d9483a/docs/sphinx/reference-sources.rst#L303-L310)). It is invoked from `async_tick` → `filter_frame` → `filter_async_video` while holding `async_mutex` ([obs-source.c L1302-1313](https://github.com/obsproject/obs-studio/blob/7d98bebe1115b12608837b622164ea5fe6d9483a/libobs/obs-source.c#L1302-L1313), [L3368-3389](https://github.com/obsproject/obs-studio/blob/7d98bebe1115b12608837b622164ea5fe6d9483a/libobs/obs-source.c#L3368-L3389)). Windows `dshow_input` is async (`OBS_SOURCE_ASYNC`, frames via `obs_source_output_video2`) ([win-dshow.cpp L2083](https://github.com/obsproject/obs-studio/blob/7d98bebe1115b12608837b622164ea5fe6d9483a/plugins/win-dshow/win-dshow.cpp#L2083)). macOS has two camera source types:
  - `macos-avcapture` ("Video Capture Device") is async.
  - `macos-avcapture-fast` ("Capture Card Device") is **synchronous**: it wraps the camera's IOSurface directly as a GPU texture, so an async filter can't attach to it ([mac-avcapture/plugin-main.m L289-325](https://github.com/obsproject/obs-studio/blob/7d98bebe1115b12608837b622164ea5fe6d9483a/plugins/mac-avcapture/plugin-main.m#L289-L325)).

### 2. GPU→CPU readback cost, per backend

The API is `gs_stagesurface_create`, then `gs_stage_texture(stage, tex)`, then `gs_stagesurface_map/unmap`. The OBS docs say of `gs_stage_texture`: "Ideally best to give this a frame to process to prevent stalling" ([reference-libobs-graphics-graphics.rst L708-714](https://github.com/obsproject/obs-studio/blob/7d98bebe1115b12608837b622164ea5fe6d9483a/docs/sphinx/reference-libobs-graphics-graphics.rst#L708-L714)). What "stalling" means depends on the backend:

- **Windows / D3D11 (default on Windows).**
  - `gs_stage_texture` only queues a `CopyResource` onto a staging texture, so it is asynchronous ([d3d11-subsystem.cpp L2228-2250](https://github.com/obsproject/obs-studio/blob/7d98bebe1115b12608837b622164ea5fe6d9483a/libobs-d3d11/d3d11-subsystem.cpp#L2228-L2250)).
  - `gs_stagesurface_map` calls `context->Map(..., D3D11_MAP_READ, 0, ...)` with no `DO_NOT_WAIT` flag ([L2863-2873](https://github.com/obsproject/obs-studio/blob/7d98bebe1115b12608837b622164ea5fe6d9483a/libobs-d3d11/d3d11-subsystem.cpp#L2863-L2873)). That **blocks the graphics thread until the GPU has finished all work up to the copy**. Mapping the surface you staged in the same frame forces a full CPU/GPU sync. Mapping one staged a frame or more earlier normally returns immediately.
- **macOS / OpenGL (the current default on macOS).** The frontend still defaults to OpenGL. Metal is offered as "Experimental" ([frontend/OBSApp.cpp L335](https://github.com/obsproject/obs-studio/blob/7d98bebe1115b12608837b622164ea5fe6d9483a/frontend/OBSApp.cpp#L335), [L1213](https://github.com/obsproject/obs-studio/blob/7d98bebe1115b12608837b622164ea5fe6d9483a/frontend/OBSApp.cpp#L1213), [OBSBasicSettings.cpp L1442](https://github.com/obsproject/obs-studio/blob/7d98bebe1115b12608837b622164ea5fe6d9483a/frontend/settings/OBSBasicSettings.cpp#L1442)).
  - On Apple, `gs_stage_texture` does `glReadPixels` into a pixel-pack buffer object (PBO), which is the async path. The source comment notes PBOs only transfer asynchronously on macOS when used this way.
  - `map` calls `glMapBuffer`, which blocks if the transfer isn't done ([gl-stagesurf.c L109-156](https://github.com/obsproject/obs-studio/blob/7d98bebe1115b12608837b622164ea5fe6d9483a/libobs-opengl/gl-stagesurf.c#L109-L156), [L202-219](https://github.com/obsproject/obs-studio/blob/7d98bebe1115b12608837b622164ea5fe6d9483a/libobs-opengl/gl-stagesurf.c#L202-L219)). The behaviour is the same as D3D11: map one frame or more later.
- **macOS / Metal (experimental, Apple Silicon only).**
  - `gs_stage_texture` flushes pending commands, encodes a blit into a shared `MTLBuffer`, then **`waitUntilCompleted()`**, so the stage call itself is synchronous ([MetalDevice.swift L673-699](https://github.com/obsproject/obs-studio/blob/7d98bebe1115b12608837b622164ea5fe6d9483a/libobs-metal/MetalDevice.swift#L673-L699), [metal-texture2d.swift L280-299](https://github.com/obsproject/obs-studio/blob/7d98bebe1115b12608837b622164ea5fe6d9483a/libobs-metal/metal-texture2d.swift#L280-L299)).
  - `map` is free: it returns `buffer.contents()` from unified memory ([metal-stagesurf.swift L100-125](https://github.com/obsproject/obs-studio/blob/7d98bebe1115b12608837b622164ea5fe6d9483a/libobs-metal/metal-stagesurf.swift#L100-L125)).
  - Here, deferring the map buys nothing. The only lever is keeping the staged texture small, which is one more reason to downscale on the GPU first.
- **What OBS itself does.** The main output pipeline stages into `NUM_TEXTURES = 2` rotating surfaces and maps the *previous* frame's surface (`prev_texture`) ([obs-video.c L866-913](https://github.com/obsproject/obs-studio/blob/7d98bebe1115b12608837b622164ea5fe6d9483a/libobs/obs-video.c#L866-L913), [obs-internal.h L49](https://github.com/obsproject/obs-studio/blob/7d98bebe1115b12608837b622164ea5fe6d9483a/libobs/obs-internal.h#L49)). The screenshot code and Advanced Scene Switcher spread render, stage and map across three successive ticks ([screenshot-helper.cpp L166-193](https://github.com/WarmUpTill/SceneSwitcher/blob/81683d1945ee8dedbbe0997c1f97f99179654cf5/lib/utils/screenshot-helper.cpp#L166-L193)).

**Size of the transfer.** These figures are arithmetic, not a measurement:

| Size | BGRA bytes per frame |
|---|---|
| 1920×1080 | 8.3 MB |
| 640×360 | 0.92 MB |
| 320×180 | 0.23 MB |

At 15 fps × 3 Cameras × 640×360 that is about 41 MB/s. That is trivial for PCIe on the RTX 4070 Ti Super box and for unified memory on the M2 Max. The real cost is pipeline stalls, not bandwidth, and deferred mapping removes them.

**I did not benchmark this.** No OBS build was run for this ticket. Confirming the per-frame cost in the OBS profiler, under the `render_video` / `tick_sources` entries or a custom `profile_start` name, belongs in the integration prototype.

### 3. Downscaling on the GPU first: yes, and the existing plugins show both approaches

- **No GPU downscale: obs-backgroundremoval's generic (Windows/ONNX) path.** It renders the target at **full base resolution** into a texrender, stages it, and maps it **in the same `video_render` call** ([obs-utils.cpp L19-76](https://github.com/royshil/obs-backgroundremoval/blob/ac26a4abaa34c9e5a8a346c2f92bb9fa3e8d5808/src/obs-utils/obs-utils.cpp#L19-L76), called from [background-filter.cpp L739-765](https://github.com/royshil/obs-backgroundremoval/blob/ac26a4abaa34c9e5a8a346c2f92bb9fa3e8d5808/src/background-filter.cpp#L739-L765)). It then resizes on the CPU with OpenCV. This is the naive pattern: a full sync stall plus about 8 MB copied every frame.
- **GPU downscale with deferred map: obs-backgroundremoval's newer macOS path.**
  - `MainEffect::packModelInput` draws the source into a small model-sized render target, 256 px wide ([MainEffect.cpp L246-254](https://github.com/royshil/obs-backgroundremoval/blob/ac26a4abaa34c9e5a8a346c2f92bb9fa3e8d5808/src/macos/BackgroundRemoval/MainEffect.cpp#L246-L254), [MediaPipeLandscapePipeline.hpp L60-63](https://github.com/royshil/obs-backgroundremoval/blob/ac26a4abaa34c9e5a8a346c2f92bb9fa3e8d5808/src/macos/BackgroundRemoval/MediaPipeLandscapePipeline.hpp#L60-L63)).
  - It stages into a 3-slot `AsyncTextureReader` and maps the newest ready slot on the *next* render ([MediaPipeLandscapePipeline.mm L459-495](https://github.com/royshil/obs-backgroundremoval/blob/ac26a4abaa34c9e5a8a346c2f92bb9fa3e8d5808/src/macos/BackgroundRemoval/MediaPipeLandscapePipeline.mm#L459-L495), [AsyncTextureReader.cpp](https://github.com/royshil/obs-backgroundremoval/blob/ac26a4abaa34c9e5a8a346c2f92bb9fa3e8d5808/src/ObsBridgeUtils/AsyncTextureReader.cpp)).
  - It also does colour packing (to FP16) in the shader, so the CPU receives the model's input tensor directly.
  - This is the pattern to copy, but by reimplementation only (see "Surprises" below).
- **How to downscale.** Advanced Scene Switcher shows the mechanism: `gs_texrender_begin(tr, w, h)` with `gs_ortho` set to the source's coordinate space ([screenshot-helper.cpp L85-108](https://github.com/WarmUpTill/SceneSwitcher/blob/81683d1945ee8dedbbe0997c1f97f99179654cf5/lib/utils/screenshot-helper.cpp#L85-L108)). Pass a smaller `w, h` than the ortho extents and the draw is a bilinear downscale.
- **Watch the scale ratio.** Ratios of 3× or more alias with plain bilinear filtering. If that hurts detection, render via OBS's bicubic or Lanczos base effects, or downscale in two halving passes. This is a tuning detail for the prototype.

### 4. When the Camera isn't in the active scene

- **Filters see nothing.** A filter only runs while its parent is drawn (§1). A Camera that is only in a non-Program scene gets no `video_render` calls unless Studio Mode Preview, a projector or Multiview is drawing it. Multiview draws *every* scene, so a filter-based design would appear to work while Multiview is open and silently stop when it closes.
- **Async sources keep ticking.** `tick_sources` ticks *every* source every frame whether or not it is shown ([obs-video.c L32-88](https://github.com/obsproject/obs-studio/blob/7d98bebe1115b12608837b622164ea5fe6d9483a/libobs/obs-video.c#L32-L88)). `async_tick` still picks the newest frame, so async sources keep frames available to a render-from-elsewhere reader.
- **Some camera sources stop producing frames when not shown.** That is why the Director needs a `showing` ref:
  - Windows DirectShow has a per-source "Deactivate when not showing" option (`deactivate_when_not_showing`). With it enabled, the device is *closed* when `show_refs` drops to 0 ([win-dshow.cpp L52](https://github.com/obsproject/obs-studio/blob/7d98bebe1115b12608837b622164ea5fe6d9483a/plugins/win-dshow/win-dshow.cpp#L52), [L2062-2075](https://github.com/obsproject/obs-studio/blob/7d98bebe1115b12608837b622164ea5fe6d9483a/plugins/win-dshow/win-dshow.cpp#L2062-L2075)).
  - macOS "Capture Card Device" (`macos-avcapture-fast`) only rebinds its IOSurface texture in `video_tick` when `obs_source_showing()` is true ([plugin-main.m L154](https://github.com/obsproject/obs-studio/blob/7d98bebe1115b12608837b622164ea5fe6d9483a/plugins/mac-avcapture/plugin-main.m#L154)). When not shown, it would hand you a stale frame.
- **How to hold the ref.** `obs_source_inc_showing` / `dec_showing` exist for exactly this case ("Typically used when drawing a source on a display manually") ([reference-sources.rst L1292-1305](https://github.com/obsproject/obs-studio/blob/7d98bebe1115b12608837b622164ea5fe6d9483a/docs/sphinx/reference-sources.rst#L1292-L1305)). Advanced Scene Switcher wraps its off-screen render in `inc_showing`/`dec_showing` ([screenshot-helper.cpp L100-103](https://github.com/WarmUpTill/SceneSwitcher/blob/81683d1945ee8dedbbe0997c1f97f99179654cf5/lib/utils/screenshot-helper.cpp#L100-L103)). The Director should hold the ref for its whole lifetime rather than per render, so devices aren't reopened between ticks. Show state is applied on the next `video_tick` ([obs-source.c L1418-1438](https://github.com/obsproject/obs-studio/blob/7d98bebe1115b12608837b622164ea5fe6d9483a/libobs/obs-source.c#L1418-L1438)).
- **Use `showing`, not `active`.** `active` (`obs_source_inc_active`) means "on the main output". Raising it would trigger "on air" side effects (e.g. media sources restarting, tally), so don't use it.

## Surprises / possible new questions

- **obs-backgroundremoval is GPL-3.0-or-later** ([LICENSE](https://github.com/royshil/obs-backgroundremoval/blob/ac26a4abaa34c9e5a8a346c2f92bb9fa3e8d5808/LICENSE), SPDX headers in each file). Copying its code (e.g. `AsyncTextureReader`) into this GPL-2.0-or-later project would force the combined work to GPL-3. Reimplement the pattern instead. Advanced Scene Switcher is GPL-2.0 and OBS is GPL-2.0-or-later, so both are compatible.
- **The Metal renderer makes staging synchronous** (`waitUntilCompleted`). It's experimental today, but if OBS flips the macOS default to Metal (the TODO in `OBSApp.cpp` says it will), readback cost on Mac becomes "stall per stage". Keeping the staged texture tiny covers this.
- **Multiview masks the filter-only failure mode.** It renders every scene, so a filter-based design would pass testing whenever Multiview happens to be open.
- **Holding `showing` on all Cameras overrides the user's "Deactivate when not showing" choice** for those cameras. That is intended, but worth a line in the spec or UI.
- **HDR / colour space.** Sources can be in linear or HDR colour spaces. OBS's own screenshot code tonemaps HDR to SDR before staging. The reader should render with `gs_texrender_begin_with_color_space(..., GS_CS_SRGB)` and a `GS_BGRA` target to get consistent 8-bit SDR pixels.
- **Not measured.** All costs above come from reading the code plus arithmetic. A quick OBS-profiler check on both target machines should be part of the first integration prototype.
