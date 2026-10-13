# What OBS offers for plugin UI

Research for [#3](https://github.com/kreuzhofer/obs-ai-director/issues/3). Sources were checked on 2026-10-13 against OBS Studio **32.2.2** (latest release), `obs-plugintemplate` master (`3e7d7ac`), Advanced Scene Switcher (`WarmUpTill/SceneSwitcher` master) and `royshil/obs-backgroundremoval` main.

## Recommendation

Build the Director as a **frontend plugin** (link `obs-frontend-api` and Qt6), not as a source or filter. Its UI has three parts:

1. **Dock.** A custom `QWidget` registered with `obs_frontend_add_dock_by_id("obs-ai-director", …)`. It shows live status, Locks and tuning. OBS lists it in the Docks menu and handles show/hide/float. This is the pattern Advanced Scene Switcher uses for its status dock.
2. **Config panel.** A custom Qt `QDialog`, parented to `obs_frontend_get_main_window()` and opened from a Tools-menu entry (`obs_frontend_add_tools_menu_qaction`). It holds Presenter/Camera binding, the Fallback scene and model settings. Advanced Scene Switcher's settings window follows the same pattern. Don't try to reuse `obs_properties` for this. OBS's properties renderer (`OBSPropertiesView`) is not exported to plugins, and OBS has no API for adding a page to its Settings dialog.
3. **Hotkeys.** Use `obs_hotkey_register_frontend` for things like "Lock current Camera", "Release Lock" and "Pause Director". They appear in Settings → Hotkeys automatically. The plugin has to **save and load the bindings itself**.

Persist Director config **per scene collection** through `obs_frontend_add_save_callback`. It writes a JSON object into the scene-collection file, so a Camera→scene binding always travels with the scenes it names. Keep only machine-level settings (model path, inference device) in `obs_module_config_path(...)`.

Cut with `obs_frontend_set_current_scene()`, which uses OBS's configured transition. Call it from the UI thread, or from a worker that holds **no lock** the UI thread might need. **Surprise:** in Studio Mode this call cuts **Program** directly and leaves Preview alone (see section 5). That matters for the open "Studio Mode behaviour" question.

## Evidence

### 1. Building blocks and the plugin template

- `obs-plugintemplate` has two opt-in CMake switches, `ENABLE_FRONTEND_API` and `ENABLE_QT`. Turning them on links `OBS::obs-frontend-api` and `Qt6::Core`/`Qt6::Widgets` and enables AUTOMOC/AUTOUIC/AUTORCC. Both are OFF by default. ([CMakeLists.txt](https://github.com/obsproject/obs-plugintemplate/blob/master/CMakeLists.txt))
- The template's `buildspec.json` pins OBS 31.1.1 sources, plus prebuilt obs-deps and the Qt6 that OBS ships for macOS and windows-x64. The plugin builds against OBS's own Qt, which it must do because it runs inside OBS's Qt process. ([buildspec.json](https://github.com/obsproject/obs-plugintemplate/blob/master/buildspec.json))
- The whole frontend API is in one header. ([obs-frontend-api.h @ 32.2.2](https://github.com/obsproject/obs-studio/blob/32.2.2/frontend/api/obs-frontend-api.h), [docs](https://github.com/obsproject/obs-studio/blob/32.2.2/docs/sphinx/reference-frontend-api.rst))
- On localisation, the header says plugin UI should not use Qt's own translator. Use `obs_module_text`, and wrap Qt UI translation in `obs_frontend_push_ui_translation`/`pop`. ([header comment](https://github.com/obsproject/obs-studio/blob/32.2.2/frontend/api/obs-frontend-api.h))

### 2. `obs_properties` (built-in settings UI) versus custom Qt

- `obs_properties` is the declarative settings model for **sources, filters, outputs, encoders and services**. Its property types are bool, int, float, text, path, list, color, button, font, editable list, frame rate, group and color-alpha ([obs-properties.h](https://github.com/obsproject/obs-studio/blob/32.2.2/libobs/obs-properties.h), [docs](https://github.com/obsproject/obs-studio/blob/32.2.2/docs/sphinx/reference-properties.rst)). OBS renders it in the Properties/Filters dialog of the object that owns it.
- The Qt renderer for properties (`shared/properties-view`) is a CMake **INTERFACE** library that is compiled into the OBS frontend. It is not exported through `obs-frontend-api`. ([CMakeLists.txt](https://github.com/obsproject/obs-studio/blob/32.2.2/shared/properties-view/CMakeLists.txt)) A plugin that wants it would have to vendor the source, which GPL permits, and track OBS internals.
- `obs-frontend-api.h` has no function for adding a page to OBS's Settings dialog. The only menu hooks are `obs_frontend_add_tools_menu_qaction` and `obs_frontend_add_tools_menu_item`. ([header](https://github.com/obsproject/obs-studio/blob/32.2.2/frontend/api/obs-frontend-api.h))
- **obs-backgroundremoval** is a filter plugin. Its entire configuration is `obs_properties` (sliders, lists, groups, modified-callbacks) in `background-filter.cpp`, registered with `obs_register_source`. Its only frontend UI is an About/startup `QDialog` shown from a frontend event callback. ([background-filter.cpp](https://github.com/royshil/obs-backgroundremoval/blob/main/src/background-filter.cpp), [plugin-main.cpp](https://github.com/royshil/obs-backgroundremoval/blob/main/src/plugin-main.cpp), [AboutDialogIntegration.cpp](https://github.com/royshil/obs-backgroundremoval/blob/main/src/UI/AboutDialogIntegration.cpp)) That fits a per-source effect. The Director is a global controller across scenes, which is the Advanced Scene Switcher shape.
- **Optional hybrid.** If the per-Camera detection needs settings (crop, sensitivity), a small "AI Director Camera" **filter** with `obs_properties` could sit on each camera source. This is an option, not a requirement.

### 3. Docks

- `bool obs_frontend_add_dock_by_id(const char *id, const char *title, void *widget)` takes a `QWidget`. OBS wraps it in its own `OBSDock`, adds a toggle to the Docks menu, and returns false if the id is already taken. When the dock is closed, the widget receives a custom `QEvent::User + QEvent::Close`, and it gets a normal `QShowEvent` when shown again. A plugin can use these to pause expensive live previews. Added in 30.0. ([docs](https://github.com/obsproject/obs-studio/blob/32.2.2/docs/sphinx/reference-frontend-api.rst), [OBSStudioAPI.cpp](https://github.com/obsproject/obs-studio/blob/32.2.2/frontend/OBSStudioAPI.cpp))
- `obs_frontend_add_custom_qdock(id, QDockWidget*)` adds a dock with no menu toggle. `obs_frontend_remove_dock(id)` removes either kind. The old `obs_frontend_add_dock` is gone from the 32.x header. ([header](https://github.com/obsproject/obs-studio/blob/32.2.2/frontend/api/obs-frontend-api.h))
- Advanced Scene Switcher registers `"advss-status-dock"` with `obs_frontend_add_dock_by_id` and has a fallback for OBS versions before 30. ([status-control.cpp](https://github.com/WarmUpTill/SceneSwitcher/blob/master/lib/utils/status-control.cpp)) It also offers per-macro docks. ([macro-dock-window.cpp](https://github.com/WarmUpTill/SceneSwitcher/blob/master/lib/macro/macro-dock-window.cpp))
- Dock layout is restored by OBS through the dock's `objectName` (the id), so **keep the id stable** across releases. I inferred this from OBS setting `objectName` to the id and saving its main-window dock state; I did not trace it line by line.

### 4. Hotkeys

- `obs_hotkey_register_frontend(name, description, func, data)` and `obs_hotkey_pair_register_frontend(...)` (for toggles such as Pause/Resume Director) register global hotkeys. They then appear in Settings → Hotkeys. ([obs-hotkey.h](https://github.com/obsproject/obs-studio/blob/32.2.2/libobs/obs-hotkey.h))
- **Bindings are not persisted for you.** The plugin calls `obs_hotkey_save(id)` to get an `obs_data_array_t`, stores it, and calls `obs_hotkey_load(id, array)` on load. Advanced Scene Switcher does exactly this inside its scene-collection save data. ([macro.cpp](https://github.com/WarmUpTill/SceneSwitcher/blob/master/lib/macro/macro.cpp), see `obs_hotkey_save`/`obs_hotkey_load` and `obs_hotkey_register_frontend`)
- **Threading.** The OBS frontend enables hotkey callback rerouting and queues each callback onto the Qt UI thread through `QMetaObject::invokeMethod`. Hotkey handlers therefore run on the UI thread and can touch widgets and call frontend APIs. ([OBSBasic_Hotkeys.cpp](https://github.com/obsproject/obs-studio/blob/32.2.2/frontend/widgets/OBSBasic_Hotkeys.cpp))

### 5. Scene switching, Studio Mode and threading

- `obs_frontend_set_current_scene(scene)` is implemented as follows ([OBSStudioAPI.cpp](https://github.com/obsproject/obs-studio/blob/32.2.2/frontend/OBSStudioAPI.cpp)):
  - Studio Mode off: it calls `OBSBasic::SetCurrentScene`.
  - Studio Mode on: it calls `OBSBasic::TransitionToScene`, which sets `programScene` and runs the current transition. ([OBSBasic_Transitions.cpp](https://github.com/obsproject/obs-studio/blob/32.2.2/frontend/widgets/OBSBasic_Transitions.cpp)) In other words it **cuts Program directly and does not change Preview**.
  - Preview-only control is available separately: `obs_frontend_set_current_preview_scene` and `obs_frontend_preview_program_trigger_transition`. `obs_frontend_preview_program_mode_active()` reports whether Studio Mode is on.
- Both paths go through `QMetaObject::invokeMethod(main, …, WaitConnection())`. `WaitConnection()` is `DirectConnection` on the UI thread and **`BlockingQueuedConnection` from any other thread**. ([qt-wrappers.hpp](https://github.com/obsproject/obs-studio/blob/32.2.2/shared/qt/wrappers/qt-wrappers.hpp)) So a call from a detector or worker thread **blocks until the UI thread runs it**. If the worker holds a mutex the UI thread also wants, OBS deadlocks. Advanced Scene Switcher calls this from its worker thread and leaves itself a warning ("be careful using switcher->m here as there is potential for deadlocks when using frontend functions such as obs_frontend_set_current_scene()"). ([advanced-scene-switcher.cpp](https://github.com/WarmUpTill/SceneSwitcher/blob/master/lib/advanced-scene-switcher.cpp), [scene-switch-helpers.cpp](https://github.com/WarmUpTill/SceneSwitcher/blob/master/lib/utils/scene-switch-helpers.cpp))
- Recommendation: the inference thread posts "cut to X" to the UI thread with a queued, non-blocking invoke, and the Director state machine applies it there.
- Transition control is also exposed: `obs_frontend_get/set_current_transition` and `obs_frontend_get/set_transition_duration`.

### 6. Persisting config

| Scope | API | Where it lives |
|---|---|---|
| Per scene collection | `obs_frontend_add_save_callback(cb, data)`. `cb(obs_data_t *save_data, bool saving, void*)` is called with `saving=true` on save and `false` on load. Write your own sub-object, e.g. `obs_data_set_obj(save_data, "obs-ai-director", …)` | Inside the scene collection JSON ([docs](https://github.com/obsproject/obs-studio/blob/32.2.2/docs/sphinx/reference-frontend-api.rst)) |
| Before a collection loads | `obs_frontend_add_preload_callback` | Same data, but called before sources load |
| Per profile | `obs_frontend_get_profile_config()` returns a `config_t*` (ini) | Profile `basic.ini` |
| Per user / app | `obs_frontend_get_user_config()` (user.ini) and `obs_frontend_get_app_config()` (global.ini), both added in 31.0. `get_global_config` is deprecated | OBS config dir |
| Plugin-private files | `obs_module_config_path("file.json")` | `plugin_config/<module>/` ([obs-module.h](https://github.com/obsproject/obs-studio/blob/32.2.2/libobs/obs-module.h)) |

- Advanced Scene Switcher keeps all its settings per scene collection under `"advanced-scene-switcher"` in the save callback. On load it stops itself, loads, then restarts. It also handles `SCENE_COLLECTION_CHANGING`, where it stops and closes its settings window, and `SCENE_COLLECTION_CLEANUP`, where it resets to defaults so settings don't leak into a new empty collection. ([advanced-scene-switcher.cpp](https://github.com/WarmUpTill/SceneSwitcher/blob/master/lib/advanced-scene-switcher.cpp)) The Director needs the same lifecycle.
- Store scenes and sources by **name or UUID** and resolve them to weak references at runtime. Scene objects are destroyed and recreated when the collection changes.

### 7. Frontend events

`obs_frontend_add_event_callback`. The full enum is in the [header](https://github.com/obsproject/obs-studio/blob/32.2.2/frontend/api/obs-frontend-api.h). The events relevant to the Director:

- **Scene:** `SCENE_CHANGED` (Program changed, either by the Director or by the user, which is how manual-override detection works), `PREVIEW_SCENE_CHANGED`, `SCENE_LIST_CHANGED` (scenes added, removed or renamed, so revalidate Camera bindings), `TRANSITION_STOPPED`.
- **Studio Mode:** `STUDIO_MODE_ENABLED` and `STUDIO_MODE_DISABLED`.
- **Outputs:** `STREAMING_STARTING/STARTED/STOPPING/STOPPED`, `RECORDING_*` (including PAUSED/UNPAUSED), `VIRTUALCAM_STARTED/STOPPED`, plus polling with `obs_frontend_streaming_active()` and `obs_frontend_recording_active()`. These are relevant to "Director behaviour when not live".
- **Lifecycle:** `FINISHED_LOADING` (safe point to start), `SCENE_COLLECTION_CHANGING`, `SCENE_COLLECTION_CHANGED`, `SCENE_COLLECTION_CLEANUP`, `PROFILE_CHANGED`, `SCRIPTING_SHUTDOWN` (Advanced Scene Switcher uses this as its shutdown hook because collection data is already cleared by `EXIT`), and `EXIT` (last chance to make frontend calls), `THEME_CHANGED` (restyle the dock).
- **Caveat:** `OBSStudioAPI::on_event` drops every event except `SCENE_COLLECTION_CLEANUP` and `EXIT` while the frontend has `disableSaving` set, which happens during collection load and teardown. So don't depend on seeing `SCENE_CHANGED` during a collection switch. ([OBSStudioAPI.cpp](https://github.com/obsproject/obs-studio/blob/32.2.2/frontend/OBSStudioAPI.cpp))

### 8. How the reference plugins are structured

- **Advanced Scene Switcher.** The plugin is split into a thin `MODULE` (`module/advanced-scene-switcher-module.c`) and a `SHARED` lib that does the real work, plus loadable sub-plugins found with `QLibrary`. ([CMakeLists.txt](https://github.com/WarmUpTill/SceneSwitcher/blob/master/CMakeLists.txt)) Init does four things: `SetupDock()`, `obs_frontend_add_save_callback`, `obs_frontend_add_event_callback`, and a Tools-menu `QAction` that opens the `QDialog` settings window parented to the main window with `WA_DeleteOnClose`. A worker `std::thread` loop with a condition variable makes the decisions and switches scenes through `obs_frontend_set_current_scene`. It also integrates with obs-websocket (`obs-websocket-api.h`). ([advanced-scene-switcher.cpp](https://github.com/WarmUpTill/SceneSwitcher/blob/master/lib/advanced-scene-switcher.cpp))
- **obs-backgroundremoval.** It is a filter (`obs_register_source`) whose UI is entirely `obs_properties`, with ML inference in the filter's video path. The only frontend-API use is a startup/About dialog. ([plugin-main.cpp](https://github.com/royshil/obs-backgroundremoval/blob/main/src/plugin-main.cpp))

## Open points this raises

- **Studio Mode.** A plain `obs_frontend_set_current_scene` cuts Program in Studio Mode, bypassing the operator's Preview. Deciding "cut to Preview vs Program" now means choosing between that and `obs_frontend_set_current_preview_scene`, optionally followed by `obs_frontend_preview_program_trigger_transition`.
- **Remote control.** obs-websocket integration (as in Advanced Scene Switcher) would give Stream Deck and remote control of Locks almost for free. It could be a natural Lock source or extension point.
- **Plugin shape.** It is undecided whether per-Camera settings belong in a small filter (rendered by `obs_properties`) or entirely in the custom config dialog.
