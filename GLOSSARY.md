# OBS AI Director

An OBS plugin that automatically cuts between scenes based on which camera a single presenter is directing their attention to.

## Language

### People and devices

**Presenter**:
The one person in front of the cameras whose attention drives the Director.
_Avoid_: User, streamer, host, talent

**Camera**:
A physical camera, feeding OBS through a video source, that the Director watches to detect whether the Presenter is attending to it.
_Avoid_: Source, input, feed

**Attention**:
Which Camera the Presenter is facing or looking into at a given moment, or none.
_Avoid_: Facial expression, gaze (unless specifically eye direction), focus

### Scenes

**Director scene**:
An OBS scene the Director is allowed to cut to: every Camera scene, the Fallback scene, and any lock-only scene such as an overhead view.
_Avoid_: Shot, managed scene

**Camera scene**:
The Director scene bound one-to-one to a Camera, cut to when the Presenter's Attention is on that Camera.
_Avoid_: Camera (when the scene is meant)

**Fallback scene**:
The Director scene cut to when the Presenter's Attention is on no Camera.
_Avoid_: Default scene, wide shot

### Director behaviour

**Director**:
The plugin's decision-maker that watches Cameras and cuts between Director scenes; it never "switches cameras".
_Avoid_: AI, auto-switcher, bot

**Lock**:
A pin on one Director scene that suspends Attention-based cutting until released; there is at most one at a time.
_Avoid_: Override, hold, freeze

**Lock source**:
Anything that can lock, release, pause or resume the Director: the dock and hotkeys in v1, voice commands and hand gestures later.
_Avoid_: Trigger, input

**Paused**:
The Director state in which it makes no cuts and leaves whatever is live alone, entered when Program shows a scene that isn't a Director scene or when paused by a Lock source.
_Avoid_: Disabled, stopped, manual mode

**Off**:
The Director state in which no analysis runs at all.
_Avoid_: Paused, disabled
