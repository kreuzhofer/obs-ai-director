# OBS AI Director

An OBS plugin that automatically cuts between camera scenes based on where a single presenter is directing their attention.

## Language

**Director**:
The plugin's decision-maker that chooses which scene is live, based on the presenter's attention.
_Avoid_: AI, auto-switcher, bot

**Presenter**:
The one person in front of the cameras whose attention drives the Director.
_Avoid_: User, streamer, host, talent

**Camera**:
A physical camera feeding OBS, bound to exactly one scene the Director can cut to.
_Avoid_: Source, input, feed

**Attention**:
Which Camera the Presenter is facing or looking into at a given moment, or none.
_Avoid_: Facial expression, gaze (unless specifically eye direction), focus

**Lock**:
A pin on one Camera that suspends Attention-based switching until it is released.
_Avoid_: Override, hold, freeze

**Lock source**:
Anything that can create or release a Lock: the dock today, voice commands and hand gestures later.
_Avoid_: Trigger, input

**Fallback scene**:
The scene the Director cuts to when the Presenter's attention is on no Camera.
_Avoid_: Default scene, wide shot
