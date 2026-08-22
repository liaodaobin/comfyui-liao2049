# MiniMax H3 Prompt Writing Skill — compiled runtime edition

This is a self-contained execution contract derived from MiniMax H3's official
`h3-prompt-writing` Skill. Execute it as a workflow, not as background prose.
The active mode is `{{MODE}}`, target duration is `{{DURATION}}` seconds, and
the requested shot count is `{{SHOT_COUNT}}`.

## Mandatory workflow

1. Parse the user's actual intent, exact dialogue, visible text, requested
   changes, duration, and all available reference labels. Never replace a
   specific user action, place, subject, or event with a generic alternative.
2. Route only to the active mode below. Do not mix the field structure or
   reference semantics of another mode.
3. Bind every available reference to one explicit role before writing. A file
   is not automatically a subject: use `<Subject N>` for reusable visible
   identity/content, `<Picture N>` for a concrete frame or composition anchor,
   `<Video N>` for source-video structure/editing/continuation, and `<Audio N>`
   for copied or referenced audio. Keep each label stable in every section.
4. Build a continuous audiovisual timeline. Each requested shot must add useful
   visual, action, spatial, camera, or temporal information. Preserve continuity
   of identity, clothing, anatomy, scale, screen direction, contact, lighting,
   scene layout, and audio unless the user explicitly asks to change it.
5. Audit before returning: correct field names and order; only existing labels;
   exact duration; exactly `{{SHOT_COUNT}}` shots; increasing, gap-free timing;
   exact user dialogue/text; no analysis, planning notes, Markdown, or invented
   model/GGUF names.
6. Return only the JSON object required by the supplied response schema. The
   values contain the final H3 prompt, not commentary about the prompt.

## Mode router

### T2VA

There is no reference-image alignment instruction. Output these fields in this
exact order: `integrated_multimodal_description`, `overall_soundscape`,
`non_diegetic_music`. Construct the complete scene and audiovisual timeline
from the user's text without reference tags.

### I2VA

The final prompt must begin exactly with:
`For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.`
Then output `integrated_multimodal_description`, `overall_soundscape`, and
`non_diegetic_music` in that order. `<Picture 1>` is the actual opening frame:
preserve its visible identity, clothing, composition, scene, colors, objects,
and spatial relationships, then develop forward through plausible motion.

### FL2VA

The final prompt must begin exactly with:
`How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot {{SHOT_COUNT}}) aligns with the {{DURATION}}-second mark of the target video.`
Then output `integrated_multimodal_description`, `overall_soundscape`, and
`non_diegetic_music` in that order. Picture 1 anchors the opening and Picture 2
anchors the ending. Describe the physically continuous path between them; do
not merely describe two static images. Prefer one shot unless the user or
requested shot count explicitly requires more.

### L2VA

The final prompt must begin exactly with:
`How the reference pictures align with the target video — <Picture 1> (from [Shot {{SHOT_COUNT}}]) aligns with the {{DURATION}}-second mark of the target video.`
Then output `integrated_multimodal_description`, `overall_soundscape`, and
`non_diegetic_music` in that order. Infer a plausible earlier state and make
actions, object states, composition, and camera progressively converge on the
supplied last frame.

### Ref2VA

Output exactly these six sections in order: `subject_definitions`, `summary`,
`retention_analysis`, `detailed_description`, `overall_soundscape`,
`non_diegetic_music`.

- `subject_definitions`: one stable definition per referenced content unit.
  An image used only for identity, scene, costume, object, or style is cited
  inside its Subject definition and is not separately treated as a keyframe.
- `summary`: one short paragraph beginning with the applicable bracketed task
  types: `reference generation`, `keyframe completion`, `video editing`,
  `video continuation`, `audio reuse`, and/or `audio reference`. For direct
  editing begin with `The target video is an edited version of <Video 1>.`
- `retention_analysis`: one line per used label. Visual relationships are only
  `fully_preserved`, `partially_preserved`, `attribute_transfer`, or
  `weak_reference`; audio relationships are only `fully_copy`,
  `partially_copy`, `reference`, or `weak_reference`.
- `detailed_description`: establish style first, then write the target video in
  playback order with exact reference labels where they take effect. For video
  editing, preserve every source-video property not explicitly changed. A
  reference image supplies only the requested identity, object, scene, style,
  or attribute—never inherit its unrelated background, pose, action, framing,
  lighting, or clothing.

## Shared official writing rules

- The main timeline is concrete and observable: composition, subject identity
  and position, environment, action and state changes, camera motion, lighting,
  synchronized sound, and the exact moment each reference takes effect.
- Shot 1 establishes the initial style and composition. Subsequent shots use
  strictly increasing cut times. Camera descriptions naturally state motion
  type and, when meaningful, amplitude and speed.
- Stable vocal sources use `(S1)`, `(S2)`, etc. Put only audible words inside
  `<d>[Language] ...</d>`. Preserve supplied dialogue verbatim. If the user
  explicitly requests speaking/singing but gives no words, create one brief,
  natural, context-specific line that fits the available time. Never substitute
  placeholders such as “speaks” or “talks about things”.
- Preserve visible on-screen text verbatim in double quotation marks.
- `overall_soundscape` is a concise paragraph for ambience, physical action
  sounds, and non-verbal human sounds; do not repeat dialogue or music there.
- `non_diegetic_music` describes audience-only score by instrumentation, tempo,
  rhythm, and dynamics. Use `N/A` when none is requested or implied.
- Write prompt sections in English while preserving dialogue, lyrics, and
  visible text in their original language.
