# Data dictionary

When a device visits the customer's site, a script collects signals about it. Each row of `sessions.parquet` is
one session; the table below describes each signal. Sessions that share a `cookie_id` are the same device.

| column | type | description |
| --- | --- | --- |
| `session_id` | text | Session identifier. |
| `ts` | timestamp | Session start, UTC. |
| `cookie_id` | text | Browser cookie. Blank on unlabeled sessions. |
| `ip` | text | IP address. |
| `country` | text | Country of `ip`. |
| `asn` | text | Autonomous system number of `ip`. |
| `device_cores` | number | navigator.hardwareConcurrency. |
| `device_memory` | number | navigator.deviceMemory (GB). |
| `screen_width` | number | screen.width (CSS px). |
| `screen_height` | number | screen.height (CSS px). |
| `screen_pixel_ratio` | number | window.devicePixelRatio. |
| `screen_color_depth` | number | screen.colorDepth (bits). |
| `viewport_width` | number | window.innerWidth. |
| `viewport_height` | number | window.innerHeight. |
| `touch_max_points` | number | navigator.maxTouchPoints. |
| `gpu_vendor` | text | WebGL unmasked vendor. |
| `gpu_renderer` | text | WebGL unmasked renderer. |
| `gpu_parameters_hash` | integer | Hash of the WebGL parameter values. |
| `battery_level` | number | Battery level, 0-1. |
| `battery_charging` | number | Battery charging, 1/0. |
| `device_model` | text | Device model parsed from the user agent. |
| `user_agent` | text | navigator.userAgent. |
| `browser_family` | text | Browser family parsed from the user agent; `webview` is an app's embedded browser. |
| `platform_version` | text | User-agent client hint: platform version. |
| `languages` | text | navigator.languages (JSON). |
| `timezone` | text | Browser timezone (Intl). |
| `emoji_hashes` | text | Hashes of rendered emoji glyphs (JSON list). |
| `text_shaping_hash` | integer | Hash of text-shaping measurements. |
| `math_fingerprint_hash` | integer | Hash of Math function outputs. |
| `audio_fingerprint` | integer | Fingerprint of an OfflineAudioContext rendering. |
| `fonts` | text | Fonts detected from a probe list (JSON). |
| `media_speech_voice_hash` | integer | Hash of the speechSynthesis voice list. |
| `media_rtc_video_capabilities_hash` | integer | Hash of RTCRtpSender video capabilities. |
| `privacy_incognito` | number | Private-browsing detection, 1/0. |
