#!/usr/bin/env bash
# One-shot build: check → picture → original score + SFX → mix (two-pass loudnorm) → mux → cover → QA
#   bash build.sh path/to/event.py           (SKIP_VIDEO=1 reuses WORK/video_noaudio.mp4; PY=… picks the interpreter)
# Env CPV_WORK / CPV_OUT override the config's WORK / OUT_DIR (handy for test builds).
set -euo pipefail
[[ $# -ge 1 ]] || { sed -n '2,4p' "$0"; exit 2; }
ENGINE=$(cd "$(dirname "$0")" && pwd)
EV=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
VENV_PY="$HOME/.cache/conf-promo-video/venv/bin/python"
PY=${PY:-$([[ -x "$VENV_PY" ]] && echo "$VENV_PY" || echo python3)}
"$PY" -c 'import numpy, cv2, PIL, scipy' 2>/dev/null || { echo "missing Python deps: run bash $ENGINE/setup_env.sh"; exit 1; }
"$PY" "$ENGINE/core.py" "$EV" --check
eval "$("$PY" "$ENGINE/core.py" "$EV" --shell)"
mkdir -p "$WORK" "$OUT_DIR"
W="$WORK"
F="$OUT_DIR/${NAME}.mp4"

[[ "${SKIP_VIDEO:-0}" == 1 ]] || "$PY" "$ENGINE/render.py" "$EV"
"$PY" "$ENGINE/music.py" "$EV"

# 17 kHz low-pass before AAC: the synth's near-Nyquist content otherwise overshoots ~6 dB after AAC decode
ffmpeg -v error -y -i "$W/music.wav" -i "$W/sfx.wav" -filter_complex \
  "[1:a]volume=${SFX_GAIN_DB}dB[s];[0:a][s]amix=inputs=2:normalize=0,lowpass=f=17000:poles=2" "$W/mix_raw.wav"
M=$(ffmpeg -hide_banner -i "$W/mix_raw.wav" -af "loudnorm=I=$LUFS:TP=$TRUE_PEAK:LRA=11:print_format=json" -f null - 2>&1 | sed -n '/{/,/}/p')
get() { echo "$M" | grep "\"$1\"" | sed 's/[^-0-9.]//g'; }
ffmpeg -v error -y -i "$W/mix_raw.wav" -af "loudnorm=I=$LUFS:TP=$TRUE_PEAK:LRA=11:measured_I=$(get input_i):measured_TP=$(get input_tp):measured_LRA=$(get input_lra):measured_thresh=$(get input_thresh):offset=$(get target_offset):linear=true,aresample=48000" -ar 48000 "$W/mix.wav"
ffmpeg -v error -y -i "$W/video_noaudio.mp4" -i "$W/mix.wav" -map 0:v -map 1:a -c:v copy -c:a aac -b:a 256k -ar 48000 -t "$TOTAL" -movflags +faststart "$F"
ffmpeg -v error -y -ss "$COVER_AT" -i "$F" -frames:v 1 -q:v 2 "$OUT_DIR/${NAME}_封面.jpg"
cp "$W/music.wav" "$OUT_DIR/${NAME}_原创配乐.wav"
cp "$W/sfx.wav" "$OUT_DIR/${NAME}_音效.wav"
[[ "$HAS_SRT" == 1 ]] && "$PY" "$ENGINE/core.py" "$EV" --srt "$OUT_DIR/${NAME}.srt"
st=0
bash "$ENGINE/qa.sh" "$F" "$TOTAL" "$HOLD_AT" "$LUFS" "$TRUE_PEAK" "$FPS" || st=1
if [[ -n "$SILENT_NAME" ]]; then   # picture-only version (no audio stream)
  ffmpeg -v error -y -i "$W/video_noaudio.mp4" -map 0:v -c:v copy -an -t "$TOTAL" -movflags +faststart "$OUT_DIR/${SILENT_NAME}.mp4"
  bash "$ENGINE/qa.sh" "$OUT_DIR/${SILENT_NAME}.mp4" "$TOTAL" "$HOLD_AT" "$LUFS" "$TRUE_PEAK" "$FPS" || st=1
fi
exit $st
