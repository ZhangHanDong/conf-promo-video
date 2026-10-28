#!/usr/bin/env bash
# Delivery QA: format, duration, black frames, end hold, loudness / true peak — each line PASS or FAIL
#   bash qa.sh FILE.mp4 [TOTAL_S] [HOLD_AT_S] [LUFS] [TRUE_PEAK] [FPS]   (duration must be within one frame)
set -uo pipefail
F="$1"; TOTAL="${2:-}"; HOLD_AT="${3:-}"; LUFS="${4:--14}"; TP="${5:--1}"; FPS="${6:-30}"
fail=0
ok() { if [[ "$1" == 1 ]]; then echo "  PASS  $2"; else echo "  FAIL  $2"; fail=1; fi; }
echo "== $F"
ffprobe -v error -show_entries format=duration:stream=codec_name,codec_type,width,height,r_frame_rate,pix_fmt,sample_rate,channels -of compact "$F"
dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$F")
if [[ -n "$TOTAL" ]]; then
  ok "$(awk -v d="$dur" -v t="$TOTAL" -v f="$FPS" 'BEGIN{print (d>=t-1/f && d<=t+1/f)}')" "duration ${dur}s (target ${TOTAL}s ± 1 frame)"
  nf=$(ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 "$F")
  ok "$(awk -v n="$nf" -v t="$TOTAL" -v f="$FPS" 'BEGIN{print (n>=t*f-1 && n<=t*f+1)}')" "video frames: $nf (target $(awk -v t="$TOTAL" -v f="$FPS" 'BEGIN{print t*f}'))"
fi
nb=$(ffmpeg -hide_banner -i "$F" -vf "blackdetect=d=0.1:pic_th=0.98:pix_th=0.06" -an -f null - 2>&1 | grep -c black_start)
ok "$([[ $nb == 0 ]] && echo 1 || echo 0)" "black segments: $nb"
if [[ -n "$HOLD_AT" ]]; then
  # a freeze that runs to EOF only reports freeze_start (no freeze_end/duration): measure it against the file end.
  FD=$(ffmpeg -hide_banner -i "$F" -vf "trim=start=$HOLD_AT,freezedetect=n=0.001:d=2" -an -f null - 2>&1)
  fs=$(echo "$FD" | grep -oE "freeze_start: [0-9.]+" | tail -1 | awk '{print $2}')
  fe=$(echo "$FD" | grep -oE "freeze_end: [0-9.]+" | tail -1 | awk '{print $2}')
  hold=$(awk -v s="${fs:-}" -v e="${fe:-}" -v d="$dur" 'BEGIN{ if (s=="") print 0; else if (e!="" && e+0>s+0) print e-s; else print d-s }')
  ok "$(awk -v h="$hold" -v s="${fs:-999}" -v a="$HOLD_AT" 'BEGIN{print (h>=3 && s<=a+1)}')" "end card holds still: ${hold}s from ${fs:-—}s (end card at ${HOLD_AT}s)"
fi
if [[ -z "$(ffprobe -v error -select_streams a -show_entries stream=index -of csv=p=0 "$F")" ]]; then
  echo "  ----  no audio stream (picture-only version): loudness checks skipped"
  exit $fail
fi
L=$(ffmpeg -hide_banner -i "$F" -af ebur128=peak=true -f null - 2>&1)
I=$(echo "$L" | grep -E "^\s+I:" | tail -1 | awk '{print $2}')
P=$(echo "$L" | grep -E "^\s+Peak:" | tail -1 | awk '{print $2}')
ok "$(awk -v i="$I" -v t="$LUFS" 'BEGIN{print (i>=t-1 && i<=t+1)}')" "integrated loudness ${I} LUFS (target ${LUFS})"
ok "$(awk -v p="$P" -v t="$TP" 'BEGIN{print (p<=t+0.5)}')" "true peak ${P} dBTP (limit ${TP})"
exit $fail
