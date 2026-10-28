#!/usr/bin/env bash
# Create the shared Python env for the engine (numpy, opencv, pillow, scipy) at ~/.cache/conf-promo-video/venv.
# ffmpeg must be on PATH (brew install ffmpeg).
set -euo pipefail
V="$HOME/.cache/conf-promo-video/venv"
command -v ffmpeg >/dev/null || echo "warning: ffmpeg not found on PATH (brew install ffmpeg)"
if command -v uv >/dev/null; then
  [[ -x "$V/bin/python" ]] || uv venv -q "$V"
  uv pip install -q --python "$V/bin/python" numpy opencv-python-headless pillow scipy
else
  [[ -x "$V/bin/python" ]] || python3 -m venv "$V"
  "$V/bin/pip" install -q numpy opencv-python-headless pillow scipy
fi
"$V/bin/python" -c 'import numpy, cv2, PIL, scipy; print("ok:", cv2.__version__, PIL.__version__)'
echo "$V/bin/python"
