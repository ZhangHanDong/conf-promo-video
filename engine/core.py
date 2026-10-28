#!/usr/bin/env python3
"""Shared loader for one event config (event.py): defaults, derived values, validation.

  python3 core.py EVENT.py --check     # validate the config and every asset it points at
  python3 core.py EVENT.py --shell     # shell assignments used by build.sh
  python3 core.py EVENT.py --srt OUT.srt   # subtitles from SUBTITLES
"""
import importlib.util, os, shlex, sys

ENGINE = os.path.dirname(os.path.abspath(__file__))

DEFAULTS = dict(
    W=1920, H=1080, FPS=30, BPM=128,
    DARK=(7, 9, 16), ACCENT=(255, 107, 44), ACCENT_SOFT=(255, 190, 150), ACCENT_SUB=(255, 196, 160),
    GLOW=(40, 16, 4), KV_SCRIM=(10, 14, 38),
    PHOTOS=None, PHOTO_DIRS=(), PHOTO_ID_LEN=3, ANCHOR={}, ANCHOR_DEFAULT=0.8, POIS={},
    RECAP=None, HUD_BAR_LABEL='小节',
    TERMINAL=None, KV=None, ENDCARD_LINES=(),
    WALLS={}, LOGO_DARK_PLATE=set(), LOGO_NAME_BESIDE=set(), LOGO_BLUR=set(),   # LOGO_BLUR: names shown blurred (compliance)
    TRACK_WALLS={}, PANEL_PLACEHOLDER='圆桌',
    CHORDS=[(45, [57, 60, 64, 69]), (41, [53, 57, 60, 65]), (48, [55, 60, 64, 67]), (43, [55, 59, 62, 67])],  # Am F C G
    LUFS=-14, TRUE_PEAK=-1, SFX_GAIN_DB=-5, COVER_AT=None, OUT_NAME='promo',
    HUD=True, ENDCARD_IMAGES=(), MUSIC_BRIGHTNESS=1.0,
    SILENT_NAME=None,      # also deliver a picture-only file with this name
    SUBTITLES=None,        # [(start_beat, end_beat, text)] -> OUT_DIR/<OUT_NAME>.srt
    HOLD_BEAT=None,        # beat where the final still hold starts (default: the endcard cue)
    ROUTE=None,            # walkthrough HUD: dict(steps=[(beat, title, short)], end=beat)
    MUSIC_STYLE='edm',     # 'edm' (燃版 supersaw EDM) | 'house' (chill house) | 'dnb' (172 BPM drum & bass)
)
REQUIRED = ('TOTAL', 'FONTS', 'CUES', 'MUSIC_SECTIONS')
FONT_KEYS = ('heavy', 'bold', 'semibold', 'medium', 'mono')
IMG_EXT = ('.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff')


def load_event(path):
    path = os.path.abspath(path)
    home = os.path.dirname(path)
    sys.path.insert(0, home)
    spec = importlib.util.spec_from_file_location('event', path)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    for k, v in DEFAULTS.items():
        if not hasattr(ev, k):
            setattr(ev, k, v)
    missing = [k for k in REQUIRED if not hasattr(ev, k)]
    if missing:
        sys.exit(f'event config {path} is missing: {", ".join(missing)}')
    ev.PATH, ev.HOME = path, home
    ev.BEAT = 60.0 / ev.BPM
    ev.WORK = os.environ.get('CPV_WORK') or getattr(ev, 'WORK', os.path.join(home, 'work'))
    ev.OUT_DIR = os.environ.get('CPV_OUT') or getattr(ev, 'OUT_DIR', os.path.join(home, 'out'))
    ev.CUES = sorted(ev.CUES, key=lambda c: c['b'])
    ends = [c['b'] for c in ev.CUES if c['kind'] == 'endcard']
    ev.ENDCARD_BEAT = ev.HOLD_BEAT if getattr(ev, 'HOLD_BEAT', None) is not None else (ends[0] if ends else None)
    if ev.COVER_AT is None:
        ev.COVER_AT = max(0.0, ev.TOTAL - 4.0)
    ev.PHOTOS = dict(ev.PHOTOS) if ev.PHOTOS else scan_photos(ev.PHOTO_DIRS, ev.PHOTO_ID_LEN)
    return ev


def scan_photos(dirs, id_len):
    """Photo id = the first `id_len` characters of the file name (e.g. 'B07_stage.jpg' -> 'B07'). Not recursive."""
    out = {}
    for d in dirs:
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(IMG_EXT) and not f.startswith('.'):
                pid = f[:id_len]
                if pid in out:
                    sys.exit(f'duplicate photo id {pid}: {out[pid]} and {os.path.join(d, f)}')
                out[pid] = os.path.join(d, f)
    return out


def cue_photo_ids(ev):
    ids = set()
    for c in ev.CUES:
        for k in ('pid', 'bg'):
            if isinstance(c.get(k), str):        # (a layers cue's bg is a dict, checked separately)
                ids.add(c[k])
    return ids


def check(ev):
    """Return a list of problems; empty means the config renders."""
    bad = []
    total_beats = ev.TOTAL / ev.BEAT
    for k in FONT_KEYS:
        p = ev.FONTS.get(k)
        if not p or not os.path.exists(p):
            bad.append(f'font {k!r} missing: {p}')
    for pid in sorted(cue_photo_ids(ev) | set(ev.ANCHOR) | set(ev.POIS)):
        if pid not in ev.PHOTOS:
            bad.append(f'photo id {pid} is used but not found in PHOTO_DIRS/PHOTOS')
    kinds = {c['kind'] for c in ev.CUES}
    if kinds & {'reveal', 'endcard'} and not (ev.KV and os.path.exists(ev.KV)):
        bad.append(f'KV image missing: {ev.KV}')
    if 'terminal' in kinds and not ev.TERMINAL:
        bad.append('a terminal cue needs TERMINAL')
    for c in ev.CUES:
        if c['b'] >= total_beats:
            bad.append(f"cue at beat {c['b']} starts after TOTAL ({total_beats:.1f} beats)")
        if c['kind'] == 'wall' and c['wall'] not in ev.WALLS:
            bad.append(f"wall {c['wall']!r} not in WALLS")
        if c['kind'] == 'trackwall' and c['track'] not in ev.TRACK_WALLS:
            bad.append(f"track {c['track']!r} not in TRACK_WALLS")
    def img_ok(ref, what):
        if not os.path.exists(ev.PHOTOS.get(ref, ref)):
            bad.append(f'{what}: image missing: {ref}')
    for c in ev.CUES:
        if c['kind'] == 'layers':
            if c.get('bg'):
                img_ok(c['bg']['img'], f"layers cue at beat {c['b']} bg")
            for L in c.get('layers', []):
                img_ok(L['img'], f"layers cue at beat {c['b']}")
        elif c['kind'] == 'carousel':
            for it in c['items']:
                img_ok(it['img'], f"carousel cue at beat {c['b']}")
        if c['kind'] == 'clip' and not os.path.exists(c.get('video', '')):
            bad.append(f"clip cue at beat {c['b']}: source video missing: {c.get('video')}")
        if c['kind'] == 'clip' and 'n' not in c:
            bad.append(f"clip cue at beat {c['b']} needs n (beats)")
        if c.get('xfade') and c is ev.CUES[0]:
            bad.append('the first cue cannot cross-fade (nothing before it)')
    for im in ev.ENDCARD_IMAGES:
        img_ok(im['img'], 'ENDCARD_IMAGES')
    for (a, z, txt) in (ev.SUBTITLES or []):
        if not 0 <= a < z <= total_beats + 1e-6:
            bad.append(f'subtitle {txt!r}: beats {a}-{z} outside 0-{total_beats:g}')
    for key, w in ev.WALLS.items():
        for row in w['rows']:
            for tier, items in row:
                if tier not in w['tiles']:
                    bad.append(f'wall {key}: tier {tier!r} has no tile size')
                for name, path in items:
                    if not os.path.exists(path):
                        bad.append(f'wall {key}: logo for {name} missing: {path}')
    if 'keynote' in kinds:
        ks = ev.keynotes()
        for c in ev.CUES:
            if c['kind'] == 'keynote' and c['k'] >= len(ks):
                bad.append(f"keynote cue k={c['k']} but keynotes() has {len(ks)}")
        for k in ks:
            p = k['photo'][0] if isinstance(k['photo'], tuple) else k['photo']
            if not os.path.exists(p):
                bad.append(f"keynote photo missing for {k['name']}: {p}")
    for code, tw in ev.TRACK_WALLS.items():
        ts = ev.track_talks(code)
        if len(ts) > 15:
            bad.append(f'track {code} has {len(ts)} talks; the wall fits 15 (5 x 3)')
        for t in ts:
            for p in t['photos']:
                p = p[0] if isinstance(p, tuple) else p
                if not os.path.exists(p):
                    bad.append(f"track {code}: photo missing for {t['names']}: {p}")
    secs = ev.MUSIC_SECTIONS
    for (a, z, *_), (a2, *_r) in zip(secs, secs[1:]):
        if z != a2:
            bad.append(f'MUSIC_SECTIONS gap/overlap between beat {z} and {a2}')
    if secs and secs[0][0] != 0:
        bad.append('MUSIC_SECTIONS must start at beat 0')
    return bad


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    ev = load_event(sys.argv[1])
    if sys.argv[2] == '--shell':
        vals = dict(WORK=ev.WORK, OUT_DIR=ev.OUT_DIR, NAME=ev.OUT_NAME, TOTAL=ev.TOTAL, COVER_AT=ev.COVER_AT,
                    LUFS=ev.LUFS, TRUE_PEAK=ev.TRUE_PEAK, SFX_GAIN_DB=ev.SFX_GAIN_DB, FPS=ev.FPS,
                    SILENT_NAME=ev.SILENT_NAME or '', HAS_SRT=int(bool(ev.SUBTITLES)),
                    HOLD_AT=(ev.ENDCARD_BEAT * ev.BEAT if ev.ENDCARD_BEAT is not None else max(0, ev.TOTAL - 6)))
        for k, v in vals.items():
            print(f'{k}={shlex.quote(str(v))}')
    elif sys.argv[2] == '--srt':     # write the subtitles (beats -> SRT timecodes) to the given path
        def tc(sec):
            ms = int(round(sec * 1000))
            return f'{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}'
        with open(sys.argv[3], 'w', encoding='utf-8') as f:
            for i, (a, z, txt) in enumerate(ev.SUBTITLES or [], 1):
                f.write(f'{i}\n{tc(a * ev.BEAT)} --> {tc(min(z * ev.BEAT, ev.TOTAL))}\n{txt}\n\n')
        print('wrote', sys.argv[3])
    elif sys.argv[2] == '--check':
        bad = check(ev)
        print(f'{ev.PATH}: {len(ev.CUES)} cues, {ev.TOTAL:g} s at {ev.BPM} BPM ({ev.TOTAL / ev.BEAT:.0f} beats), {len(ev.PHOTOS)} photos found')
        for b in bad:
            print('  ✗', b)
        print('OK' if not bad else f'{len(bad)} problem(s)')
        sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
