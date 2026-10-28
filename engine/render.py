#!/usr/bin/env python3
"""Conference promo video · frame renderer (dark high-contrast, beat-locked cuts, giant word slams, ff-tracking HUD).

Everything event-specific comes from the event config (see templates/event_template.py).

  python3 render.py EVENT.py              # -> WORK/video_noaudio.mp4
  python3 render.py EVENT.py --stills     # two check frames per cue + contact sheets -> WORK/stills/
  python3 render.py EVENT.py --at 63.2    # one frame at 63.2 s -> WORK/frame_63.20.png
"""
import argparse, math, os, subprocess, sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_event, check  # noqa: E402

WHITE = (255, 255, 255)
GREEN = (63, 185, 80)
MUTED = (200, 206, 222)
E = None


def setup(event_path):
    """Load an event config and build the module-level render state. Must run before render_frame()."""
    global E, W, H, FPS, BEAT, TOTAL, NF, CW, CH, CUES, DARK, ACCENT, ACCENT_SOFT, ACCENT_SUB
    global VIGNETTE, GRAIN, xx, yy, HOLD_FROM, F_MONO_T, F_MONO_S, F_MONO_M, F_TAG, F_HUD, F_SUB, F_PANO
    E = load_event(event_path)
    W, H, FPS, BEAT, TOTAL, CUES = E.W, E.H, E.FPS, E.BEAT, E.TOTAL, E.CUES
    NF = int(TOTAL * FPS)
    CW, CH = int(round(W * 1.1)), int(round(H * 1.1))      # 1.1x canvas for zoom headroom
    DARK, ACCENT, ACCENT_SOFT, ACCENT_SUB = E.DARK, E.ACCENT, E.ACCENT_SOFT, E.ACCENT_SUB
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    VIGNETTE = (1.0 - 0.42 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) ** 1.2).clip(0.45, 1.0)[..., None]
    rng = np.random.default_rng(11)
    GRAIN = [cv2.resize((rng.standard_normal((H // 2, W // 2)) * 4.0).astype(np.float32), (W, H)) for _ in range(6)]
    HOLD_FROM = E.ENDCARD_BEAT * BEAT + 5 / FPS if E.ENDCARD_BEAT is not None else TOTAL + 1   # flash done -> true still hold
    F_MONO_T, F_MONO_S, F_MONO_M = font('mono', 52), font('mono', 16), font('mono', 20)
    F_TAG, F_HUD, F_SUB, F_PANO = font('semibold', 26), font('medium', 20), font('semibold', 44), font('heavy', 86)
    return E


# ---------------------------------------------------------------- helpers
_fonts = {}


def font(key, size):
    if (key, size) not in _fonts:
        _fonts[(key, size)] = ImageFont.truetype(E.FONTS[key], size)
    return _fonts[(key, size)]


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * x)


def ease_out(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def rgba(c, a):
    return (c[0], c[1], c[2], int(max(0, min(255, a))))


# ---------------------------------------------------------------- photos
_canvas = {}


def load_rgb(path):
    """Open an image upright (phone photos often carry an EXIF rotation instead of rotated pixels)."""
    return ImageOps.exif_transpose(Image.open(path)).convert('RGB')


def grade(a):
    a = a.astype(np.float32)
    a = (a - 128) * 1.14 + 122
    return np.clip(a, 0, 255).astype(np.uint8)


def canvas(pid):
    """1.1x cover canvas (CW x CH) + mapping photo-normalised -> canvas coords."""
    if pid in _canvas:
        return _canvas[pid]
    im = load_rgb(E.PHOTOS[pid])
    iw, ih = im.size
    if ih > iw:  # portrait: blurred fill + full photo centred
        bg = im.resize((CW, int(ih * CW / iw)), Image.BILINEAR)
        top = (bg.height - CH) // 2
        bgc = np.asarray(bg)[top:top + CH]
        bgc = cv2.GaussianBlur(bgc, (0, 0), 28) * 0.35
        fh = CH
        fw = int(iw * fh / ih)
        fg = np.asarray(im.resize((fw, fh), Image.LANCZOS))
        out = bgc.astype(np.uint8).copy()
        x0 = (CW - fw) // 2
        out[:, x0:x0 + fw] = fg
        mp = (x0, 0, fw, fh)
    else:
        sc = max(CW / iw, CH / ih)
        rw, rh = int(round(iw * sc)), int(round(ih * sc))
        big = np.asarray(im.resize((rw, rh), Image.LANCZOS))
        a = E.ANCHOR.get(pid, E.ANCHOR_DEFAULT)
        oy = int((rh - CH) * a)
        ox = (rw - CW) // 2
        out = big[oy:oy + CH, ox:ox + CW].copy()
        mp = (-ox, -oy, rw, rh)
    _canvas[pid] = (grade(out), mp)
    return _canvas[pid]


_blur = {}


def blurred(pid):
    if pid not in _blur:
        c, _ = canvas(pid)
        small = cv2.resize(c, (CW // 4, CH // 4), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 6)
        b = cv2.resize(small, (CW, CH), interpolation=cv2.INTER_LINEAR).astype(np.float32) * 0.30
        _blur[pid] = b.astype(np.uint8)
    return _blur[pid]


def affine(scale, dx=0.0, dy=0.0, rot=0.0):
    """canvas (CW x CH) -> screen (W x H), scale relative to the 'fit' scale."""
    s = (W / CW) * scale
    M = cv2.getRotationMatrix2D((CW / 2, CH / 2), rot, s)
    M[0, 2] += W / 2 - CW / 2 + dx
    M[1, 2] += H / 2 - CH / 2 + dy
    return M


def warp(img, M):
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def to_screen(M, pts):
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))]
    return p @ M.T


def dark_bg(tl):
    img = np.empty((H, W, 3), np.float32)
    img[:] = DARK
    # faint grid + a slow accent glow
    img[::48, :] += 10
    img[:, ::48] += 10
    g = np.exp(-(((xx - W * (0.5 + 0.1 * math.sin(tl))) / 700) ** 2 + ((yy - H * 0.62) / 420) ** 2))[..., None]
    img += g * np.array(E.GLOW, np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- text sprites
_sprites = {}


def sprite(text, f, color=WHITE):
    key = (text, f.path, f.size, color)
    if key not in _sprites:
        bb = f.getbbox(text)
        w, h = bb[2] - bb[0] + 40, bb[3] - bb[1] + 40
        im = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.text((20 - bb[0], 20 - bb[1]), text, font=f, fill=color + (255,))
        _sprites[key] = np.asarray(im).astype(np.float32) / 255.0
    return _sprites[key]


def paste(frame, spr, cx, cy, scale=1.0, alpha=1.0):
    if scale != 1.0:
        spr = cv2.resize(spr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    h, w = spr.shape[:2]
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if xa >= xb or ya >= yb:
        return
    s = spr[ya - y0:yb - y0, xa - x0:xb - x0]
    a = s[..., 3:4] * alpha
    reg = frame[ya:yb, xa:xb].astype(np.float32)
    frame[ya:yb, xa:xb] = (reg * (1 - a) + s[..., :3] * 255 * a).astype(np.uint8)


def wrap_text(text, f, maxw, maxlines=3):
    """Keep latin words together, split CJK per character; ellipsis past maxlines."""
    lines, cur, toks, buf = [], '', [], ''
    for ch in text:
        if ch.isascii() and ch not in ' ':
            buf += ch
        else:
            if buf:
                toks.append(buf)
                buf = ''
            toks.append(ch)
    if buf:
        toks.append(buf)
    for t in toks:
        if f.getlength(cur + t) > maxw and cur.strip():
            lines.append(cur.strip())
            cur = t.lstrip()
        else:
            cur += t
    if cur.strip():
        lines.append(cur.strip())
    if len(lines) > maxlines:
        lines = lines[:maxlines]
        while f.getlength(lines[-1] + '…') > maxw and lines[-1]:
            lines[-1] = lines[-1][:-1]
        lines[-1] += '…'
    return lines


# ---------------------------------------------------------------- HUD (PIL overlay)
def brackets(d, r, col, a, lw=3, L=18):
    x0, y0, x1, y1 = r
    L = min(L, (x1 - x0) / 2.5, (y1 - y0) / 2.5)
    for (px, py, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x1, y1, -1, -1), (x0, y1, 1, -1)):
        d.line([(px, py), (px + sx * L, py)], fill=rgba(col, a), width=lw)
        d.line([(px, py), (px, py + sy * L)], fill=rgba(col, a), width=lw)


def coord(d, r, a, col=WHITE):
    cx, cy = (r[0] + r[2]) / 2, (r[1] + r[3]) / 2
    t = f'x: {int(cx)} y: {int(cy)}'
    tw = F_MONO_S.getlength(t)
    tx, ty = min(max(r[0], 8), W - tw - 12), r[1] - 26
    d.rectangle([tx - 5, ty - 3, tx + tw + 5, ty + 20], fill=rgba((0, 0, 0), a * 0.6))
    d.text((tx, ty), t, font=F_MONO_S, fill=rgba(col, a))


def hatch(d, r, col, a, step=9):
    x0, y0, x1, y1 = [int(v) for v in r]
    for k in range(-(y1 - y0), x1 - x0, step):
        xa, ya, xb, yb = x0 + k, y0, x0 + k + (y1 - y0), y1
        if xa < x0:
            ya += x0 - xa
            xa = x0
        if xb > x1:
            yb -= xb - x1
            xb = x1
        if xa <= xb:
            d.line([(xa, ya), (xb, yb)], fill=rgba(col, a), width=1)


def hud_common(d, beat, tg, recap=True, alpha=255):
    if recap and E.RECAP and E.RECAP[0] <= beat < E.RECAP[1]:
        tag = E.RECAP[2]
        tw = F_TAG.getlength(tag)
        d.rounded_rectangle([64, 52, 64 + tw + 58, 98], radius=23, fill=rgba((0, 0, 0), alpha * 0.55))
        d.ellipse([82, 68, 96, 82], fill=rgba(ACCENT, alpha))
        d.text((106, 58), tag, font=F_TAG, fill=rgba(WHITE, alpha))
    if not E.HUD:
        return
    bar, bt = int(beat // 4) + 1, int(beat % 4) + 1
    lx = 64
    if E.HUD_BAR_LABEL:
        d.text((64, H - 58), E.HUD_BAR_LABEL, font=F_HUD, fill=rgba(WHITE, alpha * 0.55))
        lx = 64 + F_HUD.getlength(E.HUD_BAR_LABEL) + 8
    d.text((lx, H - 58), f'{bar:02d}.{bt}   {E.BPM:g} BPM', font=F_MONO_M, fill=rgba(WHITE, alpha * 0.55))
    tc = f'{int(tg // 60):02d}:{tg % 60:05.2f}'
    d.text((W - 64 - F_MONO_M.getlength(tc), H - 58), tc, font=F_MONO_M, fill=rgba(WHITE, alpha * 0.55))
    for k in range(4):  # beat ticks
        on = k == int(beat % 4)
        d.rectangle([W / 2 - 54 + k * 30, H - 50, W / 2 - 34 + k * 30, H - 44], fill=rgba(ACCENT if on else WHITE, alpha * (0.95 if on else 0.3)))


# ---------------------------------------------------------------- cue renderers
def r_photo(c, tl, beat, tg):
    img, mp = canvas(c['pid'])
    d_ = c['n'] * BEAT
    m = c.get('motion', 'punch')
    if m == 'punch':
        s = 1.0 + 0.11 * (1 - ease_out(tl / 0.30)) + 0.03 * (tl / d_)
    elif m == 'push':
        s = 1.0 + 0.07 * ease(tl / d_)
    else:  # flash
        s = 1.06 - 0.03 * (tl / d_)
    lb = tl / BEAT   # beat pulse inside longer shots
    pulse = 0.012 * max(0.0, 1 - (lb - math.floor(lb)) * 5) if lb >= 1 else 0.0
    rng = np.random.default_rng(int(c['b'] * 7))
    dx, dy = (rng.random() - 0.5) * 30 * (tl / d_), (rng.random() - 0.5) * 16 * (tl / d_)
    M = affine(s + pulse, dx, dy)
    frame = warp(img, M)
    rects = []
    for (u, v, w, h) in E.POIS.get(c['pid'], []):
        ox, oy, rw, rh = mp
        pts = [(ox + (u - w / 2) * rw, oy + (v - h / 2) * rh), (ox + (u + w / 2) * rw, oy + (v + h / 2) * rh)]
        q = to_screen(M, pts)
        rects.append((q[0, 0], q[0, 1], q[1, 0], q[1, 1]))
    return frame, rects


def r_slam(c, tl, beat, tg):
    if c.get('bg'):
        frame = warp(blurred(c['bg']), affine(1.04 + 0.03 * tl))
    else:
        frame = dark_bg(tg)
    spr = sprite(c['text'], font('heavy', c.get('size', 150)))
    fi = tl * FPS
    k = ease_out(fi / 6)
    sc = 1.45 - 0.45 * k
    a = min(1.0, fi / 3 + 0.2)
    amp = 14 * max(0.0, 1 - fi / 8)  # shake
    sx, sy = math.sin(fi * 2.1) * amp, math.cos(fi * 2.7) * amp
    cy = H / 2 - (30 if c.get('sub') else 0)
    for g, ga in ((0.22, 0.18), (0.11, 0.3)):  # motion ghosts during the slam
        if fi < 6:
            paste(frame, spr, W / 2 + sx, cy + sy, sc + g * (1 - k), a * ga)
    paste(frame, spr, W / 2 + sx, cy + sy, sc, a)
    tw = spr.shape[1] * sc - 40  # accent bar grows under the word
    bw = tw * ease_out(fi / 7)
    y = int(cy + spr.shape[0] * sc / 2 + 6 + sy)
    x0 = int(W / 2 - bw / 2 + sx)
    frame[y:y + 10, max(0, x0):min(W, x0 + int(bw))] = ACCENT
    if c.get('sub'):
        ss = sprite(c['sub'], F_SUB, ACCENT_SUB)
        paste(frame, ss, W / 2 + sx, y + 60, 1.0, min(1.0, max(0.0, (fi - 4) / 4)))
    if fi < 4 and c.get('cut', 'flash') == 'flash':  # glitch slices on entry
        rng = np.random.default_rng(int(c['b'] * 13 + fi))
        for _ in range(5):
            y0 = int(rng.uniform(H * 0.3, H * 0.7))
            h = int(rng.uniform(8, 40))
            frame[y0:y0 + h] = np.roll(frame[y0:y0 + h], int(rng.uniform(-60, 60)), axis=1)
    return frame, []


def r_terminal(c, tl, beat, tg):
    """A typed command, then result lines one per beat from beat 4, tracker lock on beat 7, glitch out (an 8-beat cue)."""
    T = E.TERMINAL
    lb = beat - c['b']
    frame = dark_bg(tg)
    base = Image.fromarray(frame).convert('RGBA')
    d = ImageDraw.Draw(base)
    x, y0, lh = 200, 330, 84
    cmd = T['cmd']
    n = int(max(0.0, tl - 0.25) * 22)
    d.text((x, y0), cmd[:n], font=F_MONO_T, fill=(235, 240, 250, 255))
    if n < len(cmd) or (int(tl * 4) % 2 == 0 and lb < 7):
        cx = x + F_MONO_T.getlength(cmd[:n])
        d.rectangle([cx + 4, y0 + 8, cx + 30, y0 + 62], fill=ACCENT + (255,))
    run_rect = None
    lines = T['lines']
    for i, (verb, rest) in enumerate(lines):
        if lb >= 4 + i:
            yy_ = y0 + lh * (i + 1)
            d.text((x, yy_), verb, font=F_MONO_T, fill=GREEN + (255,))
            d.text((x + F_MONO_T.getlength(verb), yy_), rest, font=F_MONO_T, fill=(200, 205, 220, 255))
            if i == len(lines) - 1 and T.get('lock') and T['lock'] in rest:
                xs = x + F_MONO_T.getlength(verb + rest[:rest.index(T['lock'])])
                run_rect = (xs - 8, yy_ - 2, xs + F_MONO_T.getlength(T['lock']) + 8, yy_ + 66)
    if run_rect is not None:
        col = ACCENT if 7.0 <= lb < 7.0 + 3 / FPS / BEAT else WHITE
        brackets(d, run_rect, col, 255)
        coord(d, run_rect, 255, col)
    frame = np.asarray(base.convert('RGB')).copy()
    if lb >= 7.6:  # glitch out into the first slam
        rng = np.random.default_rng(int(tl * 100))
        for _ in range(8):
            yy0 = int(rng.uniform(0, H))
            frame[yy0:yy0 + 24] = np.roll(frame[yy0:yy0 + 24], int(rng.uniform(-120, 120)), axis=1)
    return frame, []


def r_pano(c, tl, beat, tg):
    key = ('pano', c['pid'])
    if key not in _canvas:
        im = load_rgb(E.PHOTOS[c['pid']])
        w = int(im.width * CH / im.height)
        _canvas[key] = grade(np.asarray(im.resize((w, CH), Image.LANCZOS)))
    img = _canvas[key]
    u = ease(tl / (c['n'] * BEAT))
    s = H / CH
    xoff = -(img.shape[1] * s - W) * (0.1 + 0.8 * u)
    M = np.array([[s, 0, xoff], [0, s, 0]], np.float32)
    frame = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    gr = np.linspace(0, 1, 360, dtype=np.float32)[:, None, None] ** 1.6   # bottom gradient
    frame[H - 360:] = (frame[H - 360:].astype(np.float32) * (1 - 0.75 * gr)).astype(np.uint8)
    spr = sprite(c['text'], F_PANO)
    a = ease((tl - 2 * BEAT) / (1.5 * BEAT))
    paste(frame, spr, W / 2, H - 170 + 12 * (1 - a), 1.0, a)
    return frame, []


_kv = None


def kv_page(scale):
    global _kv
    if _kv is None:
        im = Image.open(E.KV).convert('RGB').resize((W, H), Image.LANCZOS)
        a = np.asarray(im).astype(np.float32)
        xs = np.linspace(0, 1, W)[None, :, None]
        ys = np.linspace(0, 1, H)[:, None, None]
        scrim = np.clip(1.15 - xs * 1.9, 0, 0.66) * np.clip(1.2 - ys * 0.9, 0.3, 1)   # darken the left for text
        _kv = (a * (1 - scrim) + np.array(E.KV_SCRIM, np.float32) * scrim).astype(np.uint8)
    M = cv2.getRotationMatrix2D((W * 0.62, H * 0.55), 0, scale)
    return cv2.warpAffine(_kv, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def r_reveal(c, tl, beat, tg):
    lb = tl / BEAT
    frame = kv_page(1.06 - 0.06 * ease_out(lb / 4))
    base = Image.fromarray(frame).convert('RGBA')
    ov = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    lock = None
    for ln in E.ENDCARD_LINES:
        f = font(*ln['font'])
        p = ease_out((lb - ln['delay']) / 0.45)
        if ln.get('lock'):
            lock = (ln, f)
        if p <= 0:
            continue
        d.text((150 - 40 * (1 - p), ln['y']), ln['text'], font=f, fill=rgba(ln.get('color', WHITE), 255 * min(1, p * 1.4)))
    for im in E.ENDCARD_IMAGES:   # e.g. a badge or venue logo, fading in with the lines
        p = ease_out((lb - im.get('delay', 0.0)) / 0.45)
        if p > 0:
            spr = pil_image(im['img'], im['w'], im.get('pixel', False))
            spr = spr.copy()
            spr.putalpha(spr.getchannel('A').point(lambda v: int(v * min(1, p * 1.4))))
            ov.alpha_composite(spr, (int(im['x']), int(im['y'])))
    if lock:  # tracker brackets lock onto one line (the URL) a beat after it lands
        ln, f = lock
        r = (134, ln['y'] - 4, 166 + f.getlength(ln['text']), ln['y'] + round(f.size * 1.42))
        t_lock = ln['delay'] + 1.0
        lp = (lb - t_lock) / 0.4
        if lp > 0:
            k = ease_out(lp)
            cx, cy = (r[0] + r[2]) / 2, (r[1] + r[3]) / 2
            hw, hh = (r[2] - r[0]) / 2 * (1 + 0.5 * (1 - k)), (r[3] - r[1]) / 2 * (1 + 0.5 * (1 - k))
            flash = t_lock <= lb < t_lock + 3 / FPS / BEAT
            brackets(d, (cx - hw, cy - hh, cx + hw, cy + hh), ACCENT if flash else ACCENT_SOFT, 255 * min(1, lp))
    base.alpha_composite(ov)
    out = np.asarray(base.convert('RGB')).copy()
    fi = tl * FPS
    if fi < 6 and c.get('impact', True):  # impact shake (impact=False for a calm reveal)
        amp = 16 * (1 - fi / 6)
        out = np.roll(out, (int(math.sin(fi * 2.3) * amp), int(math.cos(fi * 1.9) * amp)), axis=(0, 1))
    return out, []


def r_endcard(c, tl, beat, tg):
    return r_reveal(c, 30.0, beat, tg)[0], []


# ---------------------------------------------------------------- sponsor walls
_logo = {}


def logo_rgba(name, path):
    """Load a logo as float RGBA and trim the empty transparent / white export margin (geometry untouched)."""
    if path not in _logo:
        im = Image.open(path).convert('RGBA')
        a = np.asarray(im).astype(np.float32) / 255.0
        if name in E.LOGO_DARK_PLATE:          # white artwork on transparency: only alpha marks ink
            ink = a[..., 3] > 0.04
        else:
            ink = (a[..., 3] > 0.04) & (a[..., :3].min(2) < 0.94)
        ys, xs = np.where(ink)
        if len(ys):
            a = a[max(ys.min() - 2, 0):ys.max() + 3, max(xs.min() - 2, 0):xs.max() + 3]
        if name in E.LOGO_BLUR:   # compliance: unreadable — pixelate, then blur hard
            h, w = a.shape[:2]
            small = cv2.resize(a, (max(1, w // 24), max(1, h // 24)), interpolation=cv2.INTER_AREA)
            a = cv2.GaussianBlur(cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR), (0, 0), max(w, h) / 40)
        _logo[path] = a
    return _logo[path]


_wall_layout = {}


def wall_layout(key):
    if key in _wall_layout:
        return _wall_layout[key]
    wd = E.WALLS[key]
    rows, tile = wd['rows'], wd['tiles']
    LBL, GAP_T, GAP_G, GAP_R = 40, 22, 90, 24
    heights = [LBL + max(tile[t][1] for t, _ in row) for row in rows]
    total = sum(heights) + GAP_R * (len(rows) - 1)
    y = 215 + max(0, (H - 40 - 215 - total) // 2)
    groups = []
    for row, hgt in zip(rows, heights):
        widths = [len(items) * tile[t][0] + (len(items) - 1) * GAP_T for t, items in row]
        x = (W - (sum(widths) + GAP_G * (len(row) - 1))) / 2
        for (tier, items), gw in zip(row, widths):
            tw, th = tile[tier]
            tiles = [(name, path, (x + i * (tw + GAP_T), y + LBL, tw, th)) for i, (name, path) in enumerate(items)]
            groups.append(dict(tier=tier, x=x, y=y, w=gw, h=LBL + th, tiles=tiles))
            x += gw + GAP_G
        y += hgt + GAP_R
    _wall_layout[key] = groups
    return groups


def draw_logo_tile(frame, name, path, rect, k, plate):
    x, y, w, h = rect
    sc = 0.82 + 0.18 * k
    cx, cy = x + w / 2, y + h / 2
    w2, h2 = w * sc, h * sc
    tile = Image.new('RGBA', (int(w2), int(h2)), (0, 0, 0, 0))
    ImageDraw.Draw(tile).rounded_rectangle([0, 0, int(w2) - 1, int(h2) - 1], radius=int(16 * sc), fill=plate + (255,))
    t = np.asarray(tile).astype(np.float32) / 255.0
    lg = logo_rgba(name, path)
    beside = name in E.LOGO_NAME_BESIDE
    if beside:  # mark on the left, organisation name typeset to its right
        mh = h2 * 0.72
        lw = lg.shape[1] * mh / lg.shape[0]
        box = (w2 * 0.06, (h2 - mh) / 2, lw, mh)
    else:
        bw, bh = w2 * 0.78, h2 * 0.64
        s_ = min(bw / lg.shape[1], bh / lg.shape[0])
        lw, lh = lg.shape[1] * s_, lg.shape[0] * s_
        box = ((w2 - lw) / 2, (h2 - lh) / 2, lw, lh)
    lr = cv2.resize(lg, (max(1, int(box[2])), max(1, int(box[3]))), interpolation=cv2.INTER_AREA)
    bx, by = int(box[0]), int(box[1])
    hh, ww = min(lr.shape[0], t.shape[0] - by), min(lr.shape[1], t.shape[1] - bx)
    reg = t[by:by + hh, bx:bx + ww]
    la = lr[:hh, :ww, 3:4] * reg[..., 3:4]
    reg[..., :3] = reg[..., :3] * (1 - la) + lr[:hh, :ww, :3] * la
    if beside:
        im = Image.fromarray((t * 255).astype(np.uint8), 'RGBA')
        d = ImageDraw.Draw(im)
        fs = int(h2 * 0.2)
        f = font('semibold', max(fs, 10))
        avail = w2 - (bx + ww) - w2 * 0.08
        while f.getlength(name) > avail and fs > 10:
            fs -= 1
            f = font('semibold', fs)
        d.text((bx + ww + w2 * 0.04, h2 / 2 - fs * 0.62), name, font=f, fill=(20, 24, 40, 255))
        t = np.asarray(im).astype(np.float32) / 255.0
    paste(frame, t, cx, cy, 1.0, k)


def r_wall(c, tl, beat, tg):
    wd = E.WALLS[c['wall']]
    frame = dark_bg(tg)
    lb = tl / BEAT
    groups = wall_layout(c['wall'])
    base = Image.fromarray(frame).convert('RGBA')
    ov = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    k = ease_out(tl * FPS / 6)   # header: pill + title slam in on the first beat
    f_pill, f_title, f_tier, f_note = font('semibold', 28), font('heavy', 76), font('semibold', 27), font('medium', 22)
    pill, title = wd['pill'], wd['title']
    pw = f_pill.getlength(pill) + 56
    ghost = wd.get('pill_style', 'accent') == 'ghost'      # ghost = a past edition, accent = this year
    pfill = (255, 255, 255, int(40 * k)) if ghost else rgba(ACCENT, 255 * k)
    d.rounded_rectangle([W / 2 - pw / 2, 52, W / 2 + pw / 2, 98], radius=23, fill=pfill, outline=rgba(WHITE, 160 * k) if ghost else None, width=2)
    d.text((W / 2 - pw / 2 + 28, 57), pill, font=f_pill, fill=rgba(WHITE, 255 * k))
    tw = f_title.getlength(title)
    d.text((W / 2 - tw / 2, 112 + 14 * (1 - k)), title, font=f_title, fill=rgba(WHITE, 255 * k))
    if wd.get('note'):
        d.text((W / 2 - f_note.getlength(wd['note']) / 2, H - 48), wd['note'], font=f_note, fill=rgba(WHITE, 150 * ease((lb - 1) / 1.5)))
    for gi, g in enumerate(groups):
        ga = ease_out((lb - (0.8 + 0.6 * gi)) * 4)
        if ga <= 0:
            continue
        lab = g['tier']
        d.text((g['x'] + g['w'] / 2 - f_tier.getlength(lab) / 2, g['y'] - 2), lab, font=f_tier, fill=rgba(ACCENT_SOFT, 255 * ga))
        since = (lb - (0.8 + 0.6 * gi)) * BEAT * FPS   # tracker brackets lock onto the tier group as it lands
        r = (g['x'] - 14, g['y'] + 30, g['x'] + g['w'] + 14, g['y'] + g['h'] + 12)
        if since < 3:
            brackets(d, r, ACCENT, 255, lw=4, L=22)
        elif since < 14:
            brackets(d, r, WHITE, 255 * (1 - (since - 3) / 11), lw=2, L=22)
    base.alpha_composite(ov)
    frame = np.asarray(base.convert('RGB')).copy()
    for gi, g in enumerate(groups):
        for ti, (name, path, rect) in enumerate(g['tiles']):
            kk = ease_out(((lb - (0.8 + 0.6 * gi)) * BEAT * FPS - ti * 2) / 6)
            if kk <= 0:
                continue
            plate = (20, 33, 61) if name in E.LOGO_DARK_PLATE else tuple(wd.get('plate', (255, 255, 255)))
            draw_logo_tile(frame, name, path, rect, kk, plate)
    return frame, []


# ---------------------------------------------------------------- speakers & talks
_face = {}


def face(photo, size, shape='round'):
    """Speaker photo as a float RGBA square. `photo` is a path or (path, crop_box)."""
    key = (photo, size, shape)
    if key in _face:
        return _face[key]
    path, box = photo if isinstance(photo, tuple) else (photo, None)
    best = load_rgb(path)
    if box:
        best = best.crop(box)
    w, h = best.size
    sq = min(w, h)
    best = best.crop(((w - sq) // 2, int((h - sq) * 0.3), (w - sq) // 2 + sq, int((h - sq) * 0.3) + sq)).resize((size, size), Image.LANCZOS)
    mask = Image.new('L', (size, size), 0)
    md = ImageDraw.Draw(mask)
    if shape == 'round':
        md.ellipse([0, 0, size - 1, size - 1], fill=255)
    else:
        md.rounded_rectangle([0, 0, size - 1, size - 1], radius=size // 14, fill=255)
    best.putalpha(mask)
    _face[key] = np.asarray(best).astype(np.float32) / 255.0
    return _face[key]


_keynotes = None


def keynote_list():
    global _keynotes
    if _keynotes is None:
        _keynotes = E.keynotes()
    return _keynotes


def r_keynote(c, tl, beat, tg):
    """One keynote card: photo left (tracker locks on beat 2), tag pill, name, role, talk title, host line."""
    k = keynote_list()[c['k']]
    frame = dark_bg(tg)
    lf = tl * FPS
    ph = face(k['photo'], 500, 'square')   # slides in and settles
    kk = ease_out(lf / 7)
    paste(frame, ph, 170 + 250 - 60 * (1 - kk), 290 + 250, 1.06 - 0.06 * kk, min(1, kk * 1.5))
    base = Image.fromarray(frame).convert('RGBA')
    ov = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    r = (170, 290, 670, 790)
    lock = BEAT * FPS <= lf < BEAT * FPS + 3
    if lf > 3:
        rr = (r[0] - 16, r[1] - 16, r[2] + 16, r[3] + 16)
        brackets(d, rr, ACCENT if lock else WHITE, 255, lw=4 if lock else 3, L=30)
        coord(d, rr, 255, ACCENT if lock else WHITE)
    x, y = 780, 300

    def line(i):
        return min(1.0, max(0.0, (lf - 2 - 2 * i) / 4))
    f_tag, f_name, f_role, f_title, f_host = font('semibold', 30), font('heavy', 96), font('medium', 32), font('bold', 54), font('medium', 26)
    a = line(0)
    tw = f_tag.getlength(k['tag'])
    d.rounded_rectangle([x, y, x + tw + 40, y + 50], radius=25, fill=rgba(ACCENT, 255 * a))
    d.text((x + 20, y + 6), k['tag'], font=f_tag, fill=rgba(WHITE, 255 * a))
    y += 78
    a = line(1)
    d.text((x + 20 * (1 - a), y), k['name'], font=f_name, fill=rgba(WHITE, 255 * a))
    y += 118
    a = line(2)
    for ln in wrap_text(k['role'], f_role, 1000, 2):
        d.text((x, y), ln, font=f_role, fill=rgba(MUTED, 255 * a))
        y += 44
    y += 18
    d.rectangle([x, y, x + 90 * line(3), y + 6], fill=rgba(ACCENT, 255))
    y += 30
    a = line(3)
    for ln in wrap_text(k['title'], f_title, 1020, 3):
        d.text((x, y), ln, font=f_title, fill=rgba(WHITE, 255 * a))
        y += 70
    if k.get('host'):
        d.text((x, y + 10), k['host'], font=f_host, fill=rgba(ACCENT_SOFT, 255 * line(4)))
    hud_common(d, beat, tg, recap=False)
    base.alpha_composite(ov)
    return np.asarray(base.convert('RGB')).copy(), []


_tracks = {}


def r_trackwall(c, tl, beat, tg):
    """All talks of one track as a 5-column wall (max 15), tiles landing one by one."""
    code = c['track']
    hd = E.TRACK_WALLS[code]
    if code not in _tracks:
        _tracks[code] = E.track_talks(code)
    ks = _tracks[code]
    frame = dark_bg(tg)
    lb = tl / BEAT
    lf = tl * FPS
    base = Image.fromarray(frame).convert('RGBA')
    ov = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    f_t, f_code, f_pill, f_sub = font('heavy', 78), font('mono', 30), font('semibold', 28), font('medium', 26)
    f_time, f_title = font('mono', 17), font('semibold', 23)
    hk = ease_out(lf / 6)
    lab = hd.get('label', '')
    cw = f_code.getlength(code) + f_pill.getlength(lab)
    d.rounded_rectangle([W / 2 - cw / 2 - 24, 48, W / 2 + cw / 2 + 24, 96], radius=24, fill=rgba(ACCENT, 255 * hk))
    d.text((W / 2 - cw / 2, 54), code, font=f_code, fill=rgba(WHITE, 255 * hk))
    d.text((W / 2 - cw / 2 + f_code.getlength(code), 55), lab, font=f_pill, fill=rgba(WHITE, 255 * hk))
    t = hd['name']
    d.text((W / 2 - f_t.getlength(t) / 2, 104 + 12 * (1 - hk)), t, font=f_t, fill=rgba(WHITE, 255 * hk))
    sub = hd.get('sub', '').format(n=len(ks))
    d.text((W / 2 - f_sub.getlength(sub) / 2, 202), sub, font=f_sub, fill=rgba(MUTED, 255 * hk))
    cols = 5
    rows = (len(ks) + cols - 1) // cols
    TWD, THT, GX, GY = 344, 226, 18, 18
    top = 262 + max(0, 3 - rows) * (THT + GY) // 2
    tiles = []
    for i, k in enumerate(ks):
        r_, c_ = divmod(i, cols)
        nrow = min(cols, len(ks) - r_ * cols)
        x0 = (W - (nrow * TWD + (nrow - 1) * GX)) / 2 + c_ * (TWD + GX)
        tiles.append((k, x0, top + r_ * (THT + GY)))
    faces = []
    for i, (k, x0, y0) in enumerate(tiles):
        appear = (lb - 0.6 - i * 0.18) * BEAT * FPS     # frames since this tile landed
        a = ease_out(appear / 5)
        if a <= 0:
            continue
        dy = 18 * (1 - a)
        d.rounded_rectangle([x0, y0 + dy, x0 + TWD, y0 + THT + dy], radius=16, fill=rgba((22, 28, 46), 235 * a), outline=rgba(WHITE, 40 * a), width=1)
        names = k['names']
        fs = 25
        f = font('bold', fs)
        while f.getlength(names) > TWD - 128 and fs > 15:
            fs -= 1
            f = font('bold', fs)
        d.text((x0 + 116, y0 + 24 + dy), names, font=f, fill=rgba(WHITE, 255 * a))
        badge = k.get('badge') or ''
        tm = k['when'] + ('  ' if badge else '')
        d.text((x0 + 116, y0 + 60 + dy), tm, font=f_time, fill=rgba(ACCENT_SOFT, 255 * a))
        if badge:
            d.text((x0 + 116 + f_time.getlength(tm), y0 + 57 + dy), badge, font=F_HUD, fill=rgba(ACCENT, 255 * a))
        yy_ = y0 + 108 + dy
        for ln in wrap_text(k['title'], f_title, TWD - 36, 3):
            d.text((x0 + 18, yy_), ln, font=f_title, fill=rgba((235, 238, 246), 255 * a))
            yy_ += 32
        if 0 <= appear < 3:
            brackets(d, (x0 - 6, y0 - 6 + dy, x0 + TWD + 6, y0 + THT + 6 + dy), ACCENT, 255, lw=3, L=18)
        faces.append((k, x0, y0 + dy, a))
    hud_common(d, beat, tg, recap=False)
    base.alpha_composite(ov)
    out = np.asarray(base.convert('RGB')).copy()
    for k, x0, y0, a in faces:
        avs = k['photos']
        if not avs:   # e.g. a panel: accent disc with a short label
            disc = Image.new('RGBA', (84, 84), (0, 0, 0, 0))
            dd = ImageDraw.Draw(disc)
            dd.ellipse([0, 0, 83, 83], fill=ACCENT + (255,))
            ff = font('bold', 26)
            dd.text((42 - ff.getlength(E.PANEL_PLACEHOLDER) / 2, 24), E.PANEL_PLACEHOLDER, font=ff, fill=(255, 255, 255, 255))
            paste(out, np.asarray(disc).astype(np.float32) / 255.0, x0 + 18 + 42, y0 + 18 + 42, 1.0, a)
        else:
            for j, av in enumerate(avs[:2]):
                sz = 84 if len(avs) == 1 else 64
                paste(out, face(av, sz), x0 + 18 + sz / 2 + j * 40, y0 + 18 + sz / 2 + j * 20, 1.0, a)
    return out, []


# ---------------------------------------------------------------- generic image layers (layers / carousel cues)
_img = {}


def img_path(ref):
    """A layer image is a file path or a photo id from PHOTOS."""
    return E.PHOTOS.get(ref, ref)


def rgba_image(ref, w, pixel=False):
    """Float RGBA (0..1) scaled to width w (aspect kept). pixel=True scales with nearest neighbour (8-bit art)."""
    key = ('rgba', ref, int(w), pixel)
    if key not in _img:
        im = ImageOps.exif_transpose(Image.open(img_path(ref))).convert('RGBA')
        h = max(1, int(round(im.height * w / im.width)))
        im = im.resize((int(w), h), Image.NEAREST if pixel else Image.LANCZOS)
        _img[key] = np.asarray(im).astype(np.float32) / 255.0
    return _img[key]


def pil_image(ref, w, pixel=False):
    key = ('pil', ref, int(w), pixel)
    if key not in _img:
        _img[key] = Image.fromarray((rgba_image(ref, w, pixel) * 255).astype(np.uint8), 'RGBA')
    return _img[key]


def framed(ref, w, border=None, radius=14, shadow=0.55, pad=48):
    """A screenshot as a floating window: rounded corners, thin border, soft drop shadow (baked once)."""
    key = ('framed', ref, int(w), border, radius, shadow)
    if key not in _img:
        src = rgba_image(ref, w)
        h, w = src.shape[0], int(w)
        out = np.zeros((h + 2 * pad, w + 2 * pad, 4), np.float32)
        sh = np.zeros(out.shape[:2], np.float32)
        cv2.rectangle(sh, (pad, pad + 14), (pad + w, pad + h + 14), 1.0, -1)
        out[..., 3] = cv2.GaussianBlur(sh, (0, 0), 18) * shadow
        mask = Image.new('L', (w, h), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=255)
        m = (np.asarray(mask).astype(np.float32) / 255.0)[..., None]
        reg = out[pad:pad + h, pad:pad + w]
        reg[..., :3] = src[..., :3] * m + reg[..., :3] * (1 - m)
        reg[..., 3:4] = np.maximum(reg[..., 3:4] * (1 - m), m * src[..., 3:4])
        if border:
            bd = Image.new('RGBA', (w, h), (0, 0, 0, 0))
            ImageDraw.Draw(bd).rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, outline=tuple(border) + (255,), width=3)
            b_ = np.asarray(bd).astype(np.float32) / 255.0
            ba = b_[..., 3:4]
            reg[..., :3] = reg[..., :3] * (1 - ba) + b_[..., :3] * ba
            reg[..., 3:4] = np.maximum(reg[..., 3:4], ba)
        _img[key] = out
    return _img[key]


def place(frame, img, cx, cy, scale=1.0, rot=0.0, alpha=1.0, nearest=False):
    """Composite a float RGBA image centred at (cx, cy) with scale/rotation; only the covered region is warped."""
    if alpha <= 0:
        return
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rot, scale)
    M[0, 2] += cx - w / 2
    M[1, 2] += cy - h / 2
    pts = to_screen(M, [(0, 0), (w, 0), (w, h), (0, h)])
    x0, y0 = np.floor(pts.min(0)).astype(int)
    x1, y1 = np.ceil(pts.max(0)).astype(int)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if xa >= xb or ya >= yb:
        return
    M[0, 2] -= xa
    M[1, 2] -= ya
    r = cv2.warpAffine(img, M, (xb - xa, yb - ya), flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    a = r[..., 3:4] * alpha
    reg = frame[ya:yb, xa:xb].astype(np.float32)
    frame[ya:yb, xa:xb] = (reg * (1 - a) + r[..., :3] * 255 * a).astype(np.uint8)


def place_quad(frame, img, quad, alpha=1.0):
    """Composite a float RGBA image onto an arbitrary quad (perspective), clockwise from top-left."""
    if alpha <= 0:
        return
    h, w = img.shape[:2]
    q = np.asarray(quad, np.float32)
    x0, y0 = np.floor(q.min(0)).astype(int)
    x1, y1 = np.ceil(q.max(0)).astype(int)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if xa >= xb or ya >= yb:
        return
    M = cv2.getPerspectiveTransform(np.float32([(0, 0), (w, 0), (w, h), (0, h)]), q - np.float32([xa, ya]))
    r = cv2.warpPerspective(img, M, (xb - xa, yb - ya), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    a = r[..., 3:4] * alpha
    reg = frame[ya:yb, xa:xb].astype(np.float32)
    frame[ya:yb, xa:xb] = (reg * (1 - a) + r[..., :3] * 255 * a).astype(np.uint8)


def pixelate(img, block):
    if block <= 1:
        return img
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


_bg = {}


def cover_bg(ref, blur=0.0, dim=1.0):
    """A full-bleed background on the 1.1x canvas (cover fit), optionally blurred and dimmed."""
    key = (ref, blur, dim)
    if key not in _bg:
        im = ImageOps.exif_transpose(Image.open(img_path(ref))).convert('RGB')
        sc = max(CW / im.width, CH / im.height)
        im = im.resize((int(round(im.width * sc)), int(round(im.height * sc))), Image.LANCZOS)
        x0, y0 = (im.width - CW) // 2, (im.height - CH) // 2
        a = np.asarray(im)[y0:y0 + CH, x0:x0 + CW].astype(np.float32)
        if blur:
            a = cv2.GaussianBlur(a, (0, 0), blur)
        _bg[key] = np.clip(a * dim, 0, 255).astype(np.uint8)
    return _bg[key]


def palette(ref):
    """The image's hot colour: mean of its most saturated and bright pixels (drives the colour wash)."""
    key = ('pal', ref)
    if key not in _img:
        a = np.asarray(Image.open(img_path(ref)).convert('RGB').resize((160, 90)))
        hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
        score = hsv[:, 1] * hsv[:, 2]
        top = a.reshape(-1, 3)[score >= np.quantile(score, 0.92)].astype(np.float32)
        _img[key] = tuple(int(v) for v in top.mean(0))
    return _img[key]


def shade_left(frame, strength):
    """Darken the left side for text (like the KV scrim)."""
    if strength <= 0:
        return frame
    k = np.clip(1.15 - np.linspace(0, 1, W, dtype=np.float32) * 1.7, 0, 1)[None, :, None] * strength
    return (frame.astype(np.float32) * (1 - k) + np.array(DARK, np.float32) * k).astype(np.uint8)


def draw_texts(d, texts, lb, lf):
    """Timed text items: slide + fade in at `at` beats; optional pill background, blinking cursor, tracker lock."""
    for t in texts:
        p = ease_out((lb - t.get('at', 0)) / t.get('dur', 0.6))
        if p <= 0 or ('until' in t and lb >= t['until']):
            continue
        f = font(*t['font'])
        txt = t['text']
        tw = f.getlength(txt)
        x = t['x'] + t.get('dx', -30) * (1 - p)
        y = t['y'] + t.get('dy', 0) * (1 - p)
        if t.get('align') == 'center':
            x -= tw / 2
        elif t.get('align') == 'right':
            x -= tw
        a = 255 * min(1.0, p * 1.4)
        px, py = t.get('pad', (22, 10))
        asc, desc = f.getmetrics()
        box = (x - px, y - py, x + tw + px, y + asc + desc + py * 0.4)
        if t.get('pill'):
            d.rounded_rectangle(box, radius=t.get('radius', int((box[3] - box[1]) / 2)),
                                fill=rgba(t['pill'], a * t.get('pill_alpha', 1.0)),
                                outline=rgba(t['outline'], a) if t.get('outline') else None, width=2)
        d.text((x, y), txt, font=f, fill=rgba(t.get('color', WHITE), a))
        if t.get('cursor') and int(lf / (BEAT * FPS / 2)) % 2 == 0:
            d.rectangle([x + tw + 10, y + asc * 0.15, x + tw + 10 + f.size * 0.45, y + asc + desc * 0.3], fill=rgba(ACCENT, a))
        if t.get('lock'):
            since = (lb - t.get('at', 0)) * BEAT * FPS
            r = (box[0] - 8, box[1] - 8, box[2] + 8, box[3] + 8)
            if since < 3:
                brackets(d, r, ACCENT, 255, lw=4, L=20)
            elif since < 16:
                brackets(d, r, WHITE, 255 * (1 - (since - 3) / 13), lw=2, L=20)


def r_layers(c, tl, beat, tg):
    """A composed scene: background (image or the dark grid), image layers with entrance moves, timed texts.

    bg     = dict(img=path|photo id, push=(s0, s1), blur=0, dim=1.0, shade=0..1)   or None for the dark grid
    layers = [dict(img, cx, cy, w, at=0, dur=1.0, dx=0, dy=0, scale=1.0, alpha=0, rot=0, spin=0, drift=0,
                   pixel=False, frame=None|dict(border=(r,g,b)), until=None)]
             dx/dy/scale/alpha/rot are the START pose, eased to rest over `dur` beats from `at` (beats into the cue);
             pixel=True resolves from big pixel blocks (8-bit); drift = slow zoom (fraction) over the whole cue
    texts  = [dict(text, font=(key, size), x, y, at, dur, dx, color, align, pill, outline, cursor, lock, until)]
    """
    lb = tl / BEAT
    lf = tl * FPS
    span = c['n'] * BEAT
    bg = c.get('bg')
    if bg:
        s0, s1 = bg.get('push', (1.0, 1.05))
        frame = warp(cover_bg(bg['img'], bg.get('blur', 0), bg.get('dim', 1.0)), affine(s0 + (s1 - s0) * ease(tl / span)))
        frame = shade_left(frame, bg.get('shade', 0))
    else:
        frame = dark_bg(tg)
    for L in c.get('layers', []):
        if 'until' in L and lb >= L['until']:
            continue
        p = ease_out((lb - L.get('at', 0)) / L.get('dur', 1.0))
        if p <= 0:
            continue
        img = framed(L['img'], L['w'], **L['frame']) if L.get('frame') else rgba_image(L['img'], L['w'], L.get('pixel', False))
        if L.get('pixel'):
            img = pixelate(img, int(round(40 * (1 - p) ** 2)) + 1)
        sc = (L.get('scale', 1.0) * (1 - p) + p) * (1 + L.get('drift', 0) * tl / span)
        a = L.get('alpha', 0.0) * (1 - p) + p
        rot = L.get('rot', 0) * (1 - p) + L.get('spin', 0) * tl
        place(frame, img, L['cx'] + L.get('dx', 0) * (1 - p), L['cy'] + L.get('dy', 0) * (1 - p), sc, rot, a,
              nearest=L.get('pixel', False))
    base = Image.fromarray(frame).convert('RGBA')
    ov = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    draw_arrows(d, c.get('arrows', []), lb)
    draw_markers(d, c.get('markers', []), lb)
    draw_texts(d, c.get('texts', []), lb, lf)
    if c.get('hud'):
        hud_common(d, beat, tg, recap=False)
    base.alpha_composite(ov)
    return np.asarray(base.convert('RGB')).copy(), []


def r_carousel(c, tl, beat, tg):
    """Theme-picker carousel: a skewed 3D card stack slides to each focus item, which expands into a framed
    full view while the whole frame washes into that item's colour, then folds back for the next one.

    items = [dict(img, name, credit)], focus = [item indices in order], header / note = optional texts, big_w
    """
    items = c['items']
    focus = c.get('focus', list(range(len(items))))
    span = c['n'] * BEAT
    per = span / len(focus)
    fi_ = min(len(focus) - 1, int(tl / per))
    u = (tl - fi_ * per) / per                       # 0..1 inside this focus phase
    cur = focus[fi_]
    prev = focus[fi_ - 1] if fi_ else cur - 1
    SL, EX, HO = 0.20, 0.36, 0.86                    # slide ends, expand ends, hold ends
    pos = prev + (cur - prev) * ease(u / SL) if u < SL else cur
    if u < SL:
        grow = 0.0
    elif u < EX:
        grow = ease((u - SL) / (EX - SL))
    elif u < HO or fi_ == len(focus) - 1:
        grow = 1.0
    else:
        grow = 1.0 - ease((u - HO) / (1 - HO))
    # background: the focused theme blurred and washed with its hot colour, cross-fading while the stack slides
    nb = int(np.floor(pos))
    k = pos - nb
    glow = np.exp(-(((xx - W * 0.5) / 900) ** 2 + ((yy - H * 0.6) / 520) ** 2))[..., None]
    frame = np.zeros((H, W, 3), np.float32)
    for idx, wgt in ((nb % len(items), 1 - k), ((nb + 1) % len(items), k)):
        if wgt <= 0.001:
            continue
        ref = items[idx]['img']
        layer = warp(cover_bg(ref, 30, 0.42), affine(1.04)).astype(np.float32)
        layer = layer * 0.75 + glow * np.array(palette(ref), np.float32) * 0.32 + np.array(DARK, np.float32) * 0.25
        frame += layer * wgt
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    # cards: coverflow quads; the focused card interpolates into the big framed view
    CWD, CHT = 840, 473
    BIGW = c.get('big_w', 1440)
    BIGH = BIGW * 9 / 16
    cy0 = H * 0.54
    drawn = []
    for i in sorted(range(len(items)), key=lambda i: -abs(i - pos)):
        kk = i - pos
        if abs(kk) > 3.2:
            continue
        s = 1.0 / (1 + 0.28 * abs(kk))
        yaw = float(np.clip(kk, -1.6, 1.6) / 1.6)
        x = W / 2 + np.sign(kk) * (350 * min(abs(kk), 1) + 250 * max(abs(kk) - 1, 0))
        hw = CWD / 2 * s * (1 - 0.38 * abs(yaw))
        hh = CHT / 2 * s
        hl, hr = (hh * (1 + 0.16 * abs(yaw)), hh * (1 - 0.16 * abs(yaw))) if kk > 0 else (hh * (1 - 0.16 * abs(yaw)), hh * (1 + 0.16 * abs(yaw)))
        sk = -18 * yaw                                # a slight slant, like the picker's stacked cards
        quad = [(x - hw + sk, cy0 - hl), (x + hw + sk, cy0 - hr), (x + hw - sk, cy0 + hr), (x - hw - sk, cy0 + hl)]
        if i == cur and grow > 0:
            top = H / 2 - BIGH / 2 + 20
            big = [(W / 2 - BIGW / 2, top), (W / 2 + BIGW / 2, top), (W / 2 + BIGW / 2, top + BIGH), (W / 2 - BIGW / 2, top + BIGH)]
            quad = [(a[0] * (1 - grow) + b[0] * grow, a[1] * (1 - grow) + b[1] * grow) for a, b in zip(quad, big)]
        alpha = 1.0 if abs(kk) < 2.5 else max(0.0, (3.2 - abs(kk)) / 0.7)
        if i != cur:
            alpha *= 1 - 0.85 * grow                  # the stack recedes while the focus card is big
        place_quad(frame, rgba_image(items[i]['img'], 1200), quad, alpha)
        if alpha > 0.3:   # outline drawn in stacking order, so nearer cards cover it
            col_i = palette(items[cur]['img']) if i == cur else (110, 114, 130)
            cv2.polylines(frame, [np.round(np.asarray(quad)).astype(np.int32)], True, col_i, 3 if i == cur else 1, cv2.LINE_AA)
        drawn.append((i, quad, alpha))
    base = Image.fromarray(frame).convert('RGBA')
    ov = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    col = palette(items[cur]['img'])
    f_name, f_small, f_head = font('heavy', 58), font('mono', 20), font('semibold', 34)
    if grow > 0.6:                                    # name plate on the big view
        a = 255 * min(1, (grow - 0.6) / 0.3)
        it = items[cur]
        bx, by = W / 2 - BIGW / 2, H / 2 + BIGH / 2 + 20
        tw = f_name.getlength(it['name'])
        d.rounded_rectangle([bx - 6, by - 100, bx + tw + 52, by - 16], radius=14, fill=rgba(DARK, a * 0.82))
        d.rectangle([bx - 6, by - 100, bx + 4, by - 16], fill=rgba(col, a))
        d.text((bx + 24, by - 96), it['name'], font=f_name, fill=rgba(WHITE, a))
        if it.get('credit'):
            cw = f_small.getlength(it['credit'])
            d.rounded_rectangle([bx + BIGW - cw - 34, by - 52, bx + BIGW - 6, by - 16], radius=10, fill=rgba(DARK, a * 0.75))
            d.text((bx + BIGW - cw - 20, by - 47), it['credit'], font=f_small, fill=rgba((220, 224, 236), a))
    if c.get('header'):
        hk = ease_out(tl * FPS / 8)
        tw = f_head.getlength(c['header'])
        d.text((W / 2 - tw / 2, 40 + 10 * (1 - hk)), c['header'], font=f_head, fill=rgba(WHITE, 255 * hk))
        d.rectangle([W / 2 - 60, 94, W / 2 + 60, 98], fill=rgba(col, 255 * hk))
    if c.get('note'):
        d.text((W / 2 - f_small.getlength(c['note']) / 2, H - 38), c['note'], font=f_small, fill=rgba(WHITE, 150))
    if c.get('hud'):
        hud_common(d, beat, tg, recap=False)
    base.alpha_composite(ov)
    return np.asarray(base.convert('RGB')).copy(), []


# ---------------------------------------------------------------- footage clips, markers, arrows, route HUD
_caps = {}


def clip_frame(c, src_t):
    """Decode the source video frame at src_t (s). One reader per cue, read forward, so a speed-ramped clip
    streams instead of seeking every frame."""
    key = id(c)
    st = _caps.get(key)
    if st is None:
        cap = cv2.VideoCapture(c['video'])
        st = _caps[key] = dict(cap=cap, fps=cap.get(cv2.CAP_PROP_FPS) or 30.0, idx=-1, frame=None,
                               n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    target = max(0, min(st['n'] - 1, int(round(src_t * st['fps']))))
    if st['frame'] is None or target < st['idx'] or target - st['idx'] > 90:
        st['cap'].set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, fr = st['cap'].read()
        st['idx'], st['frame'] = target, (fr if ok else st['frame'])
    while st['idx'] < target:
        ok, fr = st['cap'].read()
        if not ok:
            break
        st['idx'] += 1
        if st['idx'] == target:
            st['frame'] = fr
    return st['frame']


def r_clip(c, tl, beat, tg):
    """Real footage: video=path, source seconds src=(t0, t1) mapped linearly onto the cue's n beats (speed ramp;
    t0 == t1 is a freeze frame). Cover-fit to the canvas, graded and sharpened, optional slow push; overlays on top.

    push=(s0, s1), markers=[...], arrows=[...], texts=[...]  (overlay timings in beats into the cue)
    """
    span = c['n'] * BEAT
    t0, t1 = c['src']
    fr = clip_frame(c, t0 + (t1 - t0) * min(1.0, tl / span))
    img = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    ih, iw = img.shape[:2]
    s0, s1 = c.get('push', (1.0, 1.04))
    sc = max(W / iw, H / ih) * (s0 + (s1 - s0) * ease(tl / span))
    M = np.float32([[sc, 0, W / 2 - iw * sc / 2], [0, sc, H / 2 - ih * sc / 2]])
    out = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    blur = cv2.GaussianBlur(out, (0, 0), 2.2)                  # unsharp mask: upscaled phone footage
    out = cv2.addWeighted(out, 1.6, blur, -0.6, 0)
    out = grade(out)
    out = (out.astype(np.float32) * (0.82 + 0.18 * VIGNETTE)).astype(np.uint8)
    base = Image.fromarray(out).convert('RGBA')
    ov = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    lb = tl / BEAT
    draw_arrows(d, c.get('arrows', []), lb)
    draw_markers(d, c.get('markers', []), lb)
    draw_texts(d, c.get('texts', []), lb, tl * FPS)
    base.alpha_composite(ov)
    return np.asarray(base.convert('RGB')).copy(), []


def draw_markers(d, markers, lb):
    """Location pins: pulsing ring + dot at (x, y), a leader line, and a label pill (optional stamp number badge).
    dict(x, y, label, sub=None, num=None, at=0, until=None, side='right'|'left', dx=150, dy=-120, x1, y1 (drift to))"""
    f_l, f_s, f_n = font('heavy', 46), font('medium', 28), font('mono', 30)
    for m in markers:
        u = lb - m.get('at', 0)
        if u < 0 or ('until' in m and lb >= m['until']):
            continue
        fade = 1.0
        if 'until' in m:
            fade = min(1.0, (m['until'] - lb) / 0.4)
        k = ease_out(u / 0.5)
        span = (m['until'] - m.get('at', 0)) if 'until' in m else 1.0
        g = min(1.0, u / max(span, 0.01))
        x = m['x'] + (m.get('x1', m['x']) - m['x']) * g
        y = m['y'] + (m.get('y1', m['y']) - m['y']) * g
        col = tuple(m.get('color', ACCENT))
        a = 255 * fade
        for ph in (0.0, 0.5):                          # two expanding rings
            r = ((u * 0.9 + ph) % 1.0)
            rr = 18 + 70 * r
            d.ellipse([x - rr, y - rr, x + rr, y + rr], outline=rgba(col, a * (1 - r) * k), width=4)
        d.ellipse([x - 14 * k, y - 14 * k, x + 14 * k, y + 14 * k], fill=rgba(col, a), outline=rgba(WHITE, a), width=3)
        side = -1 if m.get('side') == 'left' else 1
        ex, ey = x + side * m.get('dx', 120), y + m.get('dy', -110)
        lk = ease_out((u - 0.15) / 0.4)
        if lk <= 0:
            continue
        d.line([(x, y), (x + (ex - x) * lk, y + (ey - y) * lk)], fill=rgba(WHITE, a), width=4)
        pk = ease_out((u - 0.35) / 0.4)
        if pk <= 0:
            continue
        num = m.get('num')
        lw = f_l.getlength(m['label'])
        sw = f_s.getlength(m['sub']) if m.get('sub') else 0
        nw = 64 if num else 0
        pw = max(lw, sw) + 48 + nw
        ph_ = 74 + (40 if m.get('sub') else 0)
        px0 = ex if side > 0 else ex - pw
        px0 = min(max(px0, 24), W - pw - 24)
        py0 = ey - ph_ / 2
        d.rounded_rectangle([px0, py0, px0 + pw, py0 + ph_], radius=18, fill=rgba(DARK, a * 0.86 * pk), outline=rgba(col, a * pk), width=3)
        if num:
            d.ellipse([px0 + 16, py0 + 17, px0 + 56, py0 + 57], fill=rgba(col, a * pk))
            d.text((px0 + 36 - f_n.getlength(str(num)) / 2, py0 + 19), str(num), font=f_n, fill=rgba(WHITE, a * pk))
        d.text((px0 + 24 + nw, py0 + 10), m['label'], font=f_l, fill=rgba(WHITE, a * pk))
        if m.get('sub'):
            d.text((px0 + 26 + nw, py0 + 70), m['sub'], font=f_s, fill=rgba(ACCENT_SOFT, a * pk))


def draw_arrows(d, arrows, lb):
    """Marching chevrons: dict(x, y, angle=-90 (up), n=3, size=56, gap=64, at=0, until=None, color)."""
    for ar in arrows:
        u = lb - ar.get('at', 0)
        if u < 0 or ('until' in ar and lb >= ar['until']):
            continue
        fade = min(1.0, u / 0.4) * (min(1.0, (ar['until'] - lb) / 0.4) if 'until' in ar else 1.0)
        ang = math.radians(ar.get('angle', -90))
        dx, dy = math.cos(ang), math.sin(ang)
        px, py = -dy, dx
        n, size, gap = ar.get('n', 3), ar.get('size', 56), ar.get('gap', 64)
        col = tuple(ar.get('color', ACCENT))
        for i in range(n):
            ph = (u * 1.5 - i / n) % 1.0                 # brightness wave travelling forward
            a = 255 * fade * (0.25 + 0.75 * max(0.0, 1 - abs(ph - 0.5) * 2.2))
            cx, cy = ar['x'] + dx * gap * (i - (n - 1) / 2), ar['y'] + dy * gap * (i - (n - 1) / 2)
            tip = (cx + dx * size * 0.5, cy + dy * size * 0.5)
            l1 = (cx - dx * size * 0.3 + px * size * 0.55, cy - dy * size * 0.3 + py * size * 0.55)
            l2 = (cx - dx * size * 0.3 - px * size * 0.55, cy - dy * size * 0.3 - py * size * 0.55)
            d.line([l1, tip, l2], fill=rgba(col, a), width=int(size * 0.2), joint='curve')


def draw_route(d, beat):
    """Walkthrough HUD: step header at the top, a route bar with nodes and a travelling dot near the bottom."""
    R_ = E.ROUTE
    steps = R_['steps']
    if beat < steps[0][0] or beat >= R_['end']:
        return
    i = max(j for j, s_ in enumerate(steps) if s_[0] <= beat)
    nxt = steps[i + 1][0] if i + 1 < len(steps) else R_['end']
    frac = (beat - steps[i][0]) / max(nxt - steps[i][0], 1e-6)
    lb = beat - steps[i][0]
    k = ease_out(lb / 0.6)
    top = R_.get('top', 150)
    f_step, f_title, f_node = font('mono', 30), font('heavy', 66), font('semibold', 26)
    tag = f'STEP {i + 1:02d} / {len(steps):02d}'
    tw = f_step.getlength(tag)
    d.rounded_rectangle([60, top, 60 + tw + 36, top + 50], radius=25, fill=rgba(ACCENT, 255 * k))
    d.text((78, top + 8), tag, font=f_step, fill=rgba(WHITE, 255 * k))
    d.text((60 + 30 * (1 - k), top + 66), steps[i][1], font=f_title, fill=rgba(WHITE, 255 * k), stroke_width=3, stroke_fill=rgba((0, 0, 0), 150 * k))
    # route bar
    y = R_.get('bar_y', H - 380)
    x0, x1 = 90, W - 90
    n = len(steps)
    xs = [x0 + (x1 - x0) * j / (n - 1) for j in range(n)]
    d.rounded_rectangle([x0 - 50, y - 46, x1 + 50, y + 82], radius=30, fill=rgba(DARK, 170))
    d.line([(x0, y), (x1, y)], fill=rgba(WHITE, 90), width=6)
    pos = xs[i] + (xs[min(i + 1, n - 1)] - xs[i]) * min(1.0, ease(min(1.0, frac * 1.0)))
    d.line([(x0, y), (pos, y)], fill=rgba(ACCENT, 255), width=6)
    for j, xj in enumerate(xs):
        done = j <= i
        r = 13 if j != i else 17
        d.ellipse([xj - r, y - r, xj + r, y + r], fill=rgba(ACCENT if done else DARK, 255), outline=rgba(WHITE if done else (150, 156, 175), 255), width=3)
        lab = steps[j][2]
        d.text((xj - f_node.getlength(lab) / 2, y + 26), lab, font=f_node, fill=rgba(WHITE if j == i else (170, 176, 195), 255))
    pr = 10 + 8 * (0.5 + 0.5 * math.sin(beat * math.pi))   # "you are here" dot, pulsing on the beat
    d.ellipse([pos - pr - 8, y - pr - 8, pos + pr + 8, y + pr + 8], outline=rgba(ACCENT_SOFT, 160), width=3)
    d.ellipse([pos - 9, y - 9, pos + 9, y + 9], fill=rgba(WHITE, 255))


R = dict(photo=r_photo, slam=r_slam, terminal=r_terminal, pano=r_pano, reveal=r_reveal, wall=r_wall, endcard=r_endcard,
         keynote=r_keynote, trackwall=r_trackwall, layers=r_layers, carousel=r_carousel, clip=r_clip)


def cue_at(beat):
    cur = CUES[0]
    for c in CUES:
        if c['b'] <= beat:
            cur = c
    return cur


def render_frame_raw(c, tg, beat):
    """A cue's own picture at global time tg (no cut accents); the end card stays on its held still."""
    if c['kind'] == 'endcard':
        return R['endcard'](c, 0, beat, tg)[0]
    return R[c['kind']](c, tg - c['b'] * BEAT, beat, tg)[0]


def render_frame(fi):
    tg = fi / FPS
    beat = tg / BEAT
    c = cue_at(beat)
    tl = tg - c['b'] * BEAT
    frame, rects = R[c['kind']](c, tl, beat, tg)
    lf = tl * FPS  # frames since cut
    xf = c.get('xfade', 0) * BEAT   # optional cross-dissolve from the previous cue (instead of a hard cut)
    i_ = CUES.index(c)
    if xf and tl < xf and i_ > 0:
        pc = CUES[i_ - 1]
        pf = render_frame_raw(pc, tg, beat)
        k = ease(tl / xf)
        frame = (pf.astype(np.float32) * (1 - k) + frame.astype(np.float32) * k).astype(np.uint8)
    if tg >= HOLD_FROM:
        return frame
    out = frame.astype(np.float32)
    if c['kind'] in ('photo', 'slam', 'pano'):
        out *= VIGNETTE
    downbeat = abs(c['b'] % 4) < 1e-6   # cut accents: flash + chromatic split
    cut = c.get('cut', 'none' if xf else 'flash')   # flash (default) | soft (gentle flash, no split) | none
    if c['kind'] != 'pano' and cut != 'none':
        fl = ([0.6, 0.32, 0.12] if downbeat else [0.28, 0.1])
        if c['kind'] == 'reveal':
            fl = [0.9, 0.6, 0.3, 0.12]
        if cut == 'soft':
            fl = [v * 0.35 for v in fl]
        if int(lf) < len(fl):
            col = np.array(WHITE if c['kind'] != 'slam' else ACCENT, np.float32)
            out = out * (1 - fl[int(lf)]) + col * fl[int(lf)]
        sp = [9, 6, 3, 1][int(lf)] if int(lf) < 4 and c['kind'] != 'terminal' and cut == 'flash' else 0
        if sp:
            out[:, :, 0] = np.roll(out[:, :, 0], sp, axis=1)
            out[:, :, 2] = np.roll(out[:, :, 2], -sp, axis=1)
    out = np.clip(out + GRAIN[fi % len(GRAIN)][..., None], 0, 255).astype(np.uint8)

    if c['kind'] in ('photo', 'slam', 'pano'):   # HUD overlay
        base = Image.fromarray(out).convert('RGBA')
        ov = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        for k, r in enumerate(rects):
            appear = (lf - 2 - 2 * k) / 4
            if appear <= 0:
                continue
            lock_lf = BEAT * FPS if c['n'] >= 2 else 0.6 * BEAT * FPS
            locked = k == 0 and lock_lf <= lf < lock_lf + 3
            col = ACCENT if locked else WHITE
            a = 255 * min(1, appear)
            hatch(d, r, col, a * 0.3)
            brackets(d, r, col, a, lw=4 if locked else 3)
            coord(d, r, a, col)
        if len(rects) > 1:
            (x1, y1), (x2, y2) = [((r[0] + r[2]) / 2, (r[1] + r[3]) / 2) for r in rects[:2]]
            d.line([(x1, y1), (x2, y2)], fill=rgba(WHITE, 150), width=2)
            ph = (tl * 1.6) % 1.0
            d.ellipse([x1 + (x2 - x1) * ph - 6, y1 + (y2 - y1) * ph - 6, x1 + (x2 - x1) * ph + 6, y1 + (y2 - y1) * ph + 6], fill=rgba(ACCENT, 255))
        hud_common(d, beat, tg)
        base.alpha_composite(ov)
        out = np.asarray(base.convert('RGB')).copy()
    if downbeat and int(lf) < 2 and c['kind'] in ('photo', 'slam') and cut == 'flash':   # scanline tear on downbeat cuts
        rng = np.random.default_rng(fi)
        for _ in range(2):
            y0 = int(rng.uniform(100, H - 160))
            out[y0:y0 + 36] = np.roll(out[y0:y0 + 36], int(rng.choice([-1, 1]) * rng.uniform(20, 50)), axis=1)
    if E.ROUTE:   # walkthrough HUD over everything
        base = Image.fromarray(out).convert('RGBA')
        ov = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        draw_route(ImageDraw.Draw(ov), beat)
        base.alpha_composite(ov)
        out = np.asarray(base.convert('RGB')).copy()
    return out


def contact_sheets(paths, out_prefix, cols=4, tw=480):
    """Tile check frames (labelled with time) into sheets of 5 rows."""
    th = int(tw * H / W)
    per = cols * 5
    f = font('mono', 18)
    outs = []
    for si in range(0, len(paths), per):
        chunk = paths[si:si + per]
        rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new('RGB', (cols * tw, rows * (th + 28)), (20, 20, 24))
        d = ImageDraw.Draw(sheet)
        for i, (p, label) in enumerate(chunk):
            r_, c_ = divmod(i, cols)
            sheet.paste(Image.open(p).resize((tw, th)), (c_ * tw, r_ * (th + 28)))
            d.text((c_ * tw + 6, r_ * (th + 28) + th + 4), label, font=f, fill=(230, 230, 230))
        op = f'{out_prefix}_{si // per + 1:02d}.jpg'
        sheet.save(op, quality=85)
        outs.append(op)
    return outs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('event')
    ap.add_argument('--stills', action='store_true', help='two check frames per cue + contact sheets')
    ap.add_argument('--at', type=float, nargs='+', help='render single frames at these times (s)')
    ap.add_argument('--frames', nargs=2, type=int, help='render only frames [a, b)')
    ap.add_argument('--out')
    a = ap.parse_args()
    setup(a.event)
    bad = check(E)
    if bad:
        sys.exit('event config problems:\n  ' + '\n  '.join(bad))
    os.makedirs(E.WORK, exist_ok=True)
    if a.at:
        for t in a.at:
            p = os.path.join(E.WORK, f'frame_{t:06.2f}.png')
            Image.fromarray(render_frame(int(round(t * FPS)))).save(p)
            print(p)
        return
    if a.stills:
        sd = os.path.join(E.WORK, 'stills')
        os.makedirs(sd, exist_ok=True)
        for f in os.listdir(sd):
            os.remove(os.path.join(sd, f))
        made = []
        for c in CUES:
            n = c['n'] if c.get('n') else (TOTAL / BEAT - c['b'])
            for frac in ((0.1, 0.6) if c['kind'] in ('slam', 'photo') else (0.5, 0.95)):
                fi = min(NF - 1, int((c['b'] + n * frac) * BEAT * FPS))
                p = os.path.join(sd, f'f{fi:05d}.jpg')
                Image.fromarray(render_frame(fi)).save(p, quality=88)
                made.append((p, f"{fi / FPS:6.2f}s  {c['kind']} b{c['b']:g}"))
        sheets = contact_sheets(made, os.path.join(E.WORK, 'stills_sheet'))
        print('stills ->', sd)
        print('sheets ->', *sheets, sep='\n  ')
        return
    out = a.out or os.path.join(E.WORK, 'video_noaudio.mp4')
    f0, f1 = a.frames if a.frames else (0, NF)
    # a keyframe right where the end card starts holding: no periodic I-frame lands inside the still hold
    kf = ['-force_key_frames', f'{HOLD_FROM:.3f}'] if HOLD_FROM < TOTAL else []
    ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
                           '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', '16', '-pix_fmt', 'yuv420p', *kf,
                           '-movflags', '+faststart', out], stdin=subprocess.PIPE)
    for fi in range(f0, f1):
        ff.stdin.write(render_frame(fi).tobytes())
        if fi % 300 == 0:
            print(f'frame {fi}/{f1}', flush=True)
    ff.stdin.close()
    ff.wait()
    print('wrote', out)


if __name__ == '__main__':
    main()
