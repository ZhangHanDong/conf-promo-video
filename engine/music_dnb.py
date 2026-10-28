#!/usr/bin/env python3
"""Drum & Bass score + SFX (MUSIC_STYLE = 'dnb'): 172 BPM two-step breaks, a detuned reese bass with a moving
filter, F-minor 9th pads, 16th arps in the drops, impacts on the picture cues. Synthesised in code, no samples.

Same contract as music.py: reads MUSIC_SECTIONS / CUES from the event, writes WORK/music.wav and sfx.wav.
Modes: pad · pulse · build · full · mid (half-time groove) · break · hit · count · reveal · end
  python3 music_dnb.py EVENT.py
"""
import os, sys, wave
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_event  # noqa: E402

E = load_event(sys.argv[1])
BEAT, TOTAL, CUES = E.BEAT, E.TOTAL, E.CUES
SR = 48000
N = int(TOTAL * SR)
NBEATS = TOTAL / BEAT
rng = np.random.default_rng(172)

SECS = []
for s in E.MUSIC_SECTIONS:
    SECS.append(dict(a=s[0], z=NBEATS if s[1] is None else s[1], mode=s[2], **(s[3] if len(s) > 3 else {})))
for i, s in enumerate(SECS):
    s['prev'] = SECS[i - 1]['mode'] if i else None
    s['next'] = SECS[i + 1]['mode'] if i + 1 < len(SECS) else None
MUSIC_END = max([s['a'] for s in SECS if s['mode'] == 'end'] or [NBEATS])
COUNT0 = min([s['a'] for s in SECS if s['mode'] == 'count'] or [NBEATS])


def sec_at(b):
    for s in SECS:
        if s['a'] <= b < s['z']:
            return s
    return SECS[-1]


def bs(b):
    return int(round(b * BEAT * SR))


class Bus:
    def __init__(self):
        self.x = np.zeros((N, 2), np.float32)

    def add(self, sig, at, gain=1.0, pan=0.0):
        if at >= N:
            return
        if at < 0:
            sig, at = sig[-at:], 0
        sig = sig[: N - at]
        if sig.ndim == 1:
            gl, gr = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
            self.x[at:at + len(sig), 0] += sig * gain * gl * 1.414
            self.x[at:at + len(sig), 1] += sig * gain * gr * 1.414
        else:
            self.x[at:at + len(sig)] += sig * gain


def filt(x, kind, fc, order=2):
    w = np.clip(np.asarray(fc, float) / (SR / 2), 1e-4, 0.95)
    w = float(w) if w.ndim == 0 else list(w)
    return sosfilt(butter(order, w, kind, output='sos'), x, axis=0).astype(np.float32)


def env(n, a, tau):
    t = np.arange(n) / SR
    return (np.minimum(1, t / max(a, 1e-5)) * np.exp(-t / tau)).astype(np.float32)


def noise(n):
    return rng.standard_normal(n).astype(np.float32)


def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def saw(f, n, ph=0.0):
    t = np.arange(n) / SR
    return (2 * ((f * t + ph) % 1.0) - 1).astype(np.float32)


# ---------------------------------------------------------------- drums (punchy, tight)
def kick():
    n = int(0.28 * SR)
    t = np.arange(n) / SR
    f = 52 + 150 * np.exp(-t / 0.02)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.16)
    click = filt(noise(n), 'high', 3500) * np.exp(-t / 0.002) * 0.5
    return np.tanh((body + click) * 1.8).astype(np.float32)


def snare():
    n = int(0.3 * SR)
    t = np.arange(n) / SR
    tone = np.sin(2 * np.pi * 200 * t) * np.exp(-t / 0.04) + np.sin(2 * np.pi * 330 * t) * np.exp(-t / 0.03) * 0.5
    nz = filt(noise(n), 'band', [1200, 9000]) * np.exp(-t / 0.11)
    return np.tanh((tone * 0.6 + nz * 1.1) * 1.4).astype(np.float32)


def hat(open_=False):
    n = int((0.18 if open_ else 0.04) * SR)
    return filt(noise(n), 'high', 8500, 3) * env(n, 0.0004, 0.06 if open_ else 0.01)


def ride():
    n = int(0.5 * SR)
    t = np.arange(n) / SR
    x = filt(noise(n), 'band', [5000, 12000]) * np.exp(-t / 0.25) * 0.5
    return (x + 0.15 * np.sin(2 * np.pi * 5200 * t) * np.exp(-t / 0.3)).astype(np.float32)


def crash():
    n = int(2.2 * SR)
    x = filt(noise(n), 'high', 3500) * env(n, 0.001, 0.8)
    return np.stack([x, filt(noise(n), 'high', 3500) * env(n, 0.001, 0.8)], 1) * 0.45


def boom(big=1.0):
    n = int(2.0 * SR)
    t = np.arange(n) / SR
    f = 36 + 80 * np.exp(-t / 0.05)
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / (0.5 * big))
    hit = filt(noise(n), 'low', 2200) * np.exp(-t / 0.07) * 0.7
    return np.tanh((sub + hit) * 1.4).astype(np.float32)


def riser(beats, top=9000):
    n = bs(beats)
    out = np.zeros(n, np.float32)
    x = noise(n)
    seg = max(1, n // 32)
    for i in range(0, n, seg):
        k = i / n
        fc = 500 + (top - 500) * k ** 2
        out[i:i + seg] = filt(x[i:i + seg], 'band', [fc * 0.6, min(fc * 1.5, 20000)])
    sweep = np.sin(2 * np.pi * np.cumsum(300 + 2500 * (np.arange(n) / n) ** 2) / SR) * 0.12
    return ((out * 0.6 + sweep) * (np.arange(n) / n) ** 2).astype(np.float32)


def whoosh(dur=0.3):
    n = int(dur * SR)
    return (filt(noise(n), 'band', [400, 7000]) * np.sin(np.linspace(0, np.pi, n)) ** 3).astype(np.float32)


def blip(f2=1975.5, dur=0.15):
    n = int(dur * SR)
    t = np.arange(n) / SR
    return (np.sin(2 * np.pi * 1318.5 * t) * env(n, 0.002, 0.04) + np.sin(2 * np.pi * f2 * t) * env(n, 0.04, 0.05) * 0.6).astype(np.float32)


# ---------------------------------------------------------------- tonal
def reese(m, n, cut0, cut1):
    """Detuned saw pair + sub, filter sweeping cut0 -> cut1 over the note, saturated."""
    f = midi(m)
    x = saw(f * 2 ** (-0.18 / 12), n, rng.random()) + saw(f * 2 ** (0.18 / 12), n, rng.random())
    x = x * 0.5
    out = np.zeros(n, np.float32)
    seg = max(1, n // 12)
    for i in range(0, n, seg):
        k = i / n
        out[i:i + seg] = filt(x[i:i + seg], 'low', cut0 + (cut1 - cut0) * k)
    t = np.arange(n) / SR
    sub = np.sin(2 * np.pi * f / 2 * t) * 0.7
    a = np.minimum(1, t / 0.006) * np.minimum(1, (n - np.arange(n)) / (0.01 * SR))
    return (np.tanh((out * 1.3 + sub) * 1.6) * a).astype(np.float32)


def pad(notes, n, cutoff=1600):
    t = np.arange(n) / SR
    out = np.zeros((n, 2), np.float32)
    for m in notes:
        for v, dt in enumerate((-0.1, 0.0, 0.1)):
            s = saw(midi(m + dt), n, rng.random())
            out[:, v % 2] += s
            out[:, (v + 1) % 2] += s * 0.4
    out /= len(notes) * 3
    a = np.minimum(1, t / 0.25)[:, None] * np.minimum(1, (n - np.arange(n)) / (0.2 * SR))[:, None]
    return filt(out * a, 'low', cutoff)


def pluck(m, dur=0.16):
    n = int(dur * SR)
    t = np.arange(n) / SR
    s = saw(midi(m), n) * 0.6 + saw(midi(m + 12.03), n) * 0.3
    k = np.exp(-t / 0.035)
    return ((filt(s, 'low', 7000) * k + filt(s, 'low', 1200) * (1 - k)) * env(n, 0.001, 0.09)).astype(np.float32)


def stab(notes, dur=0.35, cut=4500):
    n = int(dur * SR)
    return pad(notes, n, cut) * env(n, 0.002, 0.12)[:, None] * 3.0


CHORDS = getattr(E, 'DNB_CHORDS', [
    (41, [56, 60, 63, 67]),   # Fm9   (Ab C Eb G over F)
    (37, [53, 56, 60, 63]),   # Dbmaj9
    (44, [55, 60, 63, 67]),   # Abmaj7 (G C Eb G over Ab)
    (39, [55, 58, 63, 65]),   # Eb add9
])


def chord_at(b):
    return CHORDS[int(b // 8) % len(CHORDS)]       # one chord per 2 bars


# ---------------------------------------------------------------- arrangement
drums, bassb, keys, padb, fx, sfx = Bus(), Bus(), Bus(), Bus(), Bus(), Bus()
K, SN, HC, HO, RD = kick(), snare(), hat(), hat(True), ride()
kicks = []

for bar in range(int(MUSIC_END) // 4 + 1):
    b0 = bar * 4
    if b0 >= MUSIC_END:
        break
    s = sec_at(b0)
    m = s['mode']
    last_bar = b0 + 4 >= s['z']
    k_ = (b0 - s['a']) / max(s['z'] - s['a'], 1)
    if m in ('full', 'build') or (m == 'mid'):
        half = m == 'mid'
        for st, g in ((0, 1.0), (10, 0.9)) if not half else ((0, 1.0),):
            drums.add(K, bs(b0 + st / 4), 0.9 * g * (0.8 if m == 'build' else 1.0))
            kicks.append(bs(b0 + st / 4))
        snares = (4, 12) if not half else (8,)
        for st in snares:
            if m == 'build' and k_ < 0.25:
                continue
            drums.add(SN, bs(b0 + st / 4), 0.62)
        if m == 'full':
            for st in (7, 15):
                drums.add(SN, bs(b0 + st / 4), 0.12, pan=0.2)          # ghost notes
            if bar % 2 == 1 and not last_bar:
                drums.add(K, bs(b0 + 2.5 / 4 + 3), 0.6)              # extra kick, two-step variation
        for st in range(16):
            if st % 2 == 0:
                drums.add(HC, bs(b0 + st / 4), 0.09 if st % 4 else 0.12, pan=-0.3)
            elif m == 'full':
                drums.add(HC, bs(b0 + st / 4), 0.045, pan=-0.3)
        if m == 'full':
            drums.add(RD, bs(b0 + 1), 0.09, pan=0.35)
            drums.add(RD, bs(b0 + 3), 0.09, pan=0.35)
            drums.add(HO, bs(b0 + 2 + 0.5), 0.1, pan=0.3)
    elif m == 'pulse':
        drums.add(K, bs(b0), 0.85)
        kicks.append(bs(b0))
        drums.add(K, bs(b0 + 2.5), 0.6)
        kicks.append(bs(b0 + 2.5))
    elif m == 'count':
        for q in range(4):
            drums.add(K, bs(b0 + q), 0.95)
            kicks.append(bs(b0 + q))
        drums.add(SN, bs(b0 + 2), 0.5)
    # snare roll into the next full / count / reveal section
    nxt = s['next']
    if last_bar and nxt in ('full', 'count', 'reveal') and m in ('build', 'full', 'mid', 'pulse', 'break', 'hit'):
        t = b0 + 2
        step = 0.25
        while t < s['z'] - 0.01:
            drums.add(SN, bs(t), 0.25 + 0.4 * (t - b0 - 2) / 2, pan=0.1)
            t += step if t < s['z'] - 1 else 0.125
    # reese bass
    if m in ('full', 'build', 'mid') and not (m == 'build' and k_ < 0.5):
        root, _ = chord_at(b0)
        for st, ln, oct_ in ((0, 6, 0), (7, 2, 12), (10, 6, 0)):
            if last_bar and st >= 10 and nxt != m:
                continue
            n = bs(ln / 4)
            bassb.add(reese(root, n, 300, 1400 if m == 'full' else 800), bs(b0 + st / 4), 0.5)
    # pads
    if m in ('pad', 'pulse', 'build', 'break', 'mid', 'hit', 'full'):
        _, notes = chord_at(b0)
        cut = {'pad': 900, 'pulse': 1200, 'build': 900 + 3500 * k_ ** 1.5, 'break': 1800, 'mid': 2200, 'hit': 2200, 'full': 2600}[m]
        g = {'pad': 0.28, 'pulse': 0.24, 'build': 0.24, 'break': 0.34, 'mid': 0.26, 'hit': 0.24, 'full': 0.18}[m]
        padb.add(pad(notes, bs(4), cut), bs(b0), g)
    # arp + stabs in full
    if m == 'full':
        _, notes = chord_at(b0)
        seq = [notes[0], notes[2], notes[3], notes[1] + 12, notes[2] + 12, notes[3], notes[1], notes[2]]
        for st in range(16):
            if last_bar and st > 11:
                break
            keys.add(pluck(seq[(bar * 16 + st) % 8] + 12), bs(b0 + st / 4), 0.08, pan=0.35 if st % 2 else -0.35)
        if bar % 2 == 0:
            keys.add(stab(notes), bs(b0), 0.25)
    if m == 'break':
        _, notes = chord_at(b0)
        for st in (0, 3, 6, 8, 11, 14):
            keys.add(pluck(notes[(st // 3) % 4] + 12, 0.4), bs(b0 + st / 4), 0.08, pan=(st % 3 - 1) * 0.5)

# transitions
for s in SECS:
    if s['mode'] in ('full', 'count') and s['prev'] not in ('full',) and s['a'] >= 4:
        fx.add(riser(4), bs(s['a'] - 4), 0.32)
    if s['mode'] == 'full' and s['prev'] != 'full':
        fx.add(crash(), bs(s['a']), 0.55)
        fx.add(boom(1.0), bs(s['a']), 0.45)

# picture cues
for c in CUES:
    b, kind = c['b'], c['kind']
    if kind == 'slam':
        late = b >= COUNT0
        fx.add(boom(1.3 if late else 0.9), bs(b), 0.6 if late else 0.42)
        sfx.add(whoosh(0.28), bs(b) - int(0.26 * SR), 0.32)
        if sec_at(b)['mode'] in ('count', 'hit', 'pulse'):
            _, notes = chord_at(b)
            keys.add(stab(notes, 0.5, 5000), bs(b), 0.4)
    elif kind == 'photo':
        if sec_at(b)['mode'] == 'pulse':
            fx.add(boom(0.5), bs(b), 0.25)
        if c['pid'] in E.POIS:
            sfx.add(blip(), bs(b + (1.0 if c['n'] >= 2 else 0.6)), 0.1, pan=0.3)
    elif kind == 'pano':
        fx.add(boom(0.8), bs(b), 0.3)
    elif kind == 'wall':
        fx.add(boom(1.0), bs(b), 0.42)
        sfx.add(whoosh(0.3), bs(b) - int(0.28 * SR), 0.3)
        for gi in range(sum(len(row) for row in E.WALLS[c['wall']]['rows'])):
            sfx.add(blip(1760), bs(b + 0.8 + 0.6 * gi), 0.14, pan=-0.3 + 0.15 * gi)
    elif kind == 'terminal':
        for i, _ in enumerate(E.TERMINAL['cmd']):
            n_ = int(0.012 * SR)
            sfx.add(filt(noise(n_), 'high', 2500) * env(n_, 0.0004, 0.002), bs(b) + int((0.25 + i / 22) * SR), 0.45, pan=-0.2 + 0.4 * i / 22)
        for t in np.arange((b + 7.6) * BEAT, (b + 8) * BEAT, 0.03):
            n_ = int(0.03 * SR)
            sfx.add(filt(noise(n_), 'band', [2000, 9000]) * env(n_, 0.0005, 0.01), int(t * SR), 0.3, pan=rng.uniform(-0.6, 0.6))
    elif kind in ('reveal', 'endcard'):
        pass

# reveal / end: biggest hit, Fm9 ringing out
root, voicing = CHORDS[0]
tonic = [root] + list(voicing) + [voicing[1] + 12]
for s in SECS:
    if s['mode'] in ('reveal', 'end'):
        R0 = bs(s['a'])
        big = s['mode'] == 'reveal'
        fx.add(boom(1.8), R0, 0.85)
        fx.add(crash(), R0, 0.75)
        tn = N - R0
        x = pad(tonic, tn, 2600) * (env(tn, 0.004, 2.8) * 0.85 + 0.15)[:, None]
        padb.add(x, R0, 0.5 if big else 0.45)
        tt = np.arange(tn) / SR
        bassb.add((np.sin(2 * np.pi * midi(root - 12) * tt) * env(tn, 0.01, 2.2)).astype(np.float32), R0, 0.45)

# ---------------------------------------------------------------- mix
duck = np.ones(N, np.float32)
dn = int(0.16 * SR)
shape = 1 - 0.7 * np.exp(-np.arange(dn) / SR / 0.05)
for k in kicks:
    e = min(N, k + dn)
    duck[k:e] = np.minimum(duck[k:e], shape[: e - k])
ir_n = int(1.6 * SR)
ir_t = np.arange(ir_n) / SR
irs = [filt((rng.standard_normal(ir_n) * np.exp(-ir_t / 0.45)).astype(np.float32), 'low', 6000) for _ in range(2)]
irs = [i_ / np.sqrt((i_ ** 2).sum()) for i_ in irs]
send = keys.x * 0.35 + padb.x * 0.25 + drums.x * 0.05
rev = np.stack([fftconvolve(send[:, 0], irs[0])[:N], fftconvolve(send[:, 1], irs[1])[:N]], 1).astype(np.float32) * 0.5

sg = np.ones(N, np.float32)


def seg(b0, b1, g0, g1=None):
    i0, i1 = bs(b0), min(N, bs(b1))
    if i1 > i0:
        sg[i0:i1] = np.linspace(g0, g0 if g1 is None else g1, i1 - i0, dtype=np.float32)


for s in SECS:
    if s['mode'] == 'pad':
        seg(s['a'], s['z'], 0.75)
    elif s['mode'] == 'build':
        seg(s['a'], s['z'], 0.6, 0.85)
    if s['mode'] in ('full', 'count', 'reveal', 'end') and s['a'] > 0:
        seg(s['a'] - 0.25, s['a'], 0.15)          # breath before the drop
music = drums.x + (bassb.x + keys.x + padb.x + rev) * duck[:, None]
music = music * sg[:, None] + fx.x
music = filt(music, 'high', 30)
fade = int(1.5 * SR)
music[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)[:, None] ** 1.5
music = np.tanh(music * 1.1) / np.tanh(1.1)


def write(path, st):
    pk = np.abs(st).max()
    st = st / max(pk, 1e-9) * 10 ** (-1.0 / 20)
    with wave.open(path, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())


os.makedirs(E.WORK, exist_ok=True)
write(os.path.join(E.WORK, 'music.wav'), music)
write(os.path.join(E.WORK, 'sfx.wav'), sfx.x)
print('wrote', os.path.join(E.WORK, 'music.wav'), 'and sfx.wav (dnb)')
