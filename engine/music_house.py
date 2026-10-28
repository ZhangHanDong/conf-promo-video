#!/usr/bin/env python3
"""Chill-house score + SFX (MUSIC_STYLE = 'house'): 7th/9th chords on an FM electric piano, swung hats,
off-beat house bass, a bell lead with ping-pong delay, vinyl texture. Synthesised in code, no samples.

Same contract as music.py: reads MUSIC_SECTIONS / CUES / ROUTE from the event, writes WORK/music.wav and sfx.wav.
Section modes: pad (EP chords + vinyl) · pulse (kick + EP) · build (drums enter, filter opens, riser) ·
full (everything) · mid (groove without lead) · break (no kick, pads + EP) · end (last chord rings out).
  python3 music_house.py EVENT.py
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
SWING = getattr(E, 'SWING', 0.14)            # delay of the off 16ths, in 16ths
rng = np.random.default_rng(2026)

SECS = []
for s in E.MUSIC_SECTIONS:
    SECS.append(dict(a=s[0], z=NBEATS if s[1] is None else s[1], mode=s[2], **(s[3] if len(s) > 3 else {})))
for i, s in enumerate(SECS):
    s['prev'] = SECS[i - 1]['mode'] if i else None
    s['next'] = SECS[i + 1]['mode'] if i + 1 < len(SECS) else None
MUSIC_END = max([s['a'] for s in SECS if s['mode'] == 'end'] or [NBEATS])


def sec_at(b):
    for s in SECS:
        if s['a'] <= b < s['z']:
            return s
    return SECS[-1]


def bs(b):
    return int(round(b * BEAT * SR))


def sw(b16):
    """Sample position of a 16th-note grid index with swing on the off 16ths."""
    return bs(b16 / 4 + (SWING / 4 if b16 % 2 else 0))


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
    w = float(w) if w.ndim == 0 else list(w)          # low/high take a scalar, band takes [lo, hi]
    return sosfilt(butter(order, w, kind, output='sos'), x, axis=0).astype(np.float32)


def env(n, a, tau):
    t = np.arange(n) / SR
    return (np.minimum(1, t / max(a, 1e-5)) * np.exp(-t / tau)).astype(np.float32)


def noise(n):
    return rng.standard_normal(n).astype(np.float32)


def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12)


# ---------------------------------------------------------------- instruments
def kick():
    n = int(0.32 * SR)
    t = np.arange(n) / SR
    f = 48 + 95 * np.exp(-t / 0.035)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.2)
    return np.tanh(body * 1.3 + filt(noise(n), 'high', 3000) * np.exp(-t / 0.002) * 0.15).astype(np.float32)


def clap():
    n = int(0.3 * SR)
    x = np.zeros(n, np.float32)
    for k, off in enumerate((0, 0.008, 0.016)):
        i = int(off * SR)
        x[i:] += filt(noise(n - i), 'band', [1100, 4200]) * env(n - i, 0.0005, 0.01 if k < 2 else 0.09)
    return x * 0.7


def hat(open_=False):
    n = int((0.22 if open_ else 0.05) * SR)
    return filt(noise(n), 'high', 8000 if not open_ else 6500, 3) * env(n, 0.0005, 0.07 if open_ else 0.012)


def shaker():
    n = int(0.07 * SR)
    return filt(noise(n), 'band', [4500, 11000]) * env(n, 0.01, 0.02)


def ep(notes, dur, bright=1.0):
    """FM electric piano (Rhodes-ish): sine carrier, 1:1 modulator with a decaying index, slight detune + tremolo."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n, np.float32)
    for m in notes:
        f = midi(m)
        idx = (1.6 * bright) * np.exp(-t / 0.25) + 0.25
        mod = np.sin(2 * np.pi * f * t) * idx
        out += np.sin(2 * np.pi * f * 1.002 * t + mod) * np.exp(-t / 1.1)
        out += 0.15 * np.sin(2 * np.pi * f * 2.0 * t) * np.exp(-t / 0.18)   # tine attack
    out *= (1 + 0.12 * np.sin(2 * np.pi * 4.5 * t)) / len(notes)
    return (out * np.minimum(1, t / 0.004) * np.minimum(1, (n - np.arange(n)) / (0.03 * SR))).astype(np.float32)


def pad(notes, n, cutoff=1400):
    t = np.arange(n) / SR
    out = np.zeros((n, 2), np.float32)
    for m in notes:
        for v, dt in enumerate((-0.08, 0.0, 0.08)):
            ph = rng.random()
            s = 2 * ((midi(m + dt) * t + ph) % 1.0) - 1
            out[:, v % 2] += s
            out[:, (v + 1) % 2] += s * 0.5
    out /= len(notes) * 3
    a = np.minimum(1, t / 0.4)[:, None] * np.minimum(1, (n - np.arange(n)) / (0.3 * SR))[:, None]
    return filt(out * a, 'low', cutoff)


def bass(m, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = midi(m)
    x = np.sin(2 * np.pi * f * t) * 0.8 + filt((2 * ((f * t) % 1.0) - 1).astype(np.float32), 'low', 500) * 0.45
    return (np.tanh(x * 1.4) * env(n, 0.004, 0.16)).astype(np.float32)


def bell(m, dur=0.5):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = midi(m)
    mod = np.sin(2 * np.pi * f * 3.5 * t) * 2.2 * np.exp(-t / 0.12)
    return (np.sin(2 * np.pi * f * t + mod) * env(n, 0.002, 0.28)).astype(np.float32)


def riser(beats):
    n = bs(beats)
    out = np.zeros(n, np.float32)
    x = noise(n)
    seg = n // 24 or 1
    for i in range(0, n, seg):
        k = i / n
        fc = 400 + 7000 * k ** 2
        out[i:i + seg] = filt(x[i:i + seg], 'band', [fc * 0.7, min(fc * 1.4, 20000)])
    return out * (np.arange(n) / n) ** 2 * 0.6


def whoosh(dur=0.45):
    n = int(dur * SR)
    return (filt(noise(n), 'band', [400, 6000]) * np.sin(np.linspace(0, np.pi, n)) ** 3).astype(np.float32)


def chime(m=84):
    return bell(m, 0.6) * 0.8 + bell(m + 7, 0.6) * 0.5


def tick():
    n = int(0.06 * SR)
    t = np.arange(n) / SR
    return (np.sin(2 * np.pi * 1760 * t) * env(n, 0.0008, 0.018)).astype(np.float32)


def soft_boom():
    n = int(1.2 * SR)
    t = np.arange(n) / SR
    f = 40 + 50 * np.exp(-t / 0.08)
    return (np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.45)).astype(np.float32)


# ---------------------------------------------------------------- harmony: Ab major, IV–iii–vi–ii in 9ths
CHORDS = getattr(E, 'HOUSE_CHORDS', [
    (37, [53, 56, 60, 63]),   # Dbmaj9 (F Ab C Eb over Db)
    (36, [51, 55, 58, 62]),   # Cm9    (Eb G Bb D over C)
    (41, [56, 60, 63, 67]),   # Fm9    (Ab C Eb G over F)
    (34, [49, 53, 56, 60]),   # Bbm9   (Db F Ab C over Bb)
])
SCALE = [68, 70, 72, 75, 77, 80, 82, 84]                   # Ab major pentatonic
MOTIF = [0, 2, 3, 2, 4, 3, 2, 1, 2, 4, 5, 4, 3, 2, 1, 0]   # 16-step lead motif (indices into SCALE)
LEAD_RHYTHM = [0, 3, 6, 8, 10, 11, 14]                     # 16ths per bar the bell plays on


def chord_at(b):
    return CHORDS[int(b // 4) % len(CHORDS)]


drums, bassb, keys, padb, lead, fx, sfx = Bus(), Bus(), Bus(), Bus(), Bus(), Bus(), Bus()
KICK, CLAP, HC, HO, SHK = kick(), clap(), hat(), hat(True), shaker()
kicks = []
for b in range(int(NBEATS) + 1):
    s = sec_at(b)
    m = s['mode']
    if b >= MUSIC_END:
        break
    k_ = (b - s['a']) / max(s['z'] - s['a'], 1)
    if m in ('pulse', 'build', 'full', 'mid'):
        drums.add(KICK, bs(b), 0.9 if m != 'pulse' else 0.75)
        kicks.append(bs(b))
    if m in ('full', 'mid') or (m == 'build' and k_ >= 0.25):
        if b % 2 == 1:
            drums.add(CLAP, bs(b), 0.42, pan=0.05)
        drums.add(HO, bs(b + 0.5), 0.13 if m != 'build' else 0.13 * k_, pan=0.25)
        for s16 in range(4):
            vel = (0.07, 0.035, 0.055, 0.03)[s16] * (1.0 if m != 'build' else k_)
            drums.add(HC, sw(b * 4 + s16), vel, pan=-0.3)
            drums.add(SHK, sw(b * 4 + s16), 0.02 if m == 'full' else 0.012, pan=0.4)
    # house bass: off-8th pushes with a pickup on the last 16th
    if m in ('full', 'mid') or (m == 'build' and k_ >= 0.5):
        root, _ = chord_at(b)
        bassb.add(bass(root, BEAT * 0.42), bs(b + 0.5), 0.5)
        if b % 4 == 3:
            bassb.add(bass(root + 12, BEAT * 0.2), sw(b * 4 + 3), 0.32)
        if b % 4 == 0:
            bassb.add(bass(root, BEAT * 0.25), bs(b), 0.28)

# EP chords: Charleston-ish stabs per bar (1, the "and" of 2, 4e), sustained in pad/break/end
for bar in range(int(MUSIC_END) // 4 + 1):
    b0 = bar * 4
    if b0 >= MUSIC_END:
        break
    s = sec_at(b0)
    m = s['mode']
    _, notes = chord_at(b0)
    if m in ('pad', 'break'):
        keys.add(ep(notes, 4 * BEAT, 0.7), bs(b0), 0.5)
    else:
        for pos, dur, g in ((0, 1.2, 0.48), (1.5, 0.7, 0.36), (3.25, 0.6, 0.3)):
            keys.add(ep(notes, dur * BEAT, 1.0 if m == 'full' else 0.8), sw(int((b0 + pos) * 4)), g, pan=-0.15)
    if m in ('full', 'break', 'mid', 'pad', 'end') or m == 'build':
        cut = 900 if m in ('pad', 'build') else 1500
        padb.add(pad([n_ - 12 for n_ in notes[:3]] + [notes[3]], bs(4), cut), bs(b0), 0.22 if m != 'full' else 0.18)

# bell lead with ping-pong delay in full sections
for s in SECS:
    if s['mode'] != 'full':
        continue
    for b in range(int(s['a']), int(s['z'])):
        bar16 = (b % 4) * 4
        for s16 in range(4):
            pos = bar16 + s16
            if pos not in LEAD_RHYTHM:
                continue
            step = (b * 4 + s16) // 2 % len(MOTIF)
            note = SCALE[MOTIF[step]]
            x = bell(note, 0.45)
            at = sw(b * 4 + s16)
            lead.add(x, at, 0.16, pan=0.15)
            d16 = bs(0.75)                                  # dotted-8th echoes, alternating sides
            lead.add(x, at + d16, 0.07, pan=-0.6)
            lead.add(x, at + 2 * d16, 0.035, pan=0.6)

# transitions: risers into full sections, soft boom on entries, final chord ring-out
for s in SECS:
    if s['mode'] == 'full' and s['prev'] != 'full' and s['a'] >= 2:
        fx.add(riser(2), bs(s['a'] - 2), 0.3)
        fx.add(soft_boom(), bs(s['a']), 0.35)
    if s['mode'] == 'end':
        _, notes = chord_at(s['a'])
        root, _ = chord_at(s['a'])
        tn = N - bs(s['a'])
        keys.add(ep(notes + [notes[1] + 12], tn / SR, 0.9) * 1.2, bs(s['a']), 0.55)
        padb.add(pad(notes, tn, 1600), bs(s['a']), 0.25)
        bassb.add(bass(root, 1.5), bs(s['a']), 0.5)
        fx.add(soft_boom(), bs(s['a']), 0.4)

# SFX: a whoosh + chime on every route step, ticks on marker pops, whoosh on cue cuts
if E.ROUTE:
    for i, st in enumerate(E.ROUTE['steps']):
        sfx.add(whoosh(0.5), bs(st[0]) - int(0.25 * SR), 0.35, pan=-0.3 + 0.1 * i)
        sfx.add(chime(84 + (i % 3) * 2), bs(st[0]) + int(0.05 * SR), 0.12, pan=0.2)
for c in CUES:
    b = c['b']
    for mk in c.get('markers', []):
        sfx.add(tick(), bs(b + mk.get('at', 0)), 0.22, pan=0.3)
        sfx.add(tick(), bs(b + mk.get('at', 0) + 0.35), 0.14, pan=0.35)
    if c['kind'] in ('clip', 'layers') and c.get('whoosh'):
        sfx.add(whoosh(0.35), bs(b) - int(0.2 * SR), 0.22)

# ---------------------------------------------------------------- vinyl, sidechain, mix
vinyl = filt(noise(N), 'band', [1500, 7000]) * 0.004
pops = np.zeros(N, np.float32)
for i in rng.integers(0, N, int(TOTAL * 6)):
    pops[i] = rng.uniform(0.1, 0.35) * rng.choice([-1, 1])
vinyl += filt(pops, 'high', 2000)
duck = np.ones(N, np.float32)
dn = int(0.3 * SR)
shape = 1 - 0.6 * np.exp(-np.arange(dn) / SR / 0.11)
for k in kicks:
    e = min(N, k + dn)
    duck[k:e] = np.minimum(duck[k:e], shape[: e - k])

ir_n = int(1.8 * SR)
ir_t = np.arange(ir_n) / SR
irs = [filt((rng.standard_normal(ir_n) * np.exp(-ir_t / 0.5)).astype(np.float32), 'low', 6000) for _ in range(2)]
irs = [i_ / np.sqrt((i_ ** 2).sum()) for i_ in irs]
send = keys.x * 0.3 + lead.x * 0.5 + padb.x * 0.2
rev = np.stack([fftconvolve(send[:, 0], irs[0])[:N], fftconvolve(send[:, 1], irs[1])[:N]], 1).astype(np.float32) * 0.5

# intro filter: everything except vinyl low-passed in the first section, opening by its end
music = drums.x + (bassb.x + keys.x + padb.x + lead.x + rev) * duck[:, None]
first = SECS[0]
if first['mode'] == 'pad':
    i1 = bs(first['z'])
    lp_part = filt(music[:i1], 'low', 1200)
    w = np.linspace(0, 1, i1, dtype=np.float32)[:, None] ** 2
    music[:i1] = lp_part * (1 - w) + music[:i1] * w
music = music + fx.x
music[:, 0] += vinyl
music[:, 1] += vinyl
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
print('wrote', os.path.join(E.WORK, 'music.wav'), 'and sfx.wav (house)')
