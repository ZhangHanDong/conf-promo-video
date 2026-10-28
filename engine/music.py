#!/usr/bin/env python3
"""Conference promo video · original score + SFX, synthesised entirely in code (oscillators + noise, no samples).

The arrangement is derived from the event's MUSIC_SECTIONS (beat ranges with a mode) and its CUES, so music and
picture stay locked to the same beat grid. Writes WORK/music.wav and WORK/sfx.wav (48 kHz stereo).

  python3 music.py EVENT.py

Section modes (see references/music.md):
  pad      soft chord bed, no drums                 (opening terminal)
  pulse    heartbeat kick under slams/teaser photos
  build    kick + rising hats, filter opens, 8th bass, riser + snare roll into the next drop
  full     four-on-the-floor, clap, hats, 16th bass, sub, supersaw, 16th arp, crashes every 16 beats
  break    half-time plucks over a chord bed, no drums (emotional panorama)
  count    kick every beat + chord stab on every slam (countdown)
  reveal   the biggest hit and a long ringing chord
  hit      cue impacts only, chord bed continues (title slam between sections)
  mid      steady mid-energy groove (keynote cards)
  end      final hit, chord rings out to the end
Optional per-section overrides (4th element dict): roll (beats, 0 = none), breath (gain, 0 = none), fill (bool).
"""
import os, sys, wave
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_event  # noqa: E402

if len(sys.argv) < 2:
    sys.exit(__doc__)
E = load_event(sys.argv[1])
if E.MUSIC_STYLE in ('house', 'dnb'):   # chill house / drum & bass scores live in their own modules
    import runpy
    runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'music_{E.MUSIC_STYLE}.py'), run_name='__main__')
    sys.exit(0)
BEAT, TOTAL, CUES = E.BEAT, E.TOTAL, E.CUES
SR = 48000
N = int(TOTAL * SR)
SPB = int(round(BEAT * SR))
NBEATS = TOTAL / BEAT
rng = np.random.default_rng(128)

SECS = []
for s in E.MUSIC_SECTIONS:
    a, z, mode = s[0], s[1], s[2]
    SECS.append(dict(a=a, z=NBEATS if z is None else z, mode=mode, **(s[3] if len(s) > 3 else {})))
for i, s in enumerate(SECS):
    s['prev'] = SECS[i - 1]['mode'] if i else None
    s['next'] = SECS[i + 1]['mode'] if i + 1 < len(SECS) else None
FULLS = [s for s in SECS if s['mode'] == 'full']
FIRST_FULL = FULLS[0]['a'] if FULLS else None
COUNTS = [s for s in SECS if s['mode'] == 'count']
COUNT0 = COUNTS[0]['a'] if COUNTS else NBEATS
MUSIC_END = max([s['a'] for s in SECS if s['mode'] == 'end'] or [NBEATS])


def sec_at(b):
    for s in SECS:
        if s['a'] <= b < s['z']:
            return s
    return SECS[-1]


def mode_at(b):
    return sec_at(b)['mode']


def tail_skip(b):
    """The last beat of a full section that runs into another full section or a break drops out (room for the fill)."""
    s = sec_at(b)
    return s['mode'] == 'full' and b == s['z'] - 1 and s['next'] in ('full', 'break')


def bs(b):
    return int(round(b * BEAT * SR))


class Bus:
    def __init__(self):
        self.l = np.zeros(N, np.float32)
        self.r = np.zeros(N, np.float32)

    def add(self, sig, at, gain=1.0, pan=0.0):
        """sig: mono (n,) or stereo (n, 2); at: sample index; pan -1..1"""
        if at >= N:
            return
        if at < 0:
            sig = sig[-at:]
            at = 0
        sig = sig[: N - at]
        if sig.ndim == 1:
            gl, gr = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
            self.l[at:at + len(sig)] += sig * gain * gl * 1.414
            self.r[at:at + len(sig)] += sig * gain * gr * 1.414
        else:
            self.l[at:at + len(sig)] += sig[:, 0] * gain
            self.r[at:at + len(sig)] += sig[:, 1] * gain

    def st(self):
        return np.stack([self.l, self.r], 1)


def lp(x, fc, order=2):
    return sosfilt(butter(order, min(fc, SR / 2 * 0.95) / (SR / 2), 'low', output='sos'), x, axis=0).astype(np.float32)


def hp(x, fc, order=2):
    return sosfilt(butter(order, fc / (SR / 2), 'high', output='sos'), x, axis=0).astype(np.float32)


def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo / (SR / 2), hi / (SR / 2)], 'band', output='sos'), x, axis=0).astype(np.float32)


def env(n, a, tau):
    t = np.arange(n) / SR
    return (np.minimum(1, t / max(a, 1e-5)) * np.exp(-t / tau)).astype(np.float32)


def noise(n):
    return rng.standard_normal(n).astype(np.float32)


def saw(freq, n, phase=0.0):
    t = np.arange(n) / SR
    return (2 * ((freq * t + phase) % 1.0) - 1).astype(np.float32)


def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12)


# ---------------------------------------------------------------- instruments
def kick():
    n = int(0.36 * SR)
    t = np.arange(n) / SR
    f = 46 + 120 * np.exp(-t / 0.028)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.24)
    click = hp(noise(n), 2500) * np.exp(-t / 0.003) * 0.4
    return np.tanh((body + click) * 1.6).astype(np.float32)


def clap():
    n = int(0.4 * SR)
    x = np.zeros(n, np.float32)
    for k, off in enumerate((0, 0.009, 0.018, 0.026)):
        i = int(off * SR)
        m = n - i
        x[i:] += bp(noise(m), 900, 3200) * env(m, 0.0005, 0.012 if k < 3 else 0.14)
    return x * 0.8


def snare():
    n = int(0.25 * SR)
    t = np.arange(n) / SR
    tone = np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.05)
    nz = bp(noise(n), 1500, 7000) * np.exp(-t / 0.08)
    return (tone * 0.5 + nz * 0.8).astype(np.float32)


def hat(open_=False):
    n = int((0.3 if open_ else 0.06) * SR)
    return hp(noise(n), 7000 if not open_ else 6000, 3) * env(n, 0.0005, 0.09 if open_ else 0.016)


def crash():
    n = int(2.6 * SR)
    x = hp(noise(n), 3500, 2) * env(n, 0.001, 0.9)
    return np.stack([x, hp(noise(n), 3500, 2) * env(n, 0.001, 0.9)], 1) * 0.45


def boom(big=1.0):
    n = int(2.4 * SR)
    t = np.arange(n) / SR
    f = 38 + 70 * np.exp(-t / 0.06)
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / (0.55 * big))
    hit = lp(noise(n), 1800) * np.exp(-t / 0.08) * 0.6
    down = bp(noise(n), 200, 3000) * np.exp(-t / 0.5) * 0.15
    return np.tanh((sub + hit + down) * 1.3).astype(np.float32)


def riser(beats, top=6000):
    n = bs(beats)
    x = noise(n)
    out = np.zeros(n, np.float32)
    seg_ = SPB // 4
    for i in range(0, n, seg_):
        fc = 300 + (top - 300) * (i / n) ** 2
        out[i:i + seg_] = bp(x[i:i + seg_], fc * 0.6, min(fc * 1.6, SR / 2 * 0.9))
    sweep = np.sin(2 * np.pi * np.cumsum(200 + 1800 * (np.arange(n) / n) ** 2) / SR) * 0.15
    return ((out * 0.6 + sweep) * (np.arange(n) / n) ** 2).astype(np.float32)


def whoosh(dur=0.35):
    n = int(dur * SR)
    return (bp(noise(n), 300, 5000) * np.sin(np.linspace(0, np.pi, n)) ** 3).astype(np.float32)


def blip(f2=1975.5, dur=0.18, tau=0.05):
    n = int(dur * SR)
    t = np.arange(n) / SR
    return (np.sin(2 * np.pi * 1318.5 * t) * env(n, 0.002, tau) + np.sin(2 * np.pi * f2 * t) * env(n, 0.05, 0.06) * 0.6).astype(np.float32)


def supersaw(notes, n, cutoff, voices=7, detune=0.14):
    out = np.zeros((n, 2), np.float32)
    for m in notes:
        for v in range(voices):
            dt = (v - (voices - 1) / 2) / ((voices - 1) / 2) * detune
            s = saw(midi(m + dt), n, rng.random())
            pan = (v / (voices - 1)) * 2 - 1
            out[:, 0] += s * (1 - pan) * 0.5
            out[:, 1] += s * (1 + pan) * 0.5
    out /= len(notes) * voices * 0.5
    return lp(out, cutoff, 2)


def pluck(m, dur=0.22):
    n = int(dur * SR)
    t = np.arange(n) / SR
    s = saw(midi(m), n) * 0.6 + saw(midi(m + 12.04), n) * 0.3
    k = np.exp(-t / 0.05)   # crude filter envelope: bright -> dark
    return ((lp(s, 6000 * BRIGHT) * k + lp(s, 900) * (1 - k)) * env(n, 0.002, 0.12)).astype(np.float32)


CHORDS = E.CHORDS
BRIGHT = E.MUSIC_BRIGHTNESS   # < 1 darkens chords and plucks (calmer venue loop), 1 = the 燃版 sound


def chord_at(beat):
    return CHORDS[int(beat // 4) % len(CHORDS)]


# ---------------------------------------------------------------- arrangement
drums, bass, synth, fx, verb_send, sfx = Bus(), Bus(), Bus(), Bus(), Bus(), Bus()
KICK, CLAP, SNARE, HC, HO = kick(), clap(), snare(), hat(), hat(True)
kick_times = []


def play_kick(b, g=1.0):
    drums.add(KICK, bs(b), 0.95 * g)
    kick_times.append(bs(b))


for b in range(0, int(NBEATS) + 1):
    s = sec_at(b)
    m = s['mode']
    if m == 'full':
        if tail_skip(b):   # the beat before a drop/break is left to the snare fill
            continue
        play_kick(b)
        if b % 2 == 1:
            drums.add(CLAP, bs(b), 0.55)
            verb_send.add(CLAP, bs(b), 0.25)
        drums.add(HO, bs(b + 0.5), 0.22, pan=0.2)
        for s16 in range(4):
            drums.add(HC, bs(b + s16 / 4), 0.10 + (0.05 if s16 == 2 else 0), pan=-0.3)
    elif m == 'build':
        play_kick(b, 0.9)
        k = (b - s['a']) / (s['z'] - s['a'])
        for s16 in range(4):
            drums.add(HC, bs(b + s16 / 4), 0.06 + 0.06 * k, pan=-0.3)
        if k >= 0.5:
            drums.add(HO, bs(b + 0.5), 0.16, pan=0.2)
    elif m == 'pulse':
        play_kick(b, 0.85)
    elif m == 'mid':
        play_kick(b, 0.9)
        drums.add(HO, bs(b + 0.5), 0.16, pan=0.2)
        for s16 in range(4):
            drums.add(HC, bs(b + s16 / 4), 0.08, pan=-0.3)
        if b % 2 == 1:
            drums.add(CLAP, bs(b), 0.4)
    elif m == 'count':
        play_kick(b, 1.0)
        drums.add(HC, bs(b + 0.5), 0.12)

# snare rolls into drops, the reveal and the final hit; fills at the end of a drop
for s in SECS:
    if s['mode'] in ('full', 'reveal', 'end'):
        L = s.get('roll', 8 if s['prev'] == 'build' else 4)
        a, z = s['a'] - L, s['a']
        t = a
        while L and t < z - 0.01:
            k = (t - a) / L
            drums.add(SNARE, bs(t), 0.15 + 0.4 * k, pan=0.1)
            t += 0.5 if k < 0.5 else (0.25 if k < 0.85 else 0.125)
    if s['mode'] == 'full' and s['next'] in ('full', 'break') and s.get('fill', True):
        z = s['z']
        hits = (z - 2, z - 1.5, z - 1, z - 0.75, z - 0.5, z - 0.25) if s['next'] == 'full' else (z - 2, z - 1.5, z - 1)
        for b in hits:
            drums.add(SNARE, bs(b), 0.35)

# bass: rolling 16ths in full parts, 8ths in build/mid; sub on the beat in full parts
for b in range(0, int(MUSIC_END)):
    s = sec_at(b)
    if s['mode'] not in ('build', 'full', 'mid') or b == MUSIC_END - 1:
        continue
    if tail_skip(b) and s.get('fill', True):   # left to the snare fill; without a fill the bass carries the transition
        continue
    root, _ = chord_at(b)
    full = s['mode'] == 'full' and not tail_skip(b)
    cutoff = 220 + 160 * (b - s['a']) / (s['z'] - s['a']) if s['mode'] == 'build' else 380
    for st in ((0.25, 0.5, 0.75) if full else (0.5,)):
        n = int(BEAT * SR / 4)
        x = saw(midi(root), n) * 0.7 + saw(midi(root + 12), n) * 0.3
        bass.add(lp(x, cutoff) * env(n, 0.003, 0.09), bs(b + st), 0.55)
    if full:
        t = np.arange(SPB) / SR
        bass.add((np.sin(2 * np.pi * midi(root - 12) * t) * env(SPB, 0.004, 0.25)).astype(np.float32), bs(b), 0.35)

# chords (supersaw) one bar at a time; the filter follows the section
for bar in range(0, int(MUSIC_END) // 4):
    b0 = bar * 4
    s = sec_at(b0)
    m = s['mode']
    if m == 'pad':
        cutoff, g = 650, 0.18 * min(1.0, (b0 - s['a'] + 4) / 8)
    elif m == 'pulse':
        cutoff, g = 900, 0.22
    elif m == 'build':
        cutoff, g = 600 + 4400 * ((b0 - s['a']) / (s['z'] - s['a'])) ** 1.5, 0.28
    elif m == 'break':
        cutoff, g = 1200, 0.26
    elif m in ('count', 'reveal', 'end'):
        continue                                   # countdown uses stabs, reveal/end their own chord
    elif m in ('mid', 'hit'):
        cutoff, g = 3200, 0.28
    else:
        cutoff, g = 6500, 0.34
    _, notes = chord_at(b0)
    x = supersaw(notes, bs(4), cutoff * BRIGHT)
    synth.add(x, bs(b0), g)
    verb_send.add(x, bs(b0), g * 0.25)

# lead arp (16ths) in the full parts (the first drop gets it after 8 beats), an octave up after the first drop
for s in FULLS:
    a0 = s['a'] + (8 if s['a'] == FIRST_FULL else 0)
    for b in range(int(a0), int(s['z']) - 1):
        _, notes = chord_at(b)
        up = 12 if b >= FIRST_FULL + 32 else 0
        seq = [notes[0], notes[1], notes[2], notes[3], notes[2], notes[1], notes[3] + 12, notes[2]]
        for s16 in range(4):
            p = pluck(seq[(b * 4 + s16) % len(seq)] + 12 + up)
            synth.add(p, bs(b + s16 / 4), 0.10, pan=0.35 if s16 % 2 else -0.35)
            verb_send.add(p, bs(b + s16 / 4), 0.05)

# breakdown plucks (half time, softer)
for s in SECS:
    if s['mode'] == 'break':
        for b in range(int(s['a']), int(s['z'])):
            _, notes = chord_at(b)
            for k, m in enumerate(notes[1:]):
                p = pluck(m + 12, 0.5)
                synth.add(p, bs(b + k * 0.25), 0.10, pan=(k - 1) * 0.5)
                verb_send.add(p, bs(b + k * 0.25), 0.12)

# impacts and SFX on the picture cues
for c in CUES:
    b, kind = c['b'], c['kind']
    if kind == 'slam':
        late = b >= COUNT0
        fx.add(boom(1.3 if late else 1.0), bs(b), 0.7 if late else 0.5)
        sfx.add(whoosh(0.32), bs(b) - int(0.30 * SR), 0.35)
        s = sec_at(b)
        if s['mode'] == 'count':  # countdown chord stabs walk the progression
            _, notes = CHORDS[int((b - s['a']) // 2) % len(CHORDS)]
            st = supersaw(notes, bs(1.5), 5000 * BRIGHT) * env(bs(1.5), 0.003, 0.35)[:, None]
            synth.add(st, bs(b), 0.45)
            verb_send.add(st, bs(b), 0.3)
    elif kind == 'photo':
        if mode_at(b) == 'pulse':
            fx.add(boom(0.6), bs(b), 0.35)
        if c['pid'] in E.POIS:   # tracker lock blip
            sfx.add(blip(1975.5, 0.16, 0.04), bs(b + (1.0 if c['n'] >= 2 else 0.6)), 0.10, pan=0.3)
    elif kind == 'keynote':      # light hit as each card lands, blip on the lock
        fx.add(boom(0.5), bs(b), 0.3)
        sfx.add(blip(1975.5), bs(b + 1), 0.14, pan=-0.2)
    elif kind == 'trackwall':    # title hit + a tick per landing tile
        fx.add(boom(0.9), bs(b), 0.4)
        sfx.add(whoosh(0.32), bs(b) - int(0.30 * SR), 0.3)
        for i in range(len(E.track_talks(c['track']))):
            n_ = int(0.02 * SR)
            sfx.add(bp(noise(n_), 2500, 8000) * env(n_, 0.0005, 0.006), bs(b + 0.6 + i * 0.18), 0.22, pan=-0.6 + 0.1 * min(i, 12))
    elif kind == 'wall':         # sponsor walls: hit on the title, a lock blip per tier group
        fx.add(boom(1.0), bs(b), 0.45)
        sfx.add(whoosh(0.32), bs(b) - int(0.30 * SR), 0.3)
        for gi in range(sum(len(row) for row in E.WALLS[c['wall']]['rows'])):
            sfx.add(blip(1760), bs(b + 0.8 + 0.6 * gi), 0.16, pan=-0.3 + 0.15 * gi)
    elif kind == 'carousel':     # per focus item: whoosh as the stack slides, soft hit as the card expands, lock blip
        per = c['n'] / len(c.get('focus', c['items']))
        for i in range(len(c.get('focus', c['items']))):
            b0 = b + i * per
            sfx.add(whoosh(0.4), bs(b0) - int(0.1 * SR), 0.3, pan=0.4 - 0.4 * (i % 3))
            fx.add(boom(0.5), bs(b0 + 0.2 * per), 0.22)
            sfx.add(blip(1760), bs(b0 + 0.36 * per), 0.11, pan=0.2)
    elif kind == 'layers':       # optional entrance hit; 8-bit arpeggio while pixel art resolves; blip on text locks
        if c.get('hit') == 'boom':
            fx.add(boom(0.9), bs(b), 0.4)
            sfx.add(whoosh(0.32), bs(b) - int(0.30 * SR), 0.3)
        elif c.get('hit') == 'whoosh':
            sfx.add(whoosh(0.4), bs(b) - int(0.2 * SR), 0.3)
        for L in c.get('layers', []):
            if L.get('pixel'):
                for j, m in enumerate((72, 76, 79, 84, 88)):
                    n_ = int(0.07 * SR)
                    sq = np.sign(np.sin(2 * np.pi * midi(m) * np.arange(n_) / SR)).astype(np.float32) * env(n_, 0.001, 0.05)
                    sfx.add(lp(sq, 7000), bs(b + L.get('at', 0)) + int(j * L.get('dur', 1.0) * BEAT * SR / 6), 0.07, pan=-0.3 + 0.15 * j)
        for t in c.get('texts', []):
            if t.get('lock'):
                sfx.add(blip(1975.5), bs(b + t.get('at', 0)), 0.1, pan=0.25)
    elif kind == 'terminal':     # typing, glitch at its end, a small riser into the first slam
        for i, _ in enumerate(E.TERMINAL['cmd']):
            n_ = int(0.012 * SR)
            sfx.add(hp(noise(n_), 2500) * env(n_, 0.0004, 0.002), bs(b) + int((0.25 + i / 22) * SR), 0.5, pan=-0.2 + 0.4 * i / 22)
        for t in np.arange((b + 7.6) * BEAT, (b + 8) * BEAT, 0.035):
            n_ = int(0.03 * SR)
            sfx.add(bp(noise(n_), 2000, 9000) * env(n_, 0.0005, 0.01), int(t * SR), 0.35, pan=rng.uniform(-0.6, 0.6))
        fx.add(riser(1.0, 3000), bs(b + 7), 0.3)

# crashes at drop starts (and every 16 beats inside long drops); risers into drops, builds, countdown and the end
for s in FULLS:
    b = s['a']
    while b == s['a'] or b + 16 <= s['z']:
        fx.add(crash(), bs(b), 0.6)
        b += 16
for s in SECS:
    if s['mode'] == 'full' and s['prev'] != 'full':
        L = 6 if s['prev'] == 'build' else 4
        fx.add(riser(L), bs(s['a'] - L), 0.35)
    elif s['mode'] in ('build', 'end'):
        fx.add(riser(4), bs(s['a'] - 4), 0.35)
    elif s['mode'] == 'count':
        fx.add(riser(s['z'] - s['a']), bs(s['a']), 0.35)

# reveal and end: the biggest hits with a sustained tonic chord and a long tail
root, voicing = CHORDS[0]
tonic = [root] + list(voicing) + [voicing[1] + 12]
for s in SECS:
    if s['mode'] in ('reveal', 'end'):
        R0 = bs(s['a'])
        big = s['mode'] == 'reveal'
        fx.add(boom(1.8), R0, 0.9 if big else 0.85)
        fx.add(crash(), R0, 0.8 if big else 0.7)
        tn = N - R0
        x = supersaw(tonic, tn, 3200 * BRIGHT, voices=7)
        x *= (env(tn, 0.004, 2.6 if big else 2.8) * 0.8 + 0.2 * env(tn, 0.4, 3.5 if big else 4.0))[:, None]
        synth.add(x, R0, 0.42)
        verb_send.add(x, R0, 0.3)
        tt = np.arange(tn) / SR
        synth.add((np.sin(2 * np.pi * midi(root - 12) * tt) * env(tn, 0.01, 2.0 if big else 2.2)).astype(np.float32), R0, 0.4)

# ---------------------------------------------------------------- sidechain + reverb + master
duck = np.ones(N, np.float32)
dn = int(0.22 * SR)
shape = 1 - 0.72 * np.exp(-np.arange(dn) / SR / 0.075)
for k in kick_times:
    if k < N:
        e = min(N, k + dn)
        duck[k:e] = np.minimum(duck[k:e], shape[: e - k])

ir_n = int(2.0 * SR)
ir_t = np.arange(ir_n) / SR
irL = lp((rng.standard_normal(ir_n) * np.exp(-ir_t / 0.55)).astype(np.float32), 5000)
irR = lp((rng.standard_normal(ir_n) * np.exp(-ir_t / 0.55)).astype(np.float32), 5000)
irL /= np.sqrt((irL ** 2).sum())
irR /= np.sqrt((irR ** 2).sum())
vs = verb_send.st()
rev = np.stack([fftconvolve(vs[:, 0], irL)[:N], fftconvolve(vs[:, 1], irR)[:N]], 1).astype(np.float32) * 0.6

# section gain automation for drums/bass/synth (impacts and risers stay full); a short breath before big entries
sg = np.ones(N, np.float32)


def seg(b0, b1, g0, g1=None):
    i0, i1 = bs(b0), min(N, bs(b1))
    if i1 > i0:
        sg[i0:i1] = np.linspace(g0, g0 if g1 is None else g1, i1 - i0, dtype=np.float32)


BREATH = dict(full=0.15, **{'break': 0.3}, count=0.15, reveal=0.1, end=0.1)
for s in SECS:
    m = s['mode']
    if m == 'pad':
        seg(s['a'], s['z'], 0.75)
    elif m == 'pulse':
        seg(s['a'], s['z'], 0.62)
    elif m == 'build':
        seg(s['a'], s['z'] - 0.25, 0.55, 0.78)
    g = s.get('breath', BREATH.get(m))
    if g and s['a'] > 0:
        seg(s['a'] - 0.25, s['a'], g)
music = (drums.st() + bass.st() * duck[:, None] + synth.st() * duck[:, None] + rev * duck[:, None]) * sg[:, None]
mix = hp(music + fx.st(), 28)
fade_n = int(2.0 * SR)
mix[-fade_n:] *= np.linspace(1, 0, fade_n, dtype=np.float32)[:, None] ** 1.5
mix = np.tanh(mix * 0.9) / np.tanh(0.9)


def write(path, st):
    pk = np.abs(st).max()
    st = st / max(pk, 1e-9) * 10 ** (-1.0 / 20)
    with wave.open(path, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())


os.makedirs(E.WORK, exist_ok=True)
write(os.path.join(E.WORK, 'music.wav'), mix.astype(np.float32))
write(os.path.join(E.WORK, 'sfx.wav'), sfx.st())
print('wrote', os.path.join(E.WORK, 'music.wav'), 'and sfx.wav')
