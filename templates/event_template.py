"""<活动名> <年份> · 宣传片 event config. Fill every TODO, then:

  PY=~/.cache/conf-promo-video/venv/bin/python; E=~/.claude/skills/conf-promo-video/engine
  $PY $E/core.py event.py --check              # every path / id / tier resolves
  $PY $E/contact_sheet.py event.py             # pick photos (ids on thumbnails)
  $PY $E/render.py event.py --stills           # review WORK/stills_sheet_*.jpg
  bash $E/build.sh event.py                    # final mp4 + cover + stems + QA

Times are in BEATS (128 BPM: 1 beat = 0.46875 s, 1 bar = 4 beats = 1.875 s). Every cut lands on a beat;
start sections on bar lines (multiples of 4) and slams on even beats.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = HERE                                   # deliverables land here
WORK = os.path.join(HERE, 'work')                # intermediates (renders, stems, stills)
OUT_NAME = 'TODO活动名2026_宣传视频'

BPM, FPS, TOTAL = 128, 30, 92.0                  # TOTAL should equal (endcard beat + ~10 beats) * 60 / BPM
COVER_AT = TOTAL - 4.0                           # cover = a frame of the held end card

FONTS = dict(heavy='TODO/MiSans-Heavy.ttf', bold='TODO/MiSans-Bold.ttf', semibold='TODO/MiSans-Semibold.ttf',
             medium='TODO/MiSans-Medium.ttf', mono='TODO/FiraCode-SemiBold.ttf')   # mono = Latin/digits only
KV = 'TODO/kv_without_text.jpg'                  # key visual, ideally without baked-in text (text is typeset on the left)
ACCENT = (255, 107, 44)                          # the brand's hot colour: bars, pills, tracker locks

# ---------------------------------------------------------------- photos (past edition)
PHOTO_DIRS = ['TODO/A_opening', 'TODO/B_talks']  # not recursive; id = first PHOTO_ID_LEN chars of the file name
PHOTO_ID_LEN = 3
ANCHOR = {}        # pid -> 0..1 vertical crop anchor for 16:9 (default 0.8); lower keeps more of the top (heads)
POIS = {}          # pid -> [(u, v, w, h)] tracker boxes on OBJECTS (screens, banners, mascots), never on faces
RECAP = (12, 120, 'TODO 2025 · 城市 · 往届现场')   # beats in which every photo is the past edition: tag stays on screen

# ---------------------------------------------------------------- text
TERMINAL = dict(cmd='$ cargo run --release',                     # 8-beat opener; a non-Rust event can use any CLI
                lines=[('   Compiling', ' todoconf v2026.10.15'),
                       ('    Finished', ' `release` profile [optimized]'),
                       ('     Running', ' `target/release/todoconf`')],
                lock='todoconf')                                 # tracker locks onto this word on beat 7
ENDCARD_LINES = [                                                # left-aligned on the KV, revealed one per half beat
    dict(text='TODOConf 2026', font=('heavy', 112), delay=0.0, y=190),
    dict(text='TODO 主题', font=('heavy', 66), color=(255, 150, 80), delay=0.5, y=336),
    dict(text='TODO 城市 · 2026.MM.DD–DD', font=('bold', 50), delay=1.0, y=432),
    dict(text='TODO 日程一句话', font=('semibold', 36), color=(240, 222, 205), delay=1.5, y=512),
    dict(text='todo.example.org', font=('mono', 38), delay=2.0, y=580, lock=True),
]


# ---------------------------------------------------------------- timeline
def P(b, n, pid, motion='punch'):                # motion: punch (default) | push (slow zoom) | flash (teaser)
    return dict(kind='photo', b=b, n=n, pid=pid, motion=motion)


def S(b, n, text, sub=None, bg=None, size=150):  # giant word slam; bg = a photo id shown blurred behind it
    return dict(kind='slam', b=b, n=n, text=text, sub=sub, bg=bg, size=size)


CUES = [
    dict(kind='terminal', b=0, n=8),
    S(8, 2, '一年一度', size=260), S(10, 2, 'TODO 一句定位', size=170),
    P(12, 1, 'A01', 'flash'), P(13, 1, 'A02', 'flash'), P(14, 1, 'A03', 'flash'), P(15, 1, 'A04', 'flash'),
    S(16, 2, 'TODO 2025 · 城市', sub='往届现场', bg='A01', size=230),
    # … build (16–32): 2-beat shots + one slam; drop 1 (32–64) and drop 2 (64–96): slam every 8 beats, 2-beat then 1-beat shots
    dict(kind='pano', b=96, n=8, pid='TODO', text='TODO 情绪句'),
    # … group photos (104–112)
    dict(kind='wall', b=112, n=8, wall='past'),
    S(120, 2, '下一站', size=260), S(122, 2, 'TODO 城市', size=340), S(124, 2, '2026.MM.DD–DD', size=200),
    S(126, 2, 'TODOConf 2026', size=150),
    dict(kind='reveal', b=128, n=8),
    S(136, 2, '2026 讲师阵容', size=200),
    *[dict(kind='keynote', b=138 + 2 * i, n=2, k=i) for i in range(7)],      # one card per keynote, 2 beats each
    dict(kind='trackwall', b=152, n=6, track='T1'),
    dict(kind='wall', b=170, n=12, wall='now'),
    dict(kind='endcard', b=182, n=None),
]

MUSIC_SECTIONS = [   # contiguous from beat 0; modes and overrides are documented in engine/music.py
    (0, 8, 'pad'), (8, 16, 'pulse'), (16, 32, 'build'), (32, 64, 'full'), (64, 96, 'full'),
    (96, 104, 'break'), (104, 120, 'full'), (120, 128, 'count'), (128, 136, 'reveal'),
    (136, 138, 'hit'), (138, 152, 'mid'), (152, 170, 'full', dict(fill=False)), (170, 182, 'full', dict(roll=0, breath=0)),
    (182, None, 'end'),
]

# ---------------------------------------------------------------- sponsor walls: one per edition, tiers as the organiser names them
WALLS = {
    'past': dict(pill='2025 · TODO 城市 · 往届', pill_style='ghost', title='感谢 2025 赞助伙伴', plate=(240, 240, 242),
                 note='以上为 TODOConf 2025（城市）赞助伙伴',
                 tiles={'钻石赞助': (360, 150), '金牌赞助': (300, 124)},          # tile (w, h) per tier
                 rows=[[('钻石赞助', [('TODO 名称', 'TODO/logo.png')])],
                       [('金牌赞助', [('TODO 名称', 'TODO/logo.png')])]]),
    'now': dict(pill='2026 · TODO 城市', title='2026 赞助伙伴', plate=(255, 255, 255),
                tiles={'钻石赞助': (400, 160)},
                rows=[[('钻石赞助', [('TODO 名称', 'TODO/logo.png')])]]),
}
LOGO_DARK_PLATE = set()      # names whose logo is white artwork
LOGO_NAME_BESIDE = set()     # names whose file is only a symbol: the name is typeset beside it

# ---------------------------------------------------------------- speakers & talks (from the agenda source of truth)
TRACK_WALLS = {'T1': dict(label=' · 分会场', name='TODO 分会场名', sub='2026.MM.DD–DD · {n} 场议题')}


def keynotes():
    """One dict per keynote card, in CUES order: tag (pill text), name, role, title, host (optional), photo (path or (path, crop_box))."""
    return [dict(tag='MM.DD · 主会场 · 主题演讲', name='TODO', role='TODO', title='TODO', host=None, photo='TODO.jpg')]


def track_talks(code):
    """All talks of one track (max 15): names, when ('MM.DD HH:MM'), badge (e.g. '闪电演讲' or None), title, photos (list)."""
    return [dict(names='TODO', when='MM.DD 14:00', badge=None, title='TODO', photos=['TODO.jpg'])]
