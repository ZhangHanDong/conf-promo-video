---
name: conf-promo-video
description: 为技术大会/社区活动做「燃」的宣传视频——往届照片回顾 → 倒数 → 今年揭晓 → 讲师与议题 → 赞助伙伴墙 → 片尾信息页，踩拍剪辑、巨字冲击、ff-tracking 跟踪框、代码合成的原创配乐，一条命令出成片（1080p mp4 + 封面 + 分轨 + 质检）。用户提到大会/活动宣传片、预热视频、回顾视频、promo video、宣传样片、把往届照片做成视频、在视频里放讲师或赞助商时使用。
---

# 大会宣传视频（燃版）

这个 skill 由一支真实的大会宣传片（92 秒，128 BPM）提炼而来。
- **引擎**（`engine/`）是通用的；
- **一场活动的全部内容**都写在一个 `event.py` 里，从 `templates/event_template.py` 起步；
- 配乐有三种风格：默认的燃版 EDM、`MUSIC_STYLE='dnb'`（Drum & Bass）、`MUSIC_STYLE='house'`（chill house）。

```
SKILL=~/.claude/skills/conf-promo-video     # 装在别处就改成实际路径
E=$SKILL/engine; PY=~/.cache/conf-promo-video/venv/bin/python
```

## 流程

### 1. 收集信息
缺什么就问用户，不要编：

| 类别 | 内容 |
|---|---|
| 活动事实 | 大会名、届次和年份、日期、城市、场地、官网、一句主题（例如「Open Source × AI」） |
| 往届照片 | 目录；最好已经按场景分好组并加上编号前缀，如 `A_开场/A01_xxx.jpg` |
| 视觉 | KV 主视觉（最好是不带文字的版本）、品牌色、字体文件（中文粗细四档 + 一个等宽字体） |
| 今年讲师和议题 | 唯一信源，例如议程网站的数据文件 |
| 赞助商 | 往届和今年分开，每家注明等级名称，logo 写清来源 |
| 规格 | 时长（默认约 90 秒）、平台、横版 16:9（引擎目前只做横版） |

### 2. 准备环境和项目
```bash
bash $E/setup_env.sh                       # 一次性：numpy/opencv/pillow/scipy -> ~/.cache/conf-promo-video/venv
mkdir -p <项目>/video_project && cp $SKILL/templates/event_template.py <项目>/video_project/event.py
```
项目目录放在用户的活动目录下，不要放在 /tmp 或 scratchpad，那里几天后会被清空。

### 3. 选片
```bash
$PY $E/contact_sheet.py event.py              # WORK/source_sheets/sheet_*.jpg + photos.csv
$PY $E/contact_sheet.py event.py --grid B02   # 带 10% 网格的大图，用来读跟踪框坐标
```
- 用 Read 看联系表，按段落挑照片。
- 必要时设置 `ANCHOR`（防止人头被裁掉）和 `POIS`（跟踪框只框物体）。

方法见 `references/storyboard.md`。

### 4. 写 `event.py`
- **结构**：照标准结构排 `CUES` 和 `MUSIC_SECTIONS`，单位是拍，见 `references/storyboard.md` 和 `references/music.md`。
- **讲师和议题**：`keynotes()` / `track_talks()` 必须从信源（议程网站的数据文件、JSON 等）实时读取，不要手抄进 event.py。
- **赞助墙**：赞助按等级填进 `WALLS`。

改完先跑：
```bash
$PY $E/core.py event.py --check
```
所有路径、照片 id、等级、讲师照片和段落连续性都通过以后再往下走。

### 5. 看检查帧（一定要看）
```bash
$PY $E/render.py event.py --stills          # WORK/stills_sheet_*.jpg
$PY $E/render.py event.py --at 63.2 75.0    # 单帧
```
用 Read 逐张检查：
- 字有没有溢出或被裁掉，人头有没有被切；
- 跟踪框是否还在物体上；
- 赞助墙是否放得下；
- 标题是否被截断。

有问题就改 event.py 重渲染，直到全部通过。

### 6. 出片
```bash
bash $E/build.sh event.py                # 约 3–4 分钟；只改了配乐时用 SKIP_VIDEO=1
```
产物在 `OUT_DIR`：
- `<NAME>.mp4`（1080p30 H.264 + AAC 256k，-14 LUFS）；
- `<NAME>_封面.jpg`；
- `<NAME>_原创配乐.wav` 和 `<NAME>_音效.wav`。

最后自动跑 `qa.sh`，检查时长、黑帧、片尾静止、响度和真峰值，每项给出 PASS 或 FAIL。**有 FAIL 就不能交付。**

### 7. 交付
- 在 OUT_DIR 写 `README_交付说明.md`：规格、结构时间点、素材来源、原创配乐说明、发布前需要确认的事项、重新渲染的命令。
- 向用户如实汇报 QA 结果。
- 未经同意不上传任何平台。

## 硬规则（详见 `references/truth-and-provenance.md`）
- 往届照片只能当往届用：`RECAP` 角标始终可见；今年的信息只在倒数之后出现。
- 不编造人数、嘉宾、赞助和议题；讲师信息每次构建都从信源读取。
- logo 只缩放、去白边，不改色、不重画；等级名照主办方的叫法；每个 logo 都要有来源记录。
- 不擦照片水印；摄影版权和肖像授权写进「发布前确认」。

## 两种片型
- **回顾向燃版**：有往届照片，用 photo / slam / pano / wall / keynote / trackwall 这些镜头。
- **实拍导览片**：素材是走场视频和现场照片，用 clip、markers、arrows、ROUTE，可以做竖版（`W, H = 1080, 1920`）。
- **工具或产品向暖场片**：素材是截图、KV 分层、徽章，用 layers / carousel 加交叉溶解。会场循环播放时，最后一个 cue 用 `xfade` 溶解回片头第一帧。

新增镜头类型和参数见 `references/storyboard.md` 末节。

## 风格旋钮
- **节奏**：`BPM`（120–140）。
- **配色**：`ACCENT`（品牌热色）、`DARK`（底色）。
- **文案**：slam 文案和字号；`motion`：punch / push / flash。
- **配乐**：`CHORDS`（调性），各段的 `roll`/`breath`/`fill`。

想做冷静、克制的版本时，降 BPM、少用 slam、多用 `push`，并把 `MUSIC_SECTIONS` 里的 `full` 换成 `mid`。

## 坑
见 `references/gotchas.md`。最常见的几个：
- 中文用了 Fira Code 会变成方块；
- AAC 过冲已经内置 17 kHz 低通，不要删；
- 讲师墙最多 15 场；
- 白色 logo 要放深色底板。
