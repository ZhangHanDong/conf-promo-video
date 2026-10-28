# conf-promo-video

一个 [Claude Code](https://claude.com/claude-code) skill：为技术大会和社区活动做「燃」的宣传 / 回顾视频。
往届照片回顾 → 倒数 → 今年揭晓 → 讲师与议题 → 赞助伙伴墙 → 片尾信息页；踩拍剪辑、巨字冲击、跟踪框、
代码合成的原创配乐，一条命令出成片（1080p mp4 + 封面 + 音乐/音效分轨 + 自动质检）。

- 画面由 Python（numpy / OpenCV / Pillow / SciPy）逐帧渲染，封装用 ffmpeg。
- 配乐完全由代码合成（振荡器、噪声、滤波器），**不含任何采样或第三方音乐**。三种风格：燃版 EDM、Drum & Bass、chill house。
- 一场活动的全部内容写在一个 `event.py` 里，从 `templates/event_template.py` 起步。

## 安装
```bash
git clone <本仓库地址> ~/.claude/skills/conf-promo-video
bash ~/.claude/skills/conf-promo-video/engine/setup_env.sh   # 一次性：建 venv 并装依赖
```
需要本机有 `ffmpeg` / `ffprobe`。在 Claude Code 里提到「大会宣传片」「回顾视频」等需求时会自动触发；
完整流程见 `SKILL.md`，分镜、配乐和踩坑记录见 `references/`。

## 你需要自备的东西
本仓库**只含引擎、模板和文档**，不含任何活动素材：
- 活动照片、KV 主视觉、赞助商 logo、讲师头像与议程数据；
- 字体文件（中文粗细多档 + 一个等宽字体）。例如 MiSans、Fira Code，请各自遵守其许可；
- 照片的**摄影版权与肖像授权**、logo 的使用授权，发布前由你自行确认。`references/truth-and-provenance.md` 有检查清单。

## 致谢
跟踪框 HUD 的视觉灵感来自 [ff-tracking](https://github.com/mrsarac/ff-tracking)（MIT，基于 Rust 视频库 fframes）。

## 许可
MIT，见 `LICENSE`。
