# 踩过的坑

| 现象 | 原因 | 处理 |
|---|---|---|
| AAC 编码后真峰值超标约 6 dB | 合成器在接近奈奎斯特频率处有能量，AAC 解码后过冲 | build.sh 在 loudnorm 前做 17 kHz 低通（已内置，不要删） |
| 中文显示成方块（豆腐块） | Fira Code 等等宽字体没有 CJK 字形 | `mono` 只用于英文和数字（终端、时间码、URL、坐标）；中文一律用 heavy/bold/semibold/medium |
| 手机照片横竖颠倒 | 方向写在 EXIF 里，像素本身没有旋转 | 引擎读图时已做 `exif_transpose` |
| 片尾信息页一直在动 | reveal 的闪光和震屏还没结束 | `endcard` cue 从 HOLD_FROM 起直接输出静帧；QA 检查 ≥3 s 静止 |
| 赞助墙一排放不下 | 某个等级的 logo 太多 | 拆成两排（`rows` 再加一行），或调小这个等级的 `tiles` 尺寸 |
| 讲师墙超过 15 场 | 5 列 × 3 行的版式 | 拆成两面墙（比如按天拆），`check` 会报错提示 |
| 白色 logo 看不见 | 白底板上放了白色图形 | 把名称加进 `LOGO_DARK_PLATE` |
| logo 四周留白太大 | 导出文件自带透明或白色边 | 引擎会自动裁掉空白边，但只在图形四周裁，不会动图形本身 |
| 渲染慢 | 逐帧用 Python/OpenCV 合成 | 92 秒约 3–4 分钟（M1）。调试用 `--stills` / `--at`，最后再整片渲染；只改了配乐就用 `SKIP_VIDEO=1` |
| 临时文件丢了 | scratchpad 几天后会被清空 | event.py、选片记录、provenance 都放进项目目录，不要放在 /tmp |
| zsh 下 glob 报 no matches | zsh 默认会对无匹配的 glob 报错 | 脚本用 bash 运行，或者先 `setopt nullglob` |

## 环境
- Python 依赖：numpy、opencv-python-headless、pillow、scipy。`engine/setup_env.sh` 会装到 `~/.cache/conf-promo-video/venv`，build.sh 会自动使用它。
- 需要 `ffmpeg`（brew），编码参数：libx264 crf 16 slow，AAC 256k / 48 kHz，响度 -14 LUFS，真峰值 -1 dBTP。
- 字体：MiSans（小米，免费商用，需遵守其字体许可）和 Fira Code（OFL）。skill 本身不带字体文件，需要用户提供路径。
