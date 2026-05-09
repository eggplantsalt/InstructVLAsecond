from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json
import textwrap

ROOT = Path("/storage/v-xiangxizheng/zy_workspace/InstructVLA")
SRC_DIR = ROOT / "outputs_vpi/bridge_full_episode"
OUT_DIR = ROOT / "outputs_vpi/bridge_full_episode_visual_centric"
OUT_DIR.mkdir(parents=True, exist_ok=True)

episode = json.loads((SRC_DIR / "episode.json").read_text(encoding="utf-8"))
TEXT = episode["instructions"][0]   # exact task text

def load_font(size=38):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            print("USING_FONT:", p)
            return ImageFont.truetype(p, size=size), p
    print("USING_FONT: PIL_DEFAULT")
    return ImageFont.load_default(), "PIL_DEFAULT"

font, font_path = load_font(38)

frame_paths = sorted(SRC_DIR.glob("frame_*.png"))
assert frame_paths, f"No frames found in {SRC_DIR}"

# 强制按较短宽度换行，避免又挤成一行
wrap_width = 14
lines = textwrap.wrap(TEXT, width=wrap_width)
print("TEXT:", TEXT)
print("WRAPPED_LINES:", lines)

for p in frame_paths:
    img = Image.open(p).convert("RGB")
    W, H = img.size

    pad_x = 10
    pad_y = 10
    line_gap = 6

    # 先测文字尺寸
    dummy = ImageDraw.Draw(img)
    line_sizes = []
    max_w = 0
    total_h = 0

    for line in lines:
        bbox = dummy.textbbox((0, 0), line, font=font, stroke_width=1)
        lw = bbox[2] - bbox[0]
        lh = bbox[3] - bbox[1]
        line_sizes.append((lw, lh))
        max_w = max(max_w, lw)
        total_h += lh

    total_h += line_gap * (len(lines) - 1)
    box_h = total_h + 2 * pad_y

    # 画一个更高、更实的白条
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rectangle([0, 0, W, box_h], fill=(255, 255, 255, 245))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    draw = ImageDraw.Draw(img)
    y = pad_y
    for line, (lw, lh) in zip(lines, line_sizes):
        x = (W - lw) // 2
        draw.text(
            (x, y),
            line,
            fill=(0, 0, 0),
            font=font,
            stroke_width=1,
            stroke_fill=(0, 0, 0),
        )
        y += lh + line_gap

    img.save(OUT_DIR / p.name)

# 输出一个对比样例
raw0 = Image.open(frame_paths[0]).convert("RGB")
vc0 = Image.open(OUT_DIR / frame_paths[0].name).convert("RGB")
side = Image.new("RGB", (raw0.width * 2, raw0.height), "white")
side.paste(raw0, (0, 0))
side.paste(vc0, (raw0.width, 0))
side.save(OUT_DIR / "sample_side_by_side_bigtext_v2.png")

(OUT_DIR / "visual_centric_meta.json").write_text(
    json.dumps({
        "text": TEXT,
        "font_path": font_path,
        "font_size": 38,
        "wrap_width": wrap_width,
        "wrapped_lines": lines,
        "num_frames": len(frame_paths),
    }, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("SAVED_DIR:", OUT_DIR)
print("SAMPLE:", OUT_DIR / "sample_side_by_side_bigtext_v2.png")
