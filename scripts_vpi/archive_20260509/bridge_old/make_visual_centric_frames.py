from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json

ROOT = Path("/storage/v-xiangxizheng/zy_workspace/InstructVLA")
SRC_DIR = ROOT / "outputs_vpi/bridge_full_episode"
OUT_DIR = ROOT / "outputs_vpi/bridge_full_episode_visual_centric"
OUT_DIR.mkdir(parents=True, exist_ok=True)

episode = json.loads((SRC_DIR / "episode.json").read_text(encoding="utf-8"))
TEXT = episode["instructions"][0].strip()

def find_font():
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/opt/conda/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf",
        "/tmp/conda_env/instructvla/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf",
    ]

    try:
        import matplotlib
        candidates.append(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf"))
        candidates.append(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"))
    except Exception:
        pass

    for p in candidates:
        if Path(p).exists():
            return Path(p)

    return None

def text_size(draw, text, font, stroke_width=0):
    box = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    return box[2] - box[0], box[3] - box[1]

font_path = find_font()

frame_paths = sorted(SRC_DIR.glob("frame_*.png"))
assert frame_paths, f"No frames found in {SRC_DIR}"

img0 = Image.open(frame_paths[0]).convert("RGB")
W, H = img0.size
dummy = ImageDraw.Draw(img0)

if font_path is not None:
    chosen_font = None
    chosen_size = None

    # 字号稍微调小一点，并且允许继续往下找，避免一行太满
    for size in range(18, 8, -1):
        font = ImageFont.truetype(str(font_path), size=size)
        tw, th = text_size(dummy, TEXT, font, stroke_width=0)
        if tw <= W - 20:   # 左右多留一点边距
            chosen_font = font
            chosen_size = size
            break

    if chosen_font is None:
        chosen_size = 10
        chosen_font = ImageFont.truetype(str(font_path), size=chosen_size)
else:
    chosen_font = ImageFont.load_default()
    chosen_size = "PIL_DEFAULT"

tw, th = text_size(dummy, TEXT, chosen_font, stroke_width=0)

# 顶部白条再薄一点，减少遮挡
pad_y = 3
box_h = th + 2 * pad_y + 2

print("TEXT:", TEXT)
print("FONT_PATH:", font_path if font_path else "PIL_DEFAULT")
print("FONT_SIZE:", chosen_size)
print("TEXT_SIZE:", (tw, th))
print("BOX_H:", box_h)

for p in frame_paths:
    img = Image.open(p).convert("RGB")
    W, H = img.size

    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rectangle([0, 0, W, box_h], fill=(255, 255, 255, 235))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    draw = ImageDraw.Draw(img)
    x = (W - tw) // 2
    y = pad_y

    draw.text(
        (x, y),
        TEXT,
        font=chosen_font,
        fill=(0, 0, 0),
    )

    img.save(OUT_DIR / p.name)

raw0 = Image.open(frame_paths[0]).convert("RGB")
vc0 = Image.open(OUT_DIR / frame_paths[0].name).convert("RGB")
side = Image.new("RGB", (raw0.width * 2, raw0.height), "white")
side.paste(raw0, (0, 0))
side.paste(vc0, (raw0.width, 0))
side.save(OUT_DIR / "sample_side_by_side.png")

(OUT_DIR / "visual_centric_meta.json").write_text(
    json.dumps({
        "text": TEXT,
        "font_path": str(font_path) if font_path else "PIL_DEFAULT",
        "font_size": chosen_size,
        "text_size": [tw, th],
        "box_h": box_h,
        "num_frames": len(frame_paths),
        "note": "original-style single-line top overlay with slightly smaller clearer TTF font",
    }, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("SAVED_DIR:", OUT_DIR)
print("SAMPLE:", OUT_DIR / "sample_side_by_side.png")
