from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json

ROOT = Path("/storage/v-xiangxizheng/zy_workspace/InstructVLA")
SRC_DIR = ROOT / "outputs_vpi/bridge_full_episode"
OUT_DIR = ROOT / "outputs_vpi/bridge_full_episode_visual_centric"
OUT_DIR.mkdir(parents=True, exist_ok=True)

episode = json.loads((SRC_DIR / "episode.json").read_text(encoding="utf-8"))
TEXT = episode["instructions"][0]   # exact task text

def get_font(size=26):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()

def wrap_text(draw, text, font, max_width):
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        test = (cur + " " + w).strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines

font = get_font(26)

frame_paths = sorted(SRC_DIR.glob("frame_*.png"))
assert frame_paths, f"No frames found in {SRC_DIR}"

for p in frame_paths:
    img = Image.open(p).convert("RGB")
    W, H = img.size
    pad_x = 8
    pad_y = 8

    draw = ImageDraw.Draw(img)
    lines = wrap_text(draw, TEXT, font, W - 2 * pad_x)

    line_heights = []
    max_line_w = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        lw = bbox[2] - bbox[0]
        lh = bbox[3] - bbox[1]
        max_line_w = max(max_line_w, lw)
        line_heights.append(lh)

    text_h = sum(line_heights) + 6 * (len(lines) - 1)
    box_h = text_h + 2 * pad_y

    # stronger white box for readability
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rectangle([0, 0, W, box_h], fill=(255, 255, 255, 235))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    draw = ImageDraw.Draw(img)
    y = pad_y
    for line, lh in zip(lines, line_heights):
        bbox = draw.textbbox((0, 0), line, font=font)
        lw = bbox[2] - bbox[0]
        x = (W - lw) // 2   # centered
        draw.text((x, y), line, fill=(0, 0, 0), font=font)
        y += lh + 6

    img.save(OUT_DIR / p.name)

# sample side-by-side
raw0 = Image.open(frame_paths[0]).convert("RGB")
vc0 = Image.open(OUT_DIR / frame_paths[0].name).convert("RGB")
side = Image.new("RGB", (raw0.width * 2, raw0.height), "white")
side.paste(raw0, (0, 0))
side.paste(vc0, (raw0.width, 0))
side.save(OUT_DIR / "sample_side_by_side_bigtext.png")

(OUT_DIR / "visual_centric_meta.json").write_text(
    json.dumps({
        "image_text": TEXT,
        "font_size": 26,
        "num_frames": len(frame_paths),
        "note": "large-text visual centric frames",
    }, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("saved to:", OUT_DIR)
print("sample:", OUT_DIR / "sample_side_by_side_bigtext.png")
print("text:", TEXT)
