from PIL import Image, ImageDraw, ImageFont
import os

os.makedirs(os.path.join(os.path.dirname(__file__), "..", "media"), exist_ok=True)
out_path = os.path.join(os.path.dirname(__file__), "..", "media", "cover.png")

W, H = 1200, 630
img = Image.new("RGB", (W, H), (20, 24, 34))
d = ImageDraw.Draw(img)

try:
    font_large = ImageFont.truetype("arial.ttf", 64)
    font_small = ImageFont.truetype("arial.ttf", 28)
except Exception:
    font_large = ImageFont.load_default()
    font_small = ImageFont.load_default()

title = "The Chaos Economy"
subtitle = "Emergent Collusion — Hackathon Submission"

# Title
w, h = d.textsize(title, font=font_large)
d.text(((W-w)/2, 160), title, fill=(255, 215, 0), font=font_large)
# Subtitle
w, h = d.textsize(subtitle, font=font_small)
d.text(((W-w)/2, 260), subtitle, fill=(200, 200, 200), font=font_small)

# Small footer
footer = "Repo: https://github.com/your/repo  •  W&B: add your run link"
w, h = d.textsize(footer, font=font_small)
d.text(((W-w)/2, 540), footer, fill=(150, 150, 150), font=font_small)

img.save(out_path)
print(f"Saved cover image to {out_path}")
