#!/usr/bin/env python3
import os
from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int):
    # Try common fonts; fall back to default
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
        except Exception:
            return ImageFont.load_default()


def make_collage(img_paths, output_path, tile_cols=4, tile_rows=2, padding=24, bg_color=(255, 255, 255), title="GC model orders 1–8", scale=2.0):
    imgs = [Image.open(p).convert("RGBA") for p in img_paths]
    # Determine target cell size from max source dimensions and upscale by `scale`
    max_w = max(i.width for i in imgs)
    max_h = max(i.height for i in imgs)
    cell_w = int(max_w * scale)
    cell_h = int(max_h * scale)

    # Preserve aspect ratio for each image; center on a cell canvas to avoid squishing
    tiles = []
    for i in imgs:
        ratio = min(cell_w / i.width, cell_h / i.height)
        new_size = (max(1, int(i.width * ratio)), max(1, int(i.height * ratio)))
        imr = i.resize(new_size, Image.LANCZOS)
        tile = Image.new("RGBA", (cell_w, cell_h), (*bg_color, 255))
        ox = (cell_w - imr.width) // 2
        oy = (cell_h - imr.height) // 2
        tile.alpha_composite(imr, (ox, oy))
        tiles.append(tile)

    # Space for labels under each tile and a title row on top
    label_h = int(0.16 * cell_h)
    title_h = int(0.22 * cell_h)
    total_w = tile_cols * cell_w + (tile_cols + 1) * padding
    total_h = title_h + tile_rows * (cell_h + label_h) + (tile_rows + 2) * padding
    canvas = Image.new("RGBA", (total_w, total_h), (*bg_color, 255))
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(max(18, int(0.09 * total_w / tile_cols)))
    label_font = _load_font(max(14, int(0.06 * cell_w)))

    # Title centered
    # Compute text sizes in a Pillow-compatible way
    try:
        tw, th = title_font.getsize(title)
    except Exception:
        tw, th = draw.textbbox((0, 0), title, font=title_font)[2:4]
    draw.text(((total_w - tw) // 2, padding + (title_h - th) // 2), title, fill=(0, 0, 0), font=title_font)

    for idx, im in enumerate(tiles):
        r = idx // tile_cols
        c = idx % tile_cols
        x = padding + c * (cell_w + padding)
        y = padding + title_h + r * (cell_h + label_h + padding) + padding
        canvas.alpha_composite(im, (x, y))
        # Label below the image
        label = f"Order {idx+1}"
        try:
            lw, lh = label_font.getsize(label)
        except Exception:
            lw, lh = draw.textbbox((0, 0), label, font=label_font)[2:4]
        draw.text((x + (cell_w - lw) // 2, y + cell_h + (label_h - lh) // 2), label, fill=(0, 0, 0), font=label_font)

    # Save as PNG
    canvas.convert("RGB").save(output_path)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    # Prefer the scalar order images which clearly reflect order size
    img_dir = os.path.join(here, "images")
    if not os.path.isdir(img_dir):
        img_dir = here
    imgs = [os.path.join(img_dir, f"gc_car_model_scalar_order_{k}.png") for k in range(1, 9)]
    imgs = [p for p in imgs if os.path.isfile(p)]
    if len(imgs) >= 8:
        out = os.path.join(img_dir, "gc_car_model_orders_1_8_collage.png")
        make_collage(imgs, out, scale=2.0)
        print(f"Saved collage to {out}")
    else:
        print("Not enough images to build collage.")


