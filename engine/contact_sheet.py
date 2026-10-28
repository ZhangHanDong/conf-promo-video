#!/usr/bin/env python3
"""Contact sheets of the source photos, each thumbnail labelled with its id, for picking shots and placing tracker boxes.

  python3 contact_sheet.py EVENT.py                 # all photos found via PHOTO_DIRS/PHOTOS -> WORK/source_sheets/
  python3 contact_sheet.py EVENT.py --grid B02 C05  # chosen photos large, with a 10% grid to read u,v,w,h for POIS
Also writes WORK/source_sheets/photos.csv (id, size, orientation, path).
"""
import argparse, csv, os, sys
from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_event  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('event')
    ap.add_argument('--grid', nargs='+', help='photo ids to show large with a coordinate grid')
    ap.add_argument('--cols', type=int, default=6)
    a = ap.parse_args()
    E = load_event(a.event)
    out = os.path.join(E.WORK, 'source_sheets')
    os.makedirs(out, exist_ok=True)
    f = ImageFont.truetype(E.FONTS['mono'], 22)
    if a.grid:
        for pid in a.grid:
            im = ImageOps.exif_transpose(Image.open(E.PHOTOS[pid])).convert('RGB')
            im.thumbnail((1600, 1600))
            d = ImageDraw.Draw(im)
            for i in range(1, 10):
                x, y = im.width * i / 10, im.height * i / 10
                d.line([(x, 0), (x, im.height)], fill=(255, 80, 0), width=1)
                d.line([(0, y), (im.width, y)], fill=(255, 80, 0), width=1)
                d.text((x + 3, 3), f'.{i}', font=f, fill=(255, 220, 0))
                d.text((3, y + 3), f'.{i}', font=f, fill=(255, 220, 0))
            p = os.path.join(out, f'grid_{pid}.jpg')
            im.save(p, quality=88)
            print(p)
        return
    ids = sorted(E.PHOTOS)
    with open(os.path.join(out, 'photos.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['id', 'width', 'height', 'orientation', 'path'])
        for pid in ids:
            with Image.open(E.PHOTOS[pid]) as im:
                iw, ih = ImageOps.exif_transpose(im).size
            w.writerow([pid, iw, ih, 'portrait' if ih > iw else 'landscape', E.PHOTOS[pid]])
    tw, th, per = 320, 213, a.cols * 5
    for si in range(0, len(ids), per):
        chunk = ids[si:si + per]
        rows = (len(chunk) + a.cols - 1) // a.cols
        sheet = Image.new('RGB', (a.cols * tw, rows * (th + 30)), (18, 18, 22))
        d = ImageDraw.Draw(sheet)
        for i, pid in enumerate(chunk):
            r, c = divmod(i, a.cols)
            im = ImageOps.exif_transpose(Image.open(E.PHOTOS[pid])).convert('RGB')
            im.thumbnail((tw - 6, th - 6))
            sheet.paste(im, (c * tw + (tw - im.width) // 2, r * (th + 30) + (th - im.height) // 2))
            d.text((c * tw + 8, r * (th + 30) + th + 3), pid, font=f, fill=(255, 200, 120))
        p = os.path.join(out, f'sheet_{si // per + 1:02d}.jpg')
        sheet.save(p, quality=85)
        print(p)


if __name__ == '__main__':
    main()
