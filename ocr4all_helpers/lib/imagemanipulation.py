from PIL import Image, ImageDraw


def cutout(im, coords):
    """
        Cut out coords from image, crop and return new image.
    """
    coords = [tuple(t) for t in coords]
    if not coords:
        return None
    maskim = Image.new('1', im.size, 0)
    ImageDraw.Draw(maskim).polygon(coords, outline=1, fill=1)
    new = Image.new(im.mode, im.size, "white")
    masked = Image.composite(im, new, maskim)
    cropped = masked.crop([
            min([x[0] for x in coords]), min([x[1] for x in coords]),
            max([x[0] for x in coords]), max([x[1] for x in coords])])
    return cropped