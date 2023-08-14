import random
import numpy as np
import PIL
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

Aug_Methods = {}


def register(func):
    Aug_Methods[func.__name__] = func
    return func


@register
def Vflip(img, _):
    return img.transpose(Image.FLIP_TOP_BOTTOM)


@register
def Hflip(img, _):
    return img.transpose(Image.FLIP_LEFT_RIGHT)


@register
def Rotate(img, v):
    return img.rotate(v)


@register
def Horizontal_Shift(img, v):
    # v > 0, shift right
    # v < 0, shift left
    assert -1 < v and v < 1
    width, height = img.size

    img = img.crop((0 - v * width, 0, width - v * width, height))

    return img


@register
def Vertical_Shift(img, v):
    assert -1 < v and v < 1
    # v > 0, shift up
    # v < 0, shift down
    width, height = img.size

    img = img.crop((0, v * height, width, height + v * height))


@register
def Zoom(img, v):
    assert -1 < v and v < 1
    w, h = img.size

    img = img.crop((v * w, v * h, w - v * w, h - v * h))
    img = img.resize((w, h), Image.BILINEAR)

    return img


@register
def AutoContrast(img, _):
    return ImageOps.autocontrast(img)


@register
def Brightness(img, v):
    assert v >= 0.0
    return ImageEnhance.Brightness(img).enhance(v)


@register
def Contrast(img, v):
    assert v >= 0.0
    return ImageEnhance.Contrast(img).enhance(v)


@register
def Sharpness(img, v):
    assert v >= 0.0
    return ImageEnhance.Sharpness(img).enhance(v)


@register
def Equalize(img, _):
    return ImageOps.equalize(img)


@register
def Identity(img, _):
    return img


@register
def Invert(img, _):
    return ImageOps.invert(img)


@register
def Posterize(img, v):
    v = int(v)
    return ImageOps.posterize(img, v)


@register
def Solarize(img, v):
    return ImageOps.solarize(img, v)


@register
def Edge_enhance(img, _):
    return img.filter(ImageFilter.EDGE_ENHANCE)


@register
def Edge_enhance_more(img, _):
    return img.filter(ImageFilter.EDGE_ENHANCE_MORE)


@register
def Smooth(img, _):
    return img.filter(ImageFilter.SMOOTH)


@register
def Smooth_more(img, _):
    return img.filter(ImageFilter.SMOOTH_MORE)


@register
def Detail(img, _):
    return img.filter(ImageFilter.DETAIL)


@register
def Blur(img, _):
    return img.filter(ImageFilter.BLUR)


@register
def Sharpen(img, _):
    return img.filter(ImageFilter.SHARPEN)


@register
def Emboss(img, _):
    return img.filter(ImageFilter.EMBOSS)


@register
def Contour(img, _):
    return img.filter(ImageFilter.CONTOUR)


@register
def Find_edges(img, _):
    return img.filter(ImageFilter.FIND_EDGES)


@register
def SolarizeAdd(img, v, threshold=128):
    v = int(v)
    if random.random() > 0.5:
        v = -v
    img_np = np.array(img).astype(np.int_)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)
