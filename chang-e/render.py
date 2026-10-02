"""Chang'e Flies to the Moon - procedural layered paper-cut animation.

1080x1920, 30fps, 20s. Every frame is rendered from vector shapes + numpy
compositing (Pillow / NumPy / SciPy). No external images.

usage:
  python render.py test 0 4.5 9.1 ...      -> frames/test_XXXX.png
  python render.py video out.mp4           -> video-only master (piped to ffmpeg)
"""
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

W, H, FPS, DUR = 1080, 1920, 30, 20.0
NFR = int(FPS * DUR)
CX, CY = W / 2, H / 2
SS = 2
HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ colour
def hx(h, a=None):
    h = h.lstrip('#')
    c = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    return c if a is None else c + (a,)


def hf(h):
    return np.array(hx(h), np.float32) / 255.0


PAL = dict(
    shadow='#0b0920', ink='#1a1430',
    ivory='#e2d6bd', ivory_d='#bfb39b', celadon='#9cc7bb', skin='#f6e8d6',
    cinnabar='#cc3a33', cin_dark='#8c1f25', coral='#f08060', gold='#ecc062',
    jade_d='#1c4446', jade='#2d6a62',
)


# ------------------------------------------------------------------ easing
def c01(x):
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


def smooth(x):
    x = c01(x)
    return x * x * (3 - 2 * x)


def eio(x):
    x = c01(x)
    return 4 * x ** 3 if x < .5 else 1 - (-2 * x + 2) ** 3 / 2


def eout(x):
    x = c01(x)
    return 1 - (1 - x) ** 3


def ein(x):
    x = c01(x)
    return x ** 3


def lin(x):
    return c01(x)


EASE = dict(s=smooth, io=eio, o=eout, i=ein, l=lin)


def kf(t, keys):
    def V(v):
        return np.array(v, float) if isinstance(v, tuple) else float(v)
    if t <= keys[0][0]:
        return V(keys[0][1])
    for k0, k1 in zip(keys, keys[1:]):
        if t <= k1[0]:
            e = k1[2] if len(k1) > 2 else 's'
            u = EASE[e]((t - k0[0]) / (k1[0] - k0[0]))
            return V(k0[1]) + (V(k1[1]) - V(k0[1])) * u
    return V(keys[-1][1])


def sstep(a, b, t):
    return smooth((t - a) / (b - a))


def bump(a, b, c, d, t):
    return sstep(a, b, t) * (1 - sstep(c, d, t))


# ------------------------------------------------------------------ paper texture
TEXN = 1024
_TEX = None


def _make_tex(seed=7):
    r = np.random.default_rng(seed)
    n = np.zeros((TEXN, TEXN), np.float32)
    for s, a in ((0.7, 0.35), (2.5, 0.3), (10, 0.25), (36, 0.3)):
        g = ndimage.gaussian_filter(r.standard_normal((TEXN, TEXN)).astype(np.float32), s, mode='wrap')
        n += a * g / g.std()
    fib = Image.new('L', (TEXN, TEXN), 0)
    d = ImageDraw.Draw(fib)
    for i in range(3200):
        x, y = r.uniform(0, TEXN, 2)
        ang = r.uniform(0, np.pi)
        L = r.uniform(8, 50)
        pts = []
        for k in range(6):
            u = k / 5
            a2 = ang + 0.5 * math.sin(u * 3 + i)
            pts.append((x + math.cos(a2) * L * u, y + math.sin(a2) * L * u))
        d.line(pts, fill=int(r.uniform(50, 170)), width=1)
    fib = np.asarray(fib.filter(ImageFilter.GaussianBlur(0.6)), np.float32) / 255.0
    return (1 + 0.03 * n + 0.075 * fib - 0.02).astype(np.float32)


def tex(h, w, ox=0, oy=0):
    global _TEX
    if _TEX is None:
        _TEX = _make_tex()
    ys = (np.arange(h) + oy) % TEXN
    xs = (np.arange(w) + ox) % TEXN
    return _TEX[np.ix_(ys, xs)]


# ------------------------------------------------------------------ canvases
class MC:
    """Supersampled drawing canvas in local units."""

    def __init__(self, x0, y0, x1, y1, ss=SS, mode='L'):
        self.x0, self.y0, self.ss = math.floor(x0), math.floor(y0), ss
        self.w1 = int(math.ceil(x1 - self.x0))
        self.h1 = int(math.ceil(y1 - self.y0))
        self.im = Image.new(mode, (self.w1 * ss, self.h1 * ss), 0)
        self.d = ImageDraw.Draw(self.im, 'RGBA' if mode == 'RGBA' else None)

    def P(self, pts):
        s = self.ss
        return [((x - self.x0) * s, (y - self.y0) * s) for x, y in pts]

    def poly(self, pts, fill=255):
        if len(pts) > 2:
            self.d.polygon(self.P(pts), fill=fill)

    def ell(self, cx, cy, rx, ry=None, fill=255):
        ry = rx if ry is None else ry
        s = self.ss
        self.d.ellipse([(cx - rx - self.x0) * s, (cy - ry - self.y0) * s,
                        (cx + rx - self.x0) * s, (cy + ry - self.y0) * s], fill=fill)

    def line(self, pts, w, fill=255):
        self.d.line(self.P(pts), fill=fill, width=max(1, int(round(w * self.ss))), joint='curve')

    def rect(self, x0, y0, x1, y1, fill=255):
        self.poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], fill)

    def alpha(self):
        im = self.im.resize((self.w1, self.h1), Image.LANCZOS)
        return np.asarray(im, np.float32) / 255.0


class Spr:
    __slots__ = ('img', 'ax', 'ay')

    def __init__(self, img, ax, ay):
        self.img, self.ax, self.ay = img, ax, ay


def shift(a, dx, dy):
    """value at (x,y) = a[y-dy, x-dx] (zero padded)."""
    out = np.zeros_like(a)
    h, w = a.shape[:2]
    xs0, xs1 = max(0, dx), min(w, w + dx)
    ys0, ys1 = max(0, dy), min(h, h + dy)
    out[ys0:ys1, xs0:xs1] = a[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    return out


def vgrad(h, c0, c1, y0=0, y1=None):
    y1 = h if y1 is None else y1
    t = np.clip((np.arange(h, dtype=np.float32) - y0) / max(1, (y1 - y0)), 0, 1)[:, None, None]
    return hf(c0)[None, None] * (1 - t) + hf(c1)[None, None] * t


def finish_rgba(rgb, a, sh=(6, 10, 9, 0.55), shcol=PAL['shadow'], blur=0.0):
    """rgb (h,w,3), a (h,w) -> RGBA image with drop shadow, plus margin."""
    h, w = a.shape
    dx, dy, sb, sa = sh
    M = int(sb * 3 + max(abs(dx), abs(dy)) + blur * 3 + 2) if sa > 0 else int(blur * 3 + 2)
    A = np.zeros((h + 2 * M, w + 2 * M), np.float32)
    A[M:M + h, M:M + w] = a
    RGB = np.zeros((h + 2 * M, w + 2 * M, 3), np.float32)
    RGB[M:M + h, M:M + w] = rgb
    if sa > 0:
        S = ndimage.gaussian_filter(shift(A, int(dx), int(dy)), sb) * sa
        out_a = A + S * (1 - A)
        out_rgb = RGB * A[..., None] + hf(shcol)[None, None] * (S * (1 - A))[..., None]
    else:
        out_a = A
        out_rgb = RGB * A[..., None]
    if blur > 0:
        out_rgb = ndimage.gaussian_filter(out_rgb, (blur, blur, 0))
        out_a = ndimage.gaussian_filter(out_a, blur)
    rgb_u = out_rgb / np.maximum(out_a[..., None], 1e-4)
    arr = np.dstack([np.clip(rgb_u, 0, 1), np.clip(out_a, 0, 1)[..., None]])
    return Image.fromarray((arr * 255 + 0.5).astype(np.uint8), 'RGBA'), M


def paper(mc, top, bot=None, inner=None, rim=('#fff1d0', 0.35), botdark=0.22,
          sh=(6, 10, 9, 0.55), blur=0.0, decor=(), texoff=(0, 0), grad=None):
    """Turn a mask canvas into a paper-cut sprite (anchor = local origin)."""
    a = mc.alpha()
    h, w = a.shape
    if grad is None:
        ys = np.where(a.max(1) > 0.01)[0]
        g0, g1 = (ys.min(), ys.max()) if len(ys) else (0, h)
    else:
        g0, g1 = grad
    rgb = vgrad(h, top, bot or top, g0, g1)
    rgb = np.broadcast_to(rgb, (h, w, 3)).copy()
    if inner is not None:
        e, c0, c1 = inner
        dist = ndimage.distance_transform_edt(a > 0.5)
        im = np.clip((dist - e) / 1.5, 0, 1)
        rgb = rgb * (1 - im[..., None]) + vgrad(h, c0, c1 or c0, g0, g1) * im[..., None]
        # tiny shadow line where inner layer starts
        edge = np.clip((dist - e + 2.5) / 1.5, 0, 1) * (1 - im)
        rgb *= (1 - 0.25 * edge)[..., None]
    for dmc, col, op in decor:
        da = dmc.alpha()
        if callable(col):
            rgb = col(rgb, da)
        else:
            rgb = rgb * (1 - op * da[..., None]) + hf(col)[None, None] * (op * da)[..., None]
    rgb *= tex(h, w, *texoff)[..., None]
    if rim:
        rc, rs = rim
        rim_a = np.clip(a - shift(a, 0, 3), 0, 1)
        rim_a = ndimage.gaussian_filter(rim_a, 0.8)
        rgb = rgb + hf(rc)[None, None] * (rim_a * rs)[..., None]
    if botdark:
        bd = np.clip(a - shift(a, 0, -3), 0, 1)
        rgb *= (1 - botdark * ndimage.gaussian_filter(bd, 1.0))[..., None]
    img, M = finish_rgba(rgb, a, sh=sh, blur=blur)
    return Spr(img, -mc.x0 + M, -mc.y0 + M)


# ------------------------------------------------------------------ blit
def blit(base, spr, sx, sy, k, theta=0.0, alpha=1.0):
    img, ax, ay = spr.img, spr.ax, spr.ay
    if k < 0.7:
        f = k / 0.95
        nw, nh = max(2, int(img.size[0] * f)), max(2, int(img.size[1] * f))
        fx, fy = nw / img.size[0], nh / img.size[1]
        img = img.resize((nw, nh), Image.LANCZOS)
        ax, ay = ax * fx, ay * fy
        k = k / fx
    w, h = img.size
    c, s = math.cos(theta), math.sin(theta)
    xs, ys = [], []
    for qx, qy in ((0, 0), (w, 0), (0, h), (w, h)):
        dx, dy = (qx - ax) * k, (qy - ay) * k
        xs.append(sx + c * dx - s * dy)
        ys.append(sy + s * dx + c * dy)
    bx0, by0 = max(0, int(math.floor(min(xs)))), max(0, int(math.floor(min(ys))))
    bx1, by1 = min(W, int(math.ceil(max(xs)))), min(H, int(math.ceil(max(ys))))
    if bx1 - bx0 < 1 or by1 - by0 < 1 or alpha <= 0.003:
        return
    data = (c / k, s / k, ax + (c * (bx0 - sx) + s * (by0 - sy)) / k,
            -s / k, c / k, ay + (-s * (bx0 - sx) + c * (by0 - sy)) / k)
    reg = img.transform((bx1 - bx0, by1 - by0), Image.AFFINE, data, resample=Image.BILINEAR)
    if alpha < 0.997:
        r, g, b, a = reg.split()
        a = a.point(lambda v: int(v * alpha))
        reg = Image.merge('RGBA', (r, g, b, a))
    base.alpha_composite(reg, (bx0, by0))


# ------------------------------------------------------------------ story timeline
def her_feet(t):
    x = kf(t, [(0, -380), (3.6, -380), (6.0, 40, 'io'), (9.4, 40), (12, 75, 's'),
               (16.6, -10, 'io'), (20, -20, 'o')])
    y = 480.0
    # glide bob while walking (opera-style round steps)
    y -= 3 * math.sin(max(0, t - 3.6) * 2 * math.pi * 1.6) * bump(3.6, 3.9, 5.6, 6.0, t)
    y -= 300 * sstep(9.4, 11.6, t)
    y -= 4100 * eio((t - 11.0) / 5.6)
    y -= 60 * eout((t - 16.6) / 3.4)
    y += 7 * math.sin((t - 16.6) * 2 * math.pi / 2.7) * sstep(16.6, 17.4, t) * (1 - sstep(17.5, 20, t))
    return x, y


def her_speed(t):
    dt = 1 / 60
    return (her_feet(t + dt)[1] - her_feet(t - dt)[1]) / (2 * dt)


def camera(t):
    z = kf(t, [(0, 1.0), (4, 1.08), (8, 1.55, 'io'), (9.2, 1.5), (11.5, 1.18, 'io'), (13.5, 1.0, 'io'),
               (15, 0.96), (16.6, 1.1, 'io'), (20, 1.16, 'o')])
    tf = min(t, 8.0)
    Ff = kf(tf, [(0, (-20, -180)), (4, (0, 0), 's'), (8, (40, 260), 'io')])
    fx, fy = her_feet(t)
    Fh = np.array([fx, fy - 220])
    w = sstep(7.0, 9.0, t)
    F = Ff * (1 - w) + Fh * w
    O = kf(t, [(0, (0, 0)), (8, (0, 40)), (9.4, (0, 40)), (11.5, (-40, -220), 'io'), (13.5, (0, -120), 'io'),
               (15, (-60, -40), 'io'), (16.6, (-150, -90), 'io'), (20, (-180, -100), 'o')])
    cam = F - O / z
    still = 1 - sstep(17.5, 20, t)
    cam = cam + np.array([7 * math.sin(0.5 * t), 5 * math.sin(0.37 * t + 1)]) * still
    sh = bump(12.3, 12.9, 15.0, 15.8, t)
    cam = cam + sh * np.array([3 * math.sin(t * 21), 4 * math.sin(t * 17 + 2)])
    return float(cam[0]), float(cam[1]), float(z)


def w2s(X, Y, p, cam):
    cx, cy, z = cam
    zp = 1 + (z - 1) * p
    return CX + zp * (X - cx * p), CY + zp * (Y - cy * p), zp


def moon_state(t):
    c = kf(t, [(0, (770, 400)), (8, (760, 410)), (12, (740, 420)), (14, (650, 560), 'i'),
               (16.6, (590, 750), 'o'), (20, (590, 760), 'o')])
    r = kf(t, [(0, 120), (8, 126), (12, 140), (14, 300, 'i'), (16.6, 470, 'o'), (20, 485, 'o')])
    return float(c[0]), float(c[1]), float(r)


# layer coordinates designed in "screen at t=0" space
def L0(sx, sy, p):
    return sx - CX - 20 * p, sy - CY - 180 * p


# ------------------------------------------------------------------ assets
A = {}


def roof_poly(cx, yb, hw, h, up):
    """Chinese upturned eave roof polygon."""
    pts = []
    # left tip up -> eave underside -> right tip
    pts.append((cx - hw - up * 0.6, yb - up))
    for u in np.linspace(0, 1, 9):
        x = cx - hw - up * 0.6 + (hw * 0.35 + up * 0.6) * u
        y = yb - up + up * (1 - (1 - u) ** 2)
        pts.append((x, y))
    pts.append((cx + hw * 0.65, yb))
    for u in np.linspace(0, 1, 9):
        x = cx + hw * 0.65 + (hw * 0.35 + up * 0.6) * u
        y = yb - up * u * u
        pts.append((x, y))
    pts.append((cx + hw + up * 0.6, yb - up - 6))
    # concave slope up to ridge
    for u in np.linspace(0, 1, 10):
        x = cx + hw + up * 0.6 - (hw * 0.62 + up * 0.6) * u
        y = yb - up - (h - up) * (u ** 1.8)
        pts.append((x, y))
    for u in np.linspace(0, 1, 10):
        x = cx - hw * 0.38 - (hw * 0.62 + up * 0.6) * u
        y = yb - h + (h - up) * (u ** 0.55)
        pts.append((x, y))
    pts.append((cx - hw - up * 0.6, yb - up - 6))
    return pts


def karst(xs, base, peaks, rng, rough=6):
    y = np.full_like(xs, base, dtype=float)
    for px, ph, pw in peaks:
        u = np.clip(1 - ((xs - px) / pw) ** 2, 0, 1)
        y = np.minimum(y, base - ph * u ** 0.55)
    y += ndimage.gaussian_filter1d(rng.standard_normal(len(xs)), 3) * rough
    return y


def mountain_sprite(seed, x0, x1, base, bottom, peaks, top, bot, inner, rimc, blur=0, pines=0, pagoda=None, sh=(4, 6, 8, 0.45)):
    rng = np.random.default_rng(seed)
    xs = np.linspace(x0, x1, 400)
    ys = karst(xs, base, peaks, rng)
    mc = MC(x0, ys.min() - 60, x1, bottom)
    pts = list(zip(xs, ys)) + [(x1, bottom), (x0, bottom)]
    mc.poly(pts)
    for i in range(pines):
        k = rng.integers(10, 390)
        px, py = xs[k], ys[k] + 3
        hgt = rng.uniform(16, 30)
        for j in range(4):
            yy = py - j * hgt * 0.22
            ww = hgt * (0.42 - j * 0.08)
            mc.poly([(px - ww, yy), (px + ww, yy), (px, yy - hgt * 0.4)])
        mc.rect(px - 1.2, py - 4, px + 1.2, py + 6)
    if pagoda:
        px, py, s = pagoda
        for j in range(5):
            yy = py - j * 14 * s
            ww = (16 - j * 2.2) * s
            mc.poly(roof_poly(px, yy - 4 * s, ww, 9 * s, 4 * s))
            mc.rect(px - ww * 0.55, yy - 4 * s, px + ww * 0.55, yy + 6 * s)
        mc.line([(px, py - 70 * s), (px, py - 88 * s)], 1.5 * s)
    return paper(mc, top, bot, inner=inner, rim=(rimc, 0.45), sh=sh, blur=blur)


def cloud_mask(seed, wid, carve=True):
    r = np.random.default_rng(seed)
    lobes = int(r.integers(4, 7))
    hgt = wid * 0.30
    mc = MC(-wid * 0.68, -hgt * 1.45, wid * 0.68, hgt * 0.35)
    xs = np.linspace(-wid * 0.42, wid * 0.42, lobes)
    mc.ell(-wid * 0.40, -hgt * 0.16, hgt * 0.24)
    mc.ell(wid * 0.40, -hgt * 0.16, hgt * 0.24)
    mc.rect(-wid * 0.40, -hgt * 0.40, wid * 0.40, hgt * 0.08)
    circles = []
    for i, x in enumerate(xs):
        bmp = math.exp(-(x / (wid * 0.32)) ** 2)
        rr = wid / lobes * (0.42 + 0.38 * bmp) * r.uniform(0.85, 1.12)
        cy = -hgt * 0.2 - rr * 0.55 * bmp - rr * 0.3
        cx = x + r.uniform(-0.1, 0.1) * rr
        mc.ell(cx, cy, rr)
        circles.append((cx, cy, rr))
    # tail curl
    sgn = 1 if r.random() < 0.5 else -1
    tx = sgn * wid * 0.44
    tail = []
    for u in np.linspace(0, 1, 24):
        ang = -math.pi / 2 + sgn * u * 3.6
        rad = hgt * (0.42 - 0.3 * u)
        tail.append((tx + sgn * wid * 0.12 * u + math.cos(ang) * rad * 0.6 * sgn, -hgt * 0.05 + math.sin(ang) * rad * 0.5 + hgt * 0.1))
    mc.line(tail, hgt * 0.09)
    if carve:
        for cx, cy, rr in circles:
            if r.random() < 0.75:
                sp = []
                turns = r.uniform(1.2, 1.8)
                d = 1 if r.random() < 0.5 else -1
                for u in np.linspace(0, 1, 60):
                    ang = d * u * turns * 2 * math.pi + r.uniform(0, 0.01)
                    rad = rr * (0.12 + 0.55 * u)
                    sp.append((cx + math.cos(ang) * rad, cy + math.sin(ang) * rad))
                mc.line(sp, max(2.0, rr * 0.055), fill=0)
        wav = [(x, -hgt * 0.02 + math.sin(x / wid * 18) * hgt * 0.03) for x in np.linspace(-wid * 0.38, wid * 0.38, 40)]
        mc.line(wav, max(2.0, hgt * 0.035), fill=0)
    return mc


CLOUD_TINTS = [
    ('#202a56', '#323f78', '#9fb4e6'),
    ('#30336a', '#4c4f8c', '#c4c4f2'),
    ('#4f4280', '#7c6aa6', '#f0d0b0'),
    ('#7a5c92', '#d8b183', '#ffe6b0'),
]


def cloud_sprite(shape, level, blur):
    key = ('cloud', shape, level, blur)
    if key not in A:
        mc = A['cloudmask'][shape]
        o, i, rc = CLOUD_TINTS[level]
        A[key] = paper(mc, o, None, inner=(mc.h1 * 0.05 + 3, i, None), rim=(rc, 0.55), botdark=0.25,
                       sh=(5, 9, 9, 0.5), blur=blur)
    return A[key]


def mist_sprite(seed, w, h, col, alpha):
    r = np.random.default_rng(seed)
    n = ndimage.gaussian_filter(r.standard_normal((h, w)).astype(np.float32), (h * 0.12, w * 0.02))
    n = (n - n.min()) / (np.ptp(n) + 1e-6)
    yy = (np.arange(h) - h / 2) / (h / 2)
    prof = np.exp(-yy ** 2 * 3.5)[:, None]
    xx = np.linspace(-1, 1, w)
    edge = np.clip((1 - np.abs(xx)) * 5, 0, 1)[None, :]
    a = np.clip(prof * (0.35 + 0.9 * n) * edge, 0, 1) * alpha
    rgb = np.broadcast_to(hf(col), (h, w, 3))
    arr = np.dstack([rgb, a[..., None]])
    return Spr(Image.fromarray((arr * 255).astype(np.uint8), 'RGBA'), w / 2, h / 2)


def glow_kernel(n=256):
    yy, xx = np.mgrid[0:n, 0:n]
    d = np.hypot(xx - n / 2 + 0.5, yy - n / 2 + 0.5) / (n / 2)
    k = np.exp(-d ** 2 * 4.5) * 0.75 + 0.25 / (1 + (d * 6) ** 2)
    k *= np.clip((1 - d) * 4, 0, 1)
    return k.astype(np.float32)


def build_moon(R=600):
    pad = 8
    mc = MC(-R - pad, -R - pad, R + pad, R + pad)
    mc.ell(0, 0, R)
    a = mc.alpha()
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx -= w / 2
    yy -= h / 2
    d = np.hypot(xx, yy) / R
    core, edge = hf('#fbeac4'), hf('#e8b464')
    t = np.clip(d, 0, 1) ** 1.8
    rgb = core[None, None] * (1 - t[..., None]) + edge[None, None] * t[..., None]
    rng = np.random.default_rng(5)
    n = ndimage.gaussian_filter(rng.standard_normal((h // 4, w // 4)), 9)
    n = np.kron(n, np.ones((4, 4)))[:h, :w]
    n = (n - n.mean()) / n.std()
    mare = np.clip(n - 0.4, 0, 1.5) * 0.07
    rgb *= (1 - mare[..., None] * np.array([0.6, 0.9, 1.4])[None, None])
    # concentric paper discs
    for kr, sh_ in ((0.93, 0.035), (0.78, 0.03), (0.6, 0.02)):
        ring = np.clip((kr * R - np.hypot(xx + 0.02 * R, yy + 0.03 * R)) / 2, 0, 1)
        edge_l = np.clip(1 - np.abs(kr * R - np.hypot(xx + 0.02 * R, yy + 0.03 * R)) / 5, 0, 1)
        rgb = rgb * (1 + sh_ * ring[..., None]) * (1 - 0.06 * edge_l[..., None])
    # osmanthus tree inside the moon (faint cut paper)
    tm = MC(-R - pad, -R - pad, R + pad, R + pad)
    tx, ty = 0.18 * R, 0.05 * R
    tm.poly([(tx - 10, ty + 0.42 * R), (tx - 4, ty), (tx - 18, ty - 0.12 * R), (tx + 4, ty - 0.1 * R),
             (tx + 8, ty), (tx + 14, ty + 0.42 * R)])
    for ang, rr, dd in ((0, 0.16, 0.0), (0.9, 0.12, 0.17), (2.3, 0.12, 0.17), (3.3, 0.1, 0.2),
                        (4.4, 0.11, 0.18), (5.5, 0.12, 0.17), (1.6, 0.11, 0.22)):
        tm.ell(tx + math.cos(ang) * dd * R, ty - 0.2 * R + math.sin(ang) * dd * R * 0.8, rr * R)
    ta = tm.alpha()
    rgb *= (1 - 0.085 * ta[..., None] * np.array([0.7, 0.9, 1.3])[None, None])
    rgb *= tex(h, w, 300, 200)[..., None] ** 0.6
    edge_dark = np.clip((d - 0.9) / 0.1, 0, 1) ** 2
    rgb *= (1 - 0.12 * edge_dark[..., None])
    arr = np.dstack([np.clip(rgb, 0, 1), a[..., None]])
    return Spr(Image.fromarray((arr * 255 + .5).astype(np.uint8), 'RGBA'), w / 2, h / 2)


def build_palace():
    mc = MC(-480, -380, 480, 30)
    win = MC(-480, -380, 480, 30)
    mc.poly([(-330, 0), (330, 0), (300, -32), (-300, -32)])
    mc.poly([(-60, 0), (60, 0), (40, -32), (-40, -32)])
    # main hall
    mc.rect(-190, -120, 190, -30)
    mc.poly(roof_poly(0, -112, 250, 70, 26))
    mc.rect(-120, -190, 120, -150)
    mc.poly(roof_poly(0, -185, 165, 55, 20))
    mc.rect(-60, -245, 60, -225)
    mc.poly(roof_poly(0, -240, 95, 48, 16))
    mc.line([(0, -285), (0, -330)], 5)
    mc.ell(0, -336, 9)
    for sx in (-1, 1):
        x = sx * 345
        mc.rect(x - 38, -95, x + 38, -30)
        mc.poly(roof_poly(x, -92, 62, 42, 14))
        mc.rect(x - 22, -140, x + 22, -125)
        mc.poly(roof_poly(x, -136, 38, 34, 10))
        mc.line([(x, -165), (x, -190)], 3)
        mc.line([(sx * 190, -60), (x - sx * 38, -60)], 6)
    # windows: cut holes (moon light shows through) + warm lattice glow
    for i in range(7):
        x = -150 + i * 50
        mc.rect(x - 14, -102, x + 14, -48, fill=0)
        win.rect(x - 14, -102, x + 14, -48)
    for i in range(4):
        x = -90 + i * 60
        mc.rect(x - 12, -183, x + 12, -157, fill=0)
        win.rect(x - 12, -183, x + 12, -157)
    for sx in (-1, 1):
        mc.rect(sx * 345 - 14, -82, sx * 345 + 14, -42, fill=0)
        win.rect(sx * 345 - 14, -82, sx * 345 + 14, -42)
    A['palace'] = paper(mc, '#2c2358', '#17122f', rim=('#ffd88a', 0.9), botdark=0.3, sh=(0, 0, 1, 0))
    wa = win.alpha()
    h, w = wa.shape
    g = ndimage.gaussian_filter(wa, 1.2)
    lat = np.zeros_like(wa)
    lat[:, ::7] = 1
    lat[::7, :] = 1
    wa2 = np.clip(wa - lat * 0.8, 0, 1)
    rgb = np.broadcast_to(hf('#ffd27a'), (h, w, 3))
    img, M = finish_rgba(rgb, np.maximum(wa2 * 0.95, g * 0.0), sh=(0, 0, 1, 0))
    A['palace_win'] = Spr(img, -win.x0 + M, -win.y0 + M)
    # cloud base under the palace
    A['palace_cloud'] = paper(cloud_mask(91, 900), CLOUD_TINTS[3][0], None,
                              inner=(14, CLOUD_TINTS[3][1], None), rim=('#fff0c8', 0.6), sh=(5, 9, 9, 0.45))


def build_scene():
    # ---------------- very far / far / mid mountains
    p = 0.1
    A['mtn0'] = (mountain_sprite(1, -500, 1600, 1110, 1700, [(150, 120, 140), (420, 90, 120), (900, 140, 160), (1250, 100, 130)],
                                 '#43608e', '#7896bc', (6, '#4f6c9a', '#86a2c6'), '#c9d8f0', blur=1.6), p)
    p = 0.2
    A['mtn1'] = (mountain_sprite(2, -500, 1600, 1180, 1900,
                                 [(-80, 240, 120), (90, 300, 95), (260, 200, 110), (640, 160, 140), (960, 330, 100),
                                  (1120, 250, 110), (1380, 220, 130)],
                                 '#2c4573', '#5d7ea8', (7, '#38558a', '#6c8db6'), '#b9cdee', blur=1.0), p)
    p = 0.4
    A['mtn2'] = (mountain_sprite(3, -500, 1600, 1330, 2300,
                                 [(-40, 170, 160), (300, 120, 200), (700, 200, 150), (980, 150, 130), (1300, 190, 160)],
                                 '#173050', '#2c4f6c', (8, '#1f3e60', '#335a78'), '#9fc2dc', pines=46,
                                 pagoda=(700, 1132, 1.0)), p)
    A['mist'] = [mist_sprite(10 + i, 2600, 260, c, a) for i, (c, a) in
                 enumerate((('#a9bcdc', 0.55), ('#8ea6cc', 0.5), ('#c2d0e8', 0.4)))]
    # ---------------- lake (p=0.5)
    p = 0.5
    mc = MC(-500, 1330, 1600, 2400)
    mc.rect(-500, 1330, 1600, 2400)
    sc = MC(-500, 1330, 1600, 2400)
    for row in range(40):
        y = 1345 + row * (14 + row * 0.9)
        if y > 2390:
            break
        sw = 26 + row * 3
        off = (row % 2) * sw / 2
        x = -500 - off
        while x < 1600:
            arc = [(x + u * sw, y - math.sin(u * math.pi) * sw * 0.18) for u in np.linspace(0, 1, 10)]
            sc.line(arc, 1.4 + row * 0.08)
            x += sw
    A['lake'] = (paper(mc, '#1d2c55', '#0e1533', decor=[(sc, '#34497a', 0.55)], rim=None, botdark=0, sh=(0, 0, 1, 0),
                       grad=(0, 400)), p)

    # ---------------- cliff + terrace (p=1, world coords)
    mc = MC(-820, 440, 330, 1250)
    mc.poly([(-820, 480), (190, 480), (215, 505), (205, 545), (240, 605), (228, 690), (262, 770), (250, 870),
             (292, 990), (280, 1250), (-820, 1250)])
    cut = MC(-820, 440, 330, 1250)
    rng = np.random.default_rng(4)
    for i in range(9):
        y0 = 540 + i * 75
        pts = [(x, y0 + 14 * math.sin(x / 90 + i) + rng.uniform(-3, 3)) for x in np.linspace(-820, 240, 30)]
        cut.line(pts, 2.5)
    A['cliff'] = paper(mc, '#262f5a', '#0d1130', inner=(10, '#2e3a6a', '#141a3e'), rim=('#c9d4ff', 0.3),
                       decor=[(cut, '#0f1434', 0.7)])
    mc = MC(-820, 460, 220, 500)
    mc.poly([(-820, 470), (195, 470), (212, 494), (-820, 494)])
    A['ledge'] = paper(mc, '#575a8a', '#3a3c6a', rim=('#fff1d0', 0.5), sh=(3, 5, 4, 0.5))

    # tree rock + osmanthus (p=0.92)
    mc = MC(280, 480, 720, 1250)
    mc.poly([(320, 570), (380, 528), (450, 516), (530, 530), (610, 562), (700, 600), (720, 1250), (290, 1250),
             (318, 900), (298, 700)])
    A['tree_rock'] = paper(mc, '#2a3460', '#11163a', inner=(9, '#334075', '#171d45'), rim=('#c9d4ff', 0.3))
    A['tree'] = build_tree()

    # pavilion
    mc = MC(-760, 80, -120, 480)
    win = MC(-760, 80, -120, 480)
    mc.rect(-640, 448, -170, 472)                        # plinth
    for x in (-560, -250):
        mc.rect(x - 11, 252, x + 11, 450)                # pillars
    mc.poly(roof_poly(-405, 252, 215, 112, 44))           # roof
    mc.rect(-570, 252, -240, 268)                        # beam
    mc.line([(-560, 150), (-250, 150)], 7)               # ridge
    for sx in (-1, 1):
        x = -405 + sx * 165
        mc.line([(x, 150), (x + sx * 14, 128), (x + sx * 4, 118)], 6)
    # lattice window
    lat = MC(-760, 80, -120, 480)
    lat.rect(-545, 275, -265, 446)
    hole = MC(-760, 80, -120, 480)
    hole.rect(-545, 275, -265, 446)
    for x in np.arange(-545, -265, 20):
        hole.rect(x - 2, 275, x + 2, 446, fill=0)
    for y in np.arange(275, 446, 20):
        hole.rect(-545, y - 2, -265, y + 2, fill=0)
    hole.ell(-405, 360, 66, fill=0)
    hole.ell(-405, 360, 56)
    for k in range(-3, 4):
        hole.line([(-405 + k * 18, 304), (-405 + k * 18, 416)], 3, fill=0)
    A['pav_glow'] = paper(hole, '#ffcc78', '#f29a48', rim=None, botdark=0, sh=(0, 0, 1, 0))
    win.rect(-545, 275, -265, 446)
    A['pav_back'] = paper(win, '#3a1e2a', '#2a1422', rim=None, botdark=0, sh=(0, 0, 1, 0))
    tiles = MC(-760, 80, -120, 480)
    for x in np.linspace(-590, -220, 26):
        tiles.line([(x, 158), (x + (x + 405) * 0.18, 240)], 2.2)
    A['pavilion'] = paper(mc, '#211c44', '#120f2a', rim=('#f6e0c0', 0.45), decor=[(tiles, '#0b0920', 0.55)])
    # brackets / pillar colour layer
    mc = MC(-760, 80, -120, 480)
    for x in (-560, -250):
        mc.rect(x - 8, 268, x + 8, 448)
    mc.rect(-566, 255, -244, 265)
    A['pillars'] = paper(mc, '#7e2a34', '#4a1622', rim=('#ffd3a0', 0.4), sh=(3, 4, 3, 0.4))
    # railing
    mc = MC(-840, 380, -60, 490)
    for x in np.arange(-820, -79, 80):
        mc.rect(x - 7, 396, x + 7, 482)
        mc.ell(x, 393, 9)
    mc.rect(-830, 404, -86, 414)
    mc.rect(-830, 446, -86, 454)
    for x in np.arange(-820, -90, 16):
        mc.line([(x, 414), (x, 446)], 3.2)
    A['railing'] = paper(mc, '#3a3268', '#211c46', rim=('#ffe6c4', 0.55), sh=(4, 7, 5, 0.6))
    # lantern
    mc = MC(-40, -10, 40, 120)
    gold = MC(-40, -10, 40, 120)
    mc.line([(0, 0), (0, 32)], 2)
    mc.ell(0, 66, 27, 33)
    mc.rect(-14, 30, 14, 38)
    mc.rect(-14, 96, 14, 104)
    for k in range(-2, 3):
        mc.line([(k * 3, 104), (k * 4, 118)], 1.6)
    gold.rect(-15, 30, 15, 38)
    gold.rect(-15, 96, 15, 104)
    for k in (-1, 0, 1):
        gold.line([(k * 13, 36), (k * 18, 66), (k * 13, 98)], 1.4)
    A['lantern'] = paper(mc, '#e2453a', '#a8221f', decor=[(gold, '#f2c46a', 1.0)], rim=('#fff0b0', 0.4), sh=(3, 5, 4, 0.4))

    # ---------------- foreground (p=1.5) designed in t0 screen coords
    p = 1.5
    ox, oy = L0(0, 0, p)
    rng = np.random.default_rng(8)
    mc = MC(-200, 1700, 1300, 2700)
    xs = np.linspace(-200, 1300, 120)
    top = 1880 - 120 * np.exp(-((xs + 60) / 260) ** 2) - 60 * np.exp(-((xs - 1150) / 200) ** 2) \
        + ndimage.gaussian_filter1d(rng.standard_normal(120), 2) * 12
    mc.poly(list(zip(xs, top)) + [(1300, 2700), (-200, 2700)])
    for i in range(70):
        k = int(rng.integers(0, 119))
        bx, by = xs[k], top[k] + 6
        hgt = rng.uniform(30, 90)
        lean = rng.uniform(-0.5, 0.5)
        mc.poly([(bx - 4, by), (bx + 4, by), (bx + lean * hgt, by - hgt)])
    A['fg_rocks'] = (paper(mc, '#100d26', '#06050f', rim=('#8090c8', 0.25), sh=(0, 0, 1, 0), blur=5.0), p, ox, oy)
    # pine branch top-left
    mc = MC(-200, -100, 760, 520)
    br = [(-200, 150), (40, 190), (220, 230), (380, 250), (520, 280), (620, 300)]
    for (x0, y0), (x1, y1), wdt in zip(br, br[1:], (26, 22, 17, 13, 9)):
        mc.line([(x0, y0), (x1, y1)], wdt)
    mc.line([(180, 225), (260, 140), (330, 110)], 10)
    mc.line([(380, 250), (430, 340), (470, 380)], 8)
    fans = [(60, 160, 95), (200, 200, 85), (310, 105, 70), (420, 240, 80), (560, 270, 70), (470, 375, 60), (-40, 140, 90),
            (650, 300, 50)]
    for fx, fy, fr in fans:
        for a in np.linspace(math.pi * 1.05, math.pi * 1.95, 14):
            mc.line([(fx, fy), (fx + math.cos(a) * fr, fy + math.sin(a) * fr * 0.8)], 3.2)
        for a in np.linspace(math.pi * 0.08, math.pi * 0.92, 9):
            mc.line([(fx, fy), (fx + math.cos(a) * fr * 0.7, fy + math.sin(a) * fr * 0.45)], 3.0)
    A['fg_pine'] = (paper(mc, '#0f0c24', '#0a0818', rim=('#8f9ee0', 0.35), sh=(0, 0, 1, 0), blur=3.0), p, ox, oy)

    # moon wisps (thin clouds crossing the moon early)
    A['wisp'] = []
    for i in range(2):
        mc = MC(-420, -40, 420, 40)
        pts_t = [(x, -14 * math.exp(-(x / 260) ** 2) - 6 * math.sin(x / 50 + i)) for x in np.linspace(-400, 400, 60)]
        pts_b = [(x, 10 + 4 * math.sin(x / 70 + i * 2)) for x in np.linspace(400, -400, 60)]
        mc.poly(pts_t + pts_b)
        A['wisp'].append(paper(mc, '#3a3f78', '#2a2f62', rim=('#e8e0ff', 0.6), sh=(3, 5, 5, 0.35)))


def build_tree():
    mc = MC(170, -60, 700, 560)
    fl = MC(170, -60, 700, 560)
    trunk_l = [(425, 545), (416, 460), (430, 380), (405, 300), (392, 240)]
    trunk_r = [(420, 236), (446, 300), (454, 380), (466, 460), (470, 545)]
    mc.poly(trunk_l + trunk_r)
    mc.line([(424, 330), (370, 270), (318, 236)], 12)
    mc.line([(446, 300), (500, 240), (556, 205)], 11)
    rng = np.random.default_rng(12)
    blobs = [(430, 150, 112), (335, 190, 82), (525, 185, 86), (375, 95, 82), (482, 85, 80), (290, 130, 62),
             (575, 125, 60), (430, 228, 70), (350, 250, 52), (515, 250, 55), (430, 40, 60)]
    for x, y, r in blobs:
        mc.ell(x, y, r)
    for i in range(70):
        x, y, r = blobs[int(rng.integers(0, len(blobs)))]
        a = rng.uniform(0, 2 * math.pi)
        d = rng.uniform(0.2, 0.92) * r
        cx, cy = x + math.cos(a) * d, y + math.sin(a) * d
        for k in range(4):
            aa = k * math.pi / 2 + a
            fl.ell(cx + math.cos(aa) * 3.2, cy + math.sin(aa) * 3.2, 2.6)
    leaf_cut = MC(170, -60, 700, 560)
    for x, y, r in blobs:
        for a in np.linspace(0.3, 2.8, 4):
            leaf_cut.line([(x + math.cos(a + 3.2) * r * 0.2, y + math.sin(a + 3.2) * r * 0.2),
                           (x + math.cos(a + 3.2) * r * 0.75, y + math.sin(a + 3.2) * r * 0.75)], 2)
    spr = paper(mc, '#1d4a4b', '#112c33', inner=(10, '#2b6560', '#1b4547'), rim=('#d8f0d0', 0.35),
                decor=[(leaf_cut, '#0f2a2e', 0.6), (fl, '#f0c362', 1.0)])
    return spr


def build_assets():
    if A:
        return
    tex(4, 4)
    build_scene()
    build_palace()
    A['moon'] = build_moon()
    A['glowk'] = glow_kernel()
    A['cloudmask'] = [cloud_mask(100 + i, 600) for i in range(8)]
    A['wispcloud'] = cloud_mask(200, 900, carve=True)
    plan_clouds()
    plan_particles()


# ------------------------------------------------------------------ cloud plan
def plan_clouds():
    rng = np.random.default_rng(21)
    clouds = []
    cam_end = camera(20.0)
    # (p, density per 1000 units, blur, scale)
    for p, dens, blur, sc in ((0.45, 3.2, 1.5, 0.75), (0.7, 3.0, 0.0, 0.95), (0.9, 2.6, 0.0, 1.1),
                              (1.25, 1.3, 3.0, 1.35), (1.7, 0.8, 7.0, 1.8)):
        ylo, yhi = -4000 * p - 900, -1350
        n = int((yhi - ylo) / 1000 * dens)
        for i in range(n):
            Y = rng.uniform(ylo, yhi)
            X = rng.uniform(-750, 750) + 30
            sx, sy, _ = w2s(X, Y, p, cam_end)
            if p > 1.0 and -400 < sy < 2300:
                continue
            if 150 < sx < 1250 and 600 < sy < 1450:
                continue
            if p > 1.0:
                # keep heavy foreground clouds off-centre most of the time
                X = X + math.copysign(220, X)
            clouds.append([p, X, Y, int(rng.integers(0, 8)), blur, sc * rng.uniform(0.75, 1.25),
                           1 if rng.random() < 0.5 else -1, rng.uniform(5, 18), 10.4])
    # scripted occlusion passes
    for tt, p, xs, scl in ((13.15, 1.25, -260, 1.9), (14.5, 1.7, 300, 2.2)):
        cx, cy, z = camera(tt)
        clouds.append([p, cx * p + xs, cy * p, 3, 3.0 if p < 1.5 else 7.0, scl, 1, 8, 12.0])
    # cloud sea for the final frame
    for p, sy_list, sc in ((0.7, (1560, 1720), 1.0), (0.9, (1790,), 1.15), (1.25, (2010,), 1.45)):
        cx, cy, z = cam_end
        zp = 1 + (z - 1) * p
        for sy in sy_list:
            for sx in np.linspace(-80, 1160, 4) + rng.uniform(-90, 90):
                X = (sx - CX) / zp + cx * p
                Y = (sy - CY) / zp + cy * p
                clouds.append([p, X, Y, int(rng.integers(0, 8)), 3.0 if p > 1 else 0.0, sc * rng.uniform(0.8, 1.1),
                               1 if rng.random() < .5 else -1, rng.uniform(4, 9), 0.0])
    clouds.sort(key=lambda c: c[0])
    A['clouds'] = clouds


def cloud_level(p, Y):
    alt = c01(-Y / (4000 * p + 1e-6))
    return min(3, int(alt * 4.0))


# ------------------------------------------------------------------ particles
def plan_particles():
    rs = np.random.default_rng(11)
    nb, nt = 110, 520
    tb = np.concatenate([9.0 + rs.uniform(0, 0.12, nb), np.sort(rs.uniform(9.3, 18.9, nt))])
    life = np.concatenate([rs.uniform(0.8, 2.0, nb), rs.uniform(1.0, 2.6, nt)])
    ang = rs.uniform(0, 2 * np.pi, nb + nt)
    spd = np.concatenate([rs.uniform(120, 520, nb), rs.uniform(10, 60, nt)])
    off = np.zeros((nb + nt, 2))
    off[nb:, 0] = rs.normal(-10, 45, nt)
    off[nb:, 1] = rs.uniform(-330, 120, nt)
    size = rs.uniform(1.2, 3.6, nb + nt)
    col = rs.random(nb + nt) < 0.75
    ph = rs.uniform(0, 6.28, nb + nt)
    A['spark'] = dict(nb=nb, tb=tb, life=life, ang=ang, spd=spd, off=off, size=size, gold=col, ph=ph)
    A['spark_origin'] = [her_feet(x) for x in tb]
    A['spark']['v0'] = np.array([her_speed(x) for x in tb])
    npet = 80
    A['petals'] = dict(x=rs.uniform(-250, 900, npet), y=rs.uniform(-260, 760, npet), vy=rs.uniform(22, 50, npet),
                       vx=rs.uniform(-55, -25, npet), ph=rs.uniform(0, 6.28, npet), s=rs.uniform(2.5, 4.5, npet),
                       p=rs.choice([0.92, 1.1, 1.3], npet))
    nst = 520
    A['stars'] = dict(x=rs.uniform(-700, 700, nst), y=rs.uniform(-1500, 400, nst), p=rs.uniform(0.02, 0.12, nst),
                      b=rs.uniform(0.25, 1.0, nst) ** 2, s=rs.uniform(0.7, 2.2, nst), ph=rs.uniform(0, 6.28, nst),
                      f=rs.uniform(0.8, 3.0, nst))
    nw = 45
    A['wind'] = dict(x=rs.uniform(-700, 700, nw), y=rs.uniform(-6500, -800, nw), p=rs.uniform(1.0, 1.6, nw))


# ------------------------------------------------------------------ figures
def rotp(pts, ang, cx, cy):
    c, s = math.cos(ang), math.sin(ang)
    return [(cx + (x - cx) * c - (y - cy) * s, cy + (x - cx) * s + (y - cy) * c) for x, y in pts]


def ribbon(anchor, L, theta, w0, w1, n=40):
    pts = [anchor]
    x, y = anchor
    ds = L / n
    angs = []
    for i in range(n):
        a = theta((i + 0.5) / n)
        x += math.cos(a) * ds
        y += math.sin(a) * ds
        pts.append((x, y))
        angs.append(a)
    angs.append(angs[-1])
    lft, rgt = [], []
    for i, (px, py) in enumerate(pts):
        s = i / n
        w = (w0 + (w1 - w0) * s) / 2 * (1 - 0.6 * max(0, s - 0.92) / 0.08)
        nx, ny = -math.sin(angs[i]), math.cos(angs[i])
        lft.append((px + nx * w, py + ny * w))
        rgt.append((px - nx * w, py - ny * w))
    return lft + rgt[::-1], pts


def limb(p0, p1, w0, w1):
    a = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    nx, ny = -math.sin(a), math.cos(a)
    return [(p0[0] + nx * w0 / 2, p0[1] + ny * w0 / 2), (p1[0] + nx * w1 / 2, p1[1] + ny * w1 / 2),
            (p1[0] - nx * w1 / 2, p1[1] - ny * w1 / 2), (p0[0] - nx * w0 / 2, p0[1] - ny * w0 / 2)]


def polar(p, ang_deg, L):
    a = math.radians(ang_deg)
    return (p[0] + math.cos(a) * L, p[1] + math.sin(a) * L)


class Fig:
    """Collects coloured parts in local units, rasterises at a given scale."""

    def __init__(self):
        self.parts = []

    def add(self, kind, data, col, shadow=True):
        self.parts.append((kind, data, col, shadow))

    def bbox(self):
        xs, ys = [], []
        for kind, data, col, sh in self.parts:
            if kind == 'poly':
                xs += [p[0] for p in data]
                ys += [p[1] for p in data]
            elif kind == 'ell':
                cx, cy, rx, ry = data
                xs += [cx - rx, cx + rx]
                ys += [cy - ry, cy + ry]
            elif kind == 'line':
                pts, w = data
                xs += [p[0] - w for p in pts] + [p[0] + w for p in pts]
                ys += [p[1] - w for p in pts] + [p[1] + w for p in pts]
        return min(xs) - 12, min(ys) - 12, max(xs) + 12, max(ys) + 12

    def render(self, scale, rim_col=(1.0, 0.93, 0.75), rim_k=0.55, rim_dir=(-2, 2), sh=(4, 7, 6, 0.32), texo=(0, 0)):
        x0, y0, x1, y1 = self.bbox()
        x0, y0, x1, y1 = x0 * scale, y0 * scale, x1 * scale, y1 * scale
        mc = MC(x0, y0, x1, y1, mode='RGBA')
        for kind, data, col, shd in self.parts:
            sc = [(tuple(int(v) for v in hx(PAL['shadow'])) + (70,))]
            if kind == 'poly':
                pts = [(x * scale, y * scale) for x, y in data]
                if shd:
                    mc.poly([(x + 1.6 * scale, y + 2.4 * scale) for x, y in pts], sc[0])
                mc.poly(pts, col)
            elif kind == 'ell':
                cx, cy, rx, ry = (v * scale for v in data)
                if shd:
                    mc.ell(cx + 1.6 * scale, cy + 2.4 * scale, rx, ry, sc[0])
                mc.ell(cx, cy, rx, ry, col)
            elif kind == 'line':
                pts, w = data
                mc.line([(x * scale, y * scale) for x, y in pts], w * scale, col)
        im = mc.im.resize((mc.w1, mc.h1), Image.LANCZOS)
        arr = np.asarray(im, np.float32) / 255.0
        a = arr[..., 3]
        rgb = arr[..., :3] / np.maximum(a[..., None], 1e-3) * (a[..., None] > 0)
        h, w = a.shape
        rgb = rgb * tex(h, w, *texo)[..., None]
        if rim_k:
            dx, dy = rim_dir
            r = np.clip(a - shift(a, int(dx), int(dy)), 0, 1)
            r = ndimage.gaussian_filter(r, 0.7)
            rgb = rgb + np.array(rim_col, np.float32)[None, None] * (r * rim_k)[..., None]
        img, M = finish_rgba(rgb, a, sh=sh)
        return Spr(img, -mc.x0 + M, -mc.y0 + M)


def C(name, a=255):
    return hx(PAL[name], a) if name in PAL else hx(name, a)


ARMS = [  # near upper, near fore, far upper, far fore
    (100, -40, 95, -25),   # 0 hold elixir at chest
    (40, -114, 90, -20),   # 1 to the lips
    (-25, -50, 160, 150),  # 2 spread, lifting
    (-62, -78, 150, 125),  # 3 reach for the moon
    (-38, -62, 140, 115),  # 4 arrival
]


_PH = None


def phases(t):
    """integrated animation phases (so rate changes never make motion jump)."""
    global _PH
    if _PH is None:
        dt = 0.004
        ts = np.arange(-0.2, 20.4, dt)
        f = np.array([sstep(9.2, 10.8, x) for x in ts])
        tr = np.array([c01(abs(her_speed(x)) / 1300.0) for x in ts])
        slow = 1 - 0.85 * np.array([sstep(17.8, 20.0, x) for x in ts])
        _PH = (ts, np.cumsum(slow) * dt, np.cumsum((2.2 + 3.5 * f + 4.5 * tr) * slow) * dt,
               np.cumsum((3 + 5 * f + 4 * tr) * slow) * dt)
    ts, w, a, s = _PH
    return float(np.interp(t, ts, w)), float(np.interp(t, ts, a)), float(np.interp(t, ts, s))


def change_pose(t):
    P = [np.array(a, float) for a in ARMS]
    arms = kf(t, [(0, tuple(P[0])), (7.7, tuple(P[0])), (8.7, tuple(P[1]), 'io'), (9.1, tuple(P[1])),
                  (10.3, tuple(P[2]), 'io'), (12.5, tuple(P[3]), 'io'), (16.6, tuple(P[3])),
                  (18.2, tuple(P[4]), 'io'), (20, tuple(P[4]))])
    still = 1 - 0.85 * sstep(18.4, 20, t)
    arms = arms + np.array([3 * math.sin(t * 1.3), 5 * math.sin(t * 1.7 + 1), 4 * math.sin(t * 1.1 + 2),
                            6 * math.sin(t * 1.5 + 3)]) * sstep(9.5, 10.5, t) * still
    head = kf(t, [(0, 8), (5.6, 8), (6.5, -12, 'io'), (7.4, -12), (8.1, 0, 'io'), (9.0, -4), (10, -16, 'io'),
                  (16, -9), (20, -12, 'o')])
    lean = kf(t, [(0, 0), (9.3, 0), (10.5, 6), (13, 12, 'io'), (16.6, 5, 'io'), (20, 3, 'o')])
    f = sstep(9.2, 10.8, t)
    tr = c01(abs(her_speed(t)) / 1300.0)
    walk = bump(3.6, 3.9, 5.6, 6.0, t)
    return dict(arms=arms, head=head, lean=lean, f=f, tr=tr, still=still, walk=walk)


def draw_change(t):
    P = change_pose(t)
    f, tr, still, walk = P['f'], P['tr'], P['still'], P['walk']
    lean = math.radians(P['lean'])
    wt, phA, phS = phases(t)
    WX, WY = 0, -230
    fig = Fig()

    def R(pts):
        return rotp(pts, lean, WX, WY)

    nu, nf, fu, ff = P['arms']
    sh_n, sh_f = (14, -304), (-14, -302)
    el_n = polar(sh_n, nu, 64)
    wr_n = polar(el_n, nf, 58)
    el_f = polar(sh_f, fu, 62)
    wr_f = polar(el_f, ff, 56)
    sh_n, sh_f, el_n, wr_n, el_f, wr_f = R([sh_n, sh_f, el_n, wr_n, el_f, wr_f])

    # ---- ribbon behind (from far elbow)
    def rib_theta(base, amp, om, ph, curl):
        def th(s):
            return math.radians(base + curl * s + amp * (0.25 + s) * math.sin(2 * math.pi * 1.25 * s - om * phA + ph))
        return th
    amp = (9 + 26 * f - 6 * tr) * (0.75 + 0.25 * still) + 6 * walk
    om = 1.0
    baseA = 96 + 22 * f + 6 * tr - 6 * walk
    LA = 430 * (0.62 + 0.38 * f + 0.1 * tr)
    polyA, _ = ribbon(el_f, LA, rib_theta(baseA, amp, om, 0.0, -25 * f), 15, 11)
    fig.add('poly', polyA, C('cinnabar'))
    pa2, _ = ribbon(el_f, LA, rib_theta(baseA, amp, om, 0.0, -25 * f), 5, 3)
    fig.add('poly', pa2, C('coral'), shadow=False)

    # hair strands
    hc = R([(-14, -352)])[0]
    for k, (L, ph) in enumerate(((110, 0.5), (90, 2.0))):
        hp, _ = ribbon(hc, L * (0.8 + 0.3 * f), rib_theta(100 + 25 * f + 10 * tr + k * 6, 8 + 15 * f, om * 0.9, ph, 10), 9, 3)
        fig.add('poly', hp, C('ink'))

    # ---- far arm + sleeve
    def sleeve(el, wr, far):
        a = math.atan2(wr[1] - el[1], wr[0] - el[0])
        nx, ny = -math.sin(a), math.cos(a)
        g = np.array([-0.35 * tr - 0.15 * f, 1.0])
        g /= np.linalg.norm(g)
        Ls = 62 + 30 * f + 18 * math.sin(phS * 0.9 + far)
        fl = 10 * math.sin(wt * 4.0 + far * 2) * (0.3 + f)
        p1 = (wr[0] + nx * 11, wr[1] + ny * 11)
        p2 = (wr[0] - nx * 11, wr[1] - ny * 11)
        lowest = p1 if p1[1] > p2[1] else p2
        tip = (lowest[0] + g[0] * Ls + fl, lowest[1] + g[1] * Ls)
        mid = ((el[0] + wr[0]) / 2 + g[0] * Ls * 0.45, (el[1] + wr[1]) / 2 + g[1] * Ls * 0.45 + 6)
        return [(el[0] + nx * 9, el[1] + ny * 9), p1, p2, tip, mid, (el[0] - nx * 9, el[1] - ny * 9)], (p1, p2, tip)

    fig.add('poly', limb(sh_f, el_f, 17, 15), C('ivory_d'))
    sp, (c1, c2, ctip) = sleeve(el_f, wr_f, 1)
    fig.add('poly', sp, C('ivory_d'))
    fig.add('line', ([c2, ctip], 3.5), C('cin_dark'))
    fig.add('ell', (wr_f[0] + (wr_f[0] - el_f[0]) * 0.14, wr_f[1] + (wr_f[1] - el_f[1]) * 0.14, 6.5, 6.5), C('#e3d2bc'))

    # ---- skirts
    def skirt(extra, phase, wide):
        hem = []
        M = 16
        for i in range(M + 1):
            u = i / M
            x = -78 * wide + 165 * wide * u + 5 * math.sin(wt * 1.7) * (1 - f)
            y = extra
            x += (-55 * f * (1 - u) - 35 * tr * (1 - 0.5 * u)) + 12 * walk * math.sin(wt * 10 + u * 3)
            y += 26 * f + 80 * tr * (0.6 + 0.4 * (1 - u))
            y += (3 + 11 * f + 6 * tr) * math.sin(2 * math.pi * 2 * u - phS + phase) * (0.6 + 0.4 * still)
            hem.append((x, y))
        lft = [(-25 - 15 * u - 40 * (1 - (1 - u) ** 2) * (0.5 + 0.5 * wide) * 0.6 - 30 * f * u ** 2,
                -230 + (hem[0][1] + 230) * u) for u in np.linspace(0, 1, 8)]
        lft = [(x + (hem[0][0] - (-25 - 15 - 24 * (0.5 + 0.5 * wide) - 30 * f)) * u ** 2, y) for (x, y), u in zip(lft, np.linspace(0, 1, 8))]
        rgt = [(23 + 22 * u + 8 * math.sin(u * 3), -230 + (hem[-1][1] + 230) * u) for u in np.linspace(1, 0, 8)]
        rgt = [(x + (hem[-1][0] - 45 - 8 * math.sin(3)) * u ** 2, y) for (x, y), u in zip(rgt, np.linspace(1, 0, 8))]
        return lft + hem + rgt

    fig.add('poly', R(skirt(12, 1.3, 1.08)), C('celadon'))
    # shoe tips
    if f < 0.98:
        fig.add('ell', R([(52 + 6 * math.sin(wt * 10) * walk, -1 + 26 * f)])[0] + (8, 4), C('cinnabar'), shadow=False)
    outer = skirt(0, 0.0, 1.0)
    fig.add('poly', R(outer), C('ivory'))
    # skirt pleat cuts
    for k in range(4):
        u = 0.2 + k * 0.2
        x0 = -10 + 30 * u
        hx_, hy_ = outer[8 + int(u * 16)]
        fig.add('line', (R([(x0, -200), ((x0 + hx_) / 2, (hy_ - 200) / 2), (hx_, hy_ - 10)]), 1.6), C('#cbbd9f'))
    # ---- torso
    torso = [(-22, -312), (16, -314), (27, -288), (22, -232), (-27, -228)]
    fig.add('poly', R(torso), C('ivory'))
    fig.add('poly', R([(-28, -246), (24, -248), (25, -224), (-29, -222)]), C('cinnabar'))
    fig.add('line', (R([(-6, -313), (18, -270)]), 4), C('cinnabar'))
    fig.add('line', (R([(15, -315), (4, -284)]), 3.5), C('cin_dark'))
    # sash ties
    st_anchor = R([(18, -232)])[0]
    for k in range(2):
        sp_, _ = ribbon(st_anchor, (95 + 25 * k) * (0.85 + 0.35 * f),
                        rib_theta(84 + 30 * f + 10 * tr - k * 10, 10 + 14 * f, om, 1.5 + k, 15), 9, 6)
        fig.add('poly', sp_, C('cinnabar') if k == 0 else C('cin_dark'))
    # ---- head (rotates about neck)
    hrot = math.radians(P['head'])
    neck = R([(4, -318)])[0]

    def Hd(pts):
        return rotp(R(pts), hrot, neck[0], neck[1])
    fig.add('poly', R([(-3, -326), (11, -326), (12, -308), (-4, -308)]), C('skin'))
    hc_ = Hd([(2, -348)])[0]
    fig.add('ell', (hc_[0], hc_[1], 23, 23), C('ink'))
    fc = Hd([(9, -341)])[0]
    fig.add('ell', (fc[0], fc[1], 18.5, 21), C('skin'), shadow=False)
    fig.add('poly', Hd([(25, -350), (31, -338), (25, -334)]), C('skin'), shadow=False)
    fig.add('poly', Hd([(-12, -356), (6, -368), (24, -358), (14, -352), (-2, -350)]), C('ink'), shadow=False)
    fig.add('line', (Hd([(15, -346), (22, -345)]), 1.8), C('ink'))
    fig.add('ell', Hd([(17, -335)])[0] + (4.5, 3.5), (236, 140, 140, 150), shadow=False)
    fig.add('ell', Hd([(26.5, -328)])[0] + (2.6, 1.8), C('cinnabar'), shadow=False)
    # buns (flying-immortal loops) with cut holes
    for (bx, by, ro, ri) in ((-6, -382, 15, 6.5), (12, -394, 11, 4.5)):
        c_ = Hd([(bx, by)])[0]
        fig.add('ell', (c_[0], c_[1], ro, ro * 1.15), C('ink'))
        fig.add('ell', (c_[0], c_[1], ri, ri * 1.2), (0, 0, 0, 0), shadow=False)
    fig.add('line', (Hd([(-20, -372), (6, -368)]), 2.4), C('gold'))
    for k in range(5):
        a = k * 2 * math.pi / 5
        c_ = Hd([(-16 + math.cos(a) * 4.5, -373 + math.sin(a) * 4.5)])[0]
        fig.add('ell', (c_[0], c_[1], 3.6, 3.6), C('cinnabar'), shadow=False)
    sway = math.radians(10 * math.sin(wt * 2.3) * (0.4 + f) + 25 * tr)
    top_ = Hd([(-21, -368)])[0]
    for k in range(3):
        L = 14 + k * 9
        e_ = (top_[0] + math.sin(sway - 0.2) * L - 4 * k, top_[1] + math.cos(sway) * L)
        fig.add('line', ([top_, e_], 1.2), C('gold'))
        fig.add('ell', (e_[0], e_[1], 2.4, 2.4), C('gold'), shadow=False)
    # ---- near arm + sleeve
    fig.add('poly', limb(sh_n, el_n, 18, 15), C('ivory'))
    sp, (c1, c2, ctip) = sleeve(el_n, wr_n, 0)
    fig.add('poly', sp, C('ivory'))
    fig.add('line', ([c2, ctip], 3.5), C('cinnabar'))
    hand = (wr_n[0] + (wr_n[0] - el_n[0]) * 0.14, wr_n[1] + (wr_n[1] - el_n[1]) * 0.14)
    fig.add('ell', (hand[0], hand[1], 7, 7), C('skin'))
    # ---- ribbon in front (from near elbow)
    baseB = 84 + 26 * f + 6 * tr + 5 * walk
    LB = 380 * (0.6 + 0.4 * f + 0.1 * tr)
    polyB, _ = ribbon(el_n, LB, rib_theta(baseB, amp * 1.1, om * 1.1, 2.1, -20 * f + 18 * (1 - f)), 14, 10)
    fig.add('poly', polyB, C('cinnabar'))
    pb2, _ = ribbon(el_n, LB, rib_theta(baseB, amp * 1.1, om * 1.1, 2.1, -20 * f + 18 * (1 - f)), 4.5, 3)
    fig.add('poly', pb2, C('coral'), shadow=False)
    mouth = Hd([(28, -330)])[0]
    elixir = ((hand[0] + wr_f[0]) / 2 + 4, (hand[1] + wr_f[1]) / 2 - 8) if t < 7.7 else hand
    if t < 9.05:
        a_ = sstep(7.7, 8.7, t)
        elixir = (elixir[0] * (1 - a_) + mouth[0] * a_, elixir[1] * (1 - a_) + mouth[1] * a_)
    return fig, dict(elixir=elixir, mouth=mouth, hand=hand)


def draw_houyi(t):
    walk = bump(9.4, 9.6, 10.9, 11.2, t)
    ph = (t - 9.4) * 2 * math.pi * 1.5
    reach = sstep(10.9, 12.0, t)
    fig = Fig()
    # bow on back
    bow = [(-30 + math.cos(a) * 26, -255 + math.sin(a) * 120) for a in np.linspace(math.radians(95), math.radians(265), 20)]
    fig.add('line', (bow, 5), C('#8a6a3a'))
    fig.add('line', ([bow[0], bow[-1]], 1.4), C('#d9c7a0'))
    hip = (-2, -190)
    for k, sgn in enumerate((1, -1)):
        a = 90 + sgn * 20 * math.sin(ph) * walk
        knee = polar(hip, a, 100)
        foot = polar(knee, a + 8 * walk * max(0, math.sin(ph + sgn)), 92)
        fig.add('poly', limb(hip, knee, 26, 20), C('#2b2340' if k else '#231c36'))
        fig.add('poly', limb(knee, foot, 20, 16), C('#2b2340' if k else '#231c36'))
        fig.add('poly', [(foot[0] - 10, foot[1] - 8), (foot[0] + 20, foot[1] - 4), (foot[0] + 22, foot[1] + 2),
                         (foot[0] - 10, foot[1] + 2)], C('#15111f'))
    far_sh = (-12, -300)
    a_far = 95 - 25 * math.sin(ph) * walk
    fe = polar(far_sh, a_far * (1 - reach) + 170 * reach, 60)
    fw = polar(fe, (a_far - 10) * (1 - reach) + 140 * reach, 55)
    fig.add('poly', limb(far_sh, fe, 16, 13), C('#2a2140'))
    fig.add('poly', limb(fe, fw, 13, 11), C('#2a2140'))
    hem = [(38 + 6 * math.sin(t * 5), -108), (-42 + 8 * math.sin(t * 4), -104)]
    fig.add('poly', [(-26, -312), (22, -314), (30, -200), hem[0], hem[1], (-34, -200)], C('#3b2f55'))
    fig.add('poly', [(-34, -212), (31, -212), (32, -198), (-35, -198)], C('cin_dark'))
    fig.add('line', ([(-6, -312), (16, -262)], 3), C('#6a5a86'))
    hp = rotp([(4, -338)], math.radians(-14 * reach), 4, -316)[0]
    fig.add('ell', (hp[0] - 4, hp[1] - 3, 23, 23), C('ink'))
    fig.add('ell', (hp[0] + 3, hp[1] + 2, 18, 20), C('#d6b394'), shadow=False)
    fig.add('poly', [(hp[0] + 19, hp[1] - 6), (hp[0] + 25, hp[1] + 5), (hp[0] + 18, hp[1] + 8)], C('#d6b394'), shadow=False)
    fig.add('ell', (hp[0] - 8, hp[1] - 30, 10, 9), C('ink'))
    fig.add('line', ([(hp[0] - 22, hp[1] - 12), (hp[0] + 18, hp[1] - 16)], 4), C('cinnabar'))
    tail, _ = ribbon((hp[0] - 22, hp[1] - 12), 46, lambda s: math.radians(160 + 25 * math.sin(t * 6 - s * 5)), 5, 3)
    fig.add('poly', tail, C('cinnabar'))
    near_sh = (12, -300)
    a_n = 90 + 25 * math.sin(ph) * walk
    ne = polar(near_sh, a_n * (1 - reach) + (-55) * reach, 62)
    nw = polar(ne, (a_n - 15) * (1 - reach) + (-72) * reach, 56)
    fig.add('poly', limb(near_sh, ne, 18, 15), C('#4a3c66'))
    fig.add('poly', limb(ne, nw, 15, 12), C('#4a3c66'))
    fig.add('ell', (nw[0], nw[1], 7, 7), C('#d6b394'))
    x = kf(t, [(9.4, -820), (11.0, -260, 'o')])
    return fig, x


def draw_rabbit(t):
    fig = Fig()
    pound = abs(math.sin(t * 2 * math.pi * 0.9)) * (1 - 0.7 * sstep(18.5, 20, t))
    fig.add('poly', [(16, 0), (44, 0), (40, -20), (20, -20)], C('#b07a4a'))
    fig.add('ell', (0, -16, 17, 15), C('#f4efe6'))
    fig.add('ell', (12, -36, 10, 9), C('#f4efe6'))
    for k, a in enumerate((-100, -82)):
        b = polar((10 + k * 4, -42), a, 22)
        fig.add('poly', limb((10 + k * 4, -42), b, 7, 4), C('#f4efe6'))
    fig.add('ell', (17, -38, 1.6, 1.6), C('cinnabar'), shadow=False)
    top = (30, -48 - 14 * pound)
    fig.add('line', ([(20, -28), top], 3), C('#f4efe6'))
    fig.add('line', ([(30, -22 - 14 * pound), (30, -54 - 14 * pound)], 4), C('#8a6a3a'))
    return fig


# ------------------------------------------------------------------ additive helpers
class Add:
    """half-res additive light buffer."""

    def __init__(self):
        self.b = np.zeros((H // 2, W // 2, 3), np.float32)

    def glow(self, x, y, r, col, k):
        if k <= 0.002 or r < 1:
            return
        x, y, r = x / 2, y / 2, r / 2
        x0, x1 = int(max(0, x - r)), int(min(W // 2, x + r + 1))
        y0, y1 = int(max(0, y - r)), int(min(H // 2, y + r + 1))
        if x1 <= x0 or y1 <= y0:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d = np.hypot(xx - x, yy - y) / r
        g = np.exp(-d * d * 4.5) * 0.75 + 0.25 / (1 + (d * 6) ** 2)
        g *= np.clip((1 - d) * 4, 0, 1)
        self.b[y0:y1, x0:x1] += g[..., None] * (np.array(col, np.float32) * k)[None, None]

    def full(self):
        im = Image.fromarray(np.clip(self.b / 4.0 * 255, 0, 255).astype(np.uint8))
        up = np.asarray(im.resize((W, H), Image.BILINEAR), np.float32) / 255.0 * 4.0
        return up


# ------------------------------------------------------------------ frame
def render_frame(fi):
    build_assets()
    t = fi / FPS
    cam = camera(t)
    cx, cy, z = cam
    alt = c01(-cy / 3900.0)
    mx, my, mr = moon_state(t)
    add = Add()

    # ---------- sky
    hy = CY + (1 + (z - 1) * 0.15) * (190 - cy * 0.15)
    yv = np.arange(H, dtype=np.float32)
    top = hf('#080a22') * (1 - alt) + hf('#120c2e') * alt
    mid = hf('#18204c') * (1 - alt) + hf('#2a1f52') * alt
    hor = hf('#35507c') * (1 - alt) + hf('#5a4478') * alt
    u = np.clip(yv / max(hy, 1), 0, 1)[:, None]
    sky = np.where(u < 0.6, top[None] * (1 - u / 0.6) + mid[None] * (u / 0.6),
                   mid[None] * (1 - (u - 0.6) / 0.4) + hor[None] * ((u - 0.6) / 0.4))
    sky = np.broadcast_to(sky[:, None, :], (H, W, 3)).copy()
    # stars
    st = A['stars']
    sl = Image.new('L', (W, H), 0)
    sd = ImageDraw.Draw(sl)
    starfade = 1 - 0.35 * sstep(14.5, 17, t)
    sx = CX + (1 + (z - 1) * st['p']) * (st['x'] - cx * st['p'])
    syy = CY + (1 + (z - 1) * st['p']) * (st['y'] - cy * st['p'])
    tw = 0.55 + 0.45 * np.sin(st['ph'] + t * st['f'] * 2)
    for x, y, b, s, w_ in zip(sx, syy, st['b'], st['s'], tw):
        if 0 <= x < W and 0 <= y < H:
            v = int(255 * b * w_ * starfade)
            sd.ellipse([x - s, y - s, x + s, y + s], fill=v)
            if b > 0.75 and s > 1.7:
                sd.line([(x - 7 * w_, y), (x + 7 * w_, y)], fill=v // 2)
                sd.line([(x, y - 7 * w_), (x, y + 7 * w_)], fill=v // 2)
    sla = np.asarray(sl.filter(ImageFilter.GaussianBlur(0.5)), np.float32) / 255.0
    sky += sla[..., None] * hf('#fff4dc')[None, None] * 1.1
    # moon halo
    mglow = 0.55 + 0.2 * sstep(12, 16.6, t) + 0.2 * bump(15.8, 16.4, 16.6, 18.2, t)
    skyadd = Add()
    skyadd.glow(mx, my, mr * 4.5, hf('#e8b070'), (0.30 + 0.12 * sstep(14, 17, t)) * mglow)
    skyadd.glow(mx, my, mr * 1.9, hf('#ffe2a8'), 0.30 * mglow)
    sky += skyadd.full()
    base = Image.fromarray((np.clip(sky, 0, 1) * 255).astype(np.uint8)).convert('RGBA')

    # ---------- moon
    blit(base, A['moon'], mx, my, mr / 600.0)
    # wisps across the moon
    for i, spr in enumerate(A['wisp']):
        wx = mx - 260 + (t * (22 + 10 * i)) + i * 300
        wy = my + 30 - i * 70
        blit(base, spr, wx, wy, 0.9 + 0.2 * i, alpha=1 - sstep(10, 12.5, t))
    # palace
    pa = sstep(13.8, 16.4, t)
    if pa > 0:
        ps = mr / 500.0 * 0.78
        px, py = mx + 0.33 * mr, my + 0.74 * mr
        blit(base, A['palace'], px, py, ps, alpha=pa)
        blit(base, A['palace_win'], px, py, ps, alpha=pa * (0.8 + 0.2 * math.sin(t * 3)))
        add.glow(px, py - 80 * ps, 420 * ps, hf('#ffcf80'), 0.18 * pa)
        rb = draw_rabbit(t).render(ps * 1.1, rim_k=0.4)
        blit(base, rb, px - 250 * ps, py + 2 * ps, 1.0, alpha=pa)
        blit(base, A['palace_cloud'], px + 10 * ps, py + 255 * ps, ps * 1.05, alpha=pa)

    # ---------- distant world
    earth = 1 - sstep(0.3, 0.62, alt)
    for key in ('mtn0', 'mtn1'):
        spr, p = A[key]
        ox, oy = L0(0, 0, p)
        sx_, sy_, zp = w2s(ox, oy, p, cam)
        if sy_ + spr.img.size[1] * zp > -100:
            blit(base, spr, sx_, sy_, zp, alpha=earth)
    for i, (p, sy0, spd, al) in enumerate(((0.25, 1180, 14, 0.9), (0.33, 1290, -10, 0.8))):
        ox, oy = L0(540 + spd * t, sy0, p)
        sx_, sy_, zp = w2s(ox, oy, p, cam)
        blit(base, A['mist'][i], sx_, sy_, zp, alpha=al * earth)
    spr, p = A['mtn2']
    ox, oy = L0(0, 0, p)
    sx_, sy_, zp = w2s(ox, oy, p, cam)
    blit(base, spr, sx_, sy_, zp, alpha=earth)
    ox, oy = L0(540 - 18 * t, 1380, 0.42)
    sx_, sy_, zp = w2s(ox, oy, 0.42, cam)
    blit(base, A['mist'][2], sx_, sy_, zp, alpha=0.75 * earth)
    spr, p = A['lake']
    ox, oy = L0(0, 0, p)
    sx_, sy_, zp = w2s(ox, oy, p, cam)
    blit(base, spr, sx_, sy_, zp, alpha=earth)
    lake_top = w2s(*L0(0, 1330, p), p, cam)[1]
    # moon reflection on the lake
    if lake_top < H:
        for i in range(34):
            yy = lake_top + 8 + i * (9 + i * 0.9) * zp
            if yy > H:
                break
            wob = math.sin(i * 1.7 + t * 2.2) * 10 + math.sin(i * 0.6 - t * 1.3) * 6
            wdt = (mr * 0.9) * (1 - i / 40) * (0.6 + 0.4 * math.sin(i * 2.1 + t * 3))
            add.glow(mx + wob, yy, max(6, wdt), hf('#ffe3a0'), 0.10 * (1 - i / 34))

    # ---------- sky clouds behind the heroine
    def draw_clouds(pmin, pmax):
        for p, X, Y, shp, blur, sc, flip, drift, tin in A['clouds']:
            if not (pmin <= p < pmax) or t < tin:
                continue
            sx_, sy_, zp = w2s(X + drift * t * flip, Y, p, cam)
            k = zp * sc
            if sy_ < -400 * k or sy_ > H + 300 * k or sx_ < -500 * k or sx_ > W + 500 * k:
                continue
            spr = cloud_sprite(shp, cloud_level(p, Y), blur)
            blit(base, spr, sx_, sy_, k, alpha=sstep(tin, tin + 0.9, t) if tin > 0 else 1.0)
    draw_clouds(0.0, 0.95)

    # ---------- terrace world (p=1 & 0.92)
    tr_spr = A['tree_rock']
    sx_, sy_, zp = w2s(0, 0, 0.92, cam)
    if sy_ + 480 * zp < H + 50:
        blit(base, tr_spr, sx_, sy_, zp)
        bx, by, _ = w2s(440, 545, 0.92, cam)
        sway = math.radians(1.1 * math.sin(t * 1.2) + 0.5 * math.sin(t * 2.7)) * (1 + 1.5 * bump(9.0, 9.5, 11, 12.5, t))
        tspr = A['tree']
        # tree sprite anchored at local origin; rotate about trunk base
        ax, ay = tspr.ax + 440, tspr.ay + 545
        blit(base, Spr(tspr.img, ax, ay), bx, by, zp, theta=sway)
    sx_, sy_, zp = w2s(0, 0, 1.0, cam)
    world_vis = sy_ + 80 * zp < H + 200
    if world_vis:
        blit(base, A['pav_back'], sx_, sy_, zp)
        flick = 0.9 + 0.1 * math.sin(t * 7.3) * math.sin(t * 3.1 + 1)
        blit(base, A['pav_glow'], sx_, sy_, zp, alpha=flick)
        gx, gy, _ = w2s(-405, 360, 1.0, cam)
        add.glow(gx, gy, 380 * zp, hf('#ffaa55'), 0.35 * flick)
        blit(base, A['pavilion'], sx_, sy_, zp)
        blit(base, A['pillars'], sx_, sy_, zp)
        blit(base, A['cliff'], sx_, sy_, zp)
        blit(base, A['ledge'], sx_, sy_, zp)
        # lantern
        lx, ly, _ = w2s(-182, 236, 1.0, cam)
        la = math.radians(5 * math.sin(t * 1.6) + 2 * math.sin(t * 3.7) + 8 * bump(9.0, 9.4, 10, 12, t) * math.sin(t * 5))
        blit(base, A['lantern'], lx, ly, zp, theta=la)
        lcx, lcy = lx - math.sin(la) * 66 * zp, ly + math.cos(la) * 66 * zp
        add.glow(lcx, lcy, 230 * zp, hf('#ff7a3a'), 0.45 * flick)
        add.glow(lcx, lcy, 60 * zp, hf('#ffd29a'), 0.5 * flick)
        # Hou Yi
        if 9.4 <= t:
            fig, hxp = draw_houyi(t)
            hsx, hsy, _ = w2s(hxp, 482, 1.0, cam)
            if hsy - 420 * zp < H:
                spr = fig.render(zp, rim_col=(1.0, 0.7, 0.45), rim_k=0.5, rim_dir=(2, 2))
                blit(base, spr, hsx, hsy, 1.0)

    # ---------- Chang'e
    fig, info = draw_change(t)
    fx, fy = her_feet(t)
    hs_x, hs_y, zp = w2s(fx, fy, 1.0, cam)
    ch_alpha = sstep(3.3, 4.3, t)
    aura = sstep(9.0, 10.0, t) * (0.6 + 0.4 * bump(12, 13, 16, 17.5, t)) * 0.45
    rimc = (1.0, 0.93, 0.75) if t < 9 else (1.0, 0.95, 0.82)
    spr = fig.render(zp, rim_col=rimc, rim_k=0.5 + 0.3 * aura, rim_dir=(-2, 2))
    if aura > 0.01:
        a_img = spr.img.split()[3].resize((spr.img.size[0] // 4 + 1, spr.img.size[1] // 4 + 1))
        a_img = a_img.filter(ImageFilter.GaussianBlur(10))
        aa = np.asarray(a_img, np.float32) / 255.0
        hb = add.b
        x0 = int((hs_x - spr.ax) / 2)
        y0 = int((hs_y - spr.ay) / 2)
        aa = np.asarray(Image.fromarray((aa * 255).astype(np.uint8)).resize((spr.img.size[0] // 2, spr.img.size[1] // 2)),
                        np.float32) / 255.0
        hh, ww = aa.shape
        xs0, ys0 = max(0, x0), max(0, y0)
        xs1, ys1 = min(W // 2, x0 + ww), min(H // 2, y0 + hh)
        if xs1 > xs0 and ys1 > ys0:
            hb[ys0:ys1, xs0:xs1] += aa[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0, None] * hf('#ffd890')[None, None] * 0.5 * aura
        add.glow(hs_x, hs_y - 220 * zp, 520 * zp, hf('#ffc878'), 0.16 * aura)
    blit(base, spr, hs_x, hs_y, 1.0, alpha=ch_alpha)
    # elixir
    el = info['elixir']
    ex, ey = hs_x + el[0] * zp, hs_y + el[1] * zp
    if t < 9.05:
        pul = 0.85 + 0.15 * math.sin(t * 6)
        ek = sstep(5.4, 6.3, t) * pul
        add.glow(ex, ey, 240 * zp, hf('#8fdcff'), 0.28 * ek)
        add.glow(ex, ey, 55 * zp, hf('#e8fbff'), 0.45 * ek)
        add.glow(ex, ey, 11 * zp, hf('#ffffff'), 1.6 * ek)
    # railing sits in front of the heroine
    if world_vis:
        blit(base, A['railing'], sx_, sy_, zp)

    # ---------- foreground clouds
    draw_clouds(0.95, 9.9)

    # ---------- foreground pine & rocks (p=1.5)
    for key in ('fg_pine', 'fg_rocks'):
        spr, p, ox, oy = A[key]
        sx2, sy2, zp2 = w2s(ox, oy, p, cam)
        th = math.radians(1.2 * math.sin(t * 1.1)) if key == 'fg_pine' else 0
        if key == 'fg_pine':
            # rotate about the branch root (left edge)
            rx, ry = sx2 + (-200 + 0) * zp2, sy2 + 150 * zp2
            blit(base, Spr(spr.img, spr.ax - 200, spr.ay + 150), rx, ry, zp2, theta=th)
        else:
            blit(base, spr, sx2, sy2, zp2)

    # ---------- particles (full-res additive, with motion blur)
    pl = Image.new('L', (W, H), 0)
    pg = Image.new('L', (W, H), 0)
    dl, dg = ImageDraw.Draw(pl), ImageDraw.Draw(pg)
    cam_prev = camera(t - 1 / 120)
    S = A['spark']
    nb = S['nb']
    for i in range(len(S['tb'])):
        age = t - S['tb'][i]
        if age <= 0 or age >= S['life'][i]:
            continue
        ox_, oy_ = A['spark_origin'][i]

        def pos(ag, cm):
            if i < nb:
                k = 2.2
                dd = S['spd'][i] * (1 - math.exp(-k * ag)) / k
                X = ox_ + 35 + math.cos(S['ang'][i]) * dd
                Y = oy_ - 330 + math.sin(S['ang'][i]) * dd + 20 * ag * ag
            else:
                kk = 1.6
                carry = S['v0'][i] * 0.8 * (1 - math.exp(-kk * ag)) / kk
                X = ox_ + S['off'][i, 0] + math.cos(S['ang'][i]) * S['spd'][i] * ag + 14 * math.sin(ag * 3 + S['ph'][i])
                Y = oy_ + S['off'][i, 1] + math.sin(S['ang'][i]) * S['spd'][i] * ag * 0.5 + 18 * ag + carry
            return w2s(X, Y, 1.0, cm)
        x1, y1, zp1 = pos(age, cam)
        x0, y0, _ = pos(age - 1 / 120, cam_prev)
        if not (-50 < x1 < W + 50 and -50 < y1 < H + 50):
            continue
        lf = math.sin(math.pi * age / S['life'][i]) * (0.7 + 0.3 * math.sin(age * 20 + S['ph'][i]))
        v = int(255 * c01(lf))
        s = S['size'][i] * zp1
        d = dg if S['gold'][i] else dl
        if math.hypot(x1 - x0, y1 - y0) > 3:
            d.line([(x0, y0), (x1, y1)], fill=v, width=max(1, int(s * 1.3)))
        d.ellipse([x1 - s, y1 - s, x1 + s, y1 + s], fill=v)
    # osmanthus petals (reflected-light paper flecks)
    Pt = A['petals']
    pf = 1 - sstep(12.5, 13.5, t)
    if pf > 0:
        lift = max(0, t - 9.2)
        for i in range(len(Pt['x'])):
            X = Pt['x'][i] + Pt['vx'][i] * t + 18 * math.sin(t * 1.3 + Pt['ph'][i]) + 40 * lift * math.sin(Pt['ph'][i])
            Y = Pt['y'][i] + Pt['vy'][i] * t - 90 * lift * lift
            Y = (Y + 300) % 1100 - 300 if lift == 0 else Y
            sx3, sy3, zp3 = w2s(X, Y, Pt['p'][i], cam)
            if not (0 <= sx3 < W and 0 <= sy3 < H):
                continue
            s = Pt['s'][i] * zp3
            a = t * 3 + Pt['ph'][i]
            dg.polygon([(sx3 + math.cos(a + k * 1.57) * s * (1.0 if k % 2 == 0 else 0.45),
                         sy3 + math.sin(a + k * 1.57) * s * (1.0 if k % 2 == 0 else 0.45)) for k in range(4)],
                       fill=int(200 * pf))
    # wind streaks during the ascent
    wk = bump(11.8, 12.8, 15.6, 16.6, t)
    if wk > 0:
        Wd = A['wind']
        for i in range(len(Wd['x'])):
            x1, y1, _ = w2s(Wd['x'][i], Wd['y'][i], Wd['p'][i], cam)
            x0, y0, _ = w2s(Wd['x'][i], Wd['y'][i], Wd['p'][i], camera(t - 1 / 70))
            if -20 < x1 < W + 20 and -400 < y1 < H + 400:
                dl.line([(x0, y0), (x1, y1)], fill=int(38 * wk), width=1)
    pla = np.asarray(pl.filter(ImageFilter.GaussianBlur(0.8)), np.float32) / 255.0
    pga = np.asarray(pg.filter(ImageFilter.GaussianBlur(0.8)), np.float32) / 255.0

    # ---------- compose float
    fr = np.asarray(base, np.float32)[..., :3] / 255.0
    fr += add.full()
    fr += pla[..., None] * hf('#f4f6ff')[None, None] * 1.2 + pga[..., None] * hf('#ffcf6a')[None, None] * 1.3
    # shock ring at the moment she swallows the elixir
    if 9.0 <= t < 10.2:
        u = (t - 9.0) / 1.2
        mxp, myp = hs_x + info['mouth'][0] * zp, hs_y + info['mouth'][1] * zp
        rr = 60 + 900 * eout(u)
        ring = Image.new('L', (W // 2, H // 2), 0)
        ImageDraw.Draw(ring).ellipse([(mxp - rr) / 2, (myp - rr) / 2, (mxp + rr) / 2, (myp + rr) / 2], outline=255,
                                     width=max(2, int(18 * (1 - u))))
        ring = np.asarray(ring.filter(ImageFilter.GaussianBlur(3)).resize((W, H), Image.BILINEAR), np.float32) / 255.0
        fr += ring[..., None] * hf('#ffe9b8')[None, None] * 0.9 * (1 - u)
    # bloom
    small = Image.fromarray(np.clip(fr * 200, 0, 255).astype(np.uint8)).resize((W // 4, H // 4), Image.BILINEAR)
    sm = np.asarray(small, np.float32) / 200.0
    br = np.clip(sm - 0.8, 0, None)
    b1 = ndimage.gaussian_filter(br, (6, 6, 0))
    b2 = ndimage.gaussian_filter(br, (22, 22, 0))
    bloom = b1 * 0.55 + b2 * 0.45
    # god rays from the moon (radial blur of bright areas)
    gr = sstep(12.5, 15.5, t) * (0.65 + 0.35 * bump(15.6, 16.3, 16.8, 19, t))
    if gr > 0.01:
        ms = Image.fromarray(np.clip(br * 255, 0, 255).astype(np.uint8))
        acc = np.zeros_like(br)
        cxq, cyq = mx / 4, my / 4
        n = 14
        for k in range(n):
            s = 1 + 0.035 * k
            tr_ = ms.transform(ms.size, Image.AFFINE, (1 / s, 0, cxq - cxq / s, 0, 1 / s, cyq - cyq / s),
                               resample=Image.BILINEAR)
            acc += np.asarray(tr_, np.float32) / 255.0 * (1 - k / n)
        acc /= n * 0.5
        bloom += ndimage.gaussian_filter(acc, (2, 2, 0)) * 0.3 * gr
    bl = Image.fromarray(np.clip(bloom / 2 * 255, 0, 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    fr += np.asarray(bl, np.float32) / 255.0 * 2
    # flash
    fl = math.exp(-max(0, t - 9.0) * 6) * (t >= 9.0) * 0.2 + 0.07 * bump(16.1, 16.35, 16.4, 17.2, t)
    fr += fl * hf('#ffe2a8')[None, None]
    # exposure / grade
    expo = (0.45 + 0.55 * sstep(0, 0.8, t))
    fr *= expo
    warm = 0.06 * sstep(12, 16.6, t)
    fr *= np.array([1 + warm, 1 + warm * 0.3, 1 - warm * 0.8], np.float32)[None, None]
    sh_t = hf('#0b0920')
    fr = fr + sh_t[None, None] * (1 - np.clip(fr, 0, 1)) * 0.35
    fr = np.where(fr < 0.78, fr, 0.78 + 0.22 * (1 - np.exp(-(fr - 0.78) / 0.22)))
    # vignette
    if 'vig' not in A:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = ((xx - CX) / (W * 0.62)) ** 2 + ((yy - CY * 0.95) / (H * 0.62)) ** 2
        A['vig'] = (1 - 0.42 * np.clip(d, 0, 1.6) ** 1.3)[..., None].astype(np.float32)
    fr *= A['vig']
    g = np.random.default_rng(1000 + fi).standard_normal((H // 2, W // 2)).astype(np.float32)
    g = np.kron(g, np.ones((2, 2), np.float32))
    fr += g[..., None] * 0.012
    return (np.clip(fr, 0, 1) * 255 + 0.5).astype(np.uint8)


# ------------------------------------------------------------------ main
def _worker_init():
    build_assets()


def main():
    mode = sys.argv[1]
    if mode == 'test':
        os.makedirs(os.path.join(HERE, 'frames'), exist_ok=True)
        for s in sys.argv[2:]:
            fi = min(NFR - 1, int(round(float(s) * FPS)))
            arr = render_frame(fi)
            Image.fromarray(arr).save(os.path.join(HERE, 'frames', f'test_{fi:04d}.png'))
            print('saved', fi, flush=True)
    elif mode == 'video':
        out = sys.argv[2]
        from multiprocessing import Pool
        cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
               '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', '14', '-profile:v', 'high', '-level', '4.2',
               '-pix_fmt', 'yuv420p', '-r', str(FPS), out]
        ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        nproc = int(os.environ.get('NPROC', '7'))
        with Pool(nproc, initializer=_worker_init) as pool:
            for i, arr in enumerate(pool.imap(render_frame, range(NFR), chunksize=2)):
                assert arr.shape == (H, W, 3)
                ff.stdin.write(arr.tobytes())
                if i % 30 == 0:
                    print('frame', i, flush=True)
        ff.stdin.close()
        ff.wait()
        print('done', ff.returncode)


if __name__ == '__main__':
    main()
