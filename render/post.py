"""Pós-processamento: legendas no estilo da referência + filtro VHS."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FONT_PATH = "C:/Windows/Fonts/arialbd.ttf"
_font_cache = {}


def font(size):
    if size not in _font_cache:
        _font_cache[size] = ImageFont.truetype(FONT_PATH, size)
    return _font_cache[size]


def wrap(draw, text, f, maxw):
    words = text.split(" ")
    lines, cur = [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if draw.textlength(test, font=f) <= maxw or not cur:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


SUB_STYLES = {
    # narrador/Arthur: amarelo da referência
    "A": dict(size=37, fill=(240, 236, 170), stroke=(20, 28, 10), shadow=(6, 10, 4)),
    # grito do Arthur
    "S": dict(size=44, fill=(250, 244, 160), stroke=(30, 20, 6), shadow=(10, 4, 2)),
    # o líder: branco frio com contorno vermelho-sangue
    "L": dict(size=37, fill=(238, 226, 226), stroke=(70, 0, 8), shadow=(28, 0, 4)),
    # detetive: branco azulado
    "D": dict(size=37, fill=(222, 236, 255), stroke=(10, 26, 62), shadow=(2, 6, 18)),
}


def draw_subtitle(img, text, style="A", yc_frac=0.655, shake=(0, 0)):
    W, H = img.size
    st = SUB_STYLES.get(style, SUB_STYLES["A"])
    size = st["size"]
    dr = ImageDraw.Draw(img)
    f = font(size)
    lines = wrap(dr, text, f, W * 0.86)
    lh = int(size * 1.18)
    y = int(H * yc_frac - lh * len(lines) / 2) + shake[1]
    for ln in lines:
        w = dr.textlength(ln, font=f)
        x = (W - w) / 2 + shake[0]
        dr.text((x + 3, y + 3), ln, font=f, fill=st["shadow"], stroke_width=3, stroke_fill=st["shadow"])
        dr.text((x, y), ln, font=f, fill=st["fill"], stroke_width=2, stroke_fill=st["stroke"])
        y += lh
    return img


OSD_FONT = "C:/Windows/Fonts/OCRAEXT.TTF"
_osd_cache = {}


def osd_font(size):
    if size not in _osd_cache:
        _osd_cache[size] = ImageFont.truetype(OSD_FONT, size)
    return _osd_cache[size]


def draw_osd(img, items):
    """texto de tela estilo VCR/camcorder. itens: dict(kind=text|play|stop|rew|rec|brackets|hline, ...)"""
    dr = ImageDraw.Draw(img)
    for it in items:
        k = it.get("kind", "text")
        col = tuple(it.get("col", (235, 235, 235)))
        sh = (10, 10, 14)
        x, y = it.get("x", 26), it.get("y", 118)
        size = it.get("size", 26)
        if k in ("play", "stop", "rew", "rec"):
            s = size
            if k == "play":
                tri = [(x, y + 2), (x, y + s - 2), (x + s * 0.8, y + s / 2)]
                dr.polygon([(a + 2, b + 2) for a, b in tri], fill=sh)
                dr.polygon(tri, fill=col)
                label, lx = it.get("label", "PLAY"), x + s * 1.15
            elif k == "stop":
                sq = [x + 2, y + 3, x + s - 4, y + s - 3]
                dr.rectangle([v + 2 for v in sq], fill=sh)
                dr.rectangle(sq, fill=col)
                label, lx = it.get("label", "STOP"), x + s * 1.1
            elif k == "rew":
                for o in (0, s * 0.62):
                    tri = [(x + o + s * 0.62, y + 2), (x + o + s * 0.62, y + s - 2), (x + o, y + s / 2)]
                    dr.polygon([(a + 2, b + 2) for a, b in tri], fill=sh)
                    dr.polygon(tri, fill=col)
                label, lx = it.get("label", "REW"), x + s * 1.5
            else:
                r = s * 0.32
                dr.ellipse([x, y + s / 2 - r, x + 2 * r, y + s / 2 + r], fill=(230, 30, 30))
                label, lx = "REC", x + s * 0.85
                col = (240, 240, 240)
            f = osd_font(size)
            dr.text((lx + 2, y + 2), label, font=f, fill=sh)
            dr.text((lx, y), label, font=f, fill=col)
        elif k == "text":
            f = osd_font(size)
            if it.get("align") == "right":
                x = x - dr.textlength(it["text"], font=f)
            dr.text((x + 2, y + 2), it["text"], font=f, fill=sh)
            dr.text((x, y), it["text"], font=f, fill=col)
        elif k == "brackets":
            x0, y0, x1, y1 = it["box"]
            L = it.get("len", 22)
            w = it.get("w", 3)
            for (cx, cy, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
                dr.line([(cx, cy), (cx + dx * L, cy)], fill=col, width=w)
                dr.line([(cx, cy), (cx, cy + dy * L)], fill=col, width=w)
        elif k == "hline":
            dr.line([(it.get("x0", 0), y), (it.get("x1", img.size[0]), y)], fill=col, width=it.get("w", 2))
    return img


# ------------------------------------------------------------------ VHS
def _hbox(x, w):
    """média móvel horizontal (eixo 1) com largura w ímpar."""
    p = w // 2
    xp = np.pad(x, ((0, 0), (p + 1, p)) + ((0, 0),) * (x.ndim - 2), mode="edge")
    c = np.cumsum(xp, axis=1)
    return (c[:, w:] - c[:, :-w]) / w


_vig_cache = {}


def _vignette(H, W):
    k = (H, W)
    if k not in _vig_cache:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        ny = (yy / H - 0.5) * 2
        nx = (xx / W - 0.5) * 2
        r = np.sqrt(nx * nx * 0.9 + ny * ny * 0.75)
        v = 1 - 0.5 * np.clip(r - 0.35, 0, None) ** 1.6
        v *= 1 - 0.25 * np.clip(-ny - 0.55, 0, None)  # topo levemente mais escuro como na referência
        _vig_cache[k] = v[..., None].astype(np.float32)
    return _vig_cache[k]


def static_noise(H, W, rng, level=1.0):
    n = rng.random((H, W // 3 + 1), dtype=np.float32)
    n = np.repeat(n, 3, axis=1)[:, :W]
    rowb = rng.random((H, 1), dtype=np.float32) * 0.5 + 0.5
    n = n * rowb
    img = np.stack([n, n * 0.97, n * 1.03], 2) * level
    return np.clip(img, 0, 1)


def vhs(x, t, rng, glitch=0.0, dropouts=True, ripple=0.0):
    """x: float32 HxWx3 em [0,1] (resolução de trabalho). Retorna float32."""
    H, W, _ = x.shape
    # bloom (luzes estouradas)
    lum = x.max(axis=2)
    bm = x * np.clip((lum - 0.72) / 0.28, 0, 1)[..., None]
    sm = Image.fromarray((bm * 255).astype(np.uint8)).resize((W // 4, H // 4), Image.BILINEAR)
    sm = sm.filter(ImageFilter.GaussianBlur(3.5))
    bloom = np.asarray(sm.resize((W, H), Image.BILINEAR), dtype=np.float32) / 255
    x = x + bloom * 0.85

    # YIQ
    R, G, B = x[..., 0], x[..., 1], x[..., 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    I = 0.596 * R - 0.274 * G - 0.322 * B
    Q = 0.211 * R - 0.523 * G + 0.312 * B
    Yb = _hbox(Y, 3)
    Y = Yb + 0.45 * (Yb - _hbox(Y, 9))          # realce de borda/halo típico de VHS
    IQ = np.stack([I, Q], 2)
    IQ = _hbox(IQ, 13)
    IQ = np.roll(IQ, 4, axis=1)                # sangramento de cor para a direita
    IQ = (IQ + np.roll(IQ, 1, axis=0) * 0.5 + np.roll(IQ, -1, axis=0) * 0.5) / 2
    IQ *= 1.35
    cn = rng.standard_normal((H, W // 20 + 1, 2)).astype(np.float32) * (0.018 + 0.05 * glitch)
    IQ += np.repeat(cn, 20, axis=1)[:, :W]
    Y = Y + rng.standard_normal((H, W)).astype(np.float32) * (0.028 + 0.06 * glitch)
    I, Q = IQ[..., 0], IQ[..., 1]
    out = np.stack([Y + 0.956 * I + 0.621 * Q, Y - 0.272 * I - 0.647 * Q, Y - 1.106 * I + 1.703 * Q], 2)

    # jitter horizontal por linha + bandas de tracking
    rows = np.arange(H)
    off = np.round(0.9 * np.sin(rows * 0.021 + t * 2.7) + 0.6 * np.sin(rows * 0.11 + t * 13.0)
                   + ripple * 7.0 * np.sin(rows * 0.035 - t * 11.0)).astype(np.int64)
    if glitch > 0.02:
        for _ in range(1 + int(glitch * 5)):
            y0 = rng.integers(0, H)
            hh = rng.integers(6, int(20 + 90 * glitch))
            sh = int(rng.integers(-40, 40) * glitch)
            off[y0:y0 + hh] += sh
            out[y0:y0 + hh] += rng.uniform(-0.1, 0.25) * glitch
        # barra de tracking rolando
        yb = int((t * 900) % (H + 200)) - 100
        band = np.exp(-((rows - yb) / (18 + 30 * glitch)) ** 2)[:, None] * glitch
        out = out + band[..., None] * rng.random((H, W, 1), dtype=np.float32) * 0.9
        off += np.round(band[:, 0] * rng.integers(-25, 25, H)).astype(np.int64)
    # ruído de troca de cabeça no rodapé
    hs = 12
    off[-hs:] += np.linspace(2, 16, hs).astype(np.int64)
    cols = np.clip(np.arange(W)[None, :] - off[:, None], 0, W - 1)
    out = out[rows[:, None], cols]
    out[-hs:] = out[-hs:] * 0.7 + rng.random((hs, W, 1), dtype=np.float32) * 0.25
    # dropouts (riscos brancos)
    for _ in range(rng.poisson(0.6 + 8 * glitch) if dropouts else 0):
        r = rng.integers(0, H); c0 = rng.integers(0, W); ln = rng.integers(10, 110)
        out[r, c0:c0 + ln] = 0.8 + rng.random() * 0.2
    # vinheta, nível de preto elevado, scanlines
    out = out * _vignette(H, W)
    out = out * 0.90 + np.array([0.028, 0.034, 0.05], np.float32)
    flick = 1.0 + 0.015 * math.sin(t * 50.0)
    out = out * flick
    out[1::2] *= 0.86
    return np.clip(out, 0, 1)


def grade(x, tint=None, fade=1.0, exposure=1.0):
    if tint is not None:
        lum = (x * np.array([0.3, 0.55, 0.15], np.float32)).sum(2, keepdims=True)
        x = x * (1 - tint[3]) + lum * np.array(tint[:3], np.float32) * tint[3]
    x = x * exposure * fade
    return np.clip(x, 0, 1)
