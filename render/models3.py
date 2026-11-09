"""Modelos da parte 3: bicicleta caída, sangue, bar (normal/ensanguentado), escritório do detetive, detetive."""
import math
import numpy as np

from engine import (Builder, FaceSet, rot_x, rot_y, rot_z, normalize, orient, orient_convex, box_polys,
                    frustum_polys, tube_polys, prism_polys, disc_poly, ik2)
from models import C, Head, build_head, build_cyclist

BLOOD = C(0.24, 0.0, 0.02)


# ================================================================ utilidades
def transform_into(dst, src, R, t, skip_bucket=None):
    for P, A in zip(src.polys, src.attrs):
        if skip_bucket is not None and abs(A[5] - skip_bucket) < 1e-6:
            continue
        dst.poly(P @ R.T + t, A[0:3], emis=A[3], dbl=A[4] > 0.5, bucket=A[5], fogmul=A[8], glow=A[9] > 0.5)


def flat_quad(b, c, ax, ay, n, col, **kw):
    """retângulo centrado em c, semi-eixos ax/ay (vetores), virado para n."""
    c = np.asarray(c, float)
    q = np.array([c - ax - ay, c + ax - ay, c + ax + ay, c - ax + ay])
    b.poly(orient(q, n), col, **kw)


def blob(b, center, r, rng, col=BLOOD, n=7, normal=C(0, 1, 0), up=C(0, 0, 1), y=0.014, bucket=2.65, emis=0.08):
    """mancha irregular (várias 'estrelas' de 8 lados sobrepostas) num plano."""
    n_ = normalize(normal)
    u = normalize(np.asarray(up, float) - n_ * np.dot(up, n_))
    s_ = np.cross(u, n_)
    center = np.asarray(center, float) + n_ * y
    for k in range(n):
        c = center + s_ * rng.normal(0, r * 0.42) + u * rng.normal(0, r * 0.42) + n_ * (k * 0.0006)
        rr = r * rng.uniform(0.32, 0.72)
        a0 = rng.uniform(0, 2 * np.pi)
        pts = []
        for i in range(8):
            a = a0 + i * 2 * np.pi / 8
            ri = rr * rng.uniform(0.55, 1.3)
            pts.append(c + s_ * ri * math.cos(a) + u * ri * math.sin(a))
        b.poly(orient(np.array(pts), n_), np.clip(col * rng.uniform(0.8, 1.25), 0, 1), emis=emis, bucket=bucket)


def footprints(b, start, end, step=0.62, rng=None, col=BLOOD, ground=C(0.42, 0.27, 0.21)):
    start = np.asarray(start, float)
    end = np.asarray(end, float)
    d = end - start
    L = np.linalg.norm(d)
    if L < 0.3:
        return
    u = d / L
    v = C(-u[2], 0, u[0])
    n = int(L / step)
    for i in range(n):
        k = i / max(n, 1)
        p = start + u * (i * step) + v * (0.12 * (-1) ** i) + C(0, 0.016, 0)
        col_i = col * (1 - 0.75 * k) + ground * 0.35 * 0.75 * k
        pts = np.array([p + u * 0.14 * math.cos(a) + v * 0.055 * math.sin(a) for a in np.arange(6) * np.pi / 3])
        b.poly(orient(pts, C(0, 1, 0)), col_i, emis=0.2, bucket=2.7)


def build_fallen_bike(b, pos, yaw, spin, lamp=0.0):
    tmp = Builder()
    lights = build_cyclist(tmp, C(0, 0, 0), 0.0, spin, dict(draw_body=False, lamp=lamp))
    R = rot_y(yaw) @ rot_z(math.radians(84))
    t = np.asarray(pos, float) + C(0, 0.035, 0)
    transform_into(b, tmp, R, t, skip_bucket=2.7)
    return [(R @ lp + t, lc, li, lr) for (lp, lc, li, lr) in lights]


# ================================================================ mundo desmoronando
_CR = {}


def crumble(fs, center, t, t0, speed=30.0, dmax=100.0, safe=5.5):
    """faces caem no vazio, do horizonte para o centro; só ficam as próximas de `center`."""
    n = len(fs.V)
    if n not in _CR:
        rng = np.random.default_rng(n)
        _CR[n] = (rng.uniform(0.6, 1.4, n), rng.uniform(-0.6, 0.6, n), rng.uniform(-0.8, 0.8, n))
    rmul, jit, sj = _CR[n]
    d = np.hypot(fs.cen[:, 0] - center[0], fs.cen[:, 2] - center[2])
    start = t0 + (dmax - np.minimum(d, dmax)) / speed + jit
    u = np.clip(t - start, 0, None)
    fall = 4.9 * u * u * rmul * (d > safe + sj) * (fs.bucket >= 1.0)
    out = FaceSet()
    for name in FaceSet.__slots__:
        setattr(out, name, getattr(fs, name))
    out.V = fs.V - C(0, 1, 0)[None, None, :] * fall[:, None, None]
    out.cen = fs.cen - C(0, 1, 0)[None, :] * fall[:, None]
    k = np.clip(u * 0.7, 0, 1)[:, None] * (fall[:, None] > 0)
    out.col = fs.col * (1 - 0.55 * k) + C(0.25, 0.0, 0.0) * 0.4 * k
    return out


# ================================================================ bar
BX, BZ, BH = 7.0, 5.0, 3.8
BAR_LAMPS = [C(-3.0, 3.0, -1.0), C(3.0, 3.0, -1.0), C(-3.0, 3.0, 2.2), C(3.0, 3.0, 2.2)]


def _room(b, rng, X, Z, H, floor_cols, wain, paper, ceil, plank=0.5, bucket_w=1.5):
    # piso de tábuas
    k = 0
    for z in np.arange(-Z, Z, plank):
        for x in np.arange(-X, X, 2.0):
            col = floor_cols[k % 2] * (1 + rng.normal(0, 0.05))
            q = np.array([C(x, 0, z), C(x + 2.0, 0, z), C(x + 2.0, 0, z + plank), C(x, 0, z + plank)])
            b.poly(orient(q, C(0, 1, 0)), np.clip(col, 0, 1), bucket=1.0)
        k += 1
    walls = [(C(0, 0, -Z), C(1, 0, 0), C(0, 0, 1), X), (C(0, 0, Z), C(1, 0, 0), C(0, 0, -1), X),
             (C(-X, 0, 0), C(0, 0, 1), C(1, 0, 0), Z), (C(X, 0, 0), C(0, 0, 1), C(-1, 0, 0), Z)]
    for base, along, nrm, half in walls:
        for a in np.arange(-half, half, 2.0):
            a1 = min(a + 2.0, half)
            for (y0, y1, col) in ((0.0, 1.1, wain), (1.1, H, paper)):
                q = np.array([base + along * a + C(0, y0, 0), base + along * a1 + C(0, y0, 0),
                              base + along * a1 + C(0, y1, 0), base + along * a + C(0, y1, 0)])
                b.poly(orient(q, nrm), np.clip(col * (1 + rng.normal(0, 0.04)), 0, 1), bucket=bucket_w)
    for x in np.arange(-X, X, 3.5):
        for z in np.arange(-Z, Z, 2.5):
            q = np.array([C(x, H, z), C(x + 3.5, H, z), C(x + 3.5, H, z + 2.5), C(x, H, z + 2.5)])
            b.poly(orient(q, C(0, -1, 0)), ceil, bucket=bucket_w)


def _table(b, x, z, wood, tipped=False):
    tb = Builder()
    tb.mesh(prism_polys(8, 0.5, 0.5, 0.05, y0=0.72), wood * 1.1)
    tb.mesh(tube_polys(C(0, 0.0, 0), C(0, 0.72, 0), 0.08), wood * 0.7)
    tb.mesh(prism_polys(8, 0.25, 0.25, 0.03), wood * 0.6)
    R = rot_z(math.pi / 2) if tipped else np.eye(3)
    t = C(x, 0.5 if tipped else 0.0, z)
    transform_into(b, tb, R, t)


def _chair(b, x, z, yaw, wood, over=False):
    cb = Builder()
    cb.mesh(box_polys(0.42, 0.05, 0.42, c=(0, 0.45, 0)), wood)
    cb.mesh(box_polys(0.42, 0.5, 0.05, c=(0, 0.72, -0.19)), wood * 0.9)
    for sx in (-0.18, 0.18):
        for sz in (-0.18, 0.18):
            cb.mesh(tube_polys(C(sx, 0, sz), C(sx, 0.45, sz), 0.035), wood * 0.7)
    R = rot_y(yaw) @ (rot_x(-math.pi / 2) if over else np.eye(3))
    t = C(x, 0.22 if over else 0.0, z)
    transform_into(b, cb, R, t)


def build_bar_world(blood=False, seed=21):
    rng = np.random.default_rng(seed)
    b = Builder(group_base=5_000_000)
    floor = (C(0.38, 0.21, 0.10), C(0.31, 0.17, 0.08))
    _room(b, rng, BX, BZ, BH, floor, C(0.24, 0.12, 0.06), C(0.34, 0.07, 0.10), C(0.13, 0.09, 0.07))
    wood = C(0.30, 0.15, 0.07)
    # balcão + prateleiras + garrafas
    b.mesh(box_polys(8.0, 1.1, 0.7, c=(-1.5, 0.55, -4.25)), C(0.22, 0.11, 0.05))
    b.mesh(box_polys(8.1, 0.06, 0.8, c=(-1.5, 1.13, -4.25)), C(0.42, 0.23, 0.10))
    for y in (1.6, 2.25):
        b.mesh(box_polys(7.0, 0.05, 0.26, c=(-1.5, y, -4.86)), C(0.28, 0.14, 0.06))
        for x in np.arange(-4.8, 1.9, 0.3):
            if rng.random() < 0.15:
                continue
            col = [C(0.10, 0.40, 0.16), C(0.62, 0.32, 0.06), C(0.60, 0.70, 0.72), C(0.45, 0.06, 0.08)][rng.integers(0, 4)]
            h = rng.uniform(0.18, 0.26)
            b.mesh(prism_polys(6, 0.045, 0.045, h, y0=y + 0.025, c=(x, 0, -4.86)), col, emis=0.25)
            b.mesh(prism_polys(6, 0.018, 0.018, 0.08, y0=y + 0.025 + h, c=(x, 0, -4.86)), col, emis=0.25)
    for x in np.arange(-4.8, 2.4, 1.2):
        b.mesh(prism_polys(8, 0.18, 0.18, 0.06, y0=0.74, c=(x, 0, -3.55)), C(0.35, 0.08, 0.08))
        b.mesh(tube_polys(C(x, 0, -3.55), C(x, 0.74, -3.55), 0.05), C(0.5, 0.48, 0.45))
    # mesas e cadeiras
    tables = [(-5.4, 2.9), (5.3, 2.9), (5.4, -1.7), (-5.6, -1.4)]
    for i, (x, z) in enumerate(tables):
        tipped = blood and i in (1, 3)
        _table(b, x, z, wood, tipped=tipped)
        for j, a in enumerate((0.0, math.pi)):
            cx, cz = x + 0.75 * math.sin(a + 0.7), z + 0.75 * math.cos(a + 0.7)
            _chair(b, cx, cz, a + 0.7 + math.pi, wood * 0.9, over=blood and (i + j) % 2 == 0)
    # janelas (parede esquerda)
    for z in (-1.0, 2.5):
        flat_quad(b, C(-BX + 0.02, 2.0, z), C(0, 0, 0.6), C(0, 0.7, 0), C(1, 0, 0), C(0.05, 0.08, 0.18), emis=0.7,
                  bucket=1.6)
        b.mesh(box_polys(0.06, 1.5, 0.08, c=(-BX + 0.04, 2.0, z)), C(0.2, 0.1, 0.05))
        b.mesh(box_polys(0.06, 0.08, 1.3, c=(-BX + 0.04, 2.0, z)), C(0.2, 0.1, 0.05))
    if blood:
        for k in range(16):
            c = C(rng.uniform(-6.0, 6.0), 0, rng.uniform(-3.2, 4.4))
            blob(b, c, rng.uniform(0.5, 1.4), rng, y=0.012, bucket=1.6)
        blob(b, C(0.3, 0, 0.6), 2.2, rng, n=12, y=0.011, bucket=1.6)
        for k in range(10):          # respingos nas paredes
            wall = rng.integers(0, 3)
            if wall == 0:
                c, n_ = C(rng.uniform(-6, 6), rng.uniform(0.5, 3.2), -BZ), C(0, 0, 1)
            elif wall == 1:
                c, n_ = C(-BX, rng.uniform(0.5, 3.2), rng.uniform(-4, 4)), C(1, 0, 0)
            else:
                c, n_ = C(BX, rng.uniform(0.5, 3.0), rng.uniform(-4, 4)), C(-1, 0, 0)
            blob(b, c, rng.uniform(0.3, 0.8), rng, normal=n_, up=C(0, 1, 0), y=0.02, bucket=1.7, n=5)
        blob(b, C(-1.5, 1.165, -4.25), 0.6, rng, n=6, y=0.0, bucket=3.0)
        for k in range(40):          # cacos de vidro
            c = C(rng.uniform(-5, 2), 0.02, rng.uniform(-3.6, -2.0))
            a = rng.uniform(0, 6.28)
            tri = np.array([c, c + C(math.cos(a) * 0.08, 0, math.sin(a) * 0.08),
                            c + C(math.cos(a + 1.9) * 0.06, 0, math.sin(a + 1.9) * 0.06)])
            b.poly(orient(tri, C(0, 1, 0)), C(0.7, 0.8, 0.8), emis=0.3, bucket=1.65)
    return b.build()


# letreiro neon "BAR" + taça (parede direita), em coordenadas (u ao longo de -z, v para cima)
_NEON = [((0, 0), (0, 1)), ((0, 1), (0.42, 0.92)), ((0.42, 0.92), (0.45, 0.6)), ((0.45, 0.6), (0, 0.52)),
         ((0, 0.52), (0.48, 0.42)), ((0.48, 0.42), (0.46, 0.06)), ((0.46, 0.06), (0, 0)),
         ((0.75, 0), (0.98, 1)), ((0.98, 1), (1.2, 0)), ((0.84, 0.42), (1.12, 0.42)),
         ((1.45, 0), (1.45, 1)), ((1.45, 1), (1.85, 0.9)), ((1.85, 0.9), (1.86, 0.58)), ((1.86, 0.58), (1.45, 0.5)),
         ((1.55, 0.5), (1.9, 0)),
         ((2.3, 1.0), (2.9, 1.0)), ((2.3, 1.0), (2.6, 0.55)), ((2.9, 1.0), (2.6, 0.55)), ((2.6, 0.55), (2.6, 0.12)),
         ((2.42, 0.08), (2.78, 0.08))]


def add_neon(b, on=1.0):
    col = C(1.0, 0.22, 0.62) * (0.25 + 0.75 * on)
    base = C(BX - 0.03, 2.35, 1.6)
    U, V = C(0, 0, -0.62), C(0, 0.62, 0)
    for (u0, v0), (u1, v1) in _NEON:
        p0 = base + U * u0 + V * v0
        p1 = base + U * u1 + V * v1
        d = normalize(p1 - p0)
        w = np.cross(d, C(-1, 0, 0)) * 0.035
        q = np.array([p0 - w, p1 - w, p1 + w, p0 + w])
        b.poly(orient(q, C(-1, 0, 0)), col, emis=1.0 * on + 0.1, glow=on > 0.3, bucket=1.75)
    return (base + U * 1.4 + V * 0.5 + C(-0.6, 0, 0), C(1.0, 0.25, 0.6), 0.9 * on, 2.5)


def add_bar_lamps(b, br, swing=0.0, t=0.0):
    lights = []
    for i, lp in enumerate(BAR_LAMPS):
        top = C(lp[0], BH, lp[2])
        ang = swing * math.sin(t * 1.7 + i) if i == 1 else 0.0
        low = top + C(math.sin(ang) * 0.8, -0.8 * math.cos(ang), 0)
        b.mesh(tube_polys(top, low, 0.02), C(0.05, 0.05, 0.05))
        b.mesh(frustum_polys(low + C(0, 0.02, 0), low - C(0, 0.25, 0), 0.12, 0.12, 0.5, 0.5), C(0.55, 0.42, 0.12))
        bri = br[i]
        b.mesh(box_polys(0.12, 0.08, 0.12, c=low - C(0, 0.26, 0)), C(1.0, 0.85, 0.55) * (0.2 + 0.8 * bri), emis=bri,
               glow=bri > 0.3)
        if bri > 0.02:
            lights.append((low - C(0, 0.45, 0), C(1.0, 0.80, 0.50), 1.7 * bri, 3.4))
    return lights


# ================================================================ escritório do detetive
OX, OZ, OH = 2.6, 2.2, 2.8
DESK_Y = 0.78
CLOCK_C = C(-1.35, 2.05, -OZ + 0.03)
LAMP_BULB = C(-0.56, 1.04, -0.78)
GUN_MUZZLE, GUN_BUTT = C(0.10, 0.0, -0.25), C(0.50, 0.0, -0.90)
CROSS_C = C(-0.12, 0.0, -0.60)


def _shotgun(b):
    y = DESK_Y + 0.022
    a = GUN_MUZZLE * C(1, 0, 1) + C(0, y, 0)
    e = GUN_BUTT * C(1, 0, 1) + C(0, y, 0)
    d = normalize(e - a)
    L = np.linalg.norm(e - a)
    s = np.cross(d, C(0, 1, 0))
    metal = C(0.22, 0.23, 0.28)
    wood = C(0.62, 0.33, 0.12)
    for o in (-0.012, 0.012):                                                    # dois canos
        b.mesh(tube_polys(a + s * o, a + d * 0.50 * L + s * o, 0.024, side=s), metal, emis=0.14)
    b.mesh(tube_polys(a + d * 0.20 * L, a + d * 0.42 * L, 0.066, 0.04, side=s), wood)          # telha
    b.mesh(tube_polys(a + d * 0.49 * L, a + d * 0.62 * L, 0.06, 0.05, side=s), metal * 1.3, emis=0.06)  # báscula
    b.mesh(tube_polys(a + d * 0.585 * L + s * 0.035, a + d * 0.585 * L + s * 0.075, 0.012, side=d), metal)  # gatilho
    b.mesh(frustum_polys(a + d * 0.61 * L, a + d * 1.0 * L, 0.05, 0.04, 0.14, 0.045, side=s), wood)    # coronha
    b.mesh(tube_polys(a + d * 0.985 * L, a + d * 1.0 * L, 0.142, 0.046, side=s), C(0.12, 0.07, 0.05))  # soleira


def _cross(b):
    c = CROSS_C * C(1, 0, 1) + C(0, DESK_Y, 0)
    silver = C(0.82, 0.80, 0.72)
    b.mesh(box_polys(0.15, 0.03, 0.11, c=c + C(0, 0.015, 0)), C(0.22, 0.12, 0.06))                   # base
    b.mesh(box_polys(0.04, 0.38, 0.03, c=c + C(0, 0.03 + 0.19, 0)), silver, emis=0.3, glow=True)     # haste
    b.mesh(box_polys(0.22, 0.04, 0.03, c=c + C(0, 0.03 + 0.28, 0)), silver, emis=0.3, glow=True)     # travessa


def build_office_world(seed=31):
    rng = np.random.default_rng(seed)
    b = Builder(group_base=6_000_000)
    _room(b, rng, OX, OZ, OH, (C(0.21, 0.13, 0.08), C(0.17, 0.11, 0.07)), C(0.18, 0.12, 0.08), C(0.26, 0.28, 0.22),
          C(0.10, 0.10, 0.09), plank=0.35)
    dw = C(0.27, 0.14, 0.07)
    # mesa
    wood_top = C(0.30, 0.16, 0.08)
    leather = C(0.08, 0.20, 0.13)
    for x in np.arange(-0.8, 0.8 - 1e-6, 0.1):
        for z in np.arange(-0.95, -0.15 - 1e-6, 0.1):
            inner = (-0.7 - 1e-6 <= x < 0.7 - 1e-6) and (-0.85 - 1e-6 <= z < -0.25 - 1e-6)
            q = np.array([C(x, DESK_Y, z), C(x + 0.1, DESK_Y, z), C(x + 0.1, DESK_Y, z + 0.1), C(x, DESK_Y, z + 0.1)])
            b.poly(orient(q, C(0, 1, 0)), leather if inner else wood_top)
    for (c, ax, ay, n_) in ((C(0, DESK_Y - 0.025, -0.15), C(0.8, 0, 0), C(0, 0.025, 0), C(0, 0, 1)),
                            (C(0, DESK_Y - 0.025, -0.95), C(0.8, 0, 0), C(0, 0.025, 0), C(0, 0, -1)),
                            (C(-0.8, DESK_Y - 0.025, -0.55), C(0, 0, 0.4), C(0, 0.025, 0), C(-1, 0, 0)),
                            (C(0.8, DESK_Y - 0.025, -0.55), C(0, 0, 0.4), C(0, 0.025, 0), C(1, 0, 0))):
        flat_quad(b, c, ax, ay, n_, wood_top * 0.8)
    for sx in (-0.56, 0.56):
        b.mesh(box_polys(0.44, DESK_Y - 0.05, 0.74, c=(sx, (DESK_Y - 0.05) / 2, -0.55)), dw)
        for y in (0.2, 0.42, 0.62):
            flat_quad(b, C(sx, y, -0.17), C(0.15, 0, 0), C(0, 0.012, 0), C(0, 0, 1), C(0.6, 0.5, 0.3), emis=0.1)
    # cadeira
    b.mesh(box_polys(0.52, 0.07, 0.5, c=(0, 0.47, -1.22)), C(0.20, 0.09, 0.05))
    b.mesh(box_polys(0.52, 0.8, 0.07, c=(0, 0.9, -1.5)), C(0.20, 0.09, 0.05))
    b.mesh(tube_polys(C(0, 0, -1.22), C(0, 0.44, -1.22), 0.06), C(0.1, 0.1, 0.1))
    # objetos na mesa
    _shotgun(b)
    _cross(b)
    b.mesh(prism_polys(6, 0.045, 0.045, 0.24, y0=DESK_Y, c=(0.66, 0, -0.36)), C(0.55, 0.28, 0.05), emis=0.2)
    b.mesh(prism_polys(6, 0.016, 0.016, 0.08, y0=DESK_Y + 0.24, c=(0.66, 0, -0.36)), C(0.55, 0.28, 0.05), emis=0.2)
    b.mesh(prism_polys(6, 0.035, 0.03, 0.08, y0=DESK_Y, c=(0.56, 0, -0.24)), C(0.6, 0.66, 0.66), emis=0.25)
    for (x, z, a) in ((-0.52, -0.38, 0.25), (-0.40, -0.32, -0.12)):
        R = rot_y(a)
        q = np.array([R @ C(-0.11, 0, -0.15), R @ C(0.11, 0, -0.15), R @ C(0.11, 0, 0.15), R @ C(-0.11, 0, 0.15)])
        b.poly(orient(q + C(x, DESK_Y + 0.003, z), C(0, 1, 0)), C(0.78, 0.76, 0.68), bucket=3.0)
    # luminária de banqueiro (lâmpada é dinâmica)
    b.mesh(box_polys(0.24, 0.03, 0.14, c=(-0.6, DESK_Y + 0.015, -0.82)), C(0.55, 0.45, 0.15))
    b.mesh(tube_polys(C(-0.6, DESK_Y, -0.82), C(-0.58, 1.09, -0.80), 0.02), C(0.55, 0.45, 0.15))
    b.mesh(frustum_polys(C(-0.56, 1.10, -0.95), C(-0.56, 1.10, -0.6), 0.36, 0.10, 0.36, 0.10), C(0.08, 0.32, 0.14),
           emis=0.15)
    # persianas: listras de luz na parede do fundo
    for k in range(9):
        y0 = 0.95 + k * 0.15
        q = np.array([C(-2.45, y0, -OZ + 0.015), C(0.15, y0 + 0.35, -OZ + 0.015),
                      C(0.15, y0 + 0.42, -OZ + 0.015), C(-2.45, y0 + 0.07, -OZ + 0.015)])
        b.poly(orient(q, C(0, 0, 1)), C(0.42, 0.50, 0.66), emis=0.75, bucket=1.6)
    # janela com persiana (parede esquerda)
    flat_quad(b, C(-OX + 0.02, 1.55, -0.6), C(0, 0, 0.55), C(0, 0.65, 0), C(1, 0, 0), C(0.20, 0.26, 0.38), emis=0.6,
              bucket=1.6)
    for k in range(10):
        b.mesh(box_polys(0.04, 0.025, 1.1, c=(-OX + 0.06, 0.95 + k * 0.13, -0.6)), C(0.55, 0.55, 0.50))
    # quadro de pistas
    flat_quad(b, C(1.45, 1.70, -OZ + 0.02), C(0.7, 0, 0), C(0, 0.45, 0), C(0, 0, 1), C(0.55, 0.38, 0.20), bucket=1.6)
    photos = [C(0.95, 1.92, 0), C(1.95, 1.95, 0), C(1.05, 1.42, 0), C(1.85, 1.40, 0), C(1.45, 1.68, 0)]
    for i, ph in enumerate(photos):
        c = ph + C(0, 0, -OZ + 0.03)
        flat_quad(b, c, C(0.10, 0, 0), C(0, 0.13, 0), C(0, 0, 1), C(0.82, 0.80, 0.72), bucket=1.65)
        flat_quad(b, c + C(0, 0.1, 0.002), C(0.012, 0, 0), C(0, 0.012, 0), C(0, 0, 1), C(0.8, 0.05, 0.05), bucket=1.7,
                  emis=0.4)
        if i == 4:      # a foto do líder: silhueta de cartola
            k = c + C(0, 0, 0.003)
            flat_quad(b, k + C(0, -0.03, 0), C(0.035, 0, 0), C(0, 0.05, 0), C(0, 0, 1), C(0.03, 0.03, 0.03), bucket=1.7)
            flat_quad(b, k + C(0, 0.03, 0), C(0.06, 0, 0), C(0, 0.008, 0), C(0, 0, 1), C(0.03, 0.03, 0.03), bucket=1.7)
            flat_quad(b, k + C(0, 0.075, 0), C(0.032, 0, 0), C(0, 0.04, 0), C(0, 0, 1), C(0.03, 0.03, 0.03), bucket=1.7)
            for sx in (-0.012, 0.012):
                flat_quad(b, k + C(sx, -0.02, 0.001), C(0.006, 0, 0), C(0, 0.005, 0), C(0, 0, 1), C(1, 1, 0.9),
                          bucket=1.75, emis=0.8)
    for i in range(4):
        p0 = photos[i] + C(0, 0.1, -OZ + 0.04)
        p1 = photos[4] + C(0, 0.1, -OZ + 0.04)
        b.mesh(tube_polys(p0, p1, 0.008, side=C(0, 0, 1)), C(0.75, 0.05, 0.05), emis=0.35)
    # relógio de parede (ponteiros dinâmicos)
    b.poly(disc_poly(CLOCK_C, C(0, 0, 1), C(0, 1, 0), 0.22, 0.22, n=8), C(0.12, 0.08, 0.05), bucket=1.65)
    b.poly(disc_poly(CLOCK_C + C(0, 0, 0.004), C(0, 0, 1), C(0, 1, 0), 0.19, 0.19, n=8), C(0.82, 0.78, 0.66),
           bucket=1.7)
    # arquivo de aço + estante
    b.mesh(box_polys(0.5, 1.3, 0.6, c=(2.2, 0.65, -1.75)), C(0.34, 0.37, 0.32))
    for y in (0.35, 0.75, 1.15):
        flat_quad(b, C(2.2, y, -1.44), C(0.2, 0, 0), C(0, 0.012, 0), C(0, 0, 1), C(0.15, 0.15, 0.13))
    for y in (0.4, 1.0, 1.6):
        b.mesh(box_polys(0.3, 0.03, 1.4, c=(-OX + 0.16, y, 1.0)), dw)
        z = 0.35
        while z < 1.65:
            w = rng.uniform(0.04, 0.08)
            h = rng.uniform(0.22, 0.32)
            col = [C(0.4, 0.08, 0.06), C(0.1, 0.2, 0.35), C(0.15, 0.3, 0.12), C(0.45, 0.38, 0.2)][rng.integers(0, 4)]
            b.mesh(box_polys(0.22, h, w, c=(-OX + 0.16, y + 0.015 + h / 2, z)), col)
            z += w + 0.01
    return b.build()


def add_office_dynamic(b, t, lamp_on=1.0, fan_ang=0.0):
    lights = []
    # ponteiros do relógio (segundos "tiquetaqueando")
    for ang, ln, wd, col in ((-math.radians(int(t) * 6.0), 0.17, 0.006, C(0.6, 0.05, 0.05)),
                             (math.radians(60), 0.15, 0.012, C(0.05, 0.05, 0.05)),
                             (math.radians(-80), 0.10, 0.016, C(0.05, 0.05, 0.05))):
        d = C(math.sin(ang), math.cos(ang), 0)
        s = C(math.cos(ang), -math.sin(ang), 0)
        c0 = CLOCK_C + C(0, 0, 0.012)
        q = np.array([c0 - s * wd, c0 + s * wd, c0 + d * ln + s * wd * 0.4, c0 + d * ln - s * wd * 0.4])
        b.poly(orient(q, C(0, 0, 1)), col, emis=0.2, bucket=1.75)
    # ventilador de teto
    hub = C(0, OH - 0.25, -0.5)
    b.mesh(tube_polys(C(0, OH, -0.5), hub, 0.03), C(0.1, 0.1, 0.1))
    b.mesh(prism_polys(8, 0.1, 0.1, 0.08, y0=-0.04, c=hub), C(0.25, 0.2, 0.12))
    for k in range(4):
        a = fan_ang + k * math.pi / 2
        R = rot_y(a)
        b.mesh(box_polys(0.62, 0.012, 0.13, c=(0.38, 0, 0)), C(0.28, 0.16, 0.08), R=R, t=hub)
    # lâmpada da luminária
    b.mesh(box_polys(0.18, 0.04, 0.06, c=LAMP_BULB), C(1.0, 0.9, 0.6) * (0.15 + 0.85 * lamp_on), emis=lamp_on,
           glow=lamp_on > 0.3)
    if lamp_on > 0.02:
        lights.append((LAMP_BULB - C(0, 0.12, -0.05), C(1.0, 0.82, 0.52), 1.9 * lamp_on, 1.05))
    return lights


# ================================================================ detetive sentado
HEAD_D = Head(0.15, 0.19, 0.16, 0.38, 1.12)
SKIN_D = C(0.84, 0.58, 0.48)
COAT_D = C(0.55, 0.46, 0.31)


def build_detective(b, pos, yaw, p=None):
    p = p or {}
    Mw = rot_y(yaw)
    pos = np.asarray(pos, float)
    tw = lambda q: np.asarray(q) @ Mw.T + pos
    side = Mw @ C(1, 0, 0)
    pel = C(0, 0.50, 0.0)
    RL = rot_x(p.get("lean", 0.2))
    L = lambda q: pel + RL @ (np.asarray(q) - pel)
    trous = C(0.15, 0.14, 0.15)
    for s in (-1, 1):
        hip, knee, ank = C(s * 0.1, 0.52, 0.03), C(s * 0.13, 0.53, 0.47), C(s * 0.14, 0.11, 0.52)
        b.mesh(frustum_polys(tw(hip), tw(knee), 0.16, 0.16, 0.13, 0.13, side=side), trous)
        b.mesh(frustum_polys(tw(knee), tw(ank), 0.12, 0.12, 0.10, 0.10, side=side), trous)
        b.mesh(box_polys(0.10, 0.08, 0.25, c=ank + C(0, -0.07, 0.06)), C(0.08, 0.05, 0.04), R=Mw, t=pos)
    chest = L(C(0, 1.05, 0.0))
    b.mesh(frustum_polys(tw(L(C(0, 0.48, 0))), tw(chest), 0.38, 0.27, 0.47, 0.27, side=side), COAT_D)
    shirt = np.array([L(C(-0.065, 1.05, 0.137)), L(C(0.065, 1.05, 0.137)), L(C(0, 0.84, 0.135))])
    b.poly(tw(orient(shirt, RL @ C(0, 0.2, 1))), C(0.85, 0.85, 0.82))
    tie = np.array([L(C(-0.02, 1.03, 0.141)), L(C(0.02, 1.03, 0.141)), L(C(0.028, 0.84, 0.139)), L(C(0, 0.80, 0.139)),
                    L(C(-0.028, 0.84, 0.139))])
    b.poly(tw(orient(tie, RL @ C(0, 0.2, 1))), C(0.42, 0.05, 0.06))
    b.mesh(frustum_polys(tw(L(C(0, 1.0, -0.01))), tw(L(C(0, 1.15, -0.03))), 0.40, 0.26, 0.30, 0.20, side=side),
           COAT_D * 0.9)
    hands = p.get("hands", (C(-0.22, 0.80, 0.40), C(0.22, 0.80, 0.40)))
    for s, tgt in zip((-1, 1), hands):
        sh = L(C(s * 0.22, 1.02, 0.0))
        elbow, wrist = ik2(sh, tgt, 0.30, 0.29, C(s * 0.8, -0.4, -0.5))
        b.mesh(frustum_polys(tw(sh), tw(elbow), 0.12, 0.12, 0.10, 0.10, side=Mw @ C(0, 1, 0)), COAT_D)
        b.mesh(frustum_polys(tw(elbow), tw(wrist), 0.10, 0.10, 0.085, 0.085, side=Mw @ C(0, 1, 0)), COAT_D)
        a = normalize(wrist - elbow)
        b.mesh(tube_polys(tw(wrist), tw(wrist + a * 0.10), 0.08, 0.035, side=side), SKIN_D)
    n0, n1 = L(C(0, 1.05, 0.01)), L(C(0, 1.16, 0.02))
    b.mesh(tube_polys(tw(n0), tw(n1), 0.075, side=side), SKIN_D)
    hc = L(C(0, 1.16 + 0.17, 0.03))
    R = Mw @ RL @ rot_y(p.get("head_yaw", 0.0)) @ rot_x(p.get("head_pitch", 0.0) - p.get("lean", 0.2))
    hp = dict(eye_r=0.042, eye_dx=0.062, eye_y=0.03, eye_open=p.get("eye_open", 0.6), pupil=p.get("pupil", 1.0),
              px=p.get("px", 0.0), py=p.get("py", 0.0), eye_emis=0.3, brow=p.get("brow", 0.35),
              brow_col=C(0.20, 0.12, 0.08), smile=p.get("smile", -0.25), mw=0.045, mouth_y=-0.095, nose_y=-0.022,
              mouth_open=p.get("mouth_open", 0.0), mustache=C(0.20, 0.12, 0.07), mouth_col=C(0.30, 0.06, 0.06))
    build_head(b, HEAD_D, R, tw(hc), SKIN_D, hp, hat="fedora")


def detective_head_world(pos, yaw, lean=0.2):
    pel = C(0, 0.50, 0.0)
    hc = pel + rot_x(lean) @ (C(0, 1.33, 0.03) - pel)
    return rot_y(yaw) @ hc + np.asarray(pos, float)
