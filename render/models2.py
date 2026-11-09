"""Modelos extras da parte 2 (gato, mãos, joia, moedas, estrela, sombra longa, árvores que se curvam, mundos)."""
import math
import numpy as np

from engine import (Builder, rot_x, rot_y, rot_z, normalize, orient, box_polys, frustum_polys, tube_polys,
                    prism_polys, ico_polys, disc_poly)
from models import (C, Head, build_head, EYE_W, PUPIL, GRASS, ROAD, _tree, _house)

# ================================================================ gato
HEAD_CAT = Head(0.07, 0.062, 0.075, 0.3, 1.0, nlon=8, nlat=5)
FUR = C(0.07, 0.07, 0.085)


def build_cat(b, pos, yaw, p=None):
    p = p or {}
    arch = p.get("arch", 0.0)
    run = p.get("run", None)
    hiss = p.get("hiss", 0.0)
    sc = p.get("scale", 1.0)
    Mw = rot_y(yaw)
    pos = np.asarray(pos, float)
    tw = lambda q: (np.asarray(q) * sc) @ Mw.T + pos
    side = Mw @ C(1, 0, 0)
    bob = 0.0 if run is None else 0.035 * math.sin(2 * run)
    hip = C(0, 0.20 + 0.07 * arch + bob, -0.17)
    mid = C(0, 0.22 + 0.17 * arch, -0.01)
    sh = C(0, 0.21 + 0.05 * arch - bob, 0.14)
    b.mesh(frustum_polys(tw(hip), tw(mid), 0.13 * sc, 0.13 * sc, 0.15 * sc, 0.16 * sc, side=side), FUR)
    b.mesh(frustum_polys(tw(mid), tw(sh), 0.15 * sc, 0.16 * sc, 0.13 * sc, 0.14 * sc, side=side), FUR)
    for zc, base, ph0 in ((-0.15, hip, 0.0), (0.13, sh, math.pi / 2)):
        for s in (-1, 1):
            if run is None:
                foot = C(s * 0.055 * (1 + 0.3 * arch), 0.0, zc)
            else:
                ph = run + ph0 + (0 if s < 0 else math.pi)
                foot = C(s * 0.05, 0.05 * max(0.0, math.sin(ph)), zc + 0.1 * math.cos(ph))
            top = base + C(s * 0.05, -0.02, 0)
            b.mesh(tube_polys(tw(top), tw(foot), 0.04 * sc, side=side), FUR)
    if run is None:
        t1 = hip + C(0, 0.10 + 0.10 * arch, -0.07)
        t2 = t1 + C(0, 0.13 + 0.04 * arch, 0.02 + 0.03 * arch)
        w = 0.035 + 0.04 * arch
    else:
        t1 = hip + C(0, 0.04, -0.15)
        t2 = t1 + C(0, 0.06 + 0.03 * math.sin(run * 2), -0.14)
        w = 0.03
    b.mesh(frustum_polys(tw(hip), tw(t1), w * sc, w * sc, w * sc, w * sc, side=side), FUR)
    b.mesh(frustum_polys(tw(t1), tw(t2), w * sc, w * sc, w * 0.6 * sc, w * 0.6 * sc, side=side), FUR)
    hc = sh + C(0, 0.07 - 0.035 * arch, 0.10)
    R = Mw @ rot_y(p.get("head_yaw", 0.0)) @ rot_x(p.get("head_pitch", 0.0))
    R = R * sc
    hp = dict(eye_r=0.022, eye_dx=0.031, eye_y=0.012, eye_col=C(0.78, 0.95, 0.25), eye_emis=1.0, glow=True,
              pupil=p.get("pupil", 1.5), pupil_shape=(0.32, 1.7), nose=False, mouth_y=-0.032, mw=0.022,
              smile=-0.6 * hiss, mouth_open=hiss * 0.8, mouth_col=C(0.45, 0.05, 0.08))
    H = tw(hc)
    build_head(b, HEAD_CAT, R, H, FUR, hp)
    for s in (-1, 1):
        b0 = C(s * 0.022, 0.045, 0.01)
        b1 = C(s * 0.064, 0.028, 0.0)
        ap = C(s * 0.054, 0.105 - 0.02 * hiss, -0.01 - 0.03 * hiss)
        b.poly(np.array([b0, b1, ap]) @ R.T + H, FUR * 1.15, dbl=True)
        inner = np.array([b0 * 0.85 + b1 * 0.15, b0 * 0.2 + b1 * 0.8, ap * 0.75 + (b0 + b1) * 0.125]) + C(0, 0.004, 0.006)
        b.poly(inner @ R.T + H, C(0.40, 0.16, 0.20), dbl=True)


# ================================================================ mãos
def build_hand(b, wrist, fwd, up, s, col, curl=0.5, handed=1, point=False, **kw):
    wrist = np.asarray(wrist, float)
    fwd = normalize(fwd)
    up = np.asarray(up, float)
    up = normalize(up - fwd * np.dot(up, fwd))
    side = np.cross(fwd, up)
    palm_end = wrist + fwd * 0.10 * s
    b.mesh(frustum_polys(wrist, palm_end, 0.08 * s, 0.032 * s, 0.096 * s, 0.028 * s, side=side), col, **kw)
    lens = [(0.044, 0.030, 0.024), (0.050, 0.034, 0.026), (0.048, 0.033, 0.025), (0.038, 0.027, 0.021)]
    for k in range(4):
        off = (k - 1.5) * 0.0245 * s * handed
        p0 = palm_end + side * off - fwd * 0.004 * s
        d = normalize(fwd + side * 0.07 * (k - 1.5) * handed)
        cc = curl
        if point:
            cc = 0.05 if k == 0 else 1.6
        for L in lens[k]:
            d = normalize(d * math.cos(cc * 0.6) + up * math.sin(cc * 0.6))
            p1 = p0 + d * L * s
            b.mesh(tube_polys(p0, p1, 0.021 * s, 0.019 * s, side=side), col, **kw)
            p0 = p1
    t0 = wrist + fwd * 0.025 * s - side * 0.040 * s * handed
    d = normalize(fwd * 0.55 - side * 0.75 * handed + up * 0.2)
    for L in (0.042, 0.033):
        d = normalize(d * math.cos(curl * 0.45) + up * math.sin(curl * 0.45))
        p1 = t0 + d * L * s
        b.mesh(tube_polys(t0, p1, 0.024 * s, side=side), col, **kw)
        t0 = p1
    return palm_end


# ================================================================ joia perfeita que se degenera
GEM = ico_polys(1.0)
GEM_R = np.random.default_rng(77).uniform(0, 1, len(GEM))
GOLD = C(1.0, 0.80, 0.36)
ASH = C(0.22, 0.20, 0.19)


def build_gem(b, center, r, rot, deg=0.0, fogmul=1.0):
    R = rot_y(rot) @ rot_x(0.5) @ rot_z(0.3)
    center = np.asarray(center, float)
    for tri, ri in zip(GEM, GEM_R):
        cen = tri.mean(0)
        n = normalize(cen)
        e = deg
        shrink = 1 - 0.45 * e * ri
        q = (tri - cen) * shrink + cen + n * (0.6 * e * (0.4 + ri))
        q = (q @ R.T) * r + center + C(0, -2.4 * (e * (0.3 + ri)) ** 2 * r, 0)
        k = min(1.0, e * 1.5)
        col = np.clip((GOLD * (1 - k) + ASH * k) * (0.8 + 0.35 * ri), 0, 1)
        b.poly(q, col, emis=0.7 * (1 - k) + 0.05, glow=k < 0.5, fogmul=fogmul)


def build_coin(b, center, spin, s=0.05):
    n = rot_y(spin) @ C(0, 0, 1)
    b.poly(disc_poly(center, n, C(0, 1, 0), s, s, n=8), C(1.0, 0.78, 0.30), emis=0.5, dbl=True, glow=True)
    b.poly(disc_poly(np.asarray(center) + n * 0.002, n, C(0, 1, 0), s * 0.62, s * 0.62, n=8), C(0.85, 0.6, 0.2),
           emis=0.5, dbl=True, glow=True)


def build_star(b, center, normal, r, col=C(1.0, 0.85, 0.30), emis=0.9):
    n_ = normalize(normal)
    up = C(0, 1, 0)
    u = normalize(up - n_ * np.dot(up, n_))
    s_ = np.cross(u, n_)
    center = np.asarray(center, float)
    pts = []
    for i in range(10):
        a = math.pi / 2 + i * math.pi / 5
        rr = r if i % 2 == 0 else r * 0.42
        pts.append(center + s_ * rr * math.cos(a) + u * rr * math.sin(a))
    inner = np.array([pts[i] for i in range(1, 10, 2)])
    b.poly(orient(inner, n_), col, emis=emis, glow=True, dbl=True)
    for i in range(0, 10, 2):
        tri = np.array([pts[(i - 1) % 10], pts[i], pts[(i + 1) % 10]])
        b.poly(orient(tri, n_), col, emis=emis, glow=True, dbl=True)


# ================================================================ sombra comprida (com cartola)
SHADOW_PROFILE = [(0.0, 0.16), (0.46, 0.15), (0.50, 0.27), (0.72, 0.29), (0.735, 0.07), (0.77, 0.07), (0.78, 0.15),
                  (0.855, 0.15), (0.86, 0.33), (0.88, 0.33), (0.885, 0.18), (1.0, 0.18)]


def long_shadow(b, foot, tip, width=1.0, col=C(0.0, 0.0, 0.0), grow=1.0):
    foot = np.asarray(foot, float) * C(1, 0, 1)
    tip = np.asarray(tip, float) * C(1, 0, 1)
    d = tip - foot
    L = np.linalg.norm(d) * grow
    u = d / np.linalg.norm(d)
    v = C(-u[2], 0, u[0])
    y = C(0, 0.018, 0)
    for (a0, w0), (a1, w1) in zip(SHADOW_PROFILE[:-1], SHADOW_PROFILE[1:]):
        p0 = foot + u * a0 * L + y
        p1 = foot + u * a1 * L + y
        q = np.array([p0 - v * w0 * width, p0 + v * w0 * width, p1 + v * w1 * width, p1 - v * w1 * width])
        b.poly(orient(q, C(0, 1, 0)), col, emis=1.0, bucket=2.75)


# ================================================================ olhos avulsos (copas, janelas)
def floating_eye(b, center, normal, rx, ry, look=(0.0, 0.0), pupil=0.32, col=EYE_W, emis=1.0, fogmul=0.25):
    n_ = normalize(normal)
    up = C(0, 1, 0)
    u = normalize(up - n_ * np.dot(up, n_))
    s_ = np.cross(u, n_)
    center = np.asarray(center, float)
    if ry < 0.02 * rx:
        return
    b.poly(disc_poly(center, n_, up, rx, ry, n=6), col, emis=emis, glow=True, fogmul=fogmul)
    pr = rx * pupil
    pc = center + n_ * 0.01 + s_ * look[0] * (rx - pr) * 0.9 + u * look[1] * (ry - min(pr, ry)) * 0.9
    b.poly(disc_poly(pc, n_, up, pr, min(pr, ry * 0.95), n=6), PUPIL, emis=1.0, glow=True, fogmul=fogmul)


def look_vec(center, normal, target):
    """direção da pupila (x, y) no plano do olho para mirar `target`."""
    n_ = normalize(normal)
    up = C(0, 1, 0)
    u = normalize(up - n_ * np.dot(up, n_))
    s_ = np.cross(u, n_)
    d = normalize(np.asarray(target, float) - np.asarray(center, float))
    return (float(np.clip(np.dot(d, s_) * 1.6, -1, 1)), float(np.clip(np.dot(d, u) * 1.6, -1, 1)))


# ================================================================ árvores que se curvam
def axis_angle(axis, ang):
    k = normalize(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


class BendTree:
    def __init__(self, rng, base, big=1.0):
        bb = Builder()
        _tree(bb, rng, 0.0, 0.0, big=big)
        self.polys, self.attrs = bb.polys, bb.attrs
        self.base = np.asarray(base, float)

    def add(self, b, tilt, toward):
        d = np.asarray(toward, float) - self.base
        d[1] = 0
        d = normalize(d)
        R = axis_angle(np.cross(C(0, 1, 0), d), tilt)
        for P, A in zip(self.polys, self.attrs):
            b.poly(P @ R.T + self.base, A[0:3], emis=A[3], dbl=A[4] > 0.5, bucket=A[5], fogmul=A[8], glow=A[9] > 0.5)


# ================================================================ mundos extras
def build_void_world():
    """chão escuro infinito (o 'arquivo' do líder)."""
    rng = np.random.default_rng(4)
    b = Builder(group_base=3_000_000)
    for x in np.arange(-60, 60, 4.0):
        for z in np.arange(-60, 60, 4.0):
            col = C(0.16, 0.17, 0.21) * (1 + rng.normal(0, 0.06))
            q = np.array([C(x, 0, z), C(x + 4, 0, z), C(x + 4, 0, z + 4), C(x, 0, z + 4)])
            b.poly(orient(q, C(0, 1, 0)), np.clip(col, 0, 1), bucket=1.0)
    return b.build()


def build_clearing_world(seed=8):
    """clareira de grama; árvores só ao longe (as próximas são dinâmicas e se curvam)."""
    rng = np.random.default_rng(seed)
    b = Builder(group_base=4_000_000)
    n = 28
    radii = [0.0, 4.0, 9.0, 15.0, 24.0, 40.0, 70.0]
    for i in range(n):
        a0, a1 = 2 * np.pi * i / n, 2 * np.pi * (i + 1) / n
        for j in range(len(radii) - 1):
            r0, r1 = radii[j], radii[j + 1]
            col = np.clip(GRASS * (1 + rng.normal(0, 0.09)), 0, 1)
            if r0 == 0:
                q = np.array([C(0, 0, 0), C(r1 * math.cos(a0), 0, r1 * math.sin(a0)),
                              C(r1 * math.cos(a1), 0, r1 * math.sin(a1))])
            else:
                q = np.array([C(r0 * math.cos(a0), 0, r0 * math.sin(a0)), C(r1 * math.cos(a0), 0, r1 * math.sin(a0)),
                              C(r1 * math.cos(a1), 0, r1 * math.sin(a1)), C(r0 * math.cos(a1), 0, r0 * math.sin(a1))])
            b.poly(orient(q, C(0, 1, 0)), col, bucket=1.0)
    for k in range(40):
        a = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(19, 45)
        _tree(b, rng, r * math.cos(a), r * math.sin(a), big=rng.uniform(1.0, 1.5))
    return b.build()


_brng = np.random.default_rng(12)
BEND_TREES = []
for _k in range(12):
    _a = 2 * np.pi * _k / 12 + _brng.uniform(-0.12, 0.12)
    _r = _brng.uniform(8.0, 11.5)
    BEND_TREES.append(BendTree(_brng, C(_r * math.cos(_a), 0, _r * math.sin(_a)), big=_brng.uniform(1.0, 1.3)))
