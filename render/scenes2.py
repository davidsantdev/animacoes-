"""Parte 2 — decupagem sincronizada com parte2_audio.wav (Arthur + o líder)."""
import json
import math
import os

import numpy as np

from engine import Camera, Builder, rot_y, normalize
from models import (C, build_cyclist, build_watcher, cyclist_head_world, watcher_head_world, yaw_to,
                    add_sky_objects, add_sky_eyes, add_lamps, dir_from, LAMP_ZS, HEAD_P, SKIN_P, WHITE,
                    TREE_CROWNS, WINDOWS, box_polys, frustum_polys)
from models2 import (build_cat, build_hand, build_gem, build_coin, build_star, long_shadow, floating_eye, look_vec,
                     BEND_TREES)
import scenes as P1

HERE = os.path.dirname(os.path.abspath(__file__))
META = json.load(open(os.path.join(HERE, "parte2_words.json"), encoding="utf-8"))
FPS = 30
DUR = META["duration"]
AUDIO = "parte2_audio.wav"
OUT = "parte2_reels.mp4"
WORLDS = ("road", "ring", "void", "clearing")

ss, lerp, lamps_off = P1.ss, P1.lerp, P1.lamps_off
CX = -0.8

# ---------------------------------------------------------------- movimento do Arthur
_T = np.arange(0, 100, 0.005)
_V = np.full_like(_T, 3.0)
_m = _T >= 42.9
_V[_m] = np.clip(3.0 * (1 - (_T[_m] - 42.9) / 0.6), 0, 3)
_m = _T >= 87.5
_V[_m] = np.minimum(1.6, (_T[_m] - 87.5) * 1.1)
_Z = 20.0 + np.cumsum(_V) * 0.005


def cz(t):
    return float(np.interp(t, _T, _Z))


def Cpos(t):
    return C(CX, 0, cz(t))


def head_of(t):
    return cyclist_head_world(Cpos(t), 0.0)


# ---------------------------------------------------------------- fala -> boca
_WA = np.array([(w["s"], w["e"]) for w in META["words"] if w["spk"] == "A"])
_WL = np.array([(w["s"], w["e"]) for w in META["words"] if w["spk"] == "L"])


def talk(t, spk="L", rate=4.5):
    W = _WL if spk == "L" else _WA
    i = int(np.searchsorted(W[:, 0], t)) - 1
    if i < 0 or t > W[i, 1] + 0.02:
        return 0.0
    return 0.35 + 0.65 * abs(math.sin(math.pi * rate * (t - W[i, 0])))


def typed(text, t0, t, cps=22.0):
    if t < t0:
        return ""
    return text[:int((t - t0) * cps) + 1]


class St(P1.St):
    def __init__(self):
        super().__init__()
        self.osd = []
        self.ripple = 0.0
        self.env["lamp_fog"] = 0.3


def put_arthur(st, t, brow=None, **p):
    if brow is not None:
        p.setdefault("head_extra", {})["brow"] = brow
    st.lights += build_cyclist(st.b, Cpos(t), 0.0, cz(t), p)


def fill_light(st, hc, k=0.4):
    st.lights.append((hc + C(0.3, -0.5, 0.7), C(1.0, 0.85, 0.6), k, 1.0))


GRAY_COAT = C(0.30, 0.30, 0.33)
GRAY_SKIN = C(0.50, 0.50, 0.53)


def arthur_standing(b, pos, yaw, **p):
    q = dict(coat=WHITE, skin=SKIN_P, head=HEAD_P, hat=False, glow=False, scale=0.68, finger=0.13,
             shoe=C(0.12, 0.08, 0.07), shirt=C(0.10, 0.10, 0.16), head_scale=1.35)
    q.update(p)
    build_watcher(b, pos, yaw, q)


def mediocre(b, pos, yaw, **p):
    q = dict(coat=GRAY_COAT, skin=GRAY_SKIN, head=HEAD_P, hat=False, glow=False, scale=0.68, finger=0.13,
             shoe=C(0.1, 0.1, 0.11), shirt=C(0.22, 0.22, 0.25), head_scale=1.35, eye_open=0.0)
    q.update(p)
    build_watcher(b, pos, yaw, q)


def osd_play(st, t, t0, t1):
    if t0 <= t < t1:
        st.osd.append(dict(kind="play", x=26, y=112, size=30))


def moon_towards(cp, pt):
    v = np.asarray(pt) - np.asarray(cp)
    return math.degrees(math.atan2(v[0], v[2])), math.degrees(math.atan2(v[1], math.hypot(v[0], v[2])))


# ---------------------------------------------------------------- planos
def s01_road(t, lt, st):
    e = st.env
    e.update(moon_az=4, moon_el=6)
    P = Cpos(t)
    k = ss(0.25, 3.5, t)
    d = lerp(16.0, 3.0, k)
    tgt = P + C(0.0, 0.95, 0)
    fov = 2 * math.degrees(math.atan(2.1 / d))
    st.cam = Camera(tgt + C(0.35, 0.55 + 0.03 * d, -d), tgt, fov)
    put_arthur(st, t, brow=0.25, smile=-0.3)
    st.static = 1.0 if t < 0.22 else 0.0
    st.fade = ss(0.22, 0.6, t)
    if 0.3 <= t < 3.3:
        st.osd.append(dict(kind="text", text="PARTE 2", x=26, y=112, size=30))


def s02_annoyed(t, lt, st):
    st.env.update(moon_az=200, moon_el=10)
    hc = head_of(t)
    st.cam = Camera(hc + C(-0.5, 0.02, 1.15), hc + C(0, -0.03, 0), fov=34)
    sigh = ss(4.9, 5.4, t) * (1 - ss(5.8, 6.4, t))
    shake = 0.12 * math.sin(t * 7) * ss(4.9, 5.1, t) * (1 - ss(5.6, 5.9, t))
    put_arthur(st, t, brow=0.6, head_pitch=-0.12 * sigh, head_yaw=shake, eye_open=0.72 - 0.2 * sigh, smile=-0.6,
               px=-0.3)
    fill_light(st, hc)


def s03_shout(t, lt, st):
    st.env.update(moon_az=200, moon_el=10)
    hc = head_of(t)
    mo = talk(t, "A", rate=5.0)
    sh = (0.015 * math.sin(t * 53) + 0.01 * math.sin(t * 71)) * (mo > 0)
    st.cam = Camera(hc + C(0.35 + sh, -0.45, 1.15), hc + C(sh, 0.02, 0), fov=40)
    put_arthur(st, t, brow=0.95, head_pitch=-0.22, head_yaw=0.18, mouth_open=mo * 1.1, eye_open=1.08, pupil=0.75,
               smile=-0.2)
    fill_light(st, hc, 0.45)


CAT0 = C(-3.05, 0, 72.5)


def s04_cat(t, lt, st):
    st.env.update(moon_az=60, moon_el=12)
    u = max(0.0, t - 9.9)
    camp = CAT0 + C(0.15 - 1.2 * u, 0.32, 2.3)
    st.cam = Camera(camp, CAT0 + C(0.05 - 1.4 * u, 0.24, 0), fov=36)
    if t < 9.9:
        build_cat(st.b, CAT0, math.pi / 2, dict(arch=ss(9.0, 9.2, t), hiss=ss(9.08, 9.25, t),
                                                head_yaw=-math.pi / 2 * 0.85, head_pitch=-0.1))
    else:
        pos = CAT0 + C(-1.0, 0, 0.0) * 2.6 * u
        build_cat(st.b, pos, -math.pi / 2, dict(run=u * 20.0))


def s05_far(t, lt, st):
    P = Cpos(t)

    def fn(i, lp, tt):
        if lp[2] < P[2] + 6:
            return 1.0
        if abs(lp[2] - (P[2] + 42)) < 16 and 13.0 < tt < 13.55:
            return float(int(tt * 25) % 2)
        return 0.0
    st.env.update(lamp_fn=fn, fogdens=0.055, moon_az=8, moon_el=9)
    st.cam = Camera(P + C(0.95, 2.1, -3.2), P + C(-0.2, 1.3, 25), fov=46)
    put_arthur(st, t, head_pitch=-0.05)


def sound_rings(st, t, center, t0, speed=14.0, n=6, gap=0.5, width=0.45, maxr=80.0):
    up = C(0, 1, 0)
    seg = 40
    for k in range(n):
        r = (t - t0 - k * gap) * speed
        if r <= 0.3 or r > maxr:
            continue
        col = C(0.70, 0.84, 1.0) * min(1.0, 1.4 * (1 - r / maxr)) * (1.0 - 0.1 * k)
        for i in range(seg):
            a0, a1 = 2 * math.pi * i / seg, 2 * math.pi * (i + 1) / seg
            q = np.array([center + C(r * math.cos(a0), 0.025, r * math.sin(a0)),
                          center + C((r + width) * math.cos(a0), 0.025, (r + width) * math.sin(a0)),
                          center + C((r + width) * math.cos(a1), 0.025, (r + width) * math.sin(a1)),
                          center + C(r * math.cos(a1), 0.025, r * math.sin(a1))])
            st.b.poly(q if np.cross(q[1] - q[0], q[2] - q[1])[1] > 0 else q[::-1], col, emis=1.0, bucket=2.6,
                      fogmul=0.35, glow=True)


def s06_wave(t, lt, st):
    P = Cpos(t)
    st.env.update(lamp_fn=lamps_off, fogdens=0.045, moon_az=8, moon_el=9)
    center = C(CX + 0.3, 0, cz(14.4) + 62)
    sound_rings(st, t, center, 14.4)
    st.cam = Camera(P + C(1.0, 5.2, -4.8), P + C(-0.2, 0.0, 13), fov=54)
    put_arthur(st, t, head_pitch=-0.05)
    hit = 14.4 + 62 / 14.0
    st.ripple = 0.25 + 0.9 * ss(hit - 0.3, hit, t) * (1 - ss(hit + 0.2, hit + 1.2, t))


def s07_knows(t, lt, st):
    st.env.update(moon_az=200, moon_el=10, lamp_fn=lamps_off)
    hc = head_of(t)
    st.cam = Camera(hc + C(0.5, 0.06, 1.1), hc + C(0, -0.03, 0), fov=34)
    dart = math.sin(t * 5.0) * 0.6 * (1 - ss(21.5, 21.7, t))
    wide = ss(21.6, 21.85, t)
    put_arthur(st, t, brow=-0.35, px=dart, eye_open=0.95 + 0.2 * wide, pupil=1.0 - 0.35 * wide, smile=-0.4)
    fill_light(st, hc)


def s08_play(t, lt, st):
    e = st.env
    P = Cpos(t)
    e.update(lamp_fn=lamps_off, fogdens=lerp(0.075, 0.05, ss(25.5, 27.2, t)), stars=0.4, clouds=0.6, moon=0.0)
    bob = 0.015 * math.sin(cz(t) / 0.34 / 2.3 * 2)
    cp = P + C(0, 1.70 + bob, 0.15)
    W = C(CX + 0.3, 0, cz(t) + 17)
    st.cam = Camera(cp, W + C(0, 2.15, 0), fov=17)
    build_watcher(st.b, W, math.pi, dict(eye_open=ss(23.95, 24.25, t), fogmul=lerp(5.0, 1.0, ss(25.6, 27.1, t)),
                                         mouth_open=talk(t) * 0.3, grin=0.4, pose="behind"))
    st.lights.append((W + C(-0.6, 3.6, -1.5), C(0.85, 0.9, 1.0), 1.2, 3.0))
    if t < 23.3:
        st.fade = 0.0
    else:
        st.fade = ss(23.3, 24.4, t)
    st.static = 0.25 * (1 - ss(22.45, 22.8, t))
    osd_play(st, t, 22.5, 25.2)


W_L1 = C(CX + 0.3, 0, cz(27.35) + 17)


def s09_certeza(t, lt, st):
    st.env.update(lamp_fn=lamps_off, fogdens=0.04, moon_az=30, moon_el=17)
    W = W_L1
    st.cam = Camera(W + C(-1.7, 1.25, -3.7), W + C(0, 2.0, 0), fov=44)
    build_watcher(st.b, W, yaw_to(W, W + C(-1.7, 0, -3.7)), dict(pose="behind", head_roll=0.14, grin=0.55,
                                                                   mouth_open=talk(t) * 0.3))
    st.lights.append((W + C(-0.8, 4.0, -1.0), C(0.85, 0.9, 1.0), 1.6, 3.2))


def s10_rousseau(t, lt, st):
    st.env.update(lamp_fn=lamps_off, fogdens=0.04, moon_az=30, moon_el=17)
    W = W_L1
    st.cam = Camera(W + C(-1.15, 1.95, -2.45), W + C(0, 2.38, 0), fov=40)
    build_watcher(st.b, W, yaw_to(W, W + C(-1.7, 0, -3.7)),
                  dict(pose_from="behind", pose="lecture", pose_k=ss(29.44, 29.95, t), head_roll=0.1, grin=0.6,
                       mouth_open=talk(t) * 0.3, head_pitch=-0.05))
    st.lights.append((W + C(-0.8, 4.0, -1.0), C(0.85, 0.9, 1.0), 1.6, 3.2))
    if t >= 30.35:
        st.osd.append(dict(kind="text", text=typed("J.-J. ROUSSEAU", 30.35, t), x=26, y=112, size=24))
    if t >= 30.9:
        st.osd.append(dict(kind="text", text=typed("EMÍLIO, 1762", 30.9, t), x=26, y=144, size=24))


G = C(0.0, 60.0, 300.0)
HAND_COL = C(0.92, 0.88, 0.80)


def s11_author(t, lt, st):
    e = st.env
    e.update(lamp_fn=lamps_off, fogdens=0.02, moon_az=150, moon_el=25, stars=1.0)
    cp = G + rot_y(0.10 * lt) @ C(1.9, 2.0, 1.4)
    st.cam = Camera(cp, G + C(0, -0.15, -0.05), fov=46)
    for sx, hd in ((1, 1), (-1, -1)):
        w = G + C(sx * 0.30, -0.33, -0.55)
        build_hand(st.b, w, C(0, 0.15, 1), C(0, 1, 0), 6.0, HAND_COL, curl=0.75, handed=hd, emis=0.22)
        st.b.mesh(frustum_polys(w, w + C(sx * 0.15, -1.2, -3.0), 0.42, 0.26, 0.6, 0.45), HAND_COL * 0.9, emis=0.08)
    ap = ss(31.95, 32.3, t)
    build_gem(st.b, G + C(0, -0.02, -0.1), 0.22 * ap, t * 0.8)
    st.lights.append((G + C(0, 0.05, -0.1), C(1.0, 0.8, 0.45), 2.2 * ap, 1.0))
    st.lights.append((G + C(1.5, 3.0, 1.0), C(0.75, 0.8, 1.0), 1.0, 3.0))
    st.exposure = 1.12 + 1.2 * (1 - ss(31.95, 32.35, t))


def s12_degenerate(t, lt, st):
    e = st.env
    e.update(lamp_fn=None, fogdens=0.045, moon_az=40, moon_el=12)
    P = Cpos(t)
    wr = P + C(0.25, 1.15, 0.55)
    fwd = C(0, 0.1, 1)
    build_hand(st.b, wr, fwd, C(0, 1, 0), 1.0, SKIN_P, curl=0.45, handed=1)
    st.b.mesh(frustum_polys(wr, wr + C(0.05, -0.08, -0.4), 0.085, 0.07, 0.1, 0.09), WHITE)
    drop = 0.35 * (1 - ss(35.78, 36.0, t))
    deg = ss(36.3, 38.0, t)
    gc = wr + normalize(fwd) * 0.055 + C(0, 0.07 + drop, 0)
    build_gem(st.b, gc, 0.042, t * 0.8, deg=deg)
    st.lights.append((gc + C(0, 0.02, 0), C(1.0, 0.8, 0.45), 1.2 * (1 - deg), 0.25))
    st.cam = Camera(wr + C(0.32, 0.34, 0.24), wr + C(0, 0.03, 0.07), fov=42)
    st.lights.append((wr + C(0.3, 0.6, 0.2), C(0.75, 0.8, 1.0), 0.5, 1.0))
    st.glitch = max(st.glitch, 0.2 * deg)


def s13_proof(t, lt, st):
    e = st.env
    e.update(lamp_fn=lamps_off, fogdens=0.04, moon_az=8, moon_el=9)
    P = Cpos(t)
    W = C(CX + 0.3, 0, cz(t) + 11)
    cp = P + C(0, 1.70, 0.15)
    k = ss(39.5, 41.2, t)
    st.cam = Camera(cp, W + C(-0.05, 2.2, 0), fov=lerp(24, 15, k))
    build_watcher(st.b, W, yaw_to(W, cp), dict(pose_from="behind", pose="point", pose_k=ss(38.7, 39.2, t),
                                               point_target=cp, grin=0.7, mouth_open=talk(t) * 0.3,
                                               eye_emis=0.82 + 0.5 * ss(40.6, 40.8, t)))
    st.lights.append((W + C(-0.6, 3.6, -1.5), C(0.85, 0.9, 1.0), 1.4, 3.0))
    st.lights.append((P + C(0, 0.85, 2.4), C(1.0, 0.95, 0.85), 0.7, 2.3))


def s14_legs(t, lt, st):
    st.env.update(lamp_fn=None, moon_az=90, moon_el=14)
    P = Cpos(t)
    sh = 0.02 * math.sin(t * 60) * (ss(43.05, 43.12, t) - ss(43.3, 43.5, t))
    st.cam = Camera(P + C(-1.35, 0.6 + sh, 0.25), P + C(0, 0.55, -0.05), fov=44)
    put_arthur(st, t)
    st.tint = (0.85, 0.9, 1.0, 0.35 * ss(43.0, 43.6, t))


def _lamp_near(z):
    return int(np.argmin(np.abs(LAMP_ZS - z)))


def s15_shadow(t, lt, st):
    P = Cpos(t)
    W = P + C(1.1, 0, 22)
    st.env.update(lamp_fn=None, fogdens=0.022, moon_az=10, moon_el=12)
    long_shadow(st.b, W, P + C(0, 0, 0.6), width=1.7, grow=ss(44.1, 44.7, t))
    build_watcher(st.b, W, math.pi, dict(pose="behind", grin=0.5))
    st.lights.append((W + C(0.0, 3.2, 2.5), C(0.85, 0.9, 1.0), 1.6, 3.0))
    st.cam = Camera(P + C(1.6, 6.0, -5.5), P + C(0, 0.3, 11), fov=56)
    put_arthur(st, t, head_pitch=-0.1)


def s16_calc(t, lt, st):
    P = Cpos(t)
    W = P + C(1.1, 0, 22)
    Li = _lamp_near(W[2] + 4)
    st.env.update(lamp_fn=lambda i, lp, tt: 1.0 if i == Li else 0.0, fogdens=0.02, moon_az=190, moon_el=10)
    hw = watcher_head_world(W, math.pi)
    tgt = P + C(0, 1.0, 0)
    st.cam = Camera(hw + C(0, 0, -0.3), tgt, fov=11)
    put_arthur(st, t, brow=-0.4, eye_open=0.9, py=0.15, smile=-0.5)
    fill_light(st, head_of(t), 0.35)
    st.tint = (0.55, 0.95, 1.0, 0.92)
    st.exposure = 1.3
    pts = [P + C(sx * 0.45, y, sz * 0.7) for sx in (-1, 1) for y in (0.0, 2.0) for sz in (-1, 1)]
    pr = st.cam.project(pts, 540, 960)
    x0, x1 = pr[:, 0].min() - 18, pr[:, 0].max() + 18
    y0, y1 = pr[:, 1].min() - 18, pr[:, 1].max() + 18
    hud = (190, 255, 255)
    st.osd.append(dict(kind="brackets", box=(x0, y0, x1, y1), col=hud, len=26, w=3))
    sy = y0 + (y1 - y0) * ((t * 0.8) % 1.0)
    st.osd.append(dict(kind="hline", y=sy, x0=x0, x1=x1, col=(150, 255, 255), w=2))
    lines = [(46.2, "ANALISANDO..."), (46.85, "ALVO: ARTHUR"), (47.45, "PULSO: 128 BPM"), (48.05, "MEDO: 73%"),
             (48.65, "RESULTADO: PREVISÍVEL")]
    for k, (t0, tx) in enumerate(lines):
        if t >= t0:
            st.osd.append(dict(kind="text", text=typed(tx, t0, t, 30), x=26, y=112 + 30 * k, size=22, col=hud))


def s17_eyes(t, lt, st):
    P = Cpos(t)
    W = P + C(1.1, 0, 22)
    st.env.update(lamp_fn=lamps_off, fogdens=0.03, moon=0.0)
    hw = watcher_head_world(W, math.pi)
    st.cam = Camera(hw + C(0, -0.05, -1.0), hw + C(0, 0.0, 0), fov=30)
    op = 1.0 - (ss(50.0, 50.08, t) - ss(50.16, 50.26, t))
    build_watcher(st.b, W, math.pi, dict(eye_open=op, grin=0.6))
    st.lights.append((hw + C(0.3, 0.6, -0.8), C(0.85, 0.9, 1.0), 0.9, 1.4))
    st.tint = (0.55, 0.95, 1.0, 0.92 * (1 - ss(49.9, 50.6, t)))


def s18_closer(t, lt, st):
    P = Cpos(t)
    k = ss(50.7, 53.6, t)
    W = C(CX + 0.3, 0, P[2] + lerp(24.0, 6.5, k))
    Li = _lamp_near(P[2] + 4)

    def fn(i, lp, tt):
        if i != Li:
            return 0.0
        return 0.25 + 0.75 * float((int(tt * 13) % 5) != 0)
    st.env.update(lamp_fn=fn, fogdens=0.045, moon_az=10, moon_el=10)
    build_watcher(st.b, W, math.pi, dict(walk=(t - 50.65) * 5.0 if k < 0.985 else None, grin=0.5))
    st.cam = Camera(P + C(0.85, 1.85, -3.0), P + C(0.1, 1.7, 9), fov=46)
    put_arthur(st, t, head_pitch=-0.06)
    st.lights.append((W + C(-0.6, 3.6, -1.5), C(0.85, 0.9, 1.0), 1.0, 3.0))


def s19_patient(t, lt, st):
    P = Cpos(t)
    W = C(CX + 0.3, 0, P[2] + 6.5)
    st.env.update(lamp_fn=lamps_off, fogdens=0.045, moon_az=20, moon_el=24)
    cp = P + C(-0.95, 1.65, 0.7)
    st.cam = Camera(cp, W + C(0, 2.15, 0), fov=34)
    build_watcher(st.b, W, yaw_to(W, P + C(0, 1.7, 0)), dict(pose="behind", head_roll=0.12 * math.sin(t * 0.8),
                                                             lean=0.03 * math.sin(t * 1.3), grin=0.6,
                                                             mouth_open=talk(t) * 0.3))
    st.lights.append((W + C(-0.8, 3.5, -1.5), C(0.85, 0.9, 1.0), 1.2, 3.0))
    osd_play(st, t, 55.85, 58.3)


def s20_dossier(t, lt, st, rew=False):
    st.env.update(lamp_fn=lamps_off, moon_az=200, moon_el=10, fogdens=0.04)
    hc = head_of(t)
    st.cam = Camera(hc + C(0, 0.02, 1.55), hc + C(0, -0.10, 0), fov=30)
    nod = ss(59.2, 59.6, t)
    put_arthur(st, t, brow=-0.35, eye_open=0.8, smile=-0.35, head_pitch=0.05 * nod, py=-0.1 * nod)
    fill_light(st, hc, 0.35)
    st.tint = (0.85, 0.9, 1.0, 0.55)
    if not rew:
        osd_play(st, t, 55.85, 58.3)
        if int(t * 2) % 2 == 0:
            st.osd.append(dict(kind="rec", x=404, y=112, size=26))
        for t0, tx, k in ((56.5, "NOME: ARTHUR CASTRO", 0), (57.75, "IDADE: 19", 1), (59.3, "REFEIÇÕES HOJE: 0", 2)):
            if t >= t0:
                st.osd.append(dict(kind="text", text=typed(tx, t0, t, 26), x=26, y=158 + 30 * k, size=22))


def s21_memory(t, lt, st):
    e = st.env
    if t < 61.15:   # rebobinando
        s20_dossier(60.8 - (t - 60.8) * 6.0, 0, st, rew=True)
        st.glitch = 1.0
        st.osd.append(dict(kind="rew", x=26, y=112, size=30))
        return
    e.update(zenith=C(0.30, 0.55, 0.95), fogcol=C(0.70, 0.82, 0.95), ambient=C(0.50, 0.52, 0.55),
             moon_col=C(0.75, 0.72, 0.62), moon_az=200, moon_el=40, fogdens=0.012, stars=0.0, clouds=1.0,
             cloud_col=C(0.95, 0.95, 0.97), moon=0.0, lamp_fn=lamps_off)
    Pm = C(CX, 0, 260 + 6.0 * (t - 61.15))
    st.cam = Camera(Pm + C(-2.6, 1.3, 1.2), Pm + C(0, 1.2, 0.4), fov=44)
    build_cyclist(st.b, Pm, 0.0, Pm[2], dict(head_pitch=0.15, eye_open=0.85, smile=-0.2, lamp=0.0,
                                             head_extra=dict(brow=0.35)))
    st.tint = (1.0, 0.80, 0.56, 0.75)
    st.osd.append(dict(kind="text", text="07:42", x=514, y=112, size=26, align="right"))


LINE_XS = [-5.4, -4.05, -2.7, -1.35, 0.0, 1.35, 2.7, 4.05, 5.4]
COIN_REST = C(-0.12, 0.93, 0.40)


def _void_env(e):
    e.update(fogdens=0.045, fogcol=C(0.02, 0.025, 0.035), zenith=C(0.01, 0.012, 0.02), ambient=C(0.09, 0.095, 0.11),
             moon_col=C(0.08, 0.09, 0.12), stars=0.0, clouds=0.0, moon=0.0, lamp_fn=lamps_off)


def s22_line(t, lt, st):
    st.world = "void"
    _void_env(st.env)
    palm = ss(65.95, 66.6, t)
    for x in LINE_XS:
        pos = C(x, 0, 0)
        kw = dict(pose="palm" if palm > 0 else None, pose_k=palm)
        if x == 0.0:
            arthur_standing(st.b, pos, 0.0, py=0.35 - 0.8 * ss(67.6, 67.95, t), **kw)
        else:
            mediocre(st.b, pos, 0.0, **kw)
        st.lights.append((C(x, 3.0, 0.9), C(0.85, 0.9, 1.0), 1.9, 1.7))
        tc = 67.11
        if t >= tc:
            y = 4.0 - 4.9 * (t - tc) ** 2
            rest = pos + COIN_REST
            if y <= rest[1]:
                bt = t - (tc + math.sqrt((4.0 - rest[1]) / 4.9))
                y = rest[1] + 0.06 * abs(math.sin(bt * 9)) * math.exp(-bt * 5)
            build_coin(st.b, C(rest[0], y, rest[2]), t * 9.0 + x, 0.05)
    star = ss(64.9, 65.2, t) * (1 - ss(69.75, 70.15, t))
    if star > 0.02:
        fl = 1.0 if t < 69.75 else float(int(t * 20) % 2)
        build_star(st.b, C(0, 2.28, 0.05), C(0, 0, 1), 0.16 * star, emis=0.9 * fl)
        st.lights.append((C(0, 2.3, 0.3), C(1.0, 0.85, 0.4), 0.8 * star, 0.8))
    if t < 65.9:
        st.cam = Camera(C(1.0, 1.6, 6.6 - 0.4 * lt), C(0, 1.3, 0), fov=50)
    else:
        xc = lerp(-4.6, 0.0, ss(65.95, 70.0, t))
        st.cam = Camera(C(xc + 2.6, 1.55, 4.6), C(xc - 0.4, 1.05, 0), fov=50)


def s24_miserable(t, lt, st):
    st.env.update(lamp_fn=lamps_off, moon_az=200, moon_el=10, fogdens=0.045)
    hc = head_of(t)
    k = ss(70.75, 73.6, t)
    st.cam = Camera(hc + C(0.18, -0.22, lerp(1.0, 0.75, k)), hc + C(0, 0.0, 0), fov=36)
    put_arthur(st, t, brow=1.0, eye_open=0.62, pupil=0.85, smile=-0.85, mouth_open=talk(t, "A") * 0.35,
               head_pitch=0.08)
    st.lights.append((hc + C(0.8, 0.1, 0.4), C(1.0, 0.15, 0.1), 0.6, 1.2))
    fill_light(st, hc, 0.25)


def _world_eyes(st, t, P, t_open, spread=0.06):
    tgt = P + C(0, 1.76, 0)
    crowns = [(c, r) for c, r in TREE_CROWNS if P[2] - 12 < c[2] < P[2] + 60]
    crowns.sort(key=lambda cr: np.linalg.norm(cr[0] - tgt))
    for k, (c, r) in enumerate(crowns):
        o = ss(t_open + spread * k, t_open + spread * k + 0.18, t)
        if o <= 0.02:
            continue
        n = normalize((tgt - c) * C(1, 0.4, 1))
        ctr = c + n * r * 1.02
        s_ = normalize(np.cross(C(0, 1, 0), n))
        for sd in (-1, 1):
            ec = ctr + s_ * sd * 0.36
            floating_eye(st.b, ec, n, 0.22, 0.14 * o, look=look_vec(ec, n, tgt), pupil=0.38)
    for k, (wc, wn) in enumerate(w for w in WINDOWS if P[2] - 15 < w[0][2] < P[2] + 70):
        o = ss(t_open + 0.3 + 0.1 * k, t_open + 0.5 + 0.1 * k, t)
        if o > 0.02:
            ec = wc + wn * 0.05
            floating_eye(st.b, ec, wn, 0.48, 0.30 * o, look=look_vec(ec, wn, tgt), pupil=0.3)


def s25_knowledge(t, lt, st):
    P = Cpos(t)
    st.env.update(lamp_fn=lamps_off, fogdens=0.03, moon_az=10, moon_el=14)
    W = C(CX + 0.3, 0, P[2] + 6.5)
    build_watcher(st.b, W, yaw_to(W, P), dict(pose="behind", grin=0.7, mouth_open=talk(t) * 0.3))
    put_arthur(st, t, head_pitch=-0.1)
    st.cam = Camera(P + C(1.0, 5.0, -7.0), P + C(0, 1.6, 12), fov=56)
    _world_eyes(st, t, P, 74.17)
    st.lights.append((W + C(-0.6, 3.6, -1.5), C(0.85, 0.9, 1.0), 1.0, 3.0))
    osd_play(st, t, 73.65, 76.0)


def s26_consent(t, lt, st):
    P = Cpos(t)
    e = st.env
    e.update(lamp_fn=lamps_off, fogdens=0.03, sky_eyes=(76.7, 78.8), sky_eye_target=P + C(0, 1.7, 0),
             sky_view=(3, 47, 30, 26), moon_az=8, moon_el=55, moon_size=1.8, moon_eye=ss(77.9, 78.5, t))
    k = ss(76.5, 78.1, t)
    pitch = lerp(0.12, 1.0, k)
    cp = P + C(0.7, 1.3, -2.6)
    st.cam = Camera.from_dir(cp, C(0.05, math.sin(pitch), math.cos(pitch)), fov=60)
    W = C(CX + 0.3, 0, P[2] + 6.5)
    build_watcher(st.b, W, yaw_to(W, P), dict(pose="behind", grin=0.7, mouth_open=talk(t) * 0.3))
    put_arthur(st, t, head_pitch=-0.4 * k, py=0.6 * k)
    _world_eyes(st, t, P, 0.0)


CROWD = [(x + 0.7 * (row % 2), z) for row, z in enumerate((-3.6, -5.6, -7.6)) for x in np.arange(-4.2, 4.3, 1.4)]


def _crowd(st, t, raise_t=81.65):
    r = ss(raise_t, raise_t + 0.35, t)
    for x, z in CROWD:
        mediocre(st.b, C(x, 0, z), 0.0, head_pitch=lerp(0.65, -0.05, r), lean=lerp(0.18, 0.0, r),
                 eye_open=ss(raise_t + 0.05, raise_t + 0.3, t), pupil=0.45, head_extra=dict(eye_emis=1.0, glow=True))


def s27_leader(t, lt, st):
    st.world = "clearing"
    e = st.env
    e.update(lamp_fn=lamps_off, fogdens=0.03, moon_size=2.3, moon_eye=1.0, clouds=0.6)
    W = C(0, 0, 0)
    cp = C(0.3, 0.45, 3.4 - 0.25 * lt)
    st.cam = Camera(cp, W + C(0, 2.2, 0), fov=58)
    hw = watcher_head_world(W, 0.0)
    az, el = moon_towards(cp, hw + C(0, 0.35, 0))
    e["moon_az"], e["moon_el"] = az, el + 3.5
    build_watcher(st.b, W, 0.0, dict(pose="open", pose_k=ss(79.37, 80.3, t), grin=0.8, mouth_open=talk(t) * 0.3,
                                     eye_emis=1.0, head_pitch=-0.08))
    _crowd(st, t)
    st.lights.append((C(0.2, 1.6, 1.6), C(0.8, 0.85, 1.0), 1.4, 2.2))


def _tape_warp(t, t0=84.85, d=0.6):
    if t <= t0:
        return t
    u = min((t - t0) / d, 1.0)
    return t0 + d * (u - u * u / 2)


def s28_nature(t, lt, st):
    st.world = "clearing"
    e = st.env
    te = _tape_warp(t)
    e.update(lamp_fn=lamps_off, fogdens=0.025, moon_az=200, moon_el=30, moon_size=2.0, moon_eye=1.0, clouds=0.6)
    tilt = 0.72 * ss(82.6, 84.6, te)
    a = 0.22 * (te - 82.4)
    cp = rot_y(a) @ C(0, 9.5, 15.5)
    st.cam = Camera(cp, C(0, 1.2, 0), fov=54)
    for tr in BEND_TREES:
        tr.add(st.b, tilt, C(0, 0, 0))
    build_watcher(st.b, C(0, 0, 0), yaw_to(C(0, 0, 0), cp), dict(pose="open", grin=0.9, head_pitch=-0.25,
                                                                 mouth_open=talk(t) * 0.3, eye_emis=1.0))
    _crowd(st, te, raise_t=0.0)
    st.lights.append((C(0, 4.0, 2.0), C(0.8, 0.85, 1.0), 1.2, 4.0))
    u = ss(84.85, 85.45, t)
    st.glitch = max(st.glitch, 0.7 * u)
    st.fade = 1 - 0.65 * u


def s29_stop(t, lt, st):
    s28_nature(85.44, 0, st)
    st.glitch = 0.0
    st.fade = 0.0
    st.static = 0.07
    st.osd = [dict(kind="stop", x=26, y=112, size=30)]


def s30_inevitable(t, lt, st):
    P = Cpos(t)
    st.env.update(lamp_fn=lamps_off, fogdens=0.07, moon=0.0, stars=0.5)
    W = C(CX + 0.3, 0, cz(86.75) + 14)
    st.cam = Camera(P + C(0.45, 1.15, -3.3), P + C(0, 1.4, 12), fov=46)
    build_watcher(st.b, W, math.pi, dict(fogmul=4.0, grin=0.5))
    put_arthur(st, t, head_pitch=0.05)
    st.fade = ss(86.75, 87.15, t) * (1 - ss(89.35, 89.75, t))


def s31_outro(t, lt, st):
    e = st.env
    e.update(fogdens=0.11, stars=0.3, clouds=0.4, moon=0.0, lamp_fn=lamps_off,
             fogcol=C(0.03, 0.045, 0.075), zenith=C(0.006, 0.008, 0.02))
    W = C(3.6, 0, 15.5)
    st.cam = Camera((0.2, 1.6, 4.0 + lt * 1.0), W + C(-0.1, 2.5, 0), fov=9)
    op = 1.0 - ss(91.2, 91.37, t)
    build_watcher(st.b, W, yaw_to(W, st.cam.pos), dict(eye_open=op, py=-0.15, grin=0.6))
    st.fade = ss(89.75, 90.0, t)
    st.static = ss(91.65, 91.8, t)


SHOTS = [
    (0.00, 3.55, s01_road),
    (3.55, 6.55, s02_annoyed),
    (6.55, 9.00, s03_shout),
    (9.00, 10.30, s04_cat),
    (10.30, 14.30, s05_far),
    (14.30, 19.20, s06_wave),
    (19.20, 22.45, s07_knows),
    (22.45, 27.35, s08_play),
    (27.35, 29.30, s09_certeza),
    (29.30, 31.95, s10_rousseau),
    (31.95, 35.75, s11_author),
    (35.75, 38.55, s12_degenerate),
    (38.55, 41.60, s13_proof),
    (41.60, 44.10, s14_legs),
    (44.10, 46.05, s15_shadow),
    (46.05, 49.25, s16_calc),
    (49.25, 50.65, s17_eyes),
    (50.65, 53.75, s18_closer),
    (53.75, 56.30, s19_patient),
    (56.30, 60.80, s20_dossier),
    (60.80, 64.00, s21_memory),
    (64.00, 70.75, s22_line),
    (70.75, 73.60, s24_miserable),
    (73.60, 76.50, s25_knowledge),
    (76.50, 79.25, s26_consent),
    (79.25, 82.40, s27_leader),
    (82.40, 85.45, s28_nature),
    (85.45, 86.75, s29_stop),
    (86.75, 89.75, s30_inevitable),
    (89.75, 999.0, s31_outro),
]

SUBS = [
    (0.55, 1.95, "A estrada se torna", "A"),
    (1.95, 3.60, "cada vez mais longa.", "A"),
    (3.85, 6.00, "Isso está me deixando aborrecido.", "A"),
    (6.70, 7.40, "QUAL É?", "S"),
    (7.45, 8.80, "TÁ ACHANDO QUE VOU TER MEDO", "S"),
    (8.80, 10.20, "IGUAL A UM GATO ASSUSTADO?", "S"),
    (10.75, 11.75, "De longe,", "A"),
    (11.75, 13.95, "aquele ser misterioso responde —", "A"),
    (14.35, 16.40, "a voz chega antes da forma,", "A"),
    (16.60, 17.60, "serena,", "A"),
    (17.70, 19.00, "quase didática,", "A"),
    (19.20, 20.60, "como quem já sabe o fim da frase", "A"),
    (20.60, 22.45, "antes de começá-la.", "A"),
    (24.45, 26.60, "Medo é uma resposta primitiva,", "L"),
    (26.60, 27.30, "Arthur.", "L"),
    (27.45, 28.90, "Eu prefiro a certeza.", "L"),
    (29.35, 31.75, "Sabe o que escreveu Jean-Jacques Rousseau?", "L"),
    (31.95, 33.45, "Tudo é perfeito", "L"),
    (33.45, 35.60, "saindo das mãos do Autor das coisas;", "L"),
    (35.90, 38.40, "tudo degenera nas mãos do homem.", "L"),
    (38.65, 41.30, "Você é a prova viva da tese.", "L"),
    (42.45, 43.95, "As pernas travaram.", "A"),
    (44.15, 45.95, "Não havia igualdade ali —", "A"),
    (46.10, 47.90, "havia apenas o cálculo frio", "A"),
    (47.90, 50.20, "de quem quer que fosse aquele ser.", "A"),
    (50.80, 52.50, "A voz ecoa de novo,", "A"),
    (52.55, 53.95, "mais perto agora,", "A"),
    (54.05, 55.50, "quase paciente:", "A"),
    (56.45, 57.50, "Arthur Castro,", "L"),
    (57.60, 58.90, "19 anos.", "L"),
    (59.10, 60.70, "Não comeu nada hoje.", "L"),
    (60.85, 62.45, "Cansado porque se apressou", "L"),
    (62.45, 63.90, "para chegar ao trabalho.", "L"),
    (64.00, 65.85, "Um funcionário exemplar.", "L"),
    (65.90, 67.85, "E ainda assim recebe", "L"),
    (67.85, 70.90, "exatamente o que recebem os medíocres.", "L"),
    (71.65, 73.10, "Miserável.", "A"),
    (74.10, 76.50, "Aquilo que existe sem meu conhecimento,", "L"),
    (76.50, 79.10, "existe sem meu consentimento.", "L"),
    (79.30, 80.70, "Eu sou o líder.", "L"),
    (80.90, 82.30, "Não por escolha vossa.", "L"),
    (82.45, 85.20, "A natureza das coisas me torna o líder.", "L"),
    (87.10, 89.60, "Ele é inevitável.", "A"),
]

GLITCHES = [(0.22, 0.9, 0.3), (6.78, 0.35, 0.2), (9.0, 0.3, 0.12), (22.45, 0.9, 0.35), (26.74, 0.3, 0.2),
            (31.95, 0.4, 0.2), (35.75, 0.3, 0.12), (40.64, 0.6, 0.3), (43.09, 0.35, 0.2), (46.05, 0.5, 0.2),
            (49.25, 0.4, 0.15), (55.85, 0.6, 0.25), (61.15, 0.5, 0.2), (64.0, 0.4, 0.2), (69.75, 0.4, 0.2),
            (73.65, 0.6, 0.25), (78.0, 0.4, 0.2), (81.65, 0.6, 0.25), (85.45, 0.9, 0.3), (89.75, 0.5, 0.2)]


def subtitle_at(t):
    for a, b_, s, style in SUBS:
        if a <= t < b_:
            if style == "S":
                return (s, style, (int(3 * math.sin(t * 61)), int(2 * math.sin(t * 47))))
            return (s, style)
    return None


def frame_state(t):
    st = St()
    for t0, t1, fn in SHOTS:
        if t0 <= t < t1:
            fn(t, t - t0, st)
            if t - t0 < 0.1 and t0 > 0:
                st.glitch = max(st.glitch, 0.3 * (1 - (t - t0) / 0.1))
            break
    for gt, gs, gd in GLITCHES:
        if gt <= t < gt + gd:
            st.glitch = max(st.glitch, gs * (1 - (t - gt) / gd))
    e = st.env
    e["moon_dir"] = dir_from(e["moon_az"], e["moon_el"])
    add_sky_objects(st.b, st.cam, e, t)
    add_sky_eyes(st.b, st.cam, e, t)
    if st.world in ("road", "ring"):
        st.lights += add_lamps(st.b, st.cam, e, t, ring=(st.world == "ring"))
    e["lights"] = st.lights
    e["exposure"] = st.exposure
    return st
