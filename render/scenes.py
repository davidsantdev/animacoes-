"""Decupagem: cada plano é uma função do tempo, sincronizada com a narração (voz.mp3)."""
import math
import numpy as np

from engine import Camera, Builder, rot_y, normalize
from models import (C, build_cyclist, build_watcher, cyclist_head_world, cyclist_eye_world, watcher_head_world,
                    yaw_to, add_sky_objects, add_sky_eyes, add_lamps, dir_from, LAMP_ZS, LAMP_HEAD_X, RING_R,
                    box_polys)

FPS = 30
DUR = 67.55
CX = -0.8
SPEED = 3.4


def cz(t):
    return 8.0 + SPEED * t


def Cpos(t):
    return C(CX, 0, cz(t))


def ss(e0, e1, x):
    k = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return k * k * (3 - 2 * k)


def lerp(a, b, k):
    return a * (1 - k) + b * k


def base_env():
    return dict(zenith=C(0.010, 0.014, 0.040), fogcol=C(0.065, 0.095, 0.15), ambient=C(0.11, 0.12, 0.18),
                moon_col=C(0.30, 0.33, 0.48), moon_az=15.0, moon_el=20.0, fogdens=0.035, silhouette=0.0,
                silcol=C(0.006, 0.008, 0.016), maxdist=120.0, stars=1.0, clouds=1.0, moon=1.0, moon_eye=0.0,
                moon_size=1.0, lamp_fn=None, lights=[], exposure=1.0, near=0.04, sky_eyes=None)


class St:
    def __init__(self):
        self.env = base_env()
        self.b = Builder()
        self.cam = None
        self.world = "road"
        self.lights = []
        self.fade = 1.0
        self.tint = None
        self.glitch = 0.0
        self.static = 0.0
        self.exposure = 1.12


def lamps_off(i, lp, t):
    return 0.0


def put_cyclist(st, t, **p):
    st.lights += build_cyclist(st.b, Cpos(t), 0.0, cz(t), p)


def watcher_look(W, body_yaw, target, extra=0.0):
    rel = yaw_to(W, target) - body_yaw
    rel = (rel + math.pi) % (2 * math.pi) - math.pi
    return max(-1.4, min(1.4, rel)) + extra


# ---------------------------------------------------------------- planos
def s_intro(t, lt, st):
    e = st.env
    e.update(fogdens=0.11, stars=0.4, clouds=0.5, moon=0.0, lamp_fn=lamps_off,
             fogcol=C(0.03, 0.045, 0.075), zenith=C(0.006, 0.008, 0.02))
    W = C(3.6, 0, 15.5)
    st.cam = Camera((0.0, 1.6, lt * 0.6), W + C(-0.25, 2.45, 0), fov=11 - 2 * lt)
    op = 1.0 - (ss(0.95, 1.02, t) - ss(1.08, 1.16, t))
    build_watcher(st.b, W, yaw_to(W, st.cam.pos), dict(eye_open=op, py=-0.15, grin=0.2))
    st.fade = ss(0.30, 1.0, t)
    st.static = 1.0 if t < 0.30 else 0.0


def s_cyc_front(t, lt, st):
    e = st.env
    e.update(moon_az=205, moon_el=14)
    P = Cpos(t)
    st.cam = Camera(P + C(1.9, 1.25, 3.8), P + C(0.15, 1.2, 0), fov=44)
    put_cyclist(st, t, px=0.85 * ss(1.6, 2.0, t), head_yaw=0.35 * ss(1.7, 2.3, t))
    W = C(4.4, 0, cz(2.2) - 7)
    build_watcher(st.b, W, yaw_to(W, P), dict(grin=0.3))


WZ3 = cz(6.0) + 2.0
W3 = C(4.3, 0, WZ3)


def s_watcher_close(t, lt, st):
    e = st.env
    e.update(moon_az=52, moon_el=13)
    camp = W3 + C(-2.7, 2.30, -1.3 + lt * 0.12)
    push = ss(8.58, 8.95, t)
    st.cam = Camera(camp, W3 + C(0, 2.28 + 0.1 * push, 0), fov=36 - 10 * push)
    away = ss(3.14, 3.6, t) * (1 - ss(8.58, 8.68, t))
    more = ss(5.34, 5.9, t) * (1 - ss(8.58, 8.68, t))
    p = dict(head_yaw=-0.42 * away - 0.15 * more, head_pitch=-0.14 * away, head_roll=0.12 * away,
             px=-0.85 * away - 0.15 * more, py=0.35 * away, pupil=0.42 + 0.3 * push, grin=0.25 + 0.7 * ss(8.6, 9.3, t))
    build_watcher(st.b, W3, yaw_to(W3, camp), p)
    st.lights.append((W3 + C(-1.3, 3.8, -0.7), C(1.0, 0.8, 0.5), 1.5, 3.0))
    st.glitch = max(st.glitch, 0.9 * (1 - ss(8.58, 8.9, t)) * (t > 8.58))


def s_cyc_face(t, lt, st):
    P = Cpos(t)
    st.env.update(moon_az=200, moon_el=10)
    hc = cyclist_head_world(P, 0)
    st.cam = Camera(hc + C(0.55, 0.02, 1.15), hc + C(0.0, -0.05, 0), fov=34)
    la = ss(10.44, 10.6, t)
    blink = 1 - (ss(11.3, 11.36, t) - ss(11.42, 11.5, t))
    put_cyclist(st, t, px=0.85 - 1.75 * la, py=-0.35 * la, head_yaw=0.25 - 0.55 * la, smile=-0.6, eye_open=blink)
    st.lights.append((hc + C(0.3, -0.5, 0.7), C(1.0, 0.85, 0.6), 0.4, 1.0))


def s_side_repeat(t, lt, st):
    P = Cpos(t)
    st.env.update(moon_az=12, moon_el=12)
    st.cam = Camera(P + C(-0.4, 2.4, -5.0), P + C(1.2, 1.6, 10.0), fov=50)
    w = 0.5 - 0.5 * math.cos(2 * math.pi * lt / 1.35)
    g = ss(0.3, 0.7, w)
    put_cyclist(st, t, head_yaw=0.75 * g, px=0.9 * g)
    z0 = cz(11.9)
    for k in range(9):
        dz = 6 + 5.0 * k
        W = C(3.5 + 0.35 * (k % 3), 0, z0 + dz)
        body = -math.pi / 2 - 0.5
        hy = watcher_look(W, body, P + C(0, 1.6, 0), -0.7 * g)
        build_watcher(st.b, W, body, dict(head_yaw=hy, px=-0.9 * g, grin=0.3))


def s_high_far(t, lt, st):
    P = Cpos(t)
    st.env.update(moon_az=8, moon_el=11)
    st.cam = Camera(P + C(0.6, 5.4 - lt * 0.35, -8.5 + lt * 0.3), P + C(0.4, 0.6, 18), fov=50)
    put_cyclist(st, t)
    W = C(CX + 0.4, 0, cz(18.0) + 24)
    build_watcher(st.b, W, math.pi, dict(head_roll=0.35 * ss(18.9, 19.5, t), grin=0.4))


def s_pov_const(t, lt, st):
    P = Cpos(t)
    st.env.update(moon_az=10, moon_el=12)
    bob = 0.02 * math.sin(cz(t) / 0.34 / 2.3 * 2)
    cp = P + C(0, 1.70 + bob, 0.15)
    st.cam = Camera(cp, cp + C(0.03 * math.sin(lt * 1.3), -0.10, 1), fov=58)
    st.lights.append((P + C(0, 0.85, 2.4), C(1.0, 0.95, 0.85), 0.7, 2.3))
    W = C(CX + 0.3, 0, cz(t) + 21)
    build_watcher(st.b, W, math.pi, dict(head_roll=0.25 * math.sin(lt * 0.9), grin=0.5))


def s_push_eye(t, lt, st):
    P = Cpos(t)
    st.env.update(moon_az=200, moon_el=10, near=0.002)
    hp = dict(pupil=1.0 + 0.95 * ss(23.4, 25.3, t), smile=-0.3)
    eye, n = cyclist_eye_world(P, 0, side=-1, p=hp)
    k = min(lt / 2.5, 1.0) ** 2.2
    d = 1.25 * (1 - k) + 0.008 * k
    off = normalize(n + C(0.28, 0.06, 0) * (1 - k))
    st.cam = Camera(eye + off * d, eye, fov=38 - 10 * k)
    put_cyclist(st, t, **hp)
    hc = cyclist_head_world(P, 0)
    st.lights.append((hc + C(0.3, -0.4, 0.8), C(1.0, 0.85, 0.6), 0.4, 1.0))
    st.fade = 1 - ss(25.2, 25.45, t)


def s_inside(t, lt, st):
    e = st.env
    e.update(fogdens=0.06, lamp_fn=lamps_off, stars=0.0, clouds=0.5, moon_az=0, moon_el=12)
    W = C(-0.2, 0, 160.0 + 1.25 * lt)
    turn = ss(26.9, 27.45, t)
    build_watcher(st.b, W, 0.0, dict(walk=lt * 6.5, head_yaw=math.pi * turn, grin=0.2 + 0.8 * turn,
                                     pupil=0.42 + 0.2 * turn))
    st.cam = Camera(W + C(0.7, 1.9, -4.2), W + C(-0.1, 1.7, 5.0), fov=55)
    st.lights.append((W + C(0.6, 3.2, -1.6), C(0.9, 0.9, 1.0), 1.4, 3.0))
    st.tint = (1.0, 0.13, 0.10, 0.92)
    st.exposure = 1.35
    st.fade = ss(25.45, 25.9, t)
    st.glitch = max(st.glitch, 0.5 * ss(27.2, 27.4, t) * (1 - ss(27.5, 27.8, t)))


def s_worse(t, lt, st):
    P = Cpos(t)
    st.env.update(moon_az=200, moon_el=12)
    hc = cyclist_head_world(P, 0)
    k = lt / 2.45
    st.cam = Camera(hc + C(0.18, -0.02, 1.55 - 0.35 * k), hc + C(0, -0.12, 0), fov=40)
    p = dict(smile=-0.4)
    if 29.1 <= t < 29.75:
        if (int(t * 20) % 3) != 0:
            p = dict(face_mode="watcher")
        st.glitch = max(st.glitch, 0.8)
    elif t >= 29.75:
        p = dict(pupil=0.42, smile=0.55, head_extra=dict(eye_emis=0.75, glow=True))
    put_cyclist(st, t, **p)
    st.lights.append((hc + C(0.1, -0.6, 0.6), C(1.0, 0.85, 0.6), 0.45, 1.1))


def _lamps_dying(i, lp, t):
    dz = abs(lp[2] - cz(t))
    off_t = 31.0 + (1 - min(dz / 90.0, 1.0)) * 2.8
    if t < off_t - 0.3:
        return 1.0
    if t < off_t:
        return float(int(t * 30 + i) % 2)
    return 0.0


def s_crane(t, lt, st):
    P = Cpos(t)
    st.env.update(lamp_fn=_lamps_dying, moon_az=30, moon_el=25)
    k = ss(30.6, 34.85, t)
    st.cam = Camera(P + C(2.6 + 1.5 * k, 1.8 + 7.5 * k, -3.2 - 4.0 * k), P + C(0, 0.4, 2.5 + 3 * k), fov=52)
    put_cyclist(st, t, py=-0.2)


VULTOS_Z = cz(39.9)
VULTOS = [(3.4, 14), (5.6, 10), (3.6, 21), (7.8, 17), (4.6, 28), (9.5, 24), (6.5, 34), (-3.5, 15), (-5.2, 22),
          (-3.6, 30), (3.5, 40), (-4.4, 38)]


def s_dark(t, lt, st):
    e = st.env
    P = Cpos(t)
    dark = ss(35.2, 38.5, t)
    sil = ss(36.4, 39.2, t)
    e.update(lamp_fn=lamps_off, moon_az=17, moon_el=11, fogdens=0.03 - 0.016 * sil, silhouette=sil,
             fogcol=lerp(C(0.065, 0.095, 0.15), C(0.12, 0.16, 0.25), dark),
             zenith=lerp(C(0.010, 0.014, 0.040), C(0.03, 0.045, 0.095), dark),
             ambient=C(0.11, 0.12, 0.18) * (1 - 0.5 * dark), clouds=1.0)
    st.cam = Camera(P + C(-1.0, 0.7, -5.5), P + C(1.2, 3.6, 8.0), fov=60)
    put_cyclist(st, t, lamp=1 - 0.85 * dark)
    for i, (x, dz) in enumerate(VULTOS):
        vis = t > 40.35 or (t > 39.8 and (int(t * 24) + i) % 3 == 0)
        if not vis:
            continue
        W = C(x, 0, VULTOS_Z + dz)
        build_watcher(st.b, W, yaw_to(W, P), dict(eye_open=0.0, scale=1.12 + 0.12 * (i % 3)))


CORR_Z = cz(41.1)


def _corridor(st, t, P, eyes_from=41.84):
    for k in range(7):
        for s in (1, -1):
            W = C(s * (3.4 + 0.3 * (k % 2)), 0, CORR_Z - 3.0 - 5.5 * k + (1.4 if s > 0 else 0))
            ot = eyes_from + 0.07 * k + (0.03 if s > 0 else 0)
            build_watcher(st.b, W, yaw_to(W, P), dict(eye_open=ss(ot, ot + 0.15, t), grin=0.4))


def s_corridor(t, lt, st):
    e = st.env
    P = Cpos(t)
    e.update(lamp_fn=lamps_off, silhouette=0.65, fogdens=0.045, fogcol=C(0.08, 0.11, 0.18), moon_az=195, moon_el=9)
    st.cam = Camera(P + C(0.25, 0.95, 4.4), P + C(0, 1.45, 0), fov=52)
    put_cyclist(st, t)
    _corridor(st, t, P)


def s_turn(t, lt, st):
    e = st.env
    P = Cpos(t)
    e.update(lamp_fn=lamps_off, silhouette=0.3, fogdens=0.05, fogcol=C(0.08, 0.11, 0.18), moon_az=195, moon_el=9)
    cp = P + C(0, 1.70, 0.15)
    turn = ss(43.3, 43.85, t)
    a = math.pi * turn
    st.cam = Camera.from_dir(cp, C(math.sin(a), -0.08 + 0.36 * turn, math.cos(a)), fov=58)
    W = P + C(0.15, 0, -2.4)
    build_watcher(st.b, W, 0.0, dict(lean=0.25, head_pitch=0.35, grin=0.9, pupil=0.5))
    _corridor(st, t, P, eyes_from=0.0)
    st.lights.append((cp + C(0, 0.5, -0.6), C(0.75, 0.8, 1.0), 0.9, 2.0))
    st.lights.append((P + C(0, 0.85, 2.4), C(1.0, 0.95, 0.85), 0.7, 2.3))
    st.glitch = max(st.glitch, 0.8 * ss(43.8, 43.9, t) * (1 - ss(44.0, 44.25, t)))


def s_reach(t, lt, st):
    e = st.env
    P = Cpos(t)
    e.update(lamp_fn=lamps_off, fogdens=0.05, fogcol=C(0.07, 0.10, 0.17), moon_az=330, moon_el=12)
    st.cam = Camera(P + C(-2.3, 1.45, 2.9 - 0.3 * ss(44.4, 48.6, t)), P + C(0.1, 2.05, -0.7), fov=60)
    k = ss(44.4, 48.6, t)
    W = P + C(0.2, 0, -1.55 + 0.75 * k)
    neck = P + C(0, 1.62, -0.05)
    rt = neck - W
    build_watcher(st.b, W, 0.0, dict(lean=0.15 + 0.2 * k, head_pitch=0.35, reach=ss(45.4, 48.3, t), reach_target=rt,
                                     grin=0.6 + 0.6 * k, pupil=0.45))
    put_cyclist(st, t, py=-0.3, px=-0.6 + 0.3 * ss(47.0, 47.6, t), head_yaw=-0.15, smile=-0.5)
    st.lights.append((W + C(0.6, 1.6, 0.8), C(1.0, 0.12, 0.1), 1.3 * ss(46.5, 48.3, t), 1.8))
    st.glitch = max(st.glitch, 0.18 * ss(46.7, 48.2, t))
    if t >= 48.72:
        st.fade = 0.0
        st.static = 0.55


def s_invite(t, lt, st):
    e = st.env
    P = Cpos(t)
    Li = int(np.argmin(np.abs(LAMP_ZS - (cz(49.25) + 17))))
    Lz = LAMP_ZS[Li]

    def fn(i, lp, tt):
        if i != Li or tt < 49.4:
            return 0.0
        if tt < 49.8:
            return float(int(tt * 25) % 2)
        return 1.0
    e.update(lamp_fn=fn, fogdens=0.045, moon_az=10, moon_el=13)
    cp = P + C(0, 1.70, 0.15)
    st.cam = Camera(cp, cp + C(0, -0.06, 1), fov=56)
    st.lights.append((P + C(0, 0.85, 2.4), C(1.0, 0.95, 0.85), 0.7, 2.3))
    W = C(LAMP_HEAD_X + 0.7, 0, Lz + 0.6)
    build_watcher(st.b, W, yaw_to(W, P), dict(beckon=ss(49.7, 50.3, t), beckon_phase=t * 4.0, grin=0.75))
    st.static = 1 - ss(49.25, 49.45, t)


def s_ring(t, lt, st):
    e = st.env
    st.world = "ring"
    e.update(lamp_fn=lamps_off, fogdens=0.012, maxdist=170.0, moon=0.0, clouds=0.0)
    a = 0.05 * lt
    dz = ss(55.4, 57.8, t)
    h = lerp(46 - 2.0 * lt, 4.6, dz ** 0.8)
    cp = rot_y(a) @ C(0, h, -12 * (h / 46) - 0.6 * dz)
    st.cam = Camera(cp, C(0, 2.45 * dz, 0), fov=54 - 14 * dz)
    phi = 0.6 + 0.29 * lt
    r = RING_R
    pos = C(r * math.cos(phi), 0, r * math.sin(phi))
    st.lights += build_cyclist(st.b, pos, -phi, r * phi, {})
    W = C(0, 0, 0)
    up = ss(54.6, 55.3, t)
    b0 = yaw_to(W, pos)
    b1 = yaw_to(W, C(cp[0], 0, cp[2]))
    dd = (b1 - b0 + math.pi) % (2 * math.pi) - math.pi
    body = b0 + dd * up
    elev = math.atan2(cp[1] - 2.5, math.hypot(cp[0], cp[2]))
    build_watcher(st.b, W, body, dict(head_pitch=-min(elev, 1.3) * up, grin=0.3 + 0.8 * up, pupil=0.55))
    st.lights.append((C(0, 3.5, 0), C(0.8, 0.85, 1.0), 0.6, 4.0))
    st.lights.append((pos + C(0, 2.5, 0), C(0.85, 0.9, 1.0), 1.3, 2.2))


def s_sky(t, lt, st):
    e = st.env
    P = Cpos(t)
    e.update(lamp_fn=lamps_off, fogdens=0.04, moon_az=165, moon_el=40, moon_eye=ss(61.3, 61.8, t), moon_size=1.6,
             sky_eyes=(59.8, 61.6), sky_eye_target=P + C(0, 1.7, 0), sky_view=(160, 36, 26, 30))
    st.cam = Camera(P + C(-1.6, 0.45, 3.2), P + C(0.6, 4.2, -2.0), fov=62)
    up = ss(61.0, 61.6, t)
    put_cyclist(st, t, head_pitch=-0.45 * up, py=0.8 * up)
    st.glitch = max(st.glitch, 0.6 * ss(61.3, 61.4, t) * (1 - ss(61.45, 61.7, t)))


def s_final(t, lt, st):
    e = st.env
    z0 = cz(62.75)
    e.update(lamp_fn=lamps_off, fogdens=0.05, moon_az=5, moon_el=27, moon_eye=1.0, moon_size=1.6,
             sky_eyes=(40.0, 40.1), sky_view=(0, 22, 32, 18))
    st.cam = Camera(C(CX + 0.6, 2.6, z0 - 4.5), C(CX, 3.4, z0 + 30), fov=60)
    put_cyclist(st, t)
    e["sky_eye_target"] = Cpos(t) + C(0, 1.7, 0)
    hb = 0.0
    if 63.5 <= t < 65.0:
        hb = 1.0 if t > 63.8 else float(int(t * 20) % 2)
    elif 65.0 <= t < 65.3:
        hb = float(int(t * 30) % 2)
    if hb > 0:
        st.b.mesh(box_polys(1.0, 0.7, 0.1, c=(0.4, 1.8, z0 + 55)), C(1.0, 0.85, 0.55), emis=1.6, fogmul=0.08,
                  glow=True)
    st.fade = 1 - ss(66.2, 66.75, t)


def s_outro(t, lt, st):
    e = st.env
    e.update(fogdens=0.11, stars=0.3, clouds=0.4, moon=0.0, lamp_fn=lamps_off,
             fogcol=C(0.03, 0.045, 0.075), zenith=C(0.006, 0.008, 0.02))
    W = C(3.6, 0, 15.5)
    st.cam = Camera((0.2, 1.6, 4.0 + lt * 1.0), W + C(-0.1, 2.5, 0), fov=9)
    op = 1.0 - ss(67.15, 67.32, t)
    build_watcher(st.b, W, yaw_to(W, st.cam.pos), dict(eye_open=op, py=-0.15, grin=0.6))
    st.fade = ss(66.75, 66.95, t)
    st.static = ss(67.40, 67.5, t)


SHOTS = [
    (0.00, 1.50, s_intro),
    (1.50, 3.00, s_cyc_front),
    (3.00, 9.50, s_watcher_close),
    (9.50, 11.90, s_cyc_face),
    (11.90, 15.95, s_side_repeat),
    (15.95, 20.05, s_high_far),
    (20.05, 22.90, s_pov_const),
    (22.90, 25.45, s_push_eye),
    (25.45, 28.15, s_inside),
    (28.15, 30.60, s_worse),
    (30.60, 34.85, s_crane),
    (34.85, 41.10, s_dark),
    (41.10, 43.00, s_corridor),
    (43.00, 44.40, s_turn),
    (44.40, 49.25, s_reach),
    (49.25, 51.95, s_invite),
    (51.95, 57.85, s_ring),
    (57.85, 62.75, s_sky),
    (62.75, 66.75, s_final),
    (66.75, 99.00, s_outro),
]

SUBS = [
    (0.00, 1.30, "Há olhos ali."),
    (1.56, 2.95, "Ao cruzar com eles,"),
    (3.00, 5.25, "ele disfarça descaradamente,"),
    (5.34, 6.30, "desviando o olhar"),
    (6.30, 8.25, "para o lado esquerdo da rua —"),
    (8.36, 9.55, "mas volta a fitar-me"),
    (9.55, 11.80, "no instante em que desvio o meu."),
    (11.96, 13.28, "Um jogo silencioso"),
    (13.28, 15.80, "que ele não se cansa de repetir."),
    (16.12, 17.35, "É distante."),
    (17.48, 18.45, "E, no entanto,"),
    (18.56, 19.95, "parece me conhecer —"),
    (20.16, 21.38, "parece saber exatamente"),
    (21.38, 22.85, "para onde vou,"),
    (22.94, 24.34, "como se já tivesse percorrido"),
    (24.34, 25.42, "este mesmo caminho"),
    (25.42, 26.82, "dentro da minha cabeça"),
    (26.82, 28.10, "antes de mim."),
    (28.26, 30.55, "Talvez eu seja pior que ele."),
    (30.70, 31.74, "Talvez seja essa"),
    (31.74, 32.90, "a única certeza"),
    (32.90, 34.60, "que ainda me resta."),
    (34.96, 36.25, "Está escuro —"),
    (36.36, 37.56, "um escuro que engole"),
    (37.56, 38.88, "os contornos das coisas"),
    (38.88, 40.95, "e deixa apenas vultos."),
    (41.22, 42.85, "Não há como desviar."),
    (43.02, 44.25, "Se der meia-volta,"),
    (44.40, 46.10, "estarei expondo meu ponto fraco"),
    (46.10, 47.16, "a toda a maldade"),
    (47.16, 49.15, "que ele possa querer me infligir —"),
    (49.36, 50.45, "um convite,"),
    (50.56, 51.85, "não uma fuga."),
    (52.08, 52.95, "Outrossim,"),
    (52.95, 54.55, "este é o meu destino,"),
    (54.68, 55.98, "selado antes mesmo"),
    (55.98, 57.80, "de eu entender a sentença:"),
    (57.96, 59.65, "a menos que eu peça ajuda,"),
    (59.86, 61.15, "vagarei nesta rua,"),
    (61.30, 62.65, "sob estes olhos,"),
    (62.84, 64.35, "até que algum socorro —"),
    (64.46, 65.80, "se é que existe —"),
    (65.92, 67.45, "se disponha a me encontrar."),
]

GLITCHES = [(0.30, 0.9, 0.3), (11.9, 0.35, 0.15), (22.9, 0.3, 0.12), (25.45, 0.6, 0.25), (34.85, 0.4, 0.2),
            (41.1, 0.3, 0.12), (48.25, 0.9, 0.5), (49.25, 0.8, 0.3), (51.95, 0.5, 0.25), (57.85, 0.3, 0.15),
            (62.75, 0.3, 0.15), (66.75, 0.5, 0.2)]


def subtitle_at(t):
    for a, b_, s in SUBS:
        if a <= t < b_:
            return s
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
    st.lights += add_lamps(st.b, st.cam, e, t, ring=(st.world == "ring"))
    e["lights"] = st.lights
    e["exposure"] = st.exposure
    return st
