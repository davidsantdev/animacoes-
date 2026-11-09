"""Parte 3 — o líder com o violino, o bar, o detetive. Sincronizado com parte3_audio.wav."""
import json
import math
import os

import numpy as np

from engine import Camera, Builder, rot_y, rot_z, normalize
from models import (C, build_watcher, watcher_head_world, yaw_to, add_sky_objects, add_sky_eyes, add_lamps, dir_from,
                    LAMP_ZS, HEAD_P, WR)
from models3 import (build_fallen_bike, blob, footprints, crumble, add_neon, add_bar_lamps, add_office_dynamic,
                     build_detective, detective_head_world, DESK_Y)
import scenes as P1

HERE = os.path.dirname(os.path.abspath(__file__))
META = json.load(open(os.path.join(HERE, "parte3_words.json"), encoding="utf-8"))
FPS = 30
DUR = META["duration"]
AUDIO = "parte3_audio.wav"
OUT = "parte3_reels.mp4"
WORLDS = ("road", "ring", "bar")
TM = META["times"]
T_BAR, T_BLOOD, T_ROOM, T_LAMP_OFF = TM["T_BAR"], TM["T_BLOOD"], TM["T_ROOM"], TM["T_LAMP_OFF"]

ss, lerp, lamps_off = P1.ss, P1.lerp, P1.lamps_off

# ---------------------------------------------------------------- música -> arco do violino / dança
NOTES = META["notes"]
_NT = np.array([n[0] for n in NOTES])
BEAT = 60.0 / 104.0
T_FIRST = NOTES[0][0]


def bow_u(t):
    i = int(np.searchsorted(_NT, t)) - 1
    if i < 0:
        return 0.5
    t0, d = NOTES[i][0], NOTES[i][1]
    s = min(max((t - t0) / d, 0.0), 1.0)
    s = 0.5 - 0.5 * math.cos(math.pi * s)
    return 0.1 + 0.8 * (s if i % 2 == 0 else 1 - s)


def beat_pos(t):
    return max(0.0, (t - T_FIRST) / BEAT)


_WD = np.array([(w["s"], w["e"]) for w in META["words"] if w["spk"] == "D"])


def talk_d(t, rate=4.5):
    i = int(np.searchsorted(_WD[:, 0], t)) - 1
    if i < 0 or t > _WD[i, 1] + 0.02:
        return 0.0
    return 0.35 + 0.65 * abs(math.sin(math.pi * rate * (t - _WD[i, 0])))


class St(P1.St):
    def __init__(self):
        super().__init__()
        self.osd = []
        self.ripple = 0.0
        self.fx = None
        self.b_safe = None
        self.cut_glitch = True
        self.env["lamp_fog"] = 0.3


def leader(b, pos, yaw, t, **p):
    q = dict(violin=dict(u=bow_u(t)), grin=0.85, eye_emis=0.95)
    q.update(p)
    build_watcher(b, pos, yaw, q)


def dance(t):
    bp = beat_pos(t)
    return dict(sway=0.10 * math.sin(bp * 2 * math.pi / 3), walk=bp * math.pi * 0.5,
                lean=0.04 * math.sin(bp * math.pi))


# ---------------------------------------------------------------- cena 1: a bicicleta, o violino, o fim do mundo
B0 = C(-0.8, 0, 161.0)
BIKE_YAW = 0.4


def W1(t):
    return B0 + C(0.35, 0, 1.4 + 0.72 * max(0.0, t - 0.3))


def bike_spin(t):
    return 3.0 * (1 - math.exp(-(t + 2.0) / 4.0))


def crime_scene(st, t, b=None, t_walk=None):
    b = b or st.b
    t_walk = t if t_walk is None else t_walk
    lamp = 0.55 + 0.45 * float(math.sin(t * 23.0) > -0.3)
    st.lights += build_fallen_bike(b, B0, BIKE_YAW, bike_spin(t), lamp=lamp)
    rng = np.random.default_rng(5)
    blob(b, B0 + C(0.15, 0, 0.25), 0.75, rng, n=7)
    blob(b, B0 + C(-0.6, 0, -0.7), 0.45, rng, n=4)
    footprints(b, B0 + C(0.3, 0, 1.0), W1(t_walk) - C(0, 0, 0.45), rng=rng)


def _scene1_env(e):
    Li = int(np.argmin(np.abs(LAMP_ZS - B0[2])))
    e.update(lamp_fn=lambda i, lp, tt: (0.6 + 0.4 * float(math.sin(tt * 17) > 0.0)) if i == Li else 0.0,
             fogdens=0.04, moon_az=4, moon_el=12)


def s1_bike(t, lt, st):
    _scene1_env(st.env)
    crime_scene(st, t)
    W = W1(t)
    leader(st.b, W, 0.0, t, walk=t * 4.6)
    st.cam = Camera(B0 + C(1.3, 0.75, -2.8), B0 + C(-0.2, 0.9, 2.2), fov=52)
    st.static = 1.0 if t < 0.25 else 0.0
    st.fade = ss(0.25, 0.7, t)
    if 0.3 <= t < 3.2:
        st.osd.append(dict(kind="text", text="PARTE 3", x=26, y=112, size=30))


def s2_wheel(t, lt, st):
    _scene1_env(st.env)
    crime_scene(st, t)
    leader(st.b, W1(t), 0.0, t, walk=t * 4.6)
    from models import rot_x as _rx
    R = rot_y(BIKE_YAW) @ rot_z(math.radians(84))
    wc = R @ C(0, WR, 0.52) + B0 + C(0, 0.035, 0)
    st.cam = Camera(wc + C(0.30, 0.95 - 0.1 * lt, -0.40), wc + C(0, 0, 0.05), fov=46)


def s3_trail(t, lt, st):
    _scene1_env(st.env)
    crime_scene(st, t)
    W = W1(t)
    leader(st.b, W, 0.0, t, walk=t * 4.6)
    k = ss(6.6, 9.6, t)
    cam = lerp(B0 + C(0.7, 0.55, 0.3), W + C(0.5, 1.4, -3.3), k)
    tgt = lerp(B0 + C(0.3, 0.0, 3.0), W + C(0, 1.9, 0), k)
    st.cam = Camera(cam, tgt, fov=50)


def s4_timelapse(t, lt, st):
    e = st.env
    _scene1_env(e)
    W = W1(t)
    k = ((t - 9.6) / 1.55) % 1.0
    e.update(sky_t=t * 55.0, star_rot=(t - 9.6) * 1.4, moon_az=180 + (k - 0.5) * 60, moon_el=8 + 40 * math.sin(math.pi * k),
             lamp_fn=lamps_off)
    leader(st.b, W, 0.0, t, walk=t * 4.6, eye_open=1.12, pupil=0.5)
    st.cam = Camera(W + C(0.55, 0.75, 3.0), W + C(0, 2.6, 0), fov=64)


def s5_collapse(t, lt, st):
    e = st.env
    W = W1(12.7)
    red = ss(12.8, 14.6, t)
    base = P1.base_env()
    e.update(lamp_fn=lamps_off, fogdens=0.016, moon_az=20, moon_el=14,
             fogcol=lerp(base["fogcol"], C(0.17, 0.015, 0.02), red), zenith=lerp(base["zenith"], C(0.04, 0.0, 0.006), red),
             moon_col_disc=lerp(C(0.93, 0.90, 0.76), C(0.95, 0.25, 0.12), red),
             moon_col=lerp(base["moon_col"], C(0.70, 0.22, 0.18), red), ambient=lerp(base["ambient"], C(0.30, 0.12, 0.12), red))
    crime_scene(st, t, t_walk=12.7)
    st.b_safe = Builder(group_base=900_000)
    leader(st.b_safe, W, 0.0, t, walk=None)
    st.fx = lambda fs: crumble(fs, W, t, t0=12.8, speed=13.0, dmax=45.0, safe=5.5)
    st.lights.append((W + C(0, 4.5, 1.5), C(1.0, 0.45, 0.4), 1.6, 4.0))
    if t < 17.0:
        k = ss(15.7, 16.9, t)
        cam = lerp(W + C(-6.5, 8.5, -9.5), W + C(-2.6, 2.4, -3.6), k)
        tgt = lerp(W + C(0, 0.0, 5.0), W + C(0, 1.9, 0), k)
        st.cam = Camera(cam, tgt, fov=58 - 10 * k)
    else:
        k = ss(17.0, 19.6, t)
        st.cam = Camera(W + C(-3.0 * (1 + k), 2.5 + 4.5 * k, -6.0 - 9.0 * k), W + C(0, 0.6, 0), fov=50)
    st.fade = 1.0 - ss(19.45, 19.7, t)


# ---------------------------------------------------------------- cena 2: o bar
L0 = C(0.3, 0, 0.6)
PAIRS = [  # (vestido, terno, pele1, pele2)
    (C(0.62, 0.08, 0.12), C(0.14, 0.14, 0.20), C(0.92, 0.62, 0.55), C(0.88, 0.42, 0.62)),
    (C(0.16, 0.28, 0.62), C(0.25, 0.17, 0.11), C(0.88, 0.42, 0.62), C(0.72, 0.46, 0.36)),
    (C(0.78, 0.62, 0.14), C(0.12, 0.12, 0.12), C(0.95, 0.72, 0.62), C(0.90, 0.55, 0.50)),
    (C(0.42, 0.14, 0.52), C(0.30, 0.30, 0.34), C(0.70, 0.45, 0.35), C(0.88, 0.42, 0.62)),
    (C(0.12, 0.48, 0.30), C(0.20, 0.12, 0.10), C(0.90, 0.55, 0.50), C(0.95, 0.72, 0.62)),
]


def _bar_env(e):
    e.update(ambient=C(0.11, 0.085, 0.075), moon_col=C(0.05, 0.05, 0.06), fogdens=0.015, fogcol=C(0.05, 0.03, 0.03),
             zenith=C(0.02, 0.01, 0.01), stars=0.0, clouds=0.0, moon=0.0, lamp_fn=lamps_off)


def dancers(st, t):
    bp = beat_pos(t)
    for k, (dress, suit, s1, s2) in enumerate(PAIRS):
        th = 2 * math.pi * k / len(PAIRS) + 0.22 * t
        c = L0 + C(3.6 * math.cos(th), 0, 2.6 * math.sin(th))
        ph = 2.0 * t + k * 1.3
        d = C(math.cos(ph), 0, math.sin(ph)) * 0.36
        pa, pb = c + d, c - d
        bob = C(0, 0.03 * abs(math.sin(bp * math.pi)), 0)
        common = dict(head=HEAD_P, hat=False, glow=False, scale=0.66, finger=0.13, head_scale=1.35, grin=0.65,
                      pose="open", pose_k=0.45, walk=bp * math.pi + k, sway=0.08 * math.sin(bp * 2.1 + k),
                      shoe=C(0.08, 0.06, 0.05))
        build_watcher(st.b, pa + bob, yaw_to(pa, pb), dict(common, coat=dress, skin=s1, shirt=dress * 1.2))
        build_watcher(st.b, pb + bob, yaw_to(pb, pa), dict(common, coat=suit, skin=s2, shirt=C(0.85, 0.85, 0.85)))


def bar_leader(st, t, face_cam=None, **p):
    q = dance(t)
    q.update(p)
    yaw = 0.9 * t if face_cam is None else yaw_to(L0, face_cam)
    pos = L0 + C(0, 0.03 * abs(math.sin(beat_pos(t) * math.pi)), 0)
    leader(st.b, pos, yaw, t, **q)


def s6_bar(t, lt, st):
    st.world = "bar"
    _bar_env(st.env)
    st.lights += add_bar_lamps(st.b, [1.0, 1.0, 1.0, 1.0])
    st.lights.append(add_neon(st.b, 1.0))
    dancers(st, t)
    if t < 22.6:
        bar_leader(st, t)
        st.cam = Camera(C(5.6, 2.7, 4.3), C(0, 0.9, -0.4), fov=62)
    elif t < 25.4:
        bar_leader(st, t, grin=1.1)
        a = 0.5 * (t - 22.6) + 2.4
        st.cam = Camera(L0 + C(2.1 * math.sin(a), 2.15, 2.1 * math.cos(a)), L0 + C(0, 2.2, 0), fov=48)
    else:
        bar_leader(st, t, face_cam=LOW_CAM, grin=1.2)
        st.cam = Camera(LOW_CAM, L0 + C(0, 2.05, 0), fov=56)


LOW_CAM = L0 + C(-1.3, 1.2, 1.7)


def s7_blood(t, lt, st):
    st.world = "bar_blood"
    st.cut_glitch = False          # corte seco, sem efeito
    _bar_env(st.env)
    fl = lambda f, o: float((int(t * f + o) % 7) != 0)
    st.lights += add_bar_lamps(st.b, [0.0, 0.85 * fl(11, 0), 0.7 * fl(9, 3), 0.0], swing=0.35, t=t)
    st.lights.append(add_neon(st.b, 0.25 + 0.75 * fl(6, 1)))
    if t < 31.9:
        bar_leader(st, t, face_cam=LOW_CAM, grin=1.2)
        st.cam = Camera(LOW_CAM, L0 + C(0, 2.05, 0), fov=56)
    elif t < 35.0:
        bar_leader(st, t, grin=1.0)
        st.cam = Camera(C(-5.8, 3.0, 4.4), C(0.3, 0.6, -0.2), fov=62)
    else:
        k = ss(35.0, 38.0, t)
        cam = L0 + lerp(C(0.3, 2.4, 2.9), C(0.15, 2.45, 1.7), k)
        op = 0.06 if t < 37.45 else 1.15
        bar_leader(st, t, face_cam=cam, grin=1.0, eye_open=op, pupil=0.4)
        hw = watcher_head_world(L0, yaw_to(L0, cam), dict(lean=0.0))
        st.cam = Camera(cam, hw + C(0, -0.12, 0), fov=lerp(46, 36, k))
        st.lights.append((cam + C(0, 0.3, 0), C(1.0, 0.75, 0.6), 0.6, 1.6))


# ---------------------------------------------------------------- cena 3: o quarto do detetive
D0 = C(0.0, 0, -1.22)


def _office_env(e):
    e.update(ambient=C(0.06, 0.065, 0.08), moon_col=C(0.12, 0.15, 0.22), moon_az=-70, moon_el=25, fogdens=0.02,
             fogcol=C(0.02, 0.025, 0.035), zenith=C(0.01, 0.012, 0.02), stars=0.0, clouds=0.0, moon=0.0,
             lamp_fn=lamps_off)


def s8_office(t, lt, st):
    st.world = "office"
    _office_env(st.env)
    lamp = 1.0 if t < T_LAMP_OFF else 0.0
    st.lights += add_office_dynamic(st.b, t, lamp_on=lamp, fan_ang=0.9 * t)
    hw = detective_head_world(D0, 0.0)
    p = dict(eye_open=0.6, py=-0.55, px=0.0, head_pitch=0.25)
    if t < 41.4:
        k = ss(38.3, 41.4, t)
        st.cam = Camera(C(1.75, 1.6, 1.7 - 0.3 * k), C(-0.1, 1.0, -1.0), fov=56)
    elif t < 43.9:
        k = ss(41.4, 43.9, t)
        st.cam = Camera(C(lerp(-0.14, 0.34, k), 1.42, -0.02), C(lerp(-0.12, 0.30, k), DESK_Y + 0.10, -0.58), fov=48)
    elif t < 44.75:
        p.update(px=-0.75 if t < 44.3 else 0.75, py=-0.6)
        st.cam = Camera(hw + C(0.02, -0.02, 0.78), hw + C(0, 0.02, 0), fov=28)
    elif t < 48.7:
        up = ss(46.85, 47.3, t)
        p.update(mouth_open=talk_d(t) * 0.45, py=lerp(-0.5, 0.15, up), head_pitch=lerp(0.25, 0.0, up),
                 brow=lerp(0.35, 0.55, up), eye_open=lerp(0.6, 0.72, up))
        st.cam = Camera(C(0.36, 1.0, 0.05), hw + C(0, -0.12, 0), fov=44)
    else:
        k = ss(48.7, 50.7, t)
        st.cam = Camera(C(0.14, lerp(1.70, 1.60, k), lerp(0.48, 0.36, k)), C(0.14, DESK_Y + 0.05, -0.60), fov=50)
        st.static = ss(50.75, 50.9, t)
    build_detective(st.b, D0, 0.0, p)


SHOTS = [
    (0.00, 4.30, s1_bike),
    (4.30, 6.60, s2_wheel),
    (6.60, 9.60, s3_trail),
    (9.60, 12.70, s4_timelapse),
    (12.70, T_BAR, s5_collapse),
    (T_BAR, T_BLOOD, s6_bar),
    (T_BLOOD, T_ROOM, s7_blood),
    (T_ROOM, 999.0, s8_office),
]

SUBS = [
    (1.55, 3.95, "Naquele ser havia uma maldade antiga —", "A"),
    (4.40, 6.45, "não a fúria passageira dos homens,", "A"),
    (6.65, 9.40, "mas algo que já existia antes de ter nome.", "A"),
    (9.75, 11.05, "Ele não dorme.", "A"),
    (11.15, 12.65, "Ele não vai morrer.", "A"),
    (12.78, 15.80, "Ele diz que, quando o mundo finalmente sucumbir,", "A"),
    (15.80, 16.90, "estará lá —", "A"),
    (16.98, 19.40, "se é que já não sucumbiu há muito tempo.", "A"),
    (29.60, 30.75, "Ele continua ali,", "A"),
    (30.90, 32.05, "dançando sozinho,", "A"),
    (32.15, 33.95, "tocando a mesma melodia.", "A"),
    (34.05, 35.28, "Alguns diriam que era", "A"),
    (35.28, 37.05, "a melhor melodia do mundo.", "A"),
    (44.75, 46.40, "Seja você o que for...", "D"),
    (46.95, 48.70, "eu serei a justiça.", "D"),
]

GLITCHES = [(0.22, 0.9, 0.3), (12.9, 0.3, 0.2), (37.45, 0.6, 0.25), (T_LAMP_OFF, 0.25, 0.12), (50.75, 0.9, 0.45)]


def subtitle_at(t):
    for a, b_, s, style in SUBS:
        if a <= t < b_:
            return (s, style)
    return None


def frame_state(t):
    st = St()
    for t0, t1, fn in SHOTS:
        if t0 <= t < t1:
            fn(t, t - t0, st)
            if st.cut_glitch and t - t0 < 0.1 and t0 > 0:
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
