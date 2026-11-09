"""Fotos de perfil (1080x1080) no mesmo estilo do vídeo.

    python profile.py   -> gera ../perfil/perfil_*.png e uma prévia em círculo
"""
import os
import sys
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from engine import Camera, Builder, FaceSet, render  # noqa: E402
from models import (C, build_watcher, build_cyclist, watcher_head_world, cyclist_head_world, yaw_to, dir_from,  # noqa: E402
                    add_sky_objects, add_lamps, sky_image, build_road_world)
import scenes  # noqa: E402
from post import vhs  # noqa: E402

S, WK, OUT = 2160, 540, 1080
WORLD = build_road_world()


def shoot(cam, env_up, build_fn, lights=(), seed=1, glitch=0.0, exposure=1.12, t=3.0):
    env = scenes.base_env()
    env.update(env_up)
    env["moon_dir"] = dir_from(env["moon_az"], env["moon_el"])
    b = Builder()
    build_fn(b)
    add_sky_objects(b, cam, env, t)
    env["lights"] = list(lights) + add_lamps(b, cam, env, t)
    env["exposure"] = exposure
    fs = FaceSet.concat([WORLD, b.build()])
    img = render(fs, cam, S, S, env, sky_image(cam, S, S, env, t)).reduce(S // WK)
    x = np.asarray(img, np.float32) / 255
    x = vhs(x, t, np.random.default_rng(seed), glitch, dropouts=False)
    return Image.fromarray((x * 255 + 0.5).astype(np.uint8)).resize((OUT, OUT), Image.LANCZOS)


def _moon_behind(cam_pos, point):
    v = np.asarray(point) - cam_pos
    az = np.degrees(np.arctan2(v[0], v[2]))
    el = np.degrees(np.arctan2(v[1], np.hypot(v[0], v[2])))
    return az, el


def v_observador():
    """O observador de cartola em contra-plongée, a lua cheia atrás do chapéu."""
    W = C(4.3, 0, 61.0)
    camp0 = W + C(-2.2, 2.5, -0.9)
    yaw = yaw_to(W, camp0)
    p = dict(grin=0.95, head_roll=0.08, head_pitch=0.12, pupil=0.62, eye_emis=0.9)
    hc = watcher_head_world(W, yaw, p)
    d = C(-2.2, 0.0, -0.9) / np.linalg.norm([2.2, 0.9])
    cp = hc + d * 1.95 + C(0, -0.62, 0)
    cam = Camera(cp, hc + C(0, 0.16, 0), fov=31)
    az, el = _moon_behind(cp, hc + C(0, 0.47, 0))
    lights = [(hc + d * 1.2 + C(0.0, 0.9, 0.0), C(0.95, 0.9, 1.0), 1.1, 1.8),
              (hc + d * 0.5 + C(0, -0.7, 0), C(1.0, 0.18, 0.12), 0.3, 1.0)]
    return shoot(cam, dict(moon_az=az, moon_el=el, moon_size=2.5, clouds=0.5, lamp_fn=scenes.lamps_off,
                           fogdens=0.03), lambda b: build_watcher(b, W, yaw, p), lights, seed=7)


def v_olhos():
    """Só os olhos e o sorriso dele saindo da escuridão; a cartola recortada no céu."""
    W = C(4.3, 0, 61.0)
    camp0 = W + C(-2.2, 2.5, -0.9)
    yaw = yaw_to(W, camp0)
    p = dict(grin=1.1, pupil=0.6, eye_emis=1.0, mouth_open=0.25, head_pitch=0.1)
    hc = watcher_head_world(W, yaw, p)
    d = C(-2.2, 0.0, -0.9) / np.linalg.norm([2.2, 0.9])
    cp = hc + d * 1.7 + C(0, -0.45, 0)
    cam = Camera(cp, hc + C(0, 0.05, 0), fov=26)
    lights = [(hc + d * 0.45 + C(0, -0.55, 0), C(0.9, 0.12, 0.10), 0.32, 0.55)]
    env = dict(moon=0.0, clouds=0.35, stars=0.6, lamp_fn=scenes.lamps_off, fogdens=0.02,
               ambient=C(0.012, 0.013, 0.02), moon_col=C(0.015, 0.015, 0.025),
               fogcol=C(0.05, 0.065, 0.11), zenith=C(0.025, 0.032, 0.065))
    return shoot(cam, env, lambda b: build_watcher(b, W, yaw, p), lights, seed=12)


def v_ciclista():
    """O protagonista 'contaminado': olhos acesos, pupilas mínimas, sorriso."""
    P = C(-0.8, 0, 90.0)
    p = dict(pupil=0.42, smile=0.75, head_pitch=0.08, head_extra=dict(eye_emis=0.8, glow=True))
    hc = cyclist_head_world(P, 0)
    cam = Camera(hc + C(0.32, 0.03, 1.35), hc + C(0, 0.02, 0), fov=30)
    lights = [(hc + C(0.15, -0.6, 0.6), C(1.0, 0.85, 0.6), 0.55, 1.1)]
    return shoot(cam, dict(moon_az=195, moon_el=7, fogdens=0.035),
                 lambda b: build_cyclist(b, P, 0.0, 90.0, p), lights, seed=3)


def circle_preview(imgs, path):
    big, small, pad = 360, 110, 30
    W = len(imgs) * (big + pad) + pad
    sheet = Image.new("RGB", (W, big + small + pad * 3), (250, 250, 250))
    for i, im in enumerate(imgs):
        for size, y in ((big, pad), (small, big + pad * 2)):
            th = im.resize((size, size), Image.LANCZOS)
            m = Image.new("L", (size * 4, size * 4), 0)
            ImageDraw.Draw(m).ellipse([0, 0, size * 4 - 1, size * 4 - 1], fill=255)
            m = m.resize((size, size), Image.LANCZOS)
            x = pad + i * (big + pad) + (big - size) // 2
            sheet.paste(th, (x, y), m)
    sheet.save(path)


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(HERE), "perfil")
    os.makedirs(out, exist_ok=True)
    imgs = []
    for name, fn in (("perfil_1_observador", v_observador), ("perfil_2_olhos", v_olhos),
                     ("perfil_3_ciclista", v_ciclista)):
        im = fn()
        im.save(os.path.join(out, name + ".png"))
        imgs.append(im)
        print("ok", name)
    circle_preview(imgs, sys.argv[1] if len(sys.argv) > 1 else os.path.join(out, "previa_circulo.png"))
