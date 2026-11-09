"""Renderiza a animação.

    python main.py preview 0.8 4.0 9.0 ...   -> salva PNGs + folha de contato (pasta de saída)
    python main.py full                      -> gera ../olhos_reels.mp4 (1080x1920, 30fps, com voz.mp3)
"""
import os
import sys
import time
import math
import importlib
import collections
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from engine import render, FaceSet  # noqa: E402
from models import build_road_world, build_ring_world, sky_image  # noqa: E402
from post import vhs, grade, draw_subtitle, draw_osd, static_noise  # noqa: E402

# SCENES=scenes (parte 1, padrão) ou SCENES=scenes2 (parte 2)
scenes = importlib.import_module(os.environ.get("SCENES", "scenes"))

OUT_W, OUT_H = 1080, 1920
RW, RH = 1080, 1920   # resolução de render (supersample 2x da resolução de trabalho)
WW, WH = 540, 960     # resolução de trabalho do filtro VHS

_WORLDS = {}


def worlds():
    if not _WORLDS:
        _WORLDS["road"] = build_road_world()
        _WORLDS["ring"] = build_ring_world()
        names = getattr(scenes, "WORLDS", ())
        if "void" in names or "clearing" in names:
            from models2 import build_void_world, build_clearing_world
            _WORLDS["void"] = build_void_world()
            _WORLDS["clearing"] = build_clearing_world()
        if "bar" in names:
            from models3 import build_bar_world, build_office_world
            _WORLDS["bar"] = build_bar_world(blood=False)
            _WORLDS["bar_blood"] = build_bar_world(blood=True)
            _WORLDS["office"] = build_office_world()
    return _WORLDS


def render_frame(i):
    t = i / scenes.FPS
    worlds()
    st = scenes.frame_state(t)
    fs = FaceSet.concat([worlds()[st.world], st.b.build()])
    if getattr(st, "fx", None) is not None:
        fs = st.fx(fs)
    if getattr(st, "b_safe", None) is not None:
        fs = FaceSet.concat([fs, st.b_safe.build()])
    bg = sky_image(st.cam, RW, RH, st.env, t)
    img = render(fs, st.cam, RW, RH, st.env, bg)
    img = img.reduce(2)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = grade(x, st.tint, st.fade, 1.0)
    rng = np.random.default_rng(1000 + i)
    if st.static > 0:
        x = x * (1 - st.static) + static_noise(WH, WW, rng) * st.static
    sub = scenes.subtitle_at(t)
    osd = getattr(st, "osd", None)
    if sub or osd:
        im = Image.fromarray((x * 255).astype(np.uint8))
        if osd:
            draw_osd(im, osd)
        if sub:
            if isinstance(sub, tuple):
                draw_subtitle(im, sub[0], sub[1], shake=sub[2] if len(sub) > 2 else (0, 0))
            else:
                draw_subtitle(im, sub)
        x = np.asarray(im, dtype=np.float32) / 255.0
    x = vhs(x, t, rng, st.glitch, ripple=getattr(st, "ripple", 0.0))
    im = Image.fromarray((x * 255 + 0.5).astype(np.uint8)).resize((OUT_W, OUT_H), Image.BILINEAR)
    return np.asarray(im)


def preview(times, outdir):
    os.makedirs(outdir, exist_ok=True)
    thumbs = []
    for tt in times:
        i = int(round(tt * scenes.FPS))
        t0 = time.time()
        fr = render_frame(i)
        dt = time.time() - t0
        Image.fromarray(fr).save(os.path.join(outdir, f"f_{tt:06.2f}.png"))
        thumbs.append(Image.fromarray(fr).resize((270, 480), Image.BILINEAR))
        print(f"t={tt:.2f}  {dt:.2f}s", flush=True)
    cols = min(len(thumbs), 6)
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 270, rows * 480), (20, 20, 20))
    for k, th in enumerate(thumbs):
        sheet.paste(th, ((k % cols) * 270, (k // cols) * 480))
    sheet.save(os.path.join(outdir, "sheet.png"))


def _init_worker():
    worlds()


def full(out_path, audio_path, workers=10, start=0, end=None):
    import av
    import multiprocessing as mp
    n = int(math.ceil(scenes.DUR * scenes.FPS))
    end = n if end is None else min(end, n)
    out = av.open(out_path, "w", options={"movflags": "+faststart"})
    vs = out.add_stream("libx264", rate=scenes.FPS)
    vs.width, vs.height = OUT_W, OUT_H
    vs.pix_fmt = "yuv420p"
    vs.options = {"crf": "20", "preset": "medium", "profile": "high", "maxrate": "12M", "bufsize": "24M", "g": "60"}

    # áudio primeiro (pequeno)
    ain = av.open(audio_path)
    astream = out.add_stream("aac", rate=44100, layout="stereo")
    astream.bit_rate = 192000
    res = av.AudioResampler(format="fltp", layout="stereo", rate=44100)
    for frame in ain.decode(audio=0):
        for f2 in res.resample(frame):
            for pkt in astream.encode(f2):
                out.mux(pkt)
    for f2 in res.resample(None):
        for pkt in astream.encode(f2):
            out.mux(pkt)
    for pkt in astream.encode(None):
        out.mux(pkt)
    ain.close()

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(workers, initializer=_init_worker)
    pending = collections.deque()
    nxt = start
    done = start
    t0 = time.time()
    window = workers * 3
    while done < end:
        while nxt < end and len(pending) < window:
            pending.append(pool.apply_async(render_frame, (nxt,)))
            nxt += 1
        fr = pending.popleft().get()
        vf = av.VideoFrame.from_ndarray(fr, format="rgb24")
        for pkt in vs.encode(vf):
            out.mux(pkt)
        done += 1
        if done % 60 == 0:
            el = time.time() - t0
            print(f"{done}/{end} frames  {el:.0f}s  (~{el / (done - start) * (end - done):.0f}s restantes)", flush=True)
    for pkt in vs.encode(None):
        out.mux(pkt)
    out.close()
    pool.close()
    pool.join()
    print("ok", out_path, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "preview":
        outdir = os.environ.get("PREVIEW_DIR", os.path.join(HERE, "preview"))
        preview([float(a) for a in sys.argv[2:]], outdir)
    elif cmd == "full":
        root = os.path.dirname(HERE)
        out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, getattr(scenes, "OUT", "olhos_reels.mp4"))
        full(out, os.path.join(root, getattr(scenes, "AUDIO", "voz.mp3")))
