"""Parte 2 — montagem do áudio (Arthur + líder) e efeito de fita cassete/terror na voz do líder.

    python audio2.py   -> ../parte2_audio.wav  +  parte2_words.json (tempos globais por palavra)
"""
import json
import math
import os
import wave
from fractions import Fraction

import av
import numpy as np

SR = 44100
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# ------------------------------------------------------------------ utilidades
def load(path):
    c = av.open(path)
    r = av.AudioResampler(format="flt", layout="mono", rate=SR)
    out = []
    for f in c.decode(audio=0):
        for g in r.resample(f):
            out.append(g.to_ndarray().reshape(-1))
    for g in r.resample(None):
        out.append(g.to_ndarray().reshape(-1))
    c.close()
    return np.concatenate(out).astype(np.float64)


def ffx(x, chain):
    """Aplica uma cadeia de filtros do ffmpeg (via PyAV) num sinal mono."""
    g = av.filter.Graph()
    src = g.add_abuffer(sample_rate=SR, format="flt", layout="mono", time_base=Fraction(1, SR))
    prev = src
    for name, args in chain:
        f = g.add(name, args)
        prev.link_to(f)
        prev = f
    fmt = g.add("aformat", f"sample_fmts=flt:sample_rates={SR}:channel_layouts=mono")
    prev.link_to(fmt)
    sink = g.add("abuffersink")
    fmt.link_to(sink)
    g.configure()
    out = []

    def drain():
        while True:
            try:
                fr = g.pull()
            except (av.error.BlockingIOError, av.error.EOFError):
                return
            out.append(fr.to_ndarray().reshape(-1))

    B = 4096
    x32 = x.astype(np.float32)
    for i in range(0, len(x32), B):
        fr = av.AudioFrame.from_ndarray(np.ascontiguousarray(x32[None, i:i + B]), format="flt", layout="mono")
        fr.sample_rate = SR
        fr.pts = i
        fr.time_base = Fraction(1, SR)
        g.push(fr)
        drain()
    g.push(None)
    drain()
    y = np.concatenate(out).astype(np.float64) if out else np.zeros(0)
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y[:len(x)]


def pitch(x, factor):
    """muda o tom mantendo a duração (asetrate + atempo)."""
    chain = [("asetrate", str(int(round(SR * factor)))), ("aresample", str(SR))]
    tempo = 1.0 / factor
    while tempo > 2.0:
        chain.append(("atempo", "2.0"))
        tempo /= 2.0
    chain.append(("atempo", f"{tempo:.6f}"))
    return ffx(x, chain)


def spectral(x, fn):
    """filtro de fase zero no domínio da frequência; fn(freqs)->ganho."""
    n = 1 << int(math.ceil(math.log2(len(x) + 1)))
    X = np.fft.rfft(x, n)
    f = np.fft.rfftfreq(n, 1 / SR)
    return np.fft.irfft(X * fn(f), n)[:len(x)]


def hp(f0, order=2):
    return lambda f: 1.0 / np.sqrt(1 + (f0 / np.maximum(f, 1e-3)) ** (2 * order))


def lp(f0, order=2):
    return lambda f: 1.0 / np.sqrt(1 + (f / f0) ** (2 * order))


def conv(x, ir):
    n = 1 << int(math.ceil(math.log2(len(x) + len(ir))))
    return np.fft.irfft(np.fft.rfft(x, n) * np.fft.rfft(ir, n), n)[:len(x) + len(ir) - 1]


def rms(x):
    return math.sqrt(float(np.mean(x * x)) + 1e-12)


def db(v):
    return 10 ** (v / 20)


def fade(x, fi=0.01, fo=0.04):
    x = x.copy()
    a, b = int(fi * SR), int(fo * SR)
    if a:
        x[:a] *= np.linspace(0, 1, a)
    if b:
        x[-b:] *= np.linspace(1, 0, b)
    return x


def make_ir(rt60, length, dark=3500.0, seed=1, predelay=0.02):
    rng = np.random.default_rng(seed)
    n = int(length * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    ir = spectral(ir, lp(dark, 1))
    # reflexões iniciais
    for d, g in ((0.011, 0.5), (0.023, 0.35), (0.037, 0.3), (0.051, 0.22)):
        ir[int(d * SR)] += g * rng.choice([-1, 1])
    ir = np.concatenate([np.zeros(int(predelay * SR)), ir])
    return ir / np.sqrt(np.sum(ir * ir))


def wow_flutter(x, seed=3, wow=0.0075, flutter=0.0016):
    """variação de velocidade de fita: atraso variável (wow lento irregular + flutter rápido)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    t = np.arange(n) / SR
    # wow: soma de senos lentos com fases aleatórias (irregular)
    d = np.zeros(n)
    for fr, amp in ((0.47, 1.0), (0.83, 0.45), (0.21, 0.6)):
        d += amp * np.sin(2 * np.pi * fr * t + rng.uniform(0, 6.28)) / (2 * np.pi * fr)
    d *= wow / 1.6
    d += flutter * np.sin(2 * np.pi * 6.3 * t + 1.0) / (2 * np.pi * 6.3)
    d += 0.4 * flutter * np.sin(2 * np.pi * 11.7 * t) / (2 * np.pi * 11.7)
    src = np.clip(t - d - d.min(), 0, (n - 1) / SR)
    return np.interp(src * SR, np.arange(n), x)


def dropouts(x, seed=5, rate=0.35):
    """quedas rápidas de nível (fita gasta)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    env = np.ones(n)
    k = rng.poisson(rate * n / SR)
    for _ in range(k):
        c = rng.integers(0, n)
        w = int(rng.uniform(0.03, 0.09) * SR)
        depth = rng.uniform(0.35, 0.7)
        a, b = max(0, c - w), min(n, c + w)
        win = np.hanning(b - a)
        env[a:b] *= 1 - depth * win
    return x * env


def hiss(n, seed=9, level=-44.0):
    rng = np.random.default_rng(seed)
    h = spectral(rng.standard_normal(n), lambda f: hp(1800, 1)(f) * lp(9000, 2)(f))
    # leve modulação lenta (cabeçote/fita irregular)
    t = np.arange(n) / SR
    h *= 1 + 0.25 * np.sin(2 * np.pi * 0.31 * t)
    return h / rms(h) * db(level)


def click(kind="play", seed=2):
    """som mecânico de botão de toca-fitas."""
    rng = np.random.default_rng(seed)
    n = int(0.12 * SR)
    t = np.arange(n) / SR
    thump = np.sin(2 * np.pi * (95 if kind == "play" else 75) * t) * np.exp(-t / 0.025)
    snap = spectral(rng.standard_normal(n), hp(1500, 2)) * np.exp(-t / 0.006)
    snap2 = np.roll(spectral(rng.standard_normal(n), hp(900, 2)) * np.exp(-t / 0.01), int(0.045 * SR)) * 0.5
    y = 0.6 * thump + 0.5 * snap / np.abs(snap).max() + 0.35 * snap2 / (np.abs(snap2).max() + 1e-9)
    return y / np.abs(y).max() * db(-14)


def tape_stop(x, dur=0.7):
    """últimos `dur` s: a fita desacelera até parar."""
    n = int(dur * SR)
    if n >= len(x):
        return x
    head, tail = x[:-n], x[-n:]
    u = np.arange(n) / n
    speed = (1 - u) ** 1.6
    pos = np.cumsum(speed)
    y = np.interp(pos, np.arange(n), tail) * (1 - u) ** 0.5
    return np.concatenate([head, y])


# ------------------------------------------------------------------ voz do líder
def leader_fx(x, distance=0.0, sub=0.35, seed=1):
    """distance: 0 = perto/seco, 1 = longe (muita reverberação e eco)."""
    x = spectral(x, hp(70, 2))
    main = pitch(x, 0.84)                                   # ~3 semitons abaixo
    low = spectral(pitch(x, 0.5), lp(900, 2))               # 1 oitava abaixo, só o "ronco"
    y = main / rms(main) + sub * low / rms(low)
    y = wow_flutter(y, seed=seed)
    # banda estreita de cassete + médio "anasalado" + um pouco de corpo grave
    y = spectral(y, lambda f: hp(160, 2)(f) * lp(6200 - 1400 * distance, 3)(f) *
                 (1 + 0.6 * np.exp(-((np.log2(np.maximum(f, 1) / 1300)) ** 2) / 0.5)) *
                 (1 + 0.35 * np.exp(-((np.log2(np.maximum(f, 1) / 220)) ** 2) / 0.4)))
    y = y / np.abs(y).max() * 0.9
    drive = 2.2
    y = np.tanh(drive * y) / math.tanh(drive)               # saturação de fita
    y = dropouts(y, seed=seed + 10)
    y = y / rms(y) * db(-19.5)
    # espaço
    rt = 1.0 + 2.6 * distance
    ir = make_ir(rt, rt * 1.2, dark=3200 - 1200 * distance, seed=seed + 20, predelay=0.02 + 0.05 * distance)
    wet = conv(y, ir)
    wet = wet / rms(wet) * rms(y)
    dry_g = 1.0 - 0.45 * distance
    wet_g = 0.22 + 0.55 * distance
    out = np.zeros(len(wet))
    out[:len(y)] += y * dry_g
    out += wet * wet_g
    if distance > 0.3:                                       # eco de longe
        for d, g in ((0.47, 0.20), (0.94, 0.09)):
            k = int(d * SR)
            e = np.zeros(len(out))
            e[k:k + len(y)] = spectral(y, lp(2200, 2))[:len(out) - k]
            out += e * g * distance
    return out, y


def reverse_swell(y, length=1.6, seed=31):
    """reverberação invertida: o som 'chega antes' da primeira palavra."""
    head = y[:int(length * SR)]
    ir = make_ir(2.8, 3.2, dark=2600, seed=seed)
    w = conv(head[::-1], ir)[::-1]
    w = w[:len(w) - len(head) + int(0.05 * SR)]               # só o que vem antes do início
    w = w / (np.abs(w).max() + 1e-9) * db(-17)
    w *= np.linspace(0, 1, len(w)) ** 2
    return w


# ------------------------------------------------------------------ voz do Arthur
def arthur_seg(x, target=-20.0, echo=False):
    y = spectral(x, hp(75, 2))
    g = db(target) / rms(y)
    g = min(g, 0.93 / (np.abs(y).max() + 1e-9))
    y = y * g
    if echo:   # o grito volta da escuridão
        out = np.zeros(len(y) + int(1.6 * SR))
        out[:len(y)] += y
        dark = spectral(y, lambda f: lp(1800, 2)(f) * hp(250, 1)(f))
        for d, gg in ((0.36, 0.16), (0.74, 0.08), (1.15, 0.04)):
            k = int(d * SR)
            out[k:k + len(y)] += dark * gg
        return out
    return y


# ------------------------------------------------------------------ montagem
# (arquivo, início, fim, falante, opções) — na ordem do roteiro
SEGMENTS = [
    ("arthur", 0.00, 5.55, "A", {}),                          # A estrada... aborrecido.
    ("arthur", 5.62, 9.12, "A", {"echo": True, "target": -17.5}),  # Qual é? ... gato assustado?
    ("arthur", 9.30, 21.25, "A", {}),                         # De longe... começá-la.
    ("lider", 0.00, 16.85, "L", {"distance": 0.42, "sub": 0.30, "swell": True, "play": True}),
    ("arthur", 21.42, 29.30, "A", {}),                        # As pernas travaram... aquele ser.
    ("arthur", 29.45, 34.08, "A", {}),                        # A voz ecoa de novo... quase paciente
    ("lider", 16.95, 31.35, "L", {"distance": 0.32, "sub": 0.38}),
    ("arthur", 34.12, 35.66, "A", {"target": -21.0}),         # miserável.
    ("lider", 31.40, 42.84, "L", {"distance": 0.12, "sub": 0.55, "stop": True}),
    ("arthur", 35.74, 38.43, "A", {}),                        # Ele é inevitável.
]
GAP_BEFORE = [0.6, 0.45, 0.55, 1.9, 1.0, 0.5, 0.9, 0.85, 0.9, 1.6]
TAIL = 2.2


def build():
    src = {"arthur": load(os.path.join(ROOT, "arthur.mp3")), "lider": load(os.path.join(ROOT, "lider.mp3"))}
    words = {"arthur": json.load(open(os.path.join(HERE, "words_arthur.json"), encoding="utf-8")),
             "lider": json.load(open(os.path.join(HERE, "words_lider.json"), encoding="utf-8"))}
    total = sum(GAP_BEFORE) + sum(b - a for _, a, b, _, _ in SEGMENTS) + TAIL + 4.0
    mix = np.zeros(int(total * SR))
    hiss_env = np.zeros(len(mix))
    out_words, out_segs = [], []
    t = 0.0
    for (name, a, b, spk, opt), gap in zip(SEGMENTS, GAP_BEFORE):
        t += gap
        x = src[name][int(a * SR):int(b * SR)]
        x = fade(x, 0.008, 0.06)
        k0 = int(t * SR)
        if spk == "L":
            y, dry = leader_fx(x, distance=opt["distance"], sub=opt["sub"], seed=int(a) + 1)
            if opt.get("stop"):
                y = tape_stop(y[:len(x) + int(0.25 * SR)], dur=0.75)
            if opt.get("swell"):
                w = reverse_swell(dry)
                s0 = k0 - len(w)
                mix[max(0, s0):k0] += w[max(0, -s0):]
            if opt.get("play"):
                c = click("play")
                s0 = k0 - int(2.0 * SR)
                mix[s0:s0 + len(c)] += c
            h0 = k0 - int((2.0 if opt.get("play") else 0.5) * SR)
            h1 = k0 + len(x) + int(0.9 * SR)
            hiss_env[h0:h1] = 1.0
            if opt.get("stop"):
                c = click("stop", seed=4)
                s1 = k0 + len(x) + int(0.35 * SR)
                mix[s1:s1 + len(c)] += c
                hiss_env[s1:] = 0.0
        else:
            y = arthur_seg(x, target=opt.get("target", -20.0), echo=opt.get("echo", False))
        mix[k0:k0 + len(y)] += y[:len(mix) - k0]
        out_segs.append({"spk": spk, "src": name, "t0": round(t, 3), "t1": round(t + (b - a), 3),
                         "a": a, "b": b})
        for w in words[name]:
            if w["s"] >= a - 0.05 and w["e"] <= b + 0.1:
                out_words.append({"w": w["w"], "s": round(t + w["s"] - a, 3), "e": round(t + w["e"] - a, 3),
                                  "spk": spk})
        t += b - a
    t += TAIL
    mix = mix[:int(t * SR)]
    hiss_env = hiss_env[:len(mix)]
    # chiado de fita acompanha a voz do líder (com rampas suaves)
    for w in (int(0.3 * SR), int(0.3 * SR)):          # média móvel dupla (rampa suave)
        c = np.cumsum(np.concatenate([np.zeros(w), hiss_env, np.zeros(w)]))
        hiss_env = (c[w:] - c[:-w])[w // 2: w // 2 + len(hiss_env)] / w
    mix += hiss(len(mix)) * hiss_env
    # limitador suave de pico
    peak = np.abs(mix).max()
    if peak > 0.95:
        over = np.abs(mix) > 0.8
        mix[over] = np.sign(mix[over]) * (0.8 + 0.15 * np.tanh((np.abs(mix[over]) - 0.8) / 0.15))
    pcm = (np.clip(mix, -1, 1) * 32767).astype(np.int16)
    path = os.path.join(ROOT, "parte2_audio.wav")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())
    meta = {"duration": round(len(mix) / SR, 3), "segments": out_segs, "words": out_words}
    json.dump(meta, open(os.path.join(HERE, "parte2_words.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return meta


if __name__ == "__main__":
    m = build()
    print("duração:", m["duration"])
    for s in m["segments"]:
        print(s)
    print(" ".join(f"{w['spk']}:{w['w']}[{w['s']:.2f}]" for w in m["words"]).encode("ascii", "replace").decode())
