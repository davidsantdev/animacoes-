"""Parte 3 — narrador + detetive + violino (valsa original sintetizada) + ambiências.

    python audio3.py   -> ../parte3_audio.wav  +  parte3_words.json (tempos globais, notas do violino)
"""
import json
import math
import os
import wave

import numpy as np

from audio2 import (SR, load, spectral, conv, make_ir, wow_flutter, hp, lp, db, rms, fade, click, hiss)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ------------------------------------------------------------------ linha do tempo (s)
T_MUSIC0 = 0.3      # violino começa
T_N1 = 1.6          # narração parte 1 (cena da bicicleta)
T_BAR = 19.7        # corte para o bar cheio
T_BLOOD = 28.5      # corte seco: bar vazio e ensanguentado
T_N2 = 29.4         # narração parte 2 ("Ele continua ali...")
T_ROOM = 38.3       # quarto do detetive (silêncio)
T_DET = 44.8        # fala do detetive
T_LAMP_OFF = 49.6   # luminária desligada
DUR = 51.2
N_SPLIT = 17.85     # onde a narração é dividida (pausa antes de "Ele continua ali")

# ------------------------------------------------------------------ valsa original em ré menor (3/4)
BPM = 104.0
BEAT = 60.0 / BPM
MELODY = [
    ("A4", 2), ("D5", 1), ("F5", 2), ("E5", 1), ("D5", 1), ("C#5", 1), ("D5", 1), ("A4", 3),
    ("Bb4", 2), ("G4", 1), ("Bb4", 1), ("D5", 1), ("G5", 1), ("F5", 1), ("E5", 1), ("C#5", 1), ("D5", 3),
    ("A5", 2), ("G#5", 1), ("A5", 1), ("F5", 1), ("D5", 1), ("E5", 2), ("F5", 1), ("E5", 1), ("D5", 1), ("C#5", 1),
    ("D5", 1), ("F5", 1), ("A5", 1), ("Bb5", 2), ("A5", 1), ("G#5", 1), ("E5", 1), ("C#5", 1), ("D5", 3),
]
_NOTE = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "Bb": 10, "B": 11}


def midi(n):
    return 12 * (int(n[-1]) + 1) + _NOTE[n[:-1]]


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def note_list(t0, t1):
    out = []
    t = t0
    k = 0
    while t < t1:
        name, beats = MELODY[k % len(MELODY)]
        d = beats * BEAT
        out.append((t, d, midi(name), k))
        t += d
        k += 1
    return out


def box_smooth(x, w):
    w = max(1, int(w))
    c = np.cumsum(np.concatenate([np.full(w, x[0]), x, np.full(w, x[-1])]))
    return (c[w:] - c[:-w])[w // 2: w // 2 + len(x)] / w


def violin(notes, n, seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    lf = np.zeros(n)
    env = np.zeros(n)
    vib_env = np.zeros(n)
    cur = math.log(hz(notes[0][2]))
    last = 0
    for (t0, d, m, k) in notes:
        a, b = int(t0 * SR), min(n, int((t0 + d) * SR))
        if a >= n:
            break
        lf[last:a] = cur
        cur = math.log(hz(m))
        lf[a:b] = cur
        last = b
        u = (np.arange(b - a) / SR)
        att = np.clip(u / 0.075, 0, 1) ** 0.7
        swell = 0.82 + 0.18 * np.sin(np.pi * np.clip(u / d, 0, 1))
        rel = np.clip((d - u) / 0.06, 0, 1) * 0.35 + 0.65       # legato: quase não solta entre notas
        env[a:b] = np.maximum(env[a:b], att * swell * rel)
        vib_env[a:b] = np.clip((u - 0.16) / 0.25, 0, 1)
    lf[last:] = cur
    lf = box_smooth(lf, 0.03 * SR)                                # portamento
    vib = 0.013 * np.sin(2 * np.pi * 5.6 * t + 0.4 * np.sin(2 * np.pi * 0.7 * t)) * vib_env
    freq = np.exp(lf + vib)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    y = np.zeros(n)
    for k in range(1, 26):
        amp = (1.0 / k ** 1.05) * (1 + 0.15 * rng.uniform(-1, 1))
        y += amp * np.sin(k * phase + rng.uniform(0, 6.28)) * (k * freq < 15000)
    env = box_smooth(env, 0.012 * SR)
    noise = spectral(rng.standard_normal(n), lambda f: hp(1800, 2)(f) * lp(7000, 2)(f))
    y = y * env + noise / rms(noise) * 0.05 * env * rms(y)

    def body(f):
        lg = lambda fc, w: np.exp(-(np.log2(np.maximum(f, 1) / fc)) ** 2 / w)
        return hp(190, 2)(f) * lp(7600, 2)(f) * (1 + 1.3 * lg(285, 0.10) + 1.0 * lg(480, 0.12) + 0.7 * lg(2900, 0.35)
                                                 - 0.35 * lg(1500, 0.2))
    y = spectral(y, body)
    y = wow_flutter(y, seed=seed + 1, wow=0.004, flutter=0.0009)
    return y / rms(y[env > 0.3]) * db(-20)


def thump(n, seed=0):
    tt = np.arange(int(0.12 * SR)) / SR
    return np.sin(2 * np.pi * 70 * tt) * np.exp(-tt / 0.03)


def clink(seed):
    rng = np.random.default_rng(seed)
    tt = np.arange(int(0.45 * SR)) / SR
    f1, f2 = rng.uniform(2600, 3200), rng.uniform(3900, 4600)
    return (np.sin(2 * np.pi * f1 * tt) + 0.6 * np.sin(2 * np.pi * f2 * tt)) * np.exp(-tt / 0.09)


def drip(seed):
    rng = np.random.default_rng(seed)
    tt = np.arange(int(0.25 * SR)) / SR
    f = 1500 * np.exp(-tt / 0.03) + 520 + rng.uniform(-60, 60)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / 0.06)


def tick(hi=True):
    rng = np.random.default_rng(1 if hi else 2)
    tt = np.arange(int(0.03 * SR)) / SR
    y = spectral(rng.standard_normal(len(tt)), hp(2500 if hi else 1500, 2)) * np.exp(-tt / 0.004)
    return y / np.abs(y).max()


def place(mix, sig, t, gain=1.0):
    a = int(t * SR)
    if a >= len(mix):
        return
    b = min(len(mix), a + len(sig))
    mix[a:b] += sig[:b - a] * gain


def voice(x, target=-20.0):
    y = spectral(x, hp(75, 2))
    return y * min(db(target) / rms(y[np.abs(y) > 0.003] if np.any(np.abs(y) > 0.003) else y),
                   0.93 / (np.abs(y).max() + 1e-9))


def build():
    n = int(DUR * SR)
    mix = np.zeros(n)
    nar = load(os.path.join(ROOT, "narrador.mp3"))
    det = load(os.path.join(ROOT, "detetive.mp3"))
    wn = json.load(open(os.path.join(HERE, "words_narrador.json"), encoding="utf-8"))
    wd = json.load(open(os.path.join(HERE, "words_detetive.json"), encoding="utf-8"))

    # ---- vozes
    k = int(N_SPLIT * SR)
    n1 = fade(nar[:k], 0.005, 0.08)
    n2 = fade(nar[k:], 0.03, 0.08)
    v1, v2 = voice(n1), voice(n2)
    vd = voice(det, -19.5)
    ir_room = make_ir(0.5, 0.7, dark=4000, seed=41)
    vd_wet = conv(vd, ir_room)
    vd_wet = vd_wet / rms(vd_wet) * rms(vd)
    vdet = np.zeros(len(vd_wet))
    vdet[:len(vd)] += vd
    vdet += 0.16 * vd_wet
    voices = np.zeros(n)
    place(voices, v1, T_N1)
    place(voices, v2, T_N2 - 0.0)
    place(voices, vdet, T_DET)

    # ---- violino (contínuo de T_MUSIC0 até o corte para o quarto)
    notes = note_list(T_MUSIC0 + 0.15, T_ROOM + 1.0)
    vl = violin([(a - T_MUSIC0, d, m, i) for a, d, m, i in notes], int((T_ROOM - T_MUSIC0 + 1.0) * SR))
    small = make_ir(1.4, 1.8, dark=3800, seed=51)
    big = make_ir(3.2, 3.8, dark=3000, seed=52)
    w_small = conv(vl, small)[:len(vl)]
    w_big = conv(vl, big)[:len(vl)]
    w_small *= rms(vl) / rms(w_small)
    w_big *= rms(vl) / rms(w_big)
    music = np.zeros(n)
    tt = np.arange(len(vl)) / SR + T_MUSIC0
    room_big = tt >= T_BLOOD
    vmix = vl * np.where(room_big, 0.72, 0.85) + np.where(room_big, 0.75 * w_big, 0.38 * w_small)
    place(music, vmix, T_MUSIC0)
    music[int(T_ROOM * SR):] = 0.0                                   # corte seco: silêncio

    # ---- ducking do violino sob as vozes
    venv = np.abs(voices)
    venv = box_smooth(box_smooth(venv, 0.12 * SR), 0.25 * SR)
    venv = np.clip(venv / (db(-24) + 1e-9), 0, 1)
    t_all = np.arange(n) / SR
    hi = np.select([t_all < T_BAR, t_all < T_BLOOD, t_all < T_ROOM], [0.55, 1.0, 0.85], 0.0)
    lo = np.select([t_all < T_BAR, t_all < T_BLOOD, t_all < T_ROOM], [0.28, 0.9, 0.36], 0.0)
    g = hi - (hi - lo) * venv
    g *= np.clip((t_all - T_MUSIC0) / 1.2, 0, 1)                     # entrada suave do violino
    music *= g

    # ---- ambiência do bar cheio (corta seco no T_BLOOD)
    amb = np.zeros(n)
    rng = np.random.default_rng(7)
    a, b = int(T_BAR * SR), int(T_BLOOD * SR)
    mur = spectral(rng.standard_normal(b - a), lambda f: hp(260, 2)(f) * lp(1900, 2)(f))
    modu = box_smooth(np.abs(rng.standard_normal(b - a)), 0.09 * SR)
    modu = 0.6 + 0.8 * modu / modu.max()
    mur = mur * modu
    amb[a:b] += mur / rms(mur) * db(-35)
    for (t0, d, m, i) in notes:
        if T_BAR <= t0 < T_BLOOD:
            beat_in_bar = round((t0 - (T_MUSIC0 + 0.15)) / BEAT) % 3
            place(amb, thump(0), t0, db(-24 if beat_in_bar == 0 else -31))
    for j, tc in enumerate(np.sort(rng.uniform(T_BAR + 0.5, T_BLOOD - 0.5, 6))):
        place(amb, clink(j), tc, db(-31))
    amb[b:] = 0.0
    # ---- bar ensanguentado: gotas + zumbido grave
    for j, tc in enumerate(np.arange(T_BLOOD + 0.7, T_ROOM - 0.2, 1.35) + rng.uniform(-0.2, 0.2, 100)[:len(
            np.arange(T_BLOOD + 0.7, T_ROOM - 0.2, 1.35))]):
        place(amb, drip(j), tc, db(-29))
    a, b = int(T_BLOOD * SR), int(T_ROOM * SR)
    tb = np.arange(b - a) / SR
    drone = (np.sin(2 * np.pi * 73.4 * tb) + 0.5 * np.sin(2 * np.pi * 73.9 * tb) + 0.25 * np.sin(2 * np.pi * 110.0 * tb))
    amb[a:b] += drone / rms(drone) * db(-36) * np.clip(tb / 1.5, 0, 1)
    # ---- quarto do detetive: relógio + chuva + interruptor
    a = int(T_ROOM * SR)
    rain = spectral(rng.standard_normal(n - a), lambda f: hp(500, 2)(f) * lp(3200, 2)(f))
    rain *= 0.7 + 0.3 * box_smooth(np.abs(rng.standard_normal(n - a)), 0.03 * SR) * 3
    amb[a:] += rain / rms(rain) * db(-41)
    for j, tc in enumerate(np.arange(T_ROOM + 0.4, DUR - 0.8, 1.0)):
        place(amb, tick(j % 2 == 0), tc, db(-27))
    place(amb, click("stop", seed=6) / db(-14) * db(-17), T_LAMP_OFF)
    # estática final
    a = int((DUR - 0.45) * SR)
    st = spectral(rng.standard_normal(n - a), hp(400, 1))
    amb[a:] += st / rms(st) * db(-24)

    mix = voices + music + amb
    peak = np.abs(mix).max()
    if peak > 0.95:
        over = np.abs(mix) > 0.8
        mix[over] = np.sign(mix[over]) * (0.8 + 0.15 * np.tanh((np.abs(mix[over]) - 0.8) / 0.15))
    pcm = (np.clip(mix, -1, 1) * 32767).astype(np.int16)
    with wave.open(os.path.join(ROOT, "parte3_audio.wav"), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())

    words = []
    for w in wn:
        if w["e"] <= N_SPLIT:
            words.append({"w": w["w"], "s": round(T_N1 + w["s"], 3), "e": round(T_N1 + w["e"], 3), "spk": "N"})
        else:
            off = T_N2 - N_SPLIT
            words.append({"w": w["w"], "s": round(off + w["s"], 3), "e": round(off + w["e"], 3), "spk": "N"})
    for w in wd:
        words.append({"w": w["w"], "s": round(T_DET + w["s"], 3), "e": round(T_DET + w["e"], 3), "spk": "D"})
    meta = dict(duration=DUR, words=words, notes=[(round(a, 4), round(d, 4), m, i) for a, d, m, i in notes],
                times=dict(T_MUSIC0=T_MUSIC0, T_N1=T_N1, T_BAR=T_BAR, T_BLOOD=T_BLOOD, T_N2=T_N2, T_ROOM=T_ROOM,
                           T_DET=T_DET, T_LAMP_OFF=T_LAMP_OFF))
    json.dump(meta, open(os.path.join(HERE, "parte3_words.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return meta


if __name__ == "__main__":
    m = build()
    print("duração", m["duration"], "notas", len(m["notes"]))
    print(" ".join(f"{w['spk']}:{w['w']}[{w['s']:.2f}]" for w in m["words"]).encode("ascii", "replace").decode())
