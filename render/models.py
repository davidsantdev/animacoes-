"""Personagens, bicicleta, cenário e céu (estilo CG low-poly anos 90, versão noturna)."""
import math
import numpy as np
from PIL import Image, ImageDraw

from engine import (orient_convex, Builder, rot_x, rot_y, rot_z, normalize, orient, box_polys, frustum_polys,
                    tube_polys, prism_polys, ellipsoid_polys, ellipsoid_radii, ico_polys, disc_poly, ik2)

C = lambda *a: np.array(a, dtype=np.float64)

SKIN_P = C(0.88, 0.42, 0.62)
SKIN_W = C(0.80, 0.79, 0.86)
WHITE = C(0.90, 0.90, 0.93)
TROUSER = C(0.80, 0.80, 0.84)
BLACK = C(0.07, 0.07, 0.09)
FRAME = C(0.86, 0.86, 0.83)
TIRE = C(0.06, 0.06, 0.07)
TRUNK = C(0.45, 0.25, 0.14)
LEAVES = [C(0.40, 0.66, 0.12), C(0.24, 0.50, 0.10), C(0.52, 0.70, 0.16)]
ROAD = C(0.42, 0.27, 0.21)
GRASS = C(0.30, 0.55, 0.16)
EYE_W = C(0.96, 0.95, 0.86)
PUPIL = C(0.015, 0.015, 0.02)
LAMP_LIT = C(1.0, 0.83, 0.56)

ROAD_HALF = 2.6
LAMP_X = -3.4
LAMP_HEAD_X = -2.25
LAMP_Y = 4.45
LAMP_ZS = np.arange(14.0, 420.0, 30.0)


# ================================================================ cabeças
class Head:
    def __init__(self, rx, ry, rz, taper, chin, nlon=10, nlat=7):
        self.rx, self.ry, self.rz, self.taper, self.chin = rx, ry, rz, taper, chin
        self.polys = ellipsoid_polys(nlon, nlat, rx, ry, rz, taper, chin)

    def sz(self, x, y):
        ax, az = ellipsoid_radii(y, self.rx, self.ry, self.rz, self.taper, self.chin)
        if ax <= 1e-6:
            return 0.0
        return az * math.sqrt(max(0.0, 1 - (x / ax) ** 2))

    def onsurf(self, pts2, off):
        return np.array([(x, y, self.sz(x, y) + off) for x, y in pts2])


HEAD_P = Head(0.165, 0.205, 0.17, 0.45, 1.15)
HEAD_W = Head(0.150, 0.270, 0.160, 0.50, 1.20)


def build_head(b, head, R, T, skin, p, hat=False, lights_face=True):
    """p: px, py, eye_open, pupil, eye_emis, glow, smile, mouth_open, eye_r, eye_dx, eye_y, mouth_y, mw, mouth_col"""
    g = b.new_group()
    b.mesh(head.polys, skin, R=R, t=T, group=g, layer=0)
    if hat == "fedora":
        hc_ = p.get("hat_col", C(0.20, 0.17, 0.14))
        by = head.ry * 0.70
        b.mesh(prism_polys(8, 0.25, 0.25, 0.018, y0=by, rot=0.2), hc_, R=R, t=T, group=g, layer=1)
        b.mesh(prism_polys(8, 0.155, 0.13, 0.15, y0=by + 0.012, rot=0.2), hc_ * 1.1, R=R, t=T, group=g, layer=1)
        b.mesh(prism_polys(8, 0.158, 0.153, 0.035, y0=by + 0.015, rot=0.2), C(0.07, 0.05, 0.04), R=R, t=T, group=g,
               layer=1)
    elif hat:
        brim_y = head.ry * 0.80
        hp = prism_polys(8, 0.25, 0.25, 0.025, y0=brim_y, rot=0.2)
        b.mesh(hp, C(0.05, 0.05, 0.06), R=R, t=T, group=g, layer=1)
        cp = prism_polys(8, 0.15, 0.165, 0.38, y0=brim_y + 0.02, rot=0.2)
        b.mesh(cp, C(0.06, 0.06, 0.07), R=R, t=T, group=g, layer=1)
        bp = prism_polys(8, 0.152, 0.152, 0.05, y0=brim_y + 0.03, rot=0.2)
        b.mesh(bp, C(0.25, 0.05, 0.06), R=R, t=T, group=g, layer=1)
    er = p.get("eye_r", 0.048)
    edx = p.get("eye_dx", 0.068)
    ey = p.get("eye_y", 0.035)
    op = max(p.get("eye_open", 1.0), 0.0)
    px, py = p.get("px", 0.0), p.get("py", 0.0)
    emis = p.get("eye_emis", 0.35)
    glow = p.get("glow", False)
    pr = er * 0.46 * p.get("pupil", 1.0)
    eye_col = p.get("eye_col", EYE_W)
    psx, psy = p.get("pupil_shape", (1.0, 1.0))
    ang = np.arange(6) * np.pi / 3
    for side in (-1, 1):
        ex = side * edx
        if op > 0.04:
            pts = [(ex + er * math.cos(a), ey + er * op * math.sin(a)) for a in ang]
            w = head.onsurf(pts, 0.006)
            w = orient(w, (0, 0, 1))
            b.poly(w @ R.T + T, eye_col, emis=emis, group=g, layer=2, glow=glow, fogmul=0.2 if glow else 1.0)
            prr = min(pr, er * 0.95)
            cx = ex + px * (er - prr) * 0.95
            cy = ey + py * (er * op - prr * op) * 0.95
            pts = [(cx + prr * psx * math.cos(a), cy + min(prr * psy, er * 0.92) * op * math.sin(a)) for a in ang]
            w = orient(head.onsurf(pts, 0.010), (0, 0, 1))
            b.poly(w @ R.T + T, PUPIL, emis=1.0, group=g, layer=3, glow=glow, fogmul=0.2 if glow else 1.0)
        else:
            pts = [(ex - er, ey + 0.004), (ex + er, ey + 0.004), (ex + er, ey - 0.006), (ex - er, ey - 0.006)]
            w = orient(head.onsurf(pts, 0.006), (0, 0, 1))
            b.poly(w @ R.T + T, skin * 0.55, group=g, layer=2)
    # sobrancelhas (brow > 0 = bravo, < 0 = preocupado)
    brow = p.get("brow", None)
    if brow is not None:
        by = ey + er * 1.32 + p.get("brow_y", 0.0)
        th = 0.0075
        for side in (-1, 1):
            xin = side * (edx - er * 0.85)
            xout = side * (edx + er * 0.95)
            yin, yout = by - brow * 0.022, by + brow * 0.014
            pts = [(xin, yin - th), (xout, yout - th), (xout, yout + th), (xin, yin + th)]
            w = orient(head.onsurf(pts, 0.009), (0, 0, 1))
            b.poly(w @ R.T + T, p.get("brow_col", skin * 0.22), emis=0.3, group=g, layer=2)
    # nariz
    if p.get("nose", True):
        ny = p.get("nose_y", -0.03)
        base = head.onsurf([(-0.022, ny - 0.02), (0.022, ny - 0.02), (0.0, ny + 0.03)], 0.0)
        apex = np.array([0.0, ny - 0.018, head.sz(0, ny) + 0.045])
        bc = base.mean(0) - np.array([0, 0, 0.02])
        for i in range(3):
            tri = np.array([base[i], base[(i + 1) % 3], apex])
            tri = orient(tri, tri.mean(0) - bc)
            b.poly(tri @ R.T + T, skin * 0.92, group=g, layer=1)
    # bigode
    if p.get("mustache") is not None:
        ny = p.get("nose_y", -0.03)
        mpts = [(-0.052, ny - 0.040), (-0.022, ny - 0.024), (0.0, ny - 0.030), (0.022, ny - 0.024), (0.052, ny - 0.040),
                (0.040, ny - 0.054), (0.0, ny - 0.046), (-0.040, ny - 0.054)]
        w = orient(head.onsurf(mpts, 0.012), (0, 0, 1))
        b.poly(w @ R.T + T, p["mustache"], emis=0.15, group=g, layer=3)
    # boca
    mw = p.get("mw", 0.05)
    my = p.get("mouth_y", -0.085)
    sm = p.get("smile", 0.0)
    mo = p.get("mouth_open", 0.0)
    xs = [-mw, -mw / 3, mw / 3, mw]
    up = [(x, my + sm * 0.03 * ((x / mw) ** 2 - 0.35)) for x in xs]
    lo = [(x, y - (0.006 + mo * 0.05 * (1 - (x / mw) ** 2))) for x, y in up][::-1]
    w = orient(head.onsurf(up + lo, 0.007), (0, 0, 1))
    b.poly(w @ R.T + T, p.get("mouth_col", C(0.45, 0.04, 0.08)), emis=0.3, group=g, layer=2)
    return g


# ================================================================ bicicleta + ciclista
WR = 0.34


def _wheel(b, tw, Mw, hub, ang):
    n = 12
    a = ang + 2 * np.pi * np.arange(n + 1) / n
    hub = np.asarray(hub)
    for side in (-1, 1):
        x = side * 0.024
        for i in range(n):
            o0 = hub + C(x, WR * math.cos(a[i]), WR * math.sin(a[i]))
            o1 = hub + C(x, WR * math.cos(a[i + 1]), WR * math.sin(a[i + 1]))
            i0 = hub + C(x, 0.29 * math.cos(a[i]), 0.29 * math.sin(a[i]))
            i1 = hub + C(x, 0.29 * math.cos(a[i + 1]), 0.29 * math.sin(a[i + 1]))
            q = orient(np.array([o0, o1, i1, i0]), C(side, 0, 0))
            b.poly(tw(q), TIRE)
    for i in range(n):
        p0 = hub + C(0.024, WR * math.cos(a[i]), WR * math.sin(a[i]))
        p1 = hub + C(0.024, WR * math.cos(a[i + 1]), WR * math.sin(a[i + 1]))
        q = np.array([p0, p1, p1 - C(0.048, 0, 0), p0 - C(0.048, 0, 0)])
        mid = hub + C(0, WR * math.cos((a[i] + a[i + 1]) / 2), WR * math.sin((a[i] + a[i + 1]) / 2))
        q = orient(q, mid - hub)
        b.poly(tw(q), TIRE * 1.5)
    for k in range(4):
        aa = ang * 1.0 + k * np.pi / 4
        p1 = hub + C(0, 0.29 * math.cos(aa), 0.29 * math.sin(aa))
        p0 = hub - C(0, 0.29 * math.cos(aa), 0.29 * math.sin(aa))
        b.mesh(tube_polys(tw(p0), tw(p1), 0.012, side=Mw @ C(1, 0, 0)), FRAME * 0.8)
    b.mesh(box_polys(0.08, 0.06, 0.06, c=hub) , FRAME * 0.6, R=Mw, t=tw(C(0, 0, 0)))


def build_cyclist(b, pos, yaw, dist, p=None):
    """Retorna lista de luzes (pos, cor, intensidade, raio). p: dicionário de pose/expressão."""
    p = p or {}
    Mw = rot_y(yaw)
    pos = np.asarray(pos, float)
    tw = lambda q: np.asarray(q) @ Mw.T + pos
    side_w = Mw @ C(1, 0, 0)
    lights = []
    draw_body = p.get("draw_body", True)
    if p.get("draw_bike", True):
        wang = -dist / WR
        RH, FH = C(0, WR, -0.52), C(0, WR, 0.52)
        _wheel(b, tw, Mw, RH, wang)
        _wheel(b, tw, Mw, FH, wang)
        BB = C(0, 0.30, -0.06); ST = C(0, 0.88, -0.24); HT = C(0, 0.92, 0.36); HB = C(0, 0.74, 0.41)
        STEM = C(0, 1.02, 0.34)
        th = 0.036
        for a_, b_ in [(BB, ST), (ST, HT), (BB, HB), (HB, HT), (HT, STEM)]:
            b.mesh(tube_polys(tw(a_), tw(b_), th, side=side_w), FRAME)
        for s in (-1, 1):
            b.mesh(tube_polys(tw(HB), tw(FH + C(s * 0.04, 0, 0)), 0.026, side=side_w), FRAME)
            b.mesh(tube_polys(tw(BB), tw(RH + C(s * 0.05, 0, 0)), 0.024, side=side_w), FRAME)
            b.mesh(tube_polys(tw(ST), tw(RH + C(s * 0.05, 0, 0)), 0.022, side=side_w), FRAME)
        b.mesh(tube_polys(tw(C(-0.27, 1.03, 0.36)), tw(C(0.27, 1.03, 0.36)), 0.03, side=Mw @ C(0, 0, 1)), FRAME * 0.9)
        b.mesh(box_polys(0.15, 0.05, 0.27, c=(0, 0.925, -0.27)), C(0.12, 0.08, 0.06), R=Mw, t=pos)
        lamp = p.get("lamp", 1.0)
        lc = C(0.25, 0.22, 0.15) * (1 - lamp) + C(1.0, 0.9, 0.6) * lamp
        b.mesh(box_polys(0.09, 0.08, 0.08, c=(0, 0.98, 0.43)), lc, R=Mw, t=pos, emis=lamp, glow=lamp > 0.3)
        if lamp > 0.01:
            lights.append((tw(C(0, 0.85, 2.4)), C(1.0, 0.95, 0.85), 0.7 * lamp, 2.3))
        crank = -dist / WR / 2.3
        pedals = []
        for s in (-1, 1):
            a = crank + (0 if s < 0 else math.pi)
            ped = BB + C(s * 0.12, -0.17 * math.cos(a), 0.17 * math.sin(a))
            b.mesh(tube_polys(tw(BB + C(s * 0.07, 0, 0)), tw(ped), 0.03, side=Mw @ C(0, 0, 1)), FRAME * 0.5)
            b.mesh(box_polys(0.1, 0.025, 0.08, c=ped), C(0.1, 0.1, 0.1), R=Mw, t=pos)
            pedals.append(ped)
        cr = disc_poly(BB + C(0.05, 0, 0), C(1, 0, 0), C(0, 1, 0), 0.1, 0.1, n=8, rot=crank)
        b.poly(tw(cr), C(0.3, 0.3, 0.3), dbl=True)
        sh = disc_poly(C(0, 0.012, 0), C(0, 1, 0), C(0, 0, 1), 0.28, 0.7, n=8)
        b.poly(tw(sh), C(0.0, 0.0, 0.0), emis=1.0, bucket=2.7)
    else:
        pedals = [C(-0.12, 0.2, -0.06), C(0.12, 0.4, -0.06)]

    face = p.get("face_mode", "normal")
    skin = SKIN_P if face == "normal" else SKIN_W
    if draw_body:
        hip_c = C(0, 0.97, -0.27)
        for s, ped in zip((-1, 1), pedals):
            hip = hip_c + C(s * 0.09, 0, 0)
            ankle = ped + C(0, 0.06, -0.02)
            knee, ank = ik2(hip, ankle, 0.45, 0.44, C(0, 0.4, 1))
            b.mesh(frustum_polys(tw(hip), tw(knee), 0.15, 0.15, 0.12, 0.12, side=side_w), TROUSER)
            b.mesh(frustum_polys(tw(knee), tw(ank), 0.12, 0.12, 0.09, 0.09, side=side_w), TROUSER)
            b.mesh(box_polys(0.1, 0.08, 0.22, c=ped + C(0, 0.045, 0.03)), C(0.12, 0.08, 0.07), R=Mw, t=pos)
        chest = C(0, 1.46, -0.10)
        b.mesh(frustum_polys(tw(hip_c), tw(chest), 0.32, 0.21, 0.44, 0.25, side=side_w), WHITE)
        # lapela escura (detalhe de terno)
        lap = np.array([chest + C(-0.045, 0.0, 0.126), chest + C(0.045, 0.0, 0.126), chest + C(0, -0.16, 0.115)])
        lap = orient(lap, C(0, 0.3, 1))
        b.poly(tw(lap), C(0.10, 0.10, 0.16), bucket=3.0)
        for s in (-1, 1):
            sh_ = chest + C(s * 0.21, -0.03, 0)
            grip = C(s * 0.25, 1.04, 0.35)
            elbow, wrist = ik2(sh_, grip, 0.31, 0.30, C(s * 0.7, -0.6, -0.4))
            b.mesh(frustum_polys(tw(sh_), tw(elbow), 0.12, 0.12, 0.1, 0.1, side=Mw @ C(0, 1, 0)), WHITE)
            b.mesh(frustum_polys(tw(elbow), tw(wrist), 0.1, 0.1, 0.085, 0.085, side=Mw @ C(0, 1, 0)), WHITE)
            b.mesh(box_polys(0.08, 0.07, 0.1, c=grip), skin, R=Mw, t=pos)
        neck0 = chest + C(0, 0.02, 0.01)
        neck1 = chest + C(0, 0.13, 0.035)
        b.mesh(tube_polys(tw(neck0), tw(neck1), 0.065, side=side_w), skin)
        if p.get("draw_head", True):
            hc = neck1 + C(0, 0.17, 0.02)
            R = Mw @ rot_y(p.get("head_yaw", 0.0)) @ rot_x(p.get("head_pitch", 0.0)) @ rot_z(p.get("head_roll", 0.0))
            hp = dict(px=p.get("px", 0.0), py=p.get("py", 0.0), eye_open=p.get("eye_open", 1.0),
                      pupil=p.get("pupil", 1.0), smile=p.get("smile", 0.0), mouth_open=p.get("mouth_open", 0.0),
                      eye_emis=0.35, glow=False)
            if face != "normal":
                hp.update(eye_emis=1.0, glow=True, pupil=0.35, smile=1.4, mw=0.075, mouth_col=C(0.03, 0.0, 0.0),
                          mouth_open=0.35)
            hp.update(p.get("head_extra", {}))
            build_head(b, HEAD_P, R, tw(hc), skin, hp)
    return lights


def cyclist_head_world(pos, yaw, p=None):
    p = p or {}
    Mw = rot_y(yaw)
    hc = C(0, 1.46, -0.10) + C(0, 0.13, 0.035) + C(0, 0.17, 0.02)
    return Mw @ hc + np.asarray(pos, float)


def cyclist_eye_world(pos, yaw, side=-1, p=None):
    """posição mundial do olho (side=-1 direito do personagem) e a normal."""
    p = p or {}
    Mw = rot_y(yaw)
    R = Mw @ rot_y(p.get("head_yaw", 0.0)) @ rot_x(p.get("head_pitch", 0.0))
    hc = cyclist_head_world(pos, yaw)
    ex, ey = side * 0.068, 0.035
    loc = np.array([ex, ey, HEAD_P.sz(ex, ey) + 0.010])
    return R @ loc + hc, R @ C(0, 0, 1)


# ================================================================ o observador
def build_watcher(b, pos, yaw, p=None):
    p = p or {}
    sc = p.get("scale", 1.0)
    Mw = rot_y(yaw)
    pos = np.asarray(pos, float)
    tw = lambda q: (np.asarray(q) * sc) @ Mw.T + pos
    side_w = Mw @ C(1, 0, 0)
    lean = p.get("lean", 0.0)
    hipc = C(0, 1.27, 0)
    RL = rot_x(lean) @ rot_z(p.get("sway", 0.0))
    L = lambda q: hipc + RL @ (np.asarray(q) - hipc)
    coat = p.get("coat", C(0.055, 0.055, 0.07))
    skin = p.get("skin", SKIN_W)
    head = p.get("head", HEAD_W)
    fing = p.get("finger", 0.17)
    walk = p.get("walk", None)
    start_mark = b.mark()
    # pernas
    for s in (-1, 1):
        hip = C(s * 0.1, 1.27, 0)
        if walk is None:
            foot = C(s * 0.12, 0.0, 0.02)
        else:
            ph = walk + (0 if s < 0 else math.pi)
            foot = C(s * 0.12, 0.10 * max(0.0, math.cos(ph)), 0.30 * math.sin(ph))
        ankle = foot + C(0, 0.08, -0.03)
        knee, ank = ik2(hip, ankle, 0.64, 0.62, C(0, 0, 1))
        b.mesh(frustum_polys(tw(hip), tw(knee), 0.13, 0.13, 0.10, 0.10, side=side_w), coat)
        b.mesh(frustum_polys(tw(knee), tw(ank), 0.10, 0.10, 0.08, 0.08, side=side_w), coat)
        b.mesh(box_polys(0.11, 0.08, 0.30, c=foot + C(0, 0.04, 0.06)), p.get("shoe", C(0.03, 0.03, 0.03)), R=Mw * sc,
               t=pos)
    # casaco
    b.mesh(frustum_polys(tw(L(C(0, 1.34, 0))), tw(C(0, 0.74, -0.02)), 0.38, 0.25, 0.54, 0.36, side=side_w), coat)
    chest = L(C(0, 2.08, 0.0))
    b.mesh(frustum_polys(tw(L(C(0, 1.30, 0))), tw(chest), 0.36, 0.24, 0.54, 0.27, side=side_w), coat)
    shirt = np.array([L(C(-0.07, 2.08, 0.137)), L(C(0.07, 2.08, 0.137)), L(C(0, 1.80, 0.13))])
    shirt = orient(shirt, RL @ C(0, 0.2, 1))
    b.poly(tw(shirt), p.get("shirt", C(0.55, 0.54, 0.58)))
    # braços
    reach = p.get("reach", 0.0)
    rt = p.get("reach_target", C(-0.05, 1.9, 1.1))
    beck = p.get("beckon", 0.0)
    ptw = p.get("point_target", None)
    pt_local = None if ptw is None else (Mw.T @ (np.asarray(ptw, float) - pos)) / sc

    def arm_pose(name, s, sh_, rest):
        if name == "behind":
            return L(C(s * 0.10, 1.32, -0.24)), C(s * 1.0, -0.1, -0.2), "open"
        if name == "open":
            return L(C(s * 1.0, 1.95, 0.32)), C(s * 0.2, -1.0, -0.3), "open"
        if name == "lecture":
            if s < 0:
                return L(C(-0.44, 2.42, 0.40)), C(-1.0, -0.6, -0.1), "point"
            return L(C(s * 0.10, 1.32, -0.24)), C(s * 1.0, -0.1, -0.2), "open"
        if name == "point" and s < 0 and pt_local is not None:
            d = normalize(pt_local - sh_)
            return sh_ + d * 1.06, C(-0.6, -0.8, 0.0), "point"
        if name == "palm" and s < 0:
            return L(C(-0.16, 1.30, 0.46)), C(-0.7, -0.6, -0.3), "open"
        return rest, C(s * 0.8, -0.4, -0.6), "open"

    pose_to = p.get("pose", None)
    pose_from = p.get("pose_from", None)
    pk = p.get("pose_k", 1.0)
    vio = p.get("violin", None)
    if vio is not None:
        vfr = _violin_frame(L, RL)
        vio_left, vio_frog, vio_bd = _violin_hands(vfr, vio.get("u", 0.5))
    for s in (-1, 1):
        sh_ = L(C(s * 0.28, 2.02, 0.0))
        rest = L(C(s * 0.34, 0.98, 0.10))
        t0_, p0_, h0_ = arm_pose(pose_from, s, sh_, rest)
        t1_, p1_, h1_ = arm_pose(pose_to, s, sh_, rest)
        tgt = t0_ * (1 - pk) + t1_ * pk
        pole_ = p0_ * (1 - pk) + p1_ * pk
        hand = h1_ if pk > 0.5 else h0_
        if s < 0 and reach > 0:
            tgt = rest * (1 - reach) + np.asarray(rt) * reach
        if vio is not None:
            if s > 0:
                tgt, pole_, hand = vio_left, C(0.5, -1.0, 0.1), "grip"
            else:
                tgt, pole_, hand = vio_frog, C(-0.9, -0.5, -0.4), "grip"
        if s < 0 and beck > 0:
            ph = p.get("beckon_phase", 0.0)
            bt = L(C(-0.32, 2.05 + 0.10 * math.sin(ph), 0.50 + 0.12 * math.cos(ph)))
            tgt = rest * (1 - beck) + bt * beck
        elbow, wrist = ik2(sh_, tgt, 0.56, 0.54, pole_)
        b.mesh(frustum_polys(tw(sh_), tw(elbow), 0.12, 0.12, 0.10, 0.10, side=Mw @ C(0, 1, 0)), coat)
        b.mesh(frustum_polys(tw(elbow), tw(wrist), 0.10, 0.10, 0.085, 0.085, side=Mw @ C(0, 1, 0)), coat)
        a = normalize(wrist - elbow)
        sv = normalize(np.cross(a, C(0, 0, 1) if abs(a[2]) < 0.9 else C(1, 0, 0)))
        palm_end = wrist + a * 0.12
        b.mesh(tube_polys(tw(wrist), tw(palm_end), 0.075, 0.03, side=Mw @ sv), skin * 0.95)
        for k in (-1, 0, 1):
            f0 = palm_end + sv * 0.025 * k
            if hand == "grip":
                f1 = f0 + normalize(a * 0.5 - np.cross(sv, a) * 1.0 + sv * 0.15 * k) * fing * 0.42
            elif hand == "point":
                ln = fing * 1.25 if k == 0 else fing * 0.3
                f1 = f0 + (a if k == 0 else normalize(a - np.cross(sv, a) * 0.9)) * ln
            else:
                f1 = f0 + normalize(a + sv * 0.25 * k) * fing
            b.mesh(tube_polys(tw(f0), tw(f1), 0.018, side=Mw @ sv), skin * 0.95)
    # pescoço e cabeça
    n0 = L(C(0, 2.08, 0.01)); n1 = L(C(0, 2.27, 0.03))
    b.mesh(tube_polys(tw(n0), tw(n1), 0.07, side=side_w), skin)
    hs = p.get("head_scale", 1.0)
    hc = L(C(0, 2.27 + head.ry * head.chin * 0.78 * hs, 0.05))
    hy, hpch, hr = p.get("head_yaw", 0.0), p.get("head_pitch", 0.0), p.get("head_roll", 0.0)
    if vio is not None:          # queixo apoiado no violino
        hy, hpch, hr = hy + 0.30, hpch + 0.22, hr - 0.30
        _violin_mesh(b, tw, vfr, vio_frog, vio_bd)
    R = Mw @ RL @ rot_y(hy) @ rot_x(hpch) @ rot_z(hr)
    if sc != 1.0 or hs != 1.0:
        R = R * (sc * hs)
    if head is HEAD_W:
        hp = dict(px=p.get("px", 0.0), py=p.get("py", 0.0), eye_open=p.get("eye_open", 1.0), pupil=p.get("pupil", 0.62),
                  eye_emis=p.get("eye_emis", 0.82), glow=p.get("glow", True), smile=p.get("grin", 0.25),
                  mouth_open=p.get("mouth_open", 0.0), eye_r=0.056, eye_dx=0.068, eye_y=0.05, mouth_y=-0.13,
                  mw=p.get("mw", 0.06), nose_y=-0.04, mouth_col=C(0.05, 0.0, 0.01))
    else:
        hp = dict(px=p.get("px", 0.0), py=p.get("py", 0.0), eye_open=p.get("eye_open", 1.0), pupil=p.get("pupil", 1.0),
                  eye_emis=p.get("eye_emis", 0.35), glow=p.get("glow", False), smile=p.get("grin", 0.0),
                  mouth_open=p.get("mouth_open", 0.0))
    hp.update(p.get("head_extra", {}))
    build_head(b, head, R, tw(hc), skin, hp, hat=p.get("hat", True))
    if p.get("fogmul", 1.0) != 1.0:
        b.scale_fog(start_mark, p["fogmul"])


VIOLIN_WOOD = C(0.66, 0.28, 0.08)
VS = 1.45                 # violino proporcional a uma figura de 2,7 m
BOW_LEN = 0.72 * VS


def _violin_frame(L, RL):
    vc = C(0.20, 2.12, 0.31)
    nd = normalize(C(0.66, 0.06, 0.75))
    n0 = C(-0.38, 0.92, 0.05)
    nrm = normalize(n0 - nd * np.dot(n0, nd))
    ac = normalize(np.cross(nrm, nd))
    return L(vc), RL @ nd, RL @ nrm, RL @ ac


def _violin_hands(fr, u):
    vc, nd, nrm, ac = fr
    left = vc + nd * 0.37 * VS - nrm * 0.03
    contact = vc - nd * 0.05 * VS + nrm * 0.04 * VS
    bd = ac if ac[0] > 0 else -ac
    frog = contact - bd * (0.12 + 0.74 * float(np.clip(u, 0, 1))) * BOW_LEN
    return left, frog, bd


def _oct_slab(c, ax, ay, az, rx, ry, z0, z1):
    a = 2 * np.pi * np.arange(8) / 8 + np.pi / 8
    ring = [c + ax * rx * math.cos(t) + ay * ry * math.sin(t) for t in a]
    r0 = np.array([q + az * z0 for q in ring])
    r1 = np.array([q + az * z1 for q in ring])
    polys = [r0, r1[::-1]]
    for i in range(8):
        k = (i + 1) % 8
        polys.append(np.array([r0[i], r0[k], r1[k], r1[i]]))
    return orient_convex(polys)


def _vbox(vc, nd, ac, nrm, x0, x1, w, z0, z1):
    """caixa no referencial do violino (x ao longo do braço, y transversal, z normal)."""
    pts = []
    for (x, y, z) in [(x0, -w, z0), (x1, -w, z0), (x1, w, z0), (x0, w, z0), (x0, -w, z1), (x1, -w, z1), (x1, w, z1),
                      (x0, w, z1)]:
        pts.append(vc + nd * x + ac * y + nrm * z)
    V = np.array(pts)
    F = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 4, 7, 3), (1, 2, 6, 5), (0, 1, 5, 4), (3, 7, 6, 2)]
    return orient_convex([V[list(f)] for f in F])


def _violin_mesh(b, tw, fr, frog, bd):
    vc, nd, nrm, ac = fr
    S = VS
    T = lambda polys: [tw(q) for q in polys]
    wood = VIOLIN_WOOD
    b.mesh(T(_oct_slab(vc - nd * 0.075 * S, nd, ac, nrm, 0.105 * S, 0.108 * S, -0.022 * S, 0.022 * S)), wood, emis=0.1)
    b.mesh(T(_oct_slab(vc + nd * 0.09 * S, nd, ac, nrm, 0.085 * S, 0.085 * S, -0.021 * S, 0.021 * S)), wood * 1.08,
           emis=0.1)
    b.mesh(T(_vbox(vc, nd, ac, nrm, 0.15 * S, 0.37 * S, 0.014 * S, 0.0, 0.024 * S)), wood * 0.75)
    b.mesh(T(_vbox(vc, nd, ac, nrm, -0.03 * S, 0.34 * S, 0.017 * S, 0.022 * S, 0.034 * S)), C(0.03, 0.03, 0.03))
    b.mesh(T(_vbox(vc, nd, ac, nrm, 0.37 * S, 0.44 * S, 0.016 * S, -0.012 * S, 0.036 * S)), wood * 0.7)
    b.mesh(T(_vbox(vc, nd, ac, nrm, -0.165 * S, -0.10 * S, 0.019 * S, 0.022 * S, 0.031 * S)), C(0.03, 0.03, 0.03))
    b.mesh(T(_vbox(vc, nd, ac, nrm, -0.05 * S, -0.038 * S, 0.022 * S, 0.022 * S, 0.048 * S)), C(0.85, 0.76, 0.58))
    for k in (-1.5, -0.5, 0.5, 1.5):          # cordas
        o = ac * k * 0.006 * S
        b.mesh(tube_polys(tw(vc - nd * 0.10 * S + nrm * 0.040 * S + o), tw(vc + nd * 0.40 * S + nrm * 0.036 * S + o),
                          0.003 * S, side=nrm), C(0.85, 0.85, 0.8), emis=0.3)
    tip = frog + bd * BOW_LEN
    b.mesh(tube_polys(tw(frog), tw(tip), 0.018, side=nrm), C(0.88, 0.85, 0.74), emis=0.2)
    b.mesh(tube_polys(tw(frog + nrm * 0.025), tw(tip + nrm * 0.02), 0.016, side=nrm), C(0.20, 0.09, 0.03))
    b.mesh([tw(q) for q in _vbox(frog + nrm * 0.014, bd, np.cross(nrm, bd), nrm, -0.03, 0.03, 0.014, -0.014,
                                 0.014)], C(0.04, 0.04, 0.04))


def watcher_head_world(pos, yaw, p=None):
    p = p or {}
    sc = p.get("scale", 1.0)
    lean = p.get("lean", 0.0)
    hipc = C(0, 1.27, 0)
    hc = hipc + rot_x(lean) @ (C(0, 2.52, 0.05) - hipc)
    return rot_y(yaw) @ (hc * sc) + np.asarray(pos, float)


def yaw_to(src, dst):
    d = np.asarray(dst, float) - np.asarray(src, float)
    return math.atan2(d[0], d[2])


# ================================================================ cenário
TREE_CROWNS = []   # (centro, raio) das árvores da primeira fileira da rua
WINDOWS = []       # (centro, normal) das janelas voltadas para a rua


def _tree(b, rng, x, z, big=1.0, record=None):
    h = rng.uniform(2.0, 3.0) * big
    b.mesh(prism_polys(6, 0.24 * big, 0.15 * big, h, c=(x, 0, z), rot=rng.uniform(0, 1)), TRUNK, jitter=0.08, rng=rng)
    col = LEAVES[rng.integers(0, 3)]
    nb = rng.integers(1, 3)
    for k in range(nb):
        r = rng.uniform(1.5, 2.3) * big
        c = (x + rng.uniform(-0.6, 0.6), h + r * 0.55 + k * 0.9, z + rng.uniform(-0.6, 0.6))
        b.mesh(ico_polys(r, scale=(1, rng.uniform(0.9, 1.35), 1), jitter=0.18, rng=rng, c=c), col, jitter=0.14, rng=rng)
        if record is not None and k == 0:
            record.append((C(*c), r))


def _house(b, rng, x, z, facing_x, record=None):
    w, d, h = rng.uniform(5, 7), rng.uniform(5, 7), rng.uniform(3.0, 4.0)
    g = b.new_group()
    wall = [C(0.55, 0.50, 0.46), C(0.46, 0.48, 0.55), C(0.60, 0.55, 0.40)][rng.integers(0, 3)]
    b.mesh(box_polys(w, h, d, c=(x, h / 2, z)), wall, group=g, jitter=0.05, rng=rng)
    rh = rng.uniform(1.5, 2.3)
    x0, x1, z0, z1 = x - w / 2 - 0.3, x + w / 2 + 0.3, z - d / 2 - 0.3, z + d / 2 + 0.3
    ridge0, ridge1 = C(x, h + rh, z0), C(x, h + rh, z1)
    roof = [np.array([C(x0, h, z0), C(x0, h, z1), ridge1, ridge0]),
            np.array([C(x1, h, z1), C(x1, h, z0), ridge0, ridge1]),
            np.array([C(x0, h, z0), ridge0, C(x1, h, z0)]),
            np.array([C(x1, h, z1), ridge1, C(x0, h, z1)])]
    cen = C(x, h + rh * 0.3, z)
    roof = [orient(q, q.mean(0) - cen) for q in roof]
    b.mesh(roof, C(0.38, 0.13, 0.10), group=g, jitter=0.06, rng=rng)
    fx = x + facing_x * (w / 2 + 0.03)
    for zz in (z - d * 0.25, z + d * 0.25):
        lit = rng.random() < 0.18
        wc = C(0.9, 0.7, 0.3) if lit else C(0.03, 0.04, 0.06)
        q = np.array([C(fx, 1.2, zz - 0.5), C(fx, 1.2, zz + 0.5), C(fx, 2.2, zz + 0.5), C(fx, 2.2, zz - 0.5)])
        b.poly(orient(q, C(facing_x, 0, 0)), wc, emis=0.8 if lit else 0.0, group=g, layer=1, glow=lit)
        if record is not None:
            record.append((C(fx, 1.7, zz), C(facing_x, 0, 0)))
    q = np.array([C(fx, 0, z - 0.45), C(fx, 0, z + 0.45), C(fx, 1.9, z + 0.45), C(fx, 1.9, z - 0.45)])
    b.poly(orient(q, C(facing_x, 0, 0)), C(0.18, 0.1, 0.07), group=g, layer=1)


def build_road_world(seed=11):
    rng = np.random.default_rng(seed)
    b = Builder(group_base=1_000_000)
    TREE_CROWNS.clear()
    WINDOWS.clear()
    Z0, Z1, seg = -80, 440, 3.0
    xs = [-70, -30, -14, -7, -4.2, -ROAD_HALF, 0.0, ROAD_HALF, 4.2, 7, 14, 30, 70]
    z = Z0
    while z < Z1:
        for i in range(len(xs) - 1):
            x0, x1 = xs[i], xs[i + 1]
            road = (-ROAD_HALF <= x0 < ROAD_HALF)
            sl = seg if abs(x0) < 20 and abs(x1) < 20 else seg * 2
            if sl > seg and (round((z - Z0) / seg) % 2):
                continue
            col = ROAD * (1 + rng.normal(0, 0.035)) if road else GRASS * (1 + rng.normal(0, 0.09))
            q = np.array([C(x0, 0, z), C(x1, 0, z), C(x1, 0, z + sl), C(x0, 0, z + sl)])
            b.poly(orient(q, C(0, 1, 0)), np.clip(col, 0, 1), bucket=2.0 if road else 1.0)
        z += seg
    # faixa central gasta
    for zz in np.arange(Z0, Z1, 7.0):
        q = np.array([C(-0.07, 0.01, zz), C(0.07, 0.01, zz), C(0.07, 0.01, zz + 2.2), C(-0.07, 0.01, zz + 2.2)])
        b.poly(orient(q, C(0, 1, 0)), C(0.62, 0.58, 0.45), bucket=2.5)
    # meio-fio
    for s in (-1, 1):
        for zz in np.arange(Z0, Z1, 6.0):
            b.mesh(box_polys(0.18, 0.12, 6.0, c=(s * (ROAD_HALF + 0.09), 0.06, zz + 3)), C(0.5, 0.48, 0.45))
    # árvores
    for s in (-1, 1):
        z = Z0
        while z < Z1:
            z += rng.uniform(6.5, 11)
            _tree(b, rng, s * rng.uniform(5.4, 10.5), z, record=TREE_CROWNS)
        z = Z0
        while z < Z1:
            z += rng.uniform(12, 22)
            _tree(b, rng, s * rng.uniform(15, 32), z, big=rng.uniform(1.1, 1.5))
        z = Z0
        while z < Z1:
            z += rng.uniform(5, 14)
            r = rng.uniform(0.45, 0.8)
            b.mesh(ico_polys(r, scale=(1.3, 0.8, 1.1), jitter=0.2, rng=rng,
                             c=(s * rng.uniform(3.3, 4.6), r * 0.5, z)), LEAVES[1] * 0.8, jitter=0.12, rng=rng)
    # casas
    for s in (-1, 1):
        z = Z0 + rng.uniform(0, 20)
        while z < Z1:
            _house(b, rng, s * rng.uniform(19, 27), z, -s, record=WINDOWS)
            z += rng.uniform(26, 48)
    # postes (sem a luminária, que é dinâmica)
    for zz in LAMP_ZS:
        b.mesh(prism_polys(6, 0.08, 0.06, LAMP_Y + 0.15, c=(LAMP_X, 0, zz)), C(0.30, 0.31, 0.33))
        b.mesh(tube_polys(C(LAMP_X, LAMP_Y + 0.1, zz), C(LAMP_HEAD_X - 0.1, LAMP_Y + 0.12, zz), 0.07), C(0.30, 0.31, 0.33))
    return b.build()


RING_R = 12.0


def build_ring_world(seed=5):
    rng = np.random.default_rng(seed)
    b = Builder(group_base=2_000_000)
    n = 32
    radii = [0.0, 6.0, RING_R - 1.75, RING_R, RING_R + 1.75, 20.0, 32.0, 60.0]
    for i in range(n):
        a0, a1 = 2 * np.pi * i / n, 2 * np.pi * (i + 1) / n
        for j in range(len(radii) - 1):
            r0, r1 = radii[j], radii[j + 1]
            road = RING_R - 1.75 <= r0 < RING_R + 1.75
            col = ROAD * (1 + rng.normal(0, 0.035)) if road else GRASS * (1 + rng.normal(0, 0.09))
            if r0 == 0:
                q = np.array([C(0, 0, 0), C(r1 * math.cos(a0), 0, r1 * math.sin(a0)), C(r1 * math.cos(a1), 0, r1 * math.sin(a1))])
            else:
                q = np.array([C(r0 * math.cos(a0), 0, r0 * math.sin(a0)), C(r1 * math.cos(a0), 0, r1 * math.sin(a0)),
                              C(r1 * math.cos(a1), 0, r1 * math.sin(a1)), C(r0 * math.cos(a1), 0, r0 * math.sin(a1))])
            b.poly(orient(q, C(0, 1, 0)), np.clip(col, 0, 1), bucket=2.0 if road else 1.0)
    for k in range(5):
        a = 2 * np.pi * k / 5 + 0.3
        r = rng.uniform(4.5, 7.0)
        _tree(b, rng, r * math.cos(a), r * math.sin(a))
    for k in range(34):
        a = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(16.5, 42)
        _tree(b, rng, r * math.cos(a), r * math.sin(a), big=rng.uniform(0.9, 1.4))
    for k in range(5):
        a = 2 * np.pi * k / 5 + 0.9
        r = rng.uniform(30, 40)
        _house(b, rng, r * math.cos(a), r * math.sin(a), 1.0)
    return b.build()


# ================================================================ céu
STAR_RNG = np.random.default_rng(3)
_sd = STAR_RNG.normal(size=(260, 3))
_sd[:, 1] = np.abs(_sd[:, 1]) + 0.12
STAR_DIRS = _sd / np.linalg.norm(_sd, axis=1, keepdims=True)
STAR_B = STAR_RNG.uniform(0.25, 0.75, 260)

CLOUDS = [(az, el, w, h) for az, el, w, h in zip(
    np.linspace(0, 360, 13)[:-1] + STAR_RNG.uniform(-12, 12, 12),
    STAR_RNG.uniform(6, 22, 12), STAR_RNG.uniform(40, 80, 12), STAR_RNG.uniform(5, 9, 12))]


def dir_from(az_deg, el_deg):
    az, el = math.radians(az_deg), math.radians(el_deg)
    return C(math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az))


def sky_image(cam, W, H, env, t):
    fpx = cam.fpx(H)
    r, u, f = cam.R
    xs = np.array([0.0, W / 2.0])
    rows = np.arange(H) + 0.5
    yc = (H / 2 - rows) / fpx
    d = u[None, :] * yc[:, None] + f[None, :]
    el = np.arcsin(np.clip(d[:, 1] / np.linalg.norm(d, axis=1), -1, 1))
    k = np.clip(el / 0.6, 0, 1) ** 0.6
    col = env["fogcol"][None, :] * (1 - k[:, None]) + env["zenith"][None, :] * k[:, None]
    arr = np.ascontiguousarray(np.broadcast_to((np.clip(col, 0, 1) * 255).astype(np.uint8)[:, None, :], (H, W, 3)))
    img = Image.fromarray(arr)
    st = env.get("stars", 1.0)
    if st > 0:
        dc = (STAR_DIRS @ rot_y(env.get("star_rot", 0.0)).T) @ cam.R.T
        vis = dc[:, 2] > 0.05
        dr = ImageDraw.Draw(img)
        sx = W / 2 + fpx * dc[:, 0] / np.maximum(dc[:, 2], 1e-3)
        sy = H / 2 - fpx * dc[:, 1] / np.maximum(dc[:, 2], 1e-3)
        tw = 0.6 + 0.4 * np.sin(t * 3.0 + np.arange(len(STAR_B)) * 1.7)
        for i in np.nonzero(vis)[0]:
            if STAR_DIRS[i, 1] < 0.08:
                continue
            bb = STAR_B[i] * tw[i] * st
            c = tuple(int(v) for v in (C(0.75, 0.8, 0.95) * bb * 255))
            x, y = sx[i], sy[i]
            if 0 <= x < W and 0 <= y < H:
                dr.rectangle([x, y, x + 2, y + 2], fill=c)
    return img


def add_sky_objects(b, cam, env, t):
    D = 400.0
    md = env["moon_dir"]
    if env.get("moon", 1.0) > 0:
        R = D * math.tan(math.radians(2.6 * env.get("moon_size", 1.0)))
        ctr = cam.pos + md * D
        mc = env.get("moon_col_disc", C(0.93, 0.90, 0.76)) * env.get("moon", 1.0)
        b.poly(disc_poly(ctr, -md, C(0, 1, 0), R, R, n=8, rot=0.2), mc, emis=1.0, bucket=0.0, fogmul=0.0, glow=True)
        me = env.get("moon_eye", 0.0)
        if me > 0:
            b.poly(disc_poly(ctr - md * 1.0, -md, C(0, 1, 0), R * 0.62 * me, R * 0.62 * me, n=8, rot=0.2),
                   C(0.55, 0.08, 0.06), emis=1.0, bucket=0.05, fogmul=0.0, glow=True)
            b.poly(disc_poly(ctr - md * 2.0, -md, C(0, 1, 0), R * 0.14 * me, R * 0.55 * me, n=6, rot=0.0),
                   C(0.0, 0.0, 0.0), emis=1.0, bucket=0.08, fogmul=0.0, glow=True)
    if env.get("clouds", 1.0) > 0:
        for az, el, w, h in CLOUDS:
            dd = dir_from(az + env.get("sky_t", t) * 0.6, el)
            ctr = cam.pos + dd * D
            nrm = -dd
            up = C(0, 1, 0)
            uu = normalize(up - nrm * np.dot(up, nrm))
            ss = np.cross(uu, nrm)
            pts = np.array([ctr - ss * w, ctr - ss * w * 0.45 + uu * h, ctr + ss * w * 0.5 + uu * h * 0.8,
                            ctr + ss * w, ctr + ss * w * 0.35 - uu * h * 0.4, ctr - ss * w * 0.55 - uu * h * 0.35])
            b.poly(orient(pts, nrm), env.get("cloud_col", C(0.17, 0.20, 0.28)) * env.get("clouds", 1.0), emis=1.0,
                   bucket=0.2, fogmul=0.0,
                   glow=True)


SKY_EYES = []
_er = np.random.default_rng(21)
for i in range(48):
    SKY_EYES.append((_er.uniform(-1, 1), _er.uniform(-1, 1), _er.uniform(1.1, 2.8), _er.uniform(0, 1),
                     _er.uniform(0, 6.28)))


def add_sky_eyes(b, cam, env, t):
    """env['sky_eyes'] = (t_start, t_full) — abrem escalonados."""
    se = env.get("sky_eyes")
    if not se:
        return
    t0, t1 = se
    D = 380.0
    target = env.get("sky_eye_target", cam.pos)
    caz, cel, saz, sel = env.get("sky_view", (0, 30, 30, 25))
    for i, (u, v, size, order, ph) in enumerate(SKY_EYES):
        az = caz + u * saz
        el = max(8.0, cel + v * sel)
        ts = t0 + (t1 - t0) * order
        o = np.clip((t - ts) / 0.35, 0, 1)
        o = o * o * (3 - 2 * o)
        blink = 1.0 - max(0.0, math.sin(t * 0.9 + ph * 3) - 0.985) * 60
        o *= np.clip(blink, 0, 1)
        if o <= 0.03:
            continue
        dd = dir_from(az, el)
        ctr = cam.pos + dd * D
        nrm = -dd
        R = D * math.tan(math.radians(size))
        up = C(0, 1, 0)
        b.poly(disc_poly(ctr, nrm, up, R, R * 0.62 * o, n=6), EYE_W * 0.92, emis=1.0, bucket=0.5, fogmul=0.0, glow=True)
        look = normalize(np.asarray(target) - ctr)
        uu = normalize(up - nrm * np.dot(up, nrm)); ss = np.cross(uu, nrm)
        off = (ss * np.dot(look, ss) + uu * np.dot(look, uu)) * R * 0.9
        pc = ctr + off - dd * 0.5 + 0 * nrm
        b.poly(disc_poly(pc, nrm, up, R * 0.28, R * 0.28 * o, n=6), PUPIL, emis=1.0, bucket=0.55, fogmul=0.0, glow=True)


def add_lamps(b, cam, env, t, lamp_zs=LAMP_ZS, ring=False):
    """Luminárias dinâmicas; devolve luzes ativas (até 5 mais próximas)."""
    fn = env.get("lamp_fn")
    cand = []
    if ring:
        pos_list = [C((RING_R + 2.4) * math.cos(a), LAMP_Y, (RING_R + 2.4) * math.sin(a)) for a in
                    2 * np.pi * np.arange(6) / 6]
    else:
        pos_list = [C(LAMP_HEAD_X, LAMP_Y, z) for z in lamp_zs]
    for i, lp in enumerate(pos_list):
        if np.linalg.norm(lp - cam.pos) > env["maxdist"]:
            continue
        br = 1.0 if fn is None else fn(i, lp, t)
        col = C(0.25, 0.25, 0.27) * (1 - br) + LAMP_LIT * br
        lf = env.get("lamp_fog", 1.0)
        b.mesh(box_polys(0.55, 0.16, 0.34, c=lp), col, emis=br, glow=br > 0.3, fogmul=1.0 - (1.0 - lf) * br)
        if ring:
            b.mesh(prism_polys(6, 0.08, 0.06, LAMP_Y, c=(lp[0], 0, lp[2])), C(0.3, 0.31, 0.33))
        if br > 0.02:
            cand.append((np.linalg.norm(lp - cam.pos), (lp - C(0, 0.25, 0), LAMP_LIT, 1.35 * br, 5.5)))
    cand.sort(key=lambda x: x[0])
    return [c for _, c in cand[:5]]
