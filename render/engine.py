"""Motor 3D low-poly minimalista (flat shading, painter's algorithm) em numpy + Pillow."""
import math
import numpy as np
from PIL import Image, ImageDraw

MAXV = 8
NEAR = 0.04


def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def newell(pts):
    p = np.asarray(pts, dtype=np.float64)
    q = np.roll(p, -1, axis=0)
    return np.array([
        np.sum((p[:, 1] - q[:, 1]) * (p[:, 2] + q[:, 2])),
        np.sum((p[:, 2] - q[:, 2]) * (p[:, 0] + q[:, 0])),
        np.sum((p[:, 0] - q[:, 0]) * (p[:, 1] + q[:, 1])),
    ])


def orient(pts, outward):
    pts = np.asarray(pts, dtype=np.float64)
    if np.dot(newell(pts), outward) < 0:
        return pts[::-1].copy()
    return pts


def orient_convex(polys):
    c = np.concatenate(polys).mean(axis=0)
    return [orient(p, p.mean(axis=0) - c) for p in polys]


# ----------------------------------------------------------------- primitives
def box_polys(sx, sy, sz, c=(0, 0, 0)):
    cx, cy, cz = c
    x0, x1 = cx - sx / 2, cx + sx / 2
    y0, y1 = cy - sy / 2, cy + sy / 2
    z0, z1 = cz - sz / 2, cz + sz / 2
    V = np.array([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                  (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)])
    F = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 4, 7, 3), (1, 2, 6, 5), (0, 1, 5, 4), (3, 7, 6, 2)]
    return orient_convex([V[list(f)] for f in F])


def frustum_polys(p0, p1, w0, d0, w1, d1, side=(1, 0, 0)):
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    a = normalize(p1 - p0)
    side = np.asarray(side, float)
    s = side - a * np.dot(side, a)
    if np.linalg.norm(s) < 1e-6:
        s = np.cross(a, [0, 0, 1.0])
        if np.linalg.norm(s) < 1e-6:
            s = np.cross(a, [1.0, 0, 0])
    s = normalize(s)
    u = np.cross(a, s)
    sq = ((-1, -1), (1, -1), (1, 1), (-1, 1))
    c0 = np.array([p0 + s * w0 / 2 * i + u * d0 / 2 * j for i, j in sq])
    c1 = np.array([p1 + s * w1 / 2 * i + u * d1 / 2 * j for i, j in sq])
    polys = [c0, c1[::-1]]
    for i in range(4):
        k = (i + 1) % 4
        polys.append(np.array([c0[i], c0[k], c1[k], c1[i]]))
    return orient_convex(polys)


def tube_polys(p0, p1, w, d=None, side=(1, 0, 0)):
    d = w if d is None else d
    return frustum_polys(p0, p1, w, d, w, d, side)


def prism_polys(n, r0, r1, h, y0=0.0, c=(0, 0, 0), rot=0.0):
    ang = rot + 2 * np.pi * np.arange(n) / n
    cx, cy, cz = c
    ring0 = np.stack([cx + r0 * np.cos(ang), np.full(n, cy + y0), cz + r0 * np.sin(ang)], 1)
    ring1 = np.stack([cx + r1 * np.cos(ang), np.full(n, cy + y0 + h), cz + r1 * np.sin(ang)], 1)
    polys = [ring0.copy(), ring1.copy()]
    for i in range(n):
        k = (i + 1) % n
        polys.append(np.array([ring0[i], ring0[k], ring1[k], ring1[i]]))
    return orient_convex(polys)


def ellipsoid_radii(y, rx, ry, rz, taper, chin):
    if y >= 0:
        s = min(y / ry, 1.0); k = 1.0
    else:
        s = max(y / (ry * chin), -1.0); k = 1.0 - taper * (-s) ** 1.5
    c = math.sqrt(max(0.0, 1 - s * s))
    return rx * k * c, rz * k * c


def ellipsoid_polys(nlon, nlat, rx, ry, rz, taper=0.0, chin=1.0):
    rings = []
    for j in range(1, nlat):
        phi = -math.pi / 2 + math.pi * j / nlat
        s = math.sin(phi)
        y = ry * s * (chin if s < 0 else 1.0)
        ax, az = ellipsoid_radii(y, rx, ry, rz, taper, chin)
        th = 2 * np.pi * np.arange(nlon) / nlon + np.pi / nlon
        rings.append(np.stack([ax * np.sin(th), np.full(nlon, y), az * np.cos(th)], 1))
    bot = np.array([0, -ry * chin, 0.0]); top = np.array([0, ry, 0.0])
    polys = []
    for i in range(nlon):
        k = (i + 1) % nlon
        polys.append(np.array([bot, rings[0][k], rings[0][i]]))
        for j in range(len(rings) - 1):
            polys.append(np.array([rings[j][i], rings[j][k], rings[j + 1][k], rings[j + 1][i]]))
        polys.append(np.array([rings[-1][i], rings[-1][k], top]))
    return orient_convex(polys)


_t = (1 + 5 ** 0.5) / 2
ICO_V = np.array([(-1, _t, 0), (1, _t, 0), (-1, -_t, 0), (1, -_t, 0), (0, -1, _t), (0, 1, _t),
                  (0, -1, -_t), (0, 1, -_t), (_t, 0, -1), (_t, 0, 1), (-_t, 0, -1), (-_t, 0, 1)], float)
ICO_V /= np.linalg.norm(ICO_V, axis=1, keepdims=True)
ICO_F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
         (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5),
         (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]


def ico_polys(r, scale=(1, 1, 1), jitter=0.0, rng=None, c=(0, 0, 0)):
    j = 1 + (jitter * rng.uniform(-1, 1, (12, 1)) if jitter else 0)
    V = ICO_V * r * j * np.asarray(scale) + np.asarray(c)
    return orient_convex([V[list(f)] for f in ICO_F])


def disc_poly(center, normal, up, rx, ry, n=6, rot=0.0):
    n_ = normalize(normal)
    up = np.asarray(up, float)
    u = up - n_ * np.dot(up, n_)
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(n_, [1.0, 0, 0])
    u = normalize(u)
    s = np.cross(u, n_)
    a = rot + 2 * np.pi * np.arange(n) / n
    pts = np.asarray(center, float) + np.outer(np.cos(a) * rx, s) + np.outer(np.sin(a) * ry, u)
    return orient(pts, n_)


def ik2(root, target, l1, l2, pole):
    root = np.asarray(root, float); target = np.asarray(target, float)
    v = target - root
    d = np.linalg.norm(v)
    d = min(max(d, 1e-4), (l1 + l2) * 0.999)
    dirv = normalize(v)
    a = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
    h = math.sqrt(max(l1 * l1 - a * a, 0.0))
    pole = np.asarray(pole, float)
    pp = pole - dirv * np.dot(pole, dirv)
    pp = normalize(pp) if np.linalg.norm(pp) > 1e-6 else np.array([0, 1.0, 0])
    joint = root + dirv * a + pp * h
    end = root + dirv * d
    return joint, end


# ----------------------------------------------------------------- face sets
class Builder:
    def __init__(self, group_base=0):
        self.polys = []
        self.attrs = []
        self._g = group_base

    def new_group(self):
        self._g += 1
        return self._g

    def poly(self, pts, col, emis=0.0, dbl=False, bucket=3.0, group=-1, layer=0, fogmul=1.0, glow=False):
        p = np.asarray(pts, dtype=np.float64)
        if len(p) > MAXV:
            raise ValueError("polygon with too many vertices")
        self.polys.append(p)
        self.attrs.append((col[0], col[1], col[2], emis, float(dbl), bucket, group, layer, fogmul, float(glow)))

    def mesh(self, polys, col, R=None, t=None, jitter=0.0, rng=None, **kw):
        col = np.asarray(col, float)
        for p in polys:
            q = p if R is None else p @ R.T
            if t is not None:
                q = q + t
            c = col
            if jitter:
                c = np.clip(col * (1 + rng.uniform(-jitter, jitter)), 0, 1)
            self.poly(q, c, **kw)

    def mark(self):
        return len(self.attrs)

    def scale_fog(self, start, factor):
        """multiplica a neblina das faces adicionadas desde `start` (exceto as que brilham)."""
        for i in range(start, len(self.attrs)):
            a = self.attrs[i]
            if a[9] < 0.5:
                self.attrs[i] = a[:8] + (a[8] * factor,) + a[9:]

    def build(self):
        return FaceSet.from_lists(self.polys, self.attrs)


class FaceSet:
    __slots__ = ("V", "cnt", "col", "emis", "dbl", "bucket", "group", "layer", "fogmul", "glow", "nrm", "cen")

    @staticmethod
    def empty():
        return FaceSet.from_lists([], [])

    @staticmethod
    def from_lists(polys, attrs):
        fs = FaceSet()
        n = len(polys)
        V = np.zeros((n, MAXV, 3))
        cnt = np.zeros(n, dtype=np.int64)
        for i, p in enumerate(polys):
            k = len(p)
            V[i, :k] = p
            V[i, k:] = p[-1]
            cnt[i] = k
        A = np.array(attrs, dtype=np.float64).reshape(n, 10)
        fs.V = V; fs.cnt = cnt
        fs.col = A[:, 0:3]; fs.emis = A[:, 3]; fs.dbl = A[:, 4] > 0.5
        fs.bucket = A[:, 5]; fs.group = A[:, 6].astype(np.int64); fs.layer = A[:, 7]
        fs.fogmul = A[:, 8]; fs.glow = A[:, 9]
        fs._derive()
        return fs

    def _derive(self):
        V = self.V
        n = len(V)
        if n == 0:
            self.nrm = np.zeros((0, 3)); self.cen = np.zeros((0, 3)); return
        Q = np.roll(V, -1, axis=1)
        nx = np.sum((V[..., 1] - Q[..., 1]) * (V[..., 2] + Q[..., 2]), 1)
        ny = np.sum((V[..., 2] - Q[..., 2]) * (V[..., 0] + Q[..., 0]), 1)
        nz = np.sum((V[..., 0] - Q[..., 0]) * (V[..., 1] + Q[..., 1]), 1)
        N = np.stack([nx, ny, nz], 1)
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
        self.nrm = N
        mask = (np.arange(MAXV)[None, :] < self.cnt[:, None]).astype(np.float64)
        self.cen = (V * mask[..., None]).sum(1) / self.cnt[:, None]

    @staticmethod
    def concat(sets):
        sets = [s for s in sets if len(s.V)]
        fs = FaceSet()
        if not sets:
            return FaceSet.empty()
        for name in FaceSet.__slots__:
            setattr(fs, name, np.concatenate([getattr(s, name) for s in sets]))
        return fs


# ----------------------------------------------------------------- camera / env
class Camera:
    def __init__(self, pos, target, fov=50.0):
        self.pos = np.asarray(pos, dtype=np.float64)
        f = normalize(np.asarray(target, dtype=np.float64) - self.pos)
        up = np.array([0, 1.0, 0])
        if abs(f[1]) > 0.999:
            up = np.array([0, 0, 1.0])
        r = normalize(np.cross(f, up))
        u = np.cross(r, f)
        self.R = np.stack([r, u, f])
        self.fov = fov

    @staticmethod
    def from_dir(pos, fwd, fov=50.0):
        return Camera(pos, np.asarray(pos, float) + np.asarray(fwd, float), fov)

    def fpx(self, H):
        return (H / 2) / math.tan(math.radians(self.fov) / 2)

    def project(self, P, W, H):
        """pontos do mundo -> (x, y, z) em pixels da imagem W x H."""
        P = np.atleast_2d(np.asarray(P, float))
        c = (P - self.pos) @ self.R.T
        f = self.fpx(H)
        z = np.maximum(c[:, 2], 1e-6)
        return np.stack([W / 2 + f * c[:, 0] / z, H / 2 - f * c[:, 1] / z, c[:, 2]], 1)


def _clip_project(P, fpx, W, H, near=NEAR):
    NEAR_ = near
    out = []
    n = len(P)
    for i in range(n):
        a = P[i]; b = P[(i + 1) % n]
        ain = a[2] >= NEAR_; bin_ = b[2] >= NEAR_
        if ain:
            out.append(a)
        if ain != bin_:
            tt = (NEAR_ - a[2]) / (b[2] - a[2])
            out.append(a + (b - a) * tt)
    return [(W / 2 + fpx * p[0] / p[2], H / 2 - fpx * p[1] / p[2]) for p in out]


def render(fs, cam, W, H, env, bg):
    """Desenha FaceSet sobre a imagem de fundo `bg` (PIL RGB)."""
    img = bg
    if len(fs.V) == 0:
        return img
    fpx = cam.fpx(H)
    NEAR = env.get("near", 0.04)
    pos = cam.pos
    dvec = fs.cen - pos
    dist = np.sqrt((dvec * dvec).sum(1))
    facing = -(fs.nrm * dvec).sum(1)
    m = ((dist < env["maxdist"]) | (fs.bucket < 1.0)) & ((facing > 0) | fs.dbl)
    m &= (fs.layer < 2) | (facing > 0.3 * dist)
    idx = np.nonzero(m)[0]
    if len(idx) == 0:
        return img
    cnt = fs.cnt[idx]
    Vc = (fs.V[idx] - pos) @ cam.R.T
    valid = np.arange(MAXV)[None, :] < cnt[:, None]
    z = Vc[..., 2]
    zmax = np.where(valid, z, -np.inf).max(1)
    zmin = np.where(valid, z, np.inf).min(1)
    keep = zmax > NEAR
    idx = idx[keep]; Vc = Vc[keep]; cnt = cnt[keep]; zmin = zmin[keep]
    if len(idx) == 0:
        return img

    # ---- iluminação
    n = fs.nrm[idx]
    fc = facing[idx]
    n = np.where(fc[:, None] < 0, -n, n)
    cen = fs.cen[idx]
    light = np.tile(env["ambient"], (len(idx), 1))
    light += env["moon_col"][None, :] * np.clip(n @ env["moon_dir"], 0, None)[:, None]
    for (lp, lc, li, lr) in env["lights"]:
        v = np.asarray(lp) - cen
        d2 = (v * v).sum(1)
        d = np.sqrt(d2) + 1e-6
        ndl = np.clip((n * v).sum(1) / d, 0, 1)
        att = li * (lr * lr) / (lr * lr + d2) * np.clip(1.6 - d / (lr * 4.0), 0, 1)
        light += np.asarray(lc)[None, :] * (ndl * att)[:, None]
    base = fs.col[idx]
    e = fs.emis[idx][:, None]
    ec = np.minimum(e, 1.0)
    color = base * light * (1 - ec) + base * e
    s = env["silhouette"] * (1 - fs.glow[idx]) * (fs.bucket[idx] >= 1.0)
    color = color * (1 - s[:, None]) + env["silcol"][None, :] * s[:, None]
    fd = dist[idx] * env["fogdens"] * fs.fogmul[idx]
    fog = 1 - np.exp(-fd * (1 + 0.35 * fd))
    color = color * (1 - fog)[:, None] + env["fogcol"][None, :] * fog[:, None]
    color = np.clip(color * env.get("exposure", 1.0), 0, 1)
    cols = [tuple(c) for c in (color * 255 + 0.5).astype(np.int64).tolist()]

    # ---- projeção
    front = zmin >= NEAR
    zs = np.where(z[keep] > NEAR * 0.5, z[keep], NEAR * 0.5)
    sx = W / 2 + fpx * Vc[..., 0] / zs
    sy = H / 2 - fpx * Vc[..., 1] / zs
    big = 1e7
    sxm = np.where(valid[keep], sx, np.nan)
    sym = np.where(valid[keep], sy, np.nan)
    off = front & ((np.nanmax(sxm, 1) < 0) | (np.nanmin(sxm, 1) > W) |
                   (np.nanmax(sym, 1) < 0) | (np.nanmin(sym, 1) > H))

    # ---- ordenação (painter): bucket, profundidade do grupo, camada, profundidade da face
    depth = dist[idx]
    grp = fs.group[idx]
    primary = depth.copy()
    gm = grp >= 0
    if gm.any():
        ug, inv = np.unique(grp[gm], return_inverse=True)
        sums = np.bincount(inv, weights=depth[gm])
        cn = np.bincount(inv)
        primary[gm] = (sums / cn)[inv]
    order = np.lexsort((-depth, fs.layer[idx], -primary, fs.bucket[idx]))

    dr = ImageDraw.Draw(img)
    sxl = sx.tolist(); syl = sy.tolist()
    frontl = front.tolist(); offl = off.tolist(); cntl = cnt.tolist()
    for o in order.tolist():
        if frontl[o]:
            if offl[o]:
                continue
            c = cntl[o]
            pts = list(zip(sxl[o][:c], syl[o][:c]))
        else:
            pts = _clip_project(Vc[o, :cntl[o]], fpx, W, H, NEAR)
            if len(pts) < 3:
                continue
            pts = [(min(max(x, -big), big), min(max(y, -big), big)) for x, y in pts]
        dr.polygon(pts, fill=cols[o])
    return img
