import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------
# Photometry helpers
# ---------------------------------------------

def build_I_gamma_phi_from_two_cplanes(gamma_deg, I_c0, I_c90):
    gamma = np.asarray(gamma_deg, float)
    I0 = np.asarray(I_c0, float)
    I90 = np.asarray(I_c90, float)
    if gamma.ndim != 1:
        raise ValueError("gamma_deg must be 1D")
    if gamma.size != I0.size or gamma.size != I90.size:
        raise ValueError("gamma_deg, I_c0, I_c90 must have same length")

    def I_of(gamma_q_deg, phi_q_deg):
        g = np.asarray(gamma_q_deg, float)
        p = np.asarray(phi_q_deg, float)
        I0g = np.interp(g, gamma, I0)
        I90g = np.interp(g, gamma, I90)
        pr = np.deg2rad(p)
        c2 = np.cos(pr) ** 2
        s2 = np.sin(pr) ** 2
        return I0g * c2 + I90g * s2

    return I_of


def _normalize(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero-length vector")
    return v / n

# ---------------------------------------------
# Lux computation core (point contribution)
# ---------------------------------------------

def illuminance_from_point_photometry_on_horizontal_plane(
    I_of_gamma_phi,
    source_pos,
    X,
    Y,
    plane_z=0.0,
    luminaire_down_axis=(0.0, 0.0, -1.0),
    luminaire_phi0_axis=(1.0, 0.0, 0.0),
):
    x0, y0, z0 = source_pos

    Z = np.full_like(X, plane_z, dtype=float)
    vx = X - x0
    vy = Y - y0
    vz = Z - z0

    r2 = vx * vx + vy * vy + vz * vz
    r = np.sqrt(r2)

    # Normalize vectors from source to points
    ux = vx / r
    uy = vy / r
    uz = vz / r

    # Orthonormal basis for luminaire coordinates
    zprime = _normalize(luminaire_down_axis)

    xprime = np.asarray(luminaire_phi0_axis, float)
    xprime = xprime - np.dot(xprime, zprime) * zprime
    if np.linalg.norm(xprime) < 1e-9:
        raise ValueError("luminaire_phi0_axis cannot be parallel to luminaire_down_axis")
    xprime = _normalize(xprime)

    yprime = _normalize(np.cross(zprime, xprime))

    # Compute gamma, phi
    cos_gamma = ux * zprime[0] + uy * zprime[1] + uz * zprime[2]
    cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
    gamma = np.rad2deg(np.arccos(cos_gamma))

    upx = ux * xprime[0] + uy * xprime[1] + uz * xprime[2]
    upy = ux * yprime[0] + uy * yprime[1] + uz * yprime[2]
    phi = (np.rad2deg(np.arctan2(upy, upx)) % 360.0)

    I = I_of_gamma_phi(gamma, phi)

    # Incidence cosine for horizontal plane
    cos_inc = (-uz)
    E = np.where(cos_inc > 0, I * cos_inc / r2, 0.0)
    return E

# ---------------------------------------------
# Tube model (line source = sum of point sources)
# ---------------------------------------------

def lux_map_tube(
    I_of_gamma_phi,
    origin=(0.0, 0.0, 0.4),
    tube_length=0.6,
    tube_axis=(1.0, 0.0, 0.0),
    plane_z=0.0,
    xlim=(-0.6, 0.6),
    ylim=(-0.6, 0.6),
    nx=241,
    ny=241,
    n_segments=60,
    lumens_to_candela_scale=1.0,
):
    x0, y0, z0 = origin
    axis = _normalize(tube_axis)

    xs = np.linspace(xlim[0], xlim[1], nx)
    ys = np.linspace(ylim[0], ylim[1], ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    s_vals = np.linspace(-tube_length / 2.0, tube_length / 2.0, n_segments)

    segment_scale = lumens_to_candela_scale / float(n_segments)

    def I_segment(gamma_deg, phi_deg):
        return segment_scale * I_of_gamma_phi(gamma_deg, phi_deg)

    lum_down = (0.0, 0.0, -1.0)
    phi0_axis = tuple(axis.tolist())

    E_total = np.zeros_like(X, dtype=float)

    for s in s_vals:
        px = x0 + s * axis[0]
        py = y0 + s * axis[1]
        pz = z0 + s * axis[2]
        E_total += illuminance_from_point_photometry_on_horizontal_plane(
            I_segment,
            (px, py, pz),
            X,
            Y,
            plane_z=plane_z,
            luminaire_down_axis=lum_down,
            luminaire_phi0_axis=phi0_axis,
        )

    return X, Y, E_total

def lux_map_multi_tube(
    I_of_gamma_phi,
    center=(0.0, 0.0, 0.4),
    n_tubes=2,
    tube_spacing=0.25,
    tube_length=0.6,
    tube_axis=(1.0, 0.0, 0.0),
    spacing_axis=(0.0, 1.0, 0.0),
    plane_z=0.0,
    xlim=(-0.7, 0.7),
    ylim=(-0.5, 0.5),
    nx=281,
    ny=201,
    n_segments=80,
    lumens_to_candela_scale=1.0,
):
    cx, cy, cz = center
    s_axis = _normalize(spacing_axis)

    idx = np.arange(n_tubes, dtype=float)
    offsets = (idx - (n_tubes - 1) / 2.0) * tube_spacing

    X = Y = E_total = None

    for off in offsets:
        origin = (cx + off * s_axis[0], cy + off * s_axis[1], cz + off * s_axis[2])
        X, Y, E = lux_map_tube(
            I_of_gamma_phi,
            origin=origin,
            tube_length=tube_length,
            tube_axis=tube_axis,
            plane_z=plane_z,
            xlim=xlim,
            ylim=ylim,
            nx=nx,
            ny=ny,
            n_segments=n_segments,
            lumens_to_candela_scale=lumens_to_candela_scale,
        )
        if E_total is None:
            E_total = np.zeros_like(E)
        E_total += E

    return X, Y, E_total

# ---------------------------------------------
# Room patch discretization (coarse)
# ---------------------------------------------

def make_room_patches(room_L, room_W, room_H, n_div=4):
    """Create coarse patches on 4 walls + ceiling (no floor patches => floor is absorbing)."""
    L2 = room_L / 2.0
    W2 = room_W / 2.0

    patches_c = []
    patches_n = []
    patches_a = []

    def centers_1d(a0, a1, n):
        edges = np.linspace(a0, a1, int(n) + 1)
        return 0.5 * (edges[:-1] + edges[1:]), float((a1 - a0) / n)

    # walls x=+L/2, x=-L/2
    ys, dy = centers_1d(-W2, W2, n_div)
    zs, dz = centers_1d(0.0, room_H, n_div)

    for y in ys:
        for z in zs:
            patches_c.append([L2, y, z])
            patches_n.append([-1.0, 0.0, 0.0])
            patches_a.append(dy * dz)

    for y in ys:
        for z in zs:
            patches_c.append([-L2, y, z])
            patches_n.append([1.0, 0.0, 0.0])
            patches_a.append(dy * dz)

    # walls y=+W/2, y=-W/2
    xs, dx = centers_1d(-L2, L2, n_div)
    for x in xs:
        for z in zs:
            patches_c.append([x, W2, z])
            patches_n.append([0.0, -1.0, 0.0])
            patches_a.append(dx * dz)

    for x in xs:
        for z in zs:
            patches_c.append([x, -W2, z])
            patches_n.append([0.0, 1.0, 0.0])
            patches_a.append(dx * dz)

    # ceiling z=H
    xs, dx = centers_1d(-L2, L2, n_div)
    ys, dy = centers_1d(-W2, W2, n_div)
    for x in xs:
        for y in ys:
            patches_c.append([x, y, room_H])
            patches_n.append([0.0, 0.0, -1.0])
            patches_a.append(dx * dy)

    return np.asarray(patches_c, float), np.asarray(patches_n, float), np.asarray(patches_a, float)
# ---------------------------------------------
# Illuminance on arbitrary target points with normals
# ---------------------------------------------

def illuminance_on_points_from_point_sources_with_normals(
    I_of_gamma_phi,
    sources_xyz,
    targets_xyz,
    target_normals_xyz,
    luminaire_down_axis=(0.0, 0.0, -1.0),
    luminaire_phi0_axis=(1.0, 0.0, 0.0),
    per_source_intensity_scale=1.0,
):
    sources = np.asarray(sources_xyz, float)
    targets = np.asarray(targets_xyz, float)
    normals = np.asarray(target_normals_xyz, float)

    if sources.ndim != 2 or sources.shape[1] != 3:
        raise ValueError("sources_xyz must have shape (N,3)")
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError("targets_xyz must have shape (M,3)")
    if normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError("target_normals_xyz must have shape (M,3)")

    zprime = _normalize(luminaire_down_axis)

    xprime = np.asarray(luminaire_phi0_axis, float)
    xprime = xprime - np.dot(xprime, zprime) * zprime
    if np.linalg.norm(xprime) < 1e-9:
        raise ValueError("luminaire_phi0_axis cannot be parallel to luminaire_down_axis")
    xprime = _normalize(xprime)

    yprime = _normalize(np.cross(zprime, xprime))

    E_total = np.zeros((targets.shape[0],), dtype=float)

    for (sx, sy, sz) in sources:
        vx = targets[:, 0] - sx
        vy = targets[:, 1] - sy
        vz = targets[:, 2] - sz

        r2 = vx * vx + vy * vy + vz * vz
        r = np.sqrt(r2)

        ux = vx / r
        uy = vy / r
        uz = vz / r

        cos_gamma = ux * zprime[0] + uy * zprime[1] + uz * zprime[2]
        cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
        gamma = np.rad2deg(np.arccos(cos_gamma))

        upx = ux * xprime[0] + uy * xprime[1] + uz * xprime[2]
        upy = ux * yprime[0] + uy * yprime[1] + uz * yprime[2]
        phi = (np.rad2deg(np.arctan2(upy, upx)) % 360.0)

        I = per_source_intensity_scale * I_of_gamma_phi(gamma, phi)

        # incoming direction at target: (source-target)/r = -u
        incx, incy, incz = -ux, -uy, -uz
        cos_inc = normals[:, 0] * incx + normals[:, 1] * incy + normals[:, 2] * incz

        E = np.where(cos_inc > 0, I * cos_inc / r2, 0.0)
        E_total += E

    return E_total

# ---------------------------------------------
# Reflections: 1-bounce diffuse walls as secondary emitters
# ---------------------------------------------

def add_first_bounce_reflections_on_plane_from_sources(
    sources_xyz,
    I_of_gamma_phi,
    X,
    Y,
    room_L,
    room_W,
    room_H,
    rho=0.99,
    n_div=4,
    per_source_intensity_scale=1.0,
):
    patches_c, patches_n, patches_a = make_room_patches(room_L, room_W, room_H, n_div=n_div)

    # 1) incident illuminance on patches from sources
    E_patch = illuminance_on_points_from_point_sources_with_normals(
        I_of_gamma_phi,
        sources_xyz=sources_xyz,
        targets_xyz=patches_c,
        target_normals_xyz=patches_n,
        per_source_intensity_scale=per_source_intensity_scale,
    )

    # 2) reflected flux per patch (lumens)
    Phi_ref = float(rho) * E_patch * patches_a

    # 3) Lambertian patch intensity I0 (cd)
    I0_patch = Phi_ref / np.pi

    # 4) contribution to plant plane
    P = np.stack([X.ravel(), Y.ravel(), np.zeros_like(X).ravel()], axis=1)
    E_ref_flat = np.zeros((P.shape[0],), dtype=float)

    n_plane = np.array([0.0, 0.0, 1.0], dtype=float)

    for (pc, pn, I0) in zip(patches_c, patches_n, I0_patch):
        if I0 <= 0:
            continue

        v = P - pc[None, :]
        r2 = np.sum(v * v, axis=1)
        r = np.sqrt(r2)
        u = v / r[:, None]

        # emission cosine from patch
        cos_emit = np.dot(u, pn)

        # incidence on plane
        cos_inc = np.dot(-u, n_plane)

        I_dir = I0 * np.maximum(cos_emit, 0.0)
        E = np.where((cos_emit > 0) & (cos_inc > 0), I_dir * cos_inc / r2, 0.0)
        E_ref_flat += E

    return E_ref_flat.reshape(X.shape), E_patch


# ---------------------------------------------
# Tube segments as explicit point sources (for reflections)
# ---------------------------------------------

def tube_to_point_sources(
    center=(0.0, 0.0, 0.4),
    tube_length=0.6,
    tube_axis=(1.0, 0.0, 0.0),
    n_segments=60,
):
    """Return (n_segments,3) points along tube length."""
    cx, cy, cz = center
    axis = _normalize(tube_axis)
    s_vals = np.linspace(-tube_length / 2.0, tube_length / 2.0, int(n_segments))
    pts = np.stack([
        cx + s_vals * axis[0],
        cy + s_vals * axis[1],
        cz + s_vals * axis[2],
    ], axis=1)
    return pts


def multi_tube_to_point_sources(
    center=(0.0, 0.0, 0.4),
    n_tubes=2,
    tube_spacing=0.25,
    tube_length=0.6,
    tube_axis=(1.0, 0.0, 0.0),
    spacing_axis=(0.0, 1.0, 0.0),
    n_segments=60,
):
    """Convert a multi-tube arrangement into a set of point sources (segments)."""
    cx, cy, cz = center
    s_axis = _normalize(spacing_axis)

    idx = np.arange(int(n_tubes), dtype=float)
    offsets = (idx - (n_tubes - 1) / 2.0) * float(tube_spacing)

    all_pts = []
    for off in offsets:
        c = (cx + off * s_axis[0], cy + off * s_axis[1], cz + off * s_axis[2])
        all_pts.append(tube_to_point_sources(
            center=c,
            tube_length=tube_length,
            tube_axis=tube_axis,
            n_segments=n_segments,
        ))

    return np.concatenate(all_pts, axis=0)


# ---------------------------------------------
# Plotting + area
# ---------------------------------------------

def plot_lux_map(X, Y, E, title="Illuminance (lux)", min_lux_to_show=None, max_lux_to_show=None, draw_threshold_contour=True, tick_step_m=0.1):
    plt.figure(figsize=(7.5, 6.5))
    Z = np.array(E, dtype=float)
    if min_lux_to_show is not None:
        Z = np.ma.masked_less(Z, float(min_lux_to_show))  # hide only below-min values

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(alpha=0.0)

    vmin = float(min_lux_to_show) if min_lux_to_show is not None else None
    vmax = float(max_lux_to_show) if max_lux_to_show is not None else None

    im = plt.pcolormesh(X, Y, Z, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, label="Lux")

    if min_lux_to_show is not None and draw_threshold_contour:
        plt.contour(X, Y, E, levels=[float(min_lux_to_show)], colors=["white"], linewidths=1.5)

    ax = plt.gca()
    ax.set(xlabel="x (m)", ylabel="y (m)", title=title, aspect="equal")
    if tick_step_m and tick_step_m > 0:
        ax.set_xticks(np.arange(np.floor(np.min(X)/tick_step_m)*tick_step_m, np.max(X)+tick_step_m, tick_step_m))
        ax.set_yticks(np.arange(np.floor(np.min(Y)/tick_step_m)*tick_step_m, np.max(Y)+tick_step_m, tick_step_m))
        ax.grid(True, color="white", alpha=0.18, linewidth=0.8)
    plt.show()

def area_above_threshold_m2(X, Y, E, threshold_lux):
    mask = (E >= float(threshold_lux))
    dx = float(np.mean(np.diff(X[0, :])))
    dy = float(np.mean(np.diff(Y[:, 0])))
    return float(np.sum(mask) * abs(dx * dy))


# ---------------------------------------------
# Example
# ---------------------------------------------

if __name__ == "__main__":
    # Example photometry (replace with real tube photometry)
    gamma_deg = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=float)
    I_c0 = 2*200/160*np.array([1325, 1250, 1120, 930, 720, 510, 320, 170, 70, 0], dtype=float)
    I_c90 = 2*200/160*np.array([1325, 1220, 1080, 880, 650, 450, 280, 150, 60, 0], dtype=float)
    I_of = build_I_gamma_phi_from_two_cplanes(gamma_deg, I_c0, I_c90)

    # ------------------------------
    # ROOM PARAMETERS
    # ------------------------------
    ROOM_L = 1.2
    ROOM_W = 0.52
    ROOM_H = 0.25

    RHO = 1.0 # reflection coefficient
    WALL_DIV = 32 # wall divisions for reflecting patches creation

    # ------------------------------
    # TUBE PARAMETERS
    # ------------------------------
    POS_X = 0
    POS_Y = 0
    TUBE_LENGTH_M = 0.6
    N_TUBES = 2
    TUBE_SPACING_M = 0.2
    HEIGHT_M = 0.05
    MIN_LUX_TO_SHOW = 10000 # min for plotting range
    MAX_LUX_TO_SHOW = 35000 # max for plotting range

    N_SEGMENTS = 40 # number of divisions of the LED tube source

    # Scaling of photometry for tube segmentation.
    # If your I_of is for the entire tube, set this to 1.0 (we divide by segments inside lux_map_tube).
    # If your I_of is per-segment, set to N_SEGMENTS.
    LUMENS_TO_CANDELA_SCALE = 1.0

    # Plant plane / plotting
    xlim = (-ROOM_L / 2.0, ROOM_L / 2.0)
    ylim = (-ROOM_W / 2.0, ROOM_W / 2.0)

    # Direct lux on plant plane
    X, Y, E_direct = lux_map_multi_tube(
        I_of,
        center=(POS_X, POS_Y, HEIGHT_M),
        n_tubes=N_TUBES,
        tube_spacing=TUBE_SPACING_M,
        tube_length=TUBE_LENGTH_M,
        tube_axis=(1, 0, 0.0),
        spacing_axis=(0.0, 1.0, 0.0),
        plane_z=0.0,
        xlim=xlim,
        ylim=ylim,
        nx=100,
        ny=100,
        n_segments=N_SEGMENTS,
        lumens_to_candela_scale=LUMENS_TO_CANDELA_SCALE,
    )

    # Convert tubes into explicit segment point sources for reflection computation
    segment_points = multi_tube_to_point_sources(
        center=(POS_X, POS_Y, HEIGHT_M),
        n_tubes=N_TUBES,
        tube_spacing=TUBE_SPACING_M,
        tube_length=TUBE_LENGTH_M,
        tube_axis=(1.0, 0, 0.0),
        spacing_axis=(0.0, 1.0, 0.0),
        n_segments=N_SEGMENTS,
    )

    # IMPORTANT: each segment should emit 1/N_SEGMENTS of tube photometry (same as lux_map_tube)
    # Also, if you have multiple tubes, segmentation already creates N_TUBES*N_SEGMENTS points.
    segment_scale = LUMENS_TO_CANDELA_SCALE / float(N_SEGMENTS)

    E_ref, E_patch = add_first_bounce_reflections_on_plane_from_sources(
        sources_xyz=segment_points,
        I_of_gamma_phi=I_of,
        X=X,
        Y=Y,
        room_L=ROOM_L,
        room_W=ROOM_W,
        room_H=ROOM_H,
        rho=RHO,
        n_div=WALL_DIV,
        per_source_intensity_scale=segment_scale,
    )

    E_total = E_direct + E_ref


    area = area_above_threshold_m2(X, Y, E_total, MIN_LUX_TO_SHOW)

    print("--- SETUP ---")
    print(f"Room: L={ROOM_L} W={ROOM_W} H={ROOM_H}")
    print(f"Tubes: N={N_TUBES}, L={TUBE_LENGTH_M} m, spacing={TUBE_SPACING_M} m, height={HEIGHT_M} m")
    print(f"Segments per tube: {N_SEGMENTS}, wall_div={WALL_DIV}, rho={RHO}")

    print("--- RESULTS ---")
    print(f"Mean direct lux: {float(np.mean(E_direct)):.1f}")
    print(f"Mean reflected lux (1-bounce): {float(np.mean(E_ref)):.1f}")
    print(f"Mean total lux: {float(np.mean(E_total)):.1f}")
    print(f"Area with E >= {MIN_LUX_TO_SHOW:.0f} lx: {area:.4f} m^2")

    plot_lux_map(
        X, Y, E_total,
        title=f"Lux map TOTAL",
        min_lux_to_show=MIN_LUX_TO_SHOW,
        max_lux_to_show=MAX_LUX_TO_SHOW
    )