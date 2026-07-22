import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------
# Photometry helper
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

# Array of points (single strip)
def build_point_array(center=(0.0, 0.0, 0.4), n_points=60, point_spacing=0.01, array_axis=(1.0, 0.0, 0.0)):
    cx, cy, cz = center
    axis = np.asarray(array_axis, float)
    axis = axis / np.linalg.norm(axis)
    idx = np.arange(int(n_points), dtype=float)
    offsets = (idx - (n_points - 1) / 2.0) * float(point_spacing)
    pts = np.stack([
        cx + offsets * axis[0],
        cy + offsets * axis[1],
        cz + offsets * axis[2],
    ], axis=1)
    return pts

# Multiple arrays of points (multiple strips)
def build_multiple_point_arrays(
    center=(0.0, 0.0, 0.4),
    n_arrays=3,
    array_spacing=0.10,
    n_points=60,
    point_spacing=0.01,
    array_axis=(1.0, 0.0, 0.0),
    spacing_axis=(0.0, 1.0, 0.0),
):
    cx, cy, cz = center
    a_axis = np.asarray(array_axis, float)
    a_axis = a_axis / np.linalg.norm(a_axis)
    s_axis = np.asarray(spacing_axis, float)
    s_axis = s_axis / np.linalg.norm(s_axis)

    idx = np.arange(int(n_arrays), dtype=float)
    array_offsets = (idx - (n_arrays - 1) / 2.0) * float(array_spacing)

    all_pts = []
    for ao in array_offsets:
        c = (cx + ao * s_axis[0], cy + ao * s_axis[1], cz + ao * s_axis[2])
        all_pts.append(
            build_point_array(center=c, n_points=n_points, point_spacing=point_spacing, array_axis=a_axis)
        )
    return np.concatenate(all_pts, axis=0)


# ---------------------------------------------
# Core illuminance on arbitrary target points (vectorized over points, loop over sources)
# ---------------------------------------------

def illuminance_on_points_from_point_sources(
    I_of_gamma_phi,
    sources_xyz,
    targets_xyz,
    luminaire_down_axis=(0.0, 0.0, -1.0),
    luminaire_phi0_axis=(1.0, 0.0, 0.0),
    per_source_intensity_scale=1.0,
):

    sources = np.asarray(sources_xyz, float)
    targets = np.asarray(targets_xyz, float)
    if sources.ndim != 2 or sources.shape[1] != 3:
        raise ValueError("sources_xyz must have shape (N,3)")
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError("targets_xyz must have shape (M,3)")

    # Build luminaire coordinate basis
    zprime = np.asarray(luminaire_down_axis, float)
    zprime = zprime / np.linalg.norm(zprime)
    xprime = np.asarray(luminaire_phi0_axis, float)
    xprime = xprime - np.dot(xprime, zprime) * zprime
    if np.linalg.norm(xprime) < 1e-9:
        raise ValueError("luminaire_phi0_axis cannot be parallel to luminaire_down_axis")
    xprime = xprime / np.linalg.norm(xprime)
    yprime = np.cross(zprime, xprime)
    yprime = yprime / np.linalg.norm(yprime)

    E_total = np.zeros((targets.shape[0],), dtype=float)

    # Surface normals for targets are not known here; we compute 'irradiance-like' cosine by
    # requiring a target normal. For generality, assume target normals are provided separately.
    # In this implementation we will use this function only for wall patches where we know normals
    # and for the plant plane where normal is (0,0,1). So we pass normals through a closure.
    raise RuntimeError("Use illuminance_on_points_from_point_sources_with_normals instead")


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

    # Basis for luminaire photometry
    zprime = np.asarray(luminaire_down_axis, float)
    zprime = zprime / np.linalg.norm(zprime)
    xprime = np.asarray(luminaire_phi0_axis, float)
    xprime = xprime - np.dot(xprime, zprime) * zprime
    if np.linalg.norm(xprime) < 1e-9:
        raise ValueError("luminaire_phi0_axis cannot be parallel to luminaire_down_axis")
    xprime = xprime / np.linalg.norm(xprime)
    yprime = np.cross(zprime, xprime)
    yprime = yprime / np.linalg.norm(yprime)

    E_total = np.zeros((targets.shape[0],), dtype=float)

    # Loop sources (kept for clarity; sources count is moderate in your setups)
    for (sx, sy, sz) in sources:
        vx = targets[:, 0] - sx
        vy = targets[:, 1] - sy
        vz = targets[:, 2] - sz

        r2 = vx * vx + vy * vy + vz * vz
        r = np.sqrt(r2)

        ux = vx / r
        uy = vy / r
        uz = vz / r

        # gamma from direction source->target relative to luminaire down axis
        cos_gamma = ux * zprime[0] + uy * zprime[1] + uz * zprime[2]
        cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
        gamma = np.rad2deg(np.arccos(cos_gamma))

        upx = ux * xprime[0] + uy * xprime[1] + uz * xprime[2]
        upy = ux * yprime[0] + uy * yprime[1] + uz * yprime[2]
        phi = (np.rad2deg(np.arctan2(upy, upx)) % 360.0)

        I = per_source_intensity_scale * I_of_gamma_phi(gamma, phi)

        # incoming direction at target is from target towards source: (source - target)/r = -u
        incx, incy, incz = -ux, -uy, -uz
        cos_inc = normals[:, 0] * incx + normals[:, 1] * incy + normals[:, 2] * incz

        E = np.where(cos_inc > 0, I * cos_inc / r2, 0.0)
        E_total += E

    return E_total


# ---------------------------------------------
# Room patch discretization (coarse)
# ---------------------------------------------

def make_room_patches(room_L, room_W, room_H, n_div=4):
    L2 = room_L / 2.0
    W2 = room_W / 2.0

    patches_c = []
    patches_n = []
    patches_a = []

    # helper to create grid centers
    def centers_1d(a0, a1, n):
        edges = np.linspace(a0, a1, int(n) + 1)
        return 0.5 * (edges[:-1] + edges[1:]), float((a1 - a0) / n)

    # Wall x=+L/2 (normal points -x)
    ys, dy = centers_1d(-W2, W2, n_div)
    zs, dz = centers_1d(0.0, room_H, n_div)
    for y in ys:
        for z in zs:
            patches_c.append([L2, y, z])
            patches_n.append([-1.0, 0.0, 0.0])
            patches_a.append(dy * dz)

    # Wall x=-L/2 (normal +x)
    for y in ys:
        for z in zs:
            patches_c.append([-L2, y, z])
            patches_n.append([1.0, 0.0, 0.0])
            patches_a.append(dy * dz)

    # Wall y=+W/2 (normal -y)
    xs, dx = centers_1d(-L2, L2, n_div)
    for x in xs:
        for z in zs:
            patches_c.append([x, W2, z])
            patches_n.append([0.0, -1.0, 0.0])
            patches_a.append(dx * dz)

    # Wall y=-W/2 (normal +y)
    for x in xs:
        for z in zs:
            patches_c.append([x, -W2, z])
            patches_n.append([0.0, 1.0, 0.0])
            patches_a.append(dx * dz)

    # Ceiling z=H (normal -z)
    xs, dx = centers_1d(-L2, L2, n_div)
    ys, dy = centers_1d(-W2, W2, n_div)
    for x in xs:
        for y in ys:
            patches_c.append([x, y, room_H])
            patches_n.append([0.0, 0.0, -1.0])
            patches_a.append(dx * dy)

    return np.asarray(patches_c, float), np.asarray(patches_n, float), np.asarray(patches_a, float)


# ---------------------------------------------
# Lux map on plant plane using original grid-based function
# ---------------------------------------------

def lux_map_from_point_sources(
    I_of_gamma_phi,
    sources_xyz,
    plane_z=0.0,
    xlim=(-0.6, 0.6),
    ylim=(-0.6, 0.6),
    nx=241,
    ny=241,
    luminaire_down_axis=(0.0, 0.0, -1.0),
    luminaire_phi0_axis=(1.0, 0.0, 0.0),
    per_source_intensity_scale=1.0,
):
    sources = np.asarray(sources_xyz, float)
    if sources.ndim != 2 or sources.shape[1] != 3:
        raise ValueError("sources_xyz must have shape (N,3)")

    # Basis
    zprime = np.asarray(luminaire_down_axis, float)
    zprime = zprime / np.linalg.norm(zprime)

    xprime = np.asarray(luminaire_phi0_axis, float)
    xprime = xprime - np.dot(xprime, zprime) * zprime
    if np.linalg.norm(xprime) < 1e-9:
        raise ValueError("luminaire_phi0_axis cannot be parallel to luminaire_down_axis")
    xprime = xprime / np.linalg.norm(xprime)

    yprime = np.cross(zprime, xprime)
    yprime = yprime / np.linalg.norm(yprime)

    xs = np.linspace(xlim[0], xlim[1], nx)
    ys = np.linspace(ylim[0], ylim[1], ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    Z = np.full_like(X, float(plane_z))

    E_total = np.zeros_like(X, dtype=float)

    for (x0, y0, z0) in sources:
        vx = X - x0
        vy = Y - y0
        vz = Z - z0

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

        cos_inc = -uz
        E = np.where(cos_inc > 0, I * cos_inc / r2, 0.0)
        E_total += E

    return X, Y, E_total


# ---------------------------------------------
# Plotting + area
# ---------------------------------------------

def plot_lux_map(
    X,
    Y,
    E,
    title="Illuminance (lux)",
    min_lux_to_show=None,
    draw_threshold_contour=True,
    tick_step_m=0.1,
):
    plt.figure(figsize=(7.5, 6.5))
    Z = np.array(E, dtype=float)
    if min_lux_to_show is not None:
        Z = np.ma.masked_less(Z, float(min_lux_to_show))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(alpha=0.0)
    im = plt.pcolormesh(X, Y, Z, shading="auto", cmap=cmap)
    plt.colorbar(im, label="Lux")
    if min_lux_to_show is not None and draw_threshold_contour:
        plt.contour(X, Y, E, levels=[float(min_lux_to_show)], colors=["white"], linewidths=1.5)
    ax = plt.gca()
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    if tick_step_m is not None and tick_step_m > 0:
        xmin, xmax = float(np.min(X)), float(np.max(X))
        ymin, ymax = float(np.min(Y)), float(np.max(Y))
        xticks = np.arange(np.floor(xmin / tick_step_m) * tick_step_m, xmax + tick_step_m, tick_step_m)
        yticks = np.arange(np.floor(ymin / tick_step_m) * tick_step_m, ymax + tick_step_m, tick_step_m)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.grid(True, which="major", color="white", alpha=0.18, linewidth=0.8)
    plt.show()


def area_above_threshold_m2(X, Y, E, threshold_lux):
    mask = (E >= float(threshold_lux))
    dx = float(np.mean(np.diff(X[0, :])))
    dy = float(np.mean(np.diff(Y[:, 0])))
    return float(np.sum(mask) * abs(dx * dy))


# ---------------------------------------------
# Reflections: 1-bounce diffuse walls as secondary emitters
# ---------------------------------------------

def add_first_bounce_reflections_on_plane(
    I_of_gamma_phi,
    sources_xyz,
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

    # Compute direct illuminance at patch centers from original light sources
    E_patch = illuminance_on_points_from_point_sources_with_normals(
        I_of_gamma_phi,
        sources_xyz=sources_xyz,
        targets_xyz=patches_c,
        target_normals_xyz=patches_n,
        luminaire_down_axis=(0.0, 0.0, -1.0),
        luminaire_phi0_axis=(1.0, 0.0, 0.0),
        per_source_intensity_scale=per_source_intensity_scale,
    )

    # Reflected flux per patch (lumens)
    Phi_ref = float(rho) * E_patch * patches_a

    # Lambertian patch on-axis intensity (cd)
    I0_patch = Phi_ref / np.pi

    # Now compute contribution of each patch to the plant plane (z=0)
    # Treat each patch as a point source with Lambertian distribution into hemisphere.
    # Lambertian intensity: I(theta) = I0*cos(theta)

    # Flatten plane points for vectorized math
    P = np.stack([X.ravel(), Y.ravel(), np.zeros_like(X).ravel()], axis=1)

    E_ref_flat = np.zeros((P.shape[0],), dtype=float)

    # plane normal is +z
    n_plane = np.array([0.0, 0.0, 1.0], dtype=float)

    for (pc, pn, I0) in zip(patches_c, patches_n, I0_patch):
        if I0 <= 0:
            continue

        # Vector from patch to point on plane
        v = P - pc[None, :]
        r2 = np.sum(v * v, axis=1)
        r = np.sqrt(r2)
        u = v / r[:, None]  # direction from patch to plane point

        # Only emit into interior hemisphere: direction must be in front of patch surface
        # For a Lambertian patch, emission is proportional to cos(theta_emit) where
        # theta_emit is angle between patch normal and outgoing direction.
        cos_emit = np.dot(u, pn)

        # Incidence cosine on the plane: incoming direction is from point to patch = -u
        cos_inc = np.dot(-u, n_plane)

        # Intensity towards that direction
        I_dir = I0 * np.maximum(cos_emit, 0.0)

        E = np.where((cos_emit > 0) & (cos_inc > 0), I_dir * cos_inc / r2, 0.0)
        E_ref_flat += E

    return E_ref_flat.reshape(X.shape), E_patch, Phi_ref


# ---------------------------------------------
# Example usage
# ---------------------------------------------

if __name__ == "__main__":
    # Photometry for a single LM301B-like chip (Lambertian)
    gamma_deg = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=float)
    I_rel = np.cos(np.deg2rad(gamma_deg))
    I0_cd = 39.0 / np.pi  # 39 lm per chip => I0=Phi/pi for Lambertian hemisphere
    I_c0 = I0_cd * I_rel
    I_c90 = I0_cd * I_rel
    I_of = build_I_gamma_phi_from_two_cplanes(gamma_deg, I_c0, I_c90)

    # ------------------------------
    # ROOM PARAMETERS
    # ------------------------------
    ROOM_L = 0.9   # length (x) in meters
    ROOM_W = 0.3   # width  (y) in meters
    ROOM_H = 0.4   # height (z) in meters, ALSO the light height in this model

    RHO = 0.85 # reflection coefficient
    WALL_DIV = 16   # wall divisions for reflecting patches creation

    # ------------------------------
    # STRIP/LIGHT PARAMETERS
    # ------------------------------
    POWER_PER_POINT = 0.3  # W per chip (for power estimate only)
    POS_X = 0
    POS_Y = 0

    HEIGHT_M = 0.01
    LENGTH = 0.4 # LED strip length
    POINTS_PER_M = 60 # LED chips per meter
    N_STRIPS = 2
    STRIP_SPACING_M = 0.05 # spacing between chips

    PHOTOMETRY_SCOPE = "point"  # "assembly" or "point"

    MIN_LUX_TO_SHOW = 2000 # for plotting

    # Build sources (strips centered in room)
    N_POINTS = int(round(POINTS_PER_M * LENGTH))
    POINT_SPACING_M = 1.0 / POINTS_PER_M

    sources = build_multiple_point_arrays(
        center=(POS_X, POS_Y, HEIGHT_M),
        n_arrays=N_STRIPS,
        array_spacing=STRIP_SPACING_M,
        n_points=N_POINTS,
        point_spacing=POINT_SPACING_M,
        array_axis=(1.0, 0.0, 0.0),
        spacing_axis=(0.0, 1.0, 0.0),
    )

    if PHOTOMETRY_SCOPE == "assembly":
        scale = 1.0 / sources.shape[0]
    else:
        scale = 1.0

    # Plot limits: use room footprint
    xlim = (-ROOM_L / 2.0, ROOM_L / 2.0)
    ylim = (-ROOM_W / 2.0, ROOM_W / 2.0)

    # Direct illuminance on plant plane
    X, Y, E_direct = lux_map_from_point_sources(
        I_of,
        sources_xyz=sources,
        plane_z=0.0,
        xlim=xlim,
        ylim=ylim,
        nx=201,
        ny=201,
        per_source_intensity_scale=scale,
    )

    # 1-bounce reflected component
    E_ref, E_patch, Phi_ref = add_first_bounce_reflections_on_plane(
        I_of,
        sources_xyz=sources,
        X=X,
        Y=Y,
        room_L=ROOM_L,
        room_W=ROOM_W,
        room_H=ROOM_H,
        rho=RHO,
        n_div=WALL_DIV,
        per_source_intensity_scale=scale,
    )

    E_total = E_direct + E_ref

    area = area_above_threshold_m2(X, Y, E_total, MIN_LUX_TO_SHOW)

    print("--- SETUP ---")
    print(f"Room: L={ROOM_L} m, W={ROOM_W} m, H={ROOM_H} m")
    print(f"Reflectance rho={RHO}, wall divisions={WALL_DIV} (coarse)")
    print(f"Total point sources: {sources.shape[0]}")
    print(f"Total consumed power (estimate): {sources.shape[0] * POWER_PER_POINT:.1f} W")

    print("--- RESULTS ---")
    print(f"Mean direct lux: {float(np.mean(E_direct)):.1f}")
    print(f"Mean reflected lux (1-bounce): {float(np.mean(E_ref)):.1f}")
    print(f"Mean total lux: {float(np.mean(E_total)):.1f}")
    print(f"Area with E >= {MIN_LUX_TO_SHOW:.0f} lx: {area:.4f} m^2")

    plot_lux_map(
        X,
        Y,
        E_total,
        title=f"Lux map TOTAL (direct + 1-bounce) | strips={round(N_STRIPS,0)}, points/strip={round(N_POINTS,0)}, H={round(HEIGHT_M,2)} m, rho={round(RHO,2)}",
        min_lux_to_show=MIN_LUX_TO_SHOW,
        draw_threshold_contour=True,
        tick_step_m=0.1,
    )