import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.collections import LineCollection
from matplotlib.patches import Arc
import argparse
import random


def generate_blue_noise_points(min_xb, max_xb, min_yb, max_yb, r, k=30):
    """
    Generate points using Bridson's algorithm for Poisson Disc Sampling
    """
    width = max_xb - min_xb
    height = max_yb - min_yb
    cell_size = r / np.sqrt(2)

    cols = int(np.ceil(width / cell_size))
    rows = int(np.ceil(height / cell_size))

    grid = [[None for _ in range(rows)] for _ in range(cols)]
    points = []
    active_list = []

    # Start with a random point
    x = random.uniform(0, width)
    y = random.uniform(0, height)
    p = np.array([x, y])

    col = int(x / cell_size)
    row = int(y / cell_size)

    grid[col][row] = p
    points.append(p)
    active_list.append(p)

    while active_list:
        idx = random.randint(0, len(active_list) - 1)
        p = active_list[idx]
        found = False

        for _ in range(k):
            theta = random.uniform(0, 2 * np.pi)
            radius = random.uniform(r, 2 * r)
            new_p = p + radius * np.array([np.cos(theta), np.sin(theta)])

            if not (0 <= new_p[0] < width and 0 <= new_p[1] < height):
                continue

            col = int(new_p[0] / cell_size)
            row = int(new_p[1] / cell_size)

            if grid[col][row] is not None:
                continue

            ok = True
            for i in range(max(0, col - 2), min(cols, col + 3)):
                for j in range(max(0, row - 2), min(rows, row + 3)):
                    neighbor = grid[i][j]
                    if neighbor is not None:
                        if np.linalg.norm(new_p - neighbor) < r:
                            ok = False
                            break
                if not ok:
                    break

            if ok:
                found = True
                grid[col][row] = new_p
                points.append(new_p)
                active_list.append(new_p)
                break

        if not found:
            active_list.pop(idx)

    # Shift to match bounds
    points = np.array(points)
    points[:, 0] += min_xb
    points[:, 1] += min_yb

    return points


def lift_points(points, radii):
    """
    Lift 2D points to 3D for power diagram computation.
    Lifted coordinate z = x^2 + y^2 - r^2.
    """
    z = np.sum(points**2, axis=1) - radii**2
    return np.column_stack((points, z))


def solve_quadratic_interval(A, B, C, min_t=0.0, max_t=1.0):
    """
    Finds the intersection of the interval [min_t, max_t] with the set where At^2 + Bt + C < 0.
    """
    if abs(A) < 1e-9:
        if abs(B) < 1e-9:
            return (min_t, max_t) if C < 0 else None

        t0 = -C / B
        if B > 0:  # t < t0
            interval = (min_t, min(max_t, t0))
        else:  # t > t0
            interval = (max(min_t, t0), max_t)

        if interval[0] >= interval[1]:
            return None
        return interval

    delta = B**2 - 4 * A * C
    if delta < 0:
        return None

    sqrt_delta = np.sqrt(delta)
    t1 = (-B - sqrt_delta) / (2 * A)
    t2 = (-B + sqrt_delta) / (2 * A)

    start = max(min_t, t1)
    end = min(max_t, t2)

    if start < end:
        return (start, end)
    return None


def normalize_angle(theta):
    """Normalize angle to [0, 2pi)"""
    return theta % (2 * np.pi)


def subtract_angular_interval(intervals, remove_start, remove_end):
    """
    Subtracts [remove_start, remove_end] from a list of intervals.
    All angles in [0, 2pi).
    Intervals can wrap around.
    """
    # Normalize inputs
    remove_start = normalize_angle(remove_start)
    remove_end = normalize_angle(remove_end)

    # Check if removal interval wraps around
    if remove_end < remove_start:
        # Split into [remove_start, 2pi) and [0, remove_end]
        intervals = subtract_angular_interval(intervals, remove_start, 2 * np.pi - 1e-9)
        intervals = subtract_angular_interval(intervals, 0, remove_end)
        return intervals

    new_intervals = []
    for start, end in intervals:
        # Check if interval wraps around
        if end < start:
            # Split current interval [start, 2pi) and [0, end]
            # Recurse for simplicity
            sub_res = subtract_angular_interval(
                [(start, 2 * np.pi - 1e-9), (0, end)], remove_start, remove_end
            )
            new_intervals.extend(sub_res)
            continue

        # Now both intervals are non-wrapping
        # Current: [start, end], Remove: [remove_start, remove_end]

        # Case 1: Disjoint
        if remove_end < start or remove_start > end:
            new_intervals.append((start, end))
            continue

        # Case 2: Overlap
        # Keep [start, remove_start] if valid
        if start < remove_start:
            new_intervals.append((start, remove_start))

        # Keep [remove_end, end] if valid
        if remove_end < end:
            new_intervals.append((remove_end, end))

    return new_intervals


def compute_union_boundary_arcs(points, radii, hull, is_lower):
    """
    Compute circular arcs for each site that are on the boundary of the union of balls.
    """
    n_points = len(points)
    active_sites = set(hull.simplices[is_lower].flatten())
    arcs = []
    adj = {i: set() for i in active_sites}

    for s_idx in np.where(is_lower)[0]:
        simplex = hull.simplices[s_idx]
        for i in range(3):
            u = simplex[i]
            v = simplex[(i + 1) % 3]
            adj[u].add(v)
            adj[v].add(u)

    for i in active_sites:
        c_i = points[i]
        r_i = radii[i]
        valid_intervals = [(0.0, 2 * np.pi - 1e-9)]
        neighbors = adj[i]
        fully_covered = False

        for j in neighbors:
            c_j = points[j]
            r_j = radii[j]
            d2 = np.sum((c_j - c_i) ** 2)
            d = np.sqrt(d2)

            if d >= r_i + r_j:
                continue
            if d <= abs(r_j - r_i):
                if r_j > r_i:
                    fully_covered = True
                    break
                else:
                    continue

            phi = np.arctan2(c_j[1] - c_i[1], c_j[0] - c_i[0])
            cos_alpha = (d2 + r_i**2 - r_j**2) / (2 * d * r_i)
            cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
            alpha = np.arccos(cos_alpha)
            valid_intervals = subtract_angular_interval(
                valid_intervals, phi - alpha, phi + alpha
            )
            if not valid_intervals:
                fully_covered = True
                break

        if fully_covered:
            continue

        for start, end in valid_intervals:
            arcs.append(
                {
                    "center": c_i,
                    "radius": r_i,
                    "start_angle": np.degrees(start),
                    "end_angle": np.degrees(end),
                }
            )

    return arcs


def compute_power_diagram_edges(points, radii):
    lifted = lift_points(points, radii)
    try:
        hull = ConvexHull(lifted)
    except Exception as e:
        print(f"ConvexHull failed: {e}")
        return [], [], [], [], {}, []

    is_lower = hull.equations[:, 2] < 0
    lower_equations = hull.equations[is_lower]
    nx = lower_equations[:, 0]
    ny = lower_equations[:, 1]
    nz = lower_equations[:, 2]
    cx = -nx / (2 * nz)
    cy = -ny / (2 * nz)
    centroids = np.column_stack((cx, cy))

    map_full_to_lower = np.full(len(hull.simplices), -1, dtype=int)
    lower_indices = np.where(is_lower)[0]
    map_full_to_lower[lower_indices] = np.arange(len(lower_indices))

    raw_edges = []

    # Compute neighbors for each site based on the Regular Triangulation (lower hull)
    site_neighbors = {i: set() for i in range(len(points))}

    for i, real_idx in enumerate(lower_indices):
        simplex = hull.simplices[real_idx]

        # Add neighbors from this simplex
        u, v, w = simplex
        site_neighbors[u].add(v)
        site_neighbors[u].add(w)
        site_neighbors[v].add(u)
        site_neighbors[v].add(w)
        site_neighbors[w].add(u)
        site_neighbors[w].add(v)

        neighbors = hull.neighbors[real_idx]
        for k in range(3):
            n_idx = neighbors[k]
            v1_idx = simplex[(k + 1) % 3]
            v2_idx = simplex[(k + 2) % 3]

            if n_idx != -1 and is_lower[n_idx]:
                if real_idx < n_idx:
                    j = map_full_to_lower[n_idx]
                    p1 = centroids[i]
                    p2 = centroids[j]
                    raw_edges.append(
                        {
                            "type": "finite",
                            "p1": p1,
                            "p2": p2,
                            "sites": (v1_idx, v2_idx),
                        }
                    )
            else:
                opposite_vertex_idx = simplex[k]
                A = points[v1_idx]
                B = points[v2_idx]
                P = points[opposite_vertex_idx]
                C = centroids[i]
                edge_vec = B - A
                perp = np.array([-edge_vec[1], edge_vec[0]])
                if np.dot(perp, (A + B) / 2 - P) < 0:
                    perp = -perp
                perp = perp / (np.linalg.norm(perp) + 1e-9)
                raw_edges.append(
                    {
                        "type": "infinite",
                        "start": C,
                        "direction": perp,
                        "sites": (v1_idx, v2_idx),
                    }
                )

    processed_pd_edges = []
    alpha_complex_edges = []

    for edge in raw_edges:
        u, v = edge["sites"]
        site_center = points[u]
        site_radius = radii[u]

        if edge["type"] == "finite":
            p1 = edge["p1"]
            p2 = edge["p2"]
            V = p2 - p1
            U = p1 - site_center
            p_start = p1
            max_t = 1.0
        else:
            p1 = edge["start"]
            V = edge["direction"] * 100.0
            U = p1 - site_center
            p_start = p1
            max_t = 100.0

        A_coef = np.dot(V, V)
        B_coef = 2 * np.dot(U, V)
        C_coef = np.dot(U, U) - site_radius**2
        interval = solve_quadratic_interval(A_coef, B_coef, C_coef, 0.0, max_t)

        if interval:
            t_start, t_end = interval
            processed_pd_edges.append(
                {"p1": p1 + t_start * V, "p2": p1 + t_end * V, "type": "restricted"}
            )
            alpha_complex_edges.append((u, v))

            if edge["type"] == "finite":
                processed_pd_edges.append({"p1": p1, "p2": p2, "type": "full_faint"})
            else:
                processed_pd_edges.append(
                    {
                        "p1": p1,
                        "p2": p1 + edge["direction"] * 50.0,
                        "type": "full_faint_infinite",
                    }
                )
        else:
            if edge["type"] == "finite":
                processed_pd_edges.append({"p1": p1, "p2": p2, "type": "full_faint"})
            else:
                processed_pd_edges.append(
                    {
                        "p1": p1,
                        "p2": p1 + edge["direction"] * 50.0,
                        "type": "full_faint_infinite",
                    }
                )

    boundary_arcs = compute_union_boundary_arcs(points, radii, hull, is_lower)
    return (
        processed_pd_edges,
        alpha_complex_edges,
        centroids,
        boundary_arcs,
        site_neighbors,
    )


def visualize(
    points,
    radii,
    normals=None,
    filename="power_diagram.png",
    show_alpha=True,
    show_full_pd=True,
    show_diameter=False,
):
    processed_edges, alpha_edges, centroids, boundary_arcs, site_neighbors = (
        compute_power_diagram_edges(points, radii)
    )

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot Circles (disabled by default as per previous user edit)
    # We can enable filling lightly to see the balls if needed, but keeping as is.
    for pt, r in zip(points, radii):
        circle = plt.Circle(pt, r, color="blue", alpha=0.05, fill=True)
        ax.add_patch(circle)

    # Plot Diameters
    if show_diameter and normals is not None:
        diameter_lines = []
        for i, (pt, r, n) in enumerate(zip(points, radii, normals)):
            # Normal n = (nx, ny). Tangent t = (-ny, nx)
            tx, ty = -n[1], n[0]
            U = np.array([tx, ty])

            # Start strict line interval [-r, r]
            t_min, t_max = -r, r

            # Clip against neighbors
            neighbors = site_neighbors.get(i, set())

            for j in neighbors:
                # Neighbor j parameters
                pt_j = points[j]
                r_j = radii[j]

                # Bisector Inequality: t * (U . W) <= B
                # W = Pj - Pi
                W = pt_j - pt

                # B = 0.5 * (|W|^2 + ri^2 - rj^2)
                B = 0.5 * (np.sum(W**2) + r**2 - r_j**2)

                A = np.dot(U, W)

                if abs(A) < 1e-9:
                    # Parallel to bisector. Check if inside.
                    if 0 > B:
                        # Entire line is outside the cell.
                        t_min = 1.0
                        t_max = -1.0
                        break
                elif A > 0:
                    # t <= B/A
                    t_max = min(t_max, B / A)
                else:  # A < 0
                    # t >= B/A
                    t_min = max(t_min, B / A)

            if t_min < t_max:
                start = pt + t_min * U
                end = pt + t_max * U
                diameter_lines.append([start, end])

        if diameter_lines:
            lc_dia = LineCollection(
                diameter_lines,
                colors="green",
                linewidths=1.5,
                linestyles="--",
                label="Diameter",
            )
            ax.add_collection(lc_dia)

        # Draw arrows for normals
        # Scaling the arrow length slightly for visibility, e.g., 0.5 units or proportional to radius
        # Using quiver is usually better for multiple arrows
        plt.scatter(points[:, 0], points[:, 1], color="green", s=15)
        ax.quiver(
            points[:, 0],
            points[:, 1],
            normals[:, 0],
            normals[:, 1],
            color="green",
            scale=None,
            scale_units="xy",
            angles="xy",
            width=0.005,
            headwidth=3,
            headlength=4,
            pivot="tail",
        )

    # Plot Power Diagram Edges
    restricted_lines = []
    faint_lines = []
    faint_infinite_lines = []

    for edge in processed_edges:
        if edge["type"] == "restricted":
            restricted_lines.append([edge["p1"], edge["p2"]])
        elif edge["type"] == "full_faint":
            faint_lines.append([edge["p1"], edge["p2"]])
        elif edge["type"] == "full_faint_infinite":
            faint_infinite_lines.append([edge["p1"], edge["p2"]])

    if show_full_pd:
        if faint_lines:
            lc_faint = LineCollection(
                faint_lines, colors="red", linewidths=2.5, linestyles="solid"
            )
            ax.add_collection(lc_faint)
        if faint_infinite_lines:
            lc_inf = LineCollection(
                faint_infinite_lines,
                colors="red",
                linewidths=2.5,
                linestyles="solid",
            )
            ax.add_collection(lc_inf)

    if show_alpha:
        if restricted_lines:
            lc_res = LineCollection(
                restricted_lines,
                colors="red",
                linewidths=2.5,
                label="Union Boundary / Restricted",
            )
            ax.add_collection(lc_res)

        # Plot Boundary Arcs
        for arc in boundary_arcs:
            p = Arc(
                xy=arc["center"],
                width=2 * arc["radius"],
                height=2 * arc["radius"],
                angle=0,
                theta1=arc["start_angle"],
                theta2=arc["end_angle"],
                color="red",
                linewidth=2.5,
            )
            ax.add_patch(p)

    ax.set_xlim(-8.0, 8.0)
    ax.set_ylim(-8.0, 8.0)
    ax.set_aspect("equal")

    plt.axis("off")
    plt.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0)
    print(f"Saved visualization to {filename}")


if __name__ == "__main__":
    from matplotlib.lines import Line2D  # Import here for legend

    parser = argparse.ArgumentParser(
        description="Visualize Union of Balls, Power Diagram, and Alpha Complex"
    )
    parser.add_argument("--alpha", action="store_true", help="Draw Alpha Complex")
    parser.add_argument(
        "--full_pd", action="store_true", help="Draw full Power Diagram (faint lines)"
    )
    parser.add_argument(
        "--draw_diameter",
        action="store_true",
        help="Draw diameter perpendicular to normal for each cell",
    )

    args = parser.parse_args()

    np.random.seed(42)
    random.seed(42)

    # Blue noise generation
    # Range -5 to 5.
    min_val, max_val = -5, 5
    # Radius for poisson disk.
    # Try r=2.5 to get around 20-30 points in 10x10 area.
    # Area=100. N ~ 100 / (pi * (r/2)^2) ? No, packing density.
    # N ~ Area / (0.8 * r^2) roughly. so 20 ~ 100 / (0.8 * r^2) => r^2 ~ 6 => r ~ 2.4

    points = generate_blue_noise_points(min_val, max_val, min_val, max_val, r=2.5)
    num_points = len(points)
    print(f"Generated {num_points} blue noise points.")

    # Generate radii related to the spacing? Or just random.
    # User had: radii = np.random.rand(num_points) * 1.5 + 1.5
    radii = np.random.rand(num_points) + 2.0

    # Generate random normals (unit vectors)
    normals = np.copy(points)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    visualize(
        points,
        radii,
        normals=normals,
        filename="power_diagram.png",
        show_alpha=args.alpha,
        show_full_pd=args.full_pd,
        show_diameter=args.draw_diameter,
    )
