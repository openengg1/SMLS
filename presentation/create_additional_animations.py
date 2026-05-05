"""
Export PNG frames for nParticle, diameter, and velocity magnitude animations.
Fixed isometric view, consistent bounds across frames.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
import pyvista as pv

# Setup
vtk_dir = Path("/home/rmishra/projects/SMLS/suc/raw_data/VTK/lagrangian/sprayCloud/Lagrangian_VTK")
base_dir = Path("/home/rmishra/projects/SMLS/presentation")

# Get all VTK files sorted
vtk_files = sorted(vtk_dir.glob("sprayCloud_*.vtk"), key=lambda f: int(f.stem.split("_")[1]))
print(f"Found {len(vtk_files)} VTK files")

# 20 frames for PDF, 40 for GIF
n_frames_pdf = 20
n_frames_gif = 40
step_pdf = max(1, len(vtk_files) // n_frames_pdf)
step_gif = max(1, len(vtk_files) // n_frames_gif)
selected_pdf = vtk_files[::step_pdf][:n_frames_pdf]
selected_gif = vtk_files[::step_gif][:n_frames_gif]

# Define the three variables to animate
variables = [
    {'name': 'nParticle', 'field': 'nParticle', 'label': 'nParticle', 'cmap': 'viridis'},
    {'name': 'diameter', 'field': 'd', 'label': 'Diameter [m]', 'cmap': 'plasma'},
    {'name': 'velocity_mag', 'field': 'U', 'label': '|U| [m/s]', 'cmap': 'coolwarm'},
]

# First pass: compute global bounds and scalar ranges
print("Computing global bounds...")
xmin, xmax = float('inf'), float('-inf')
ymin, ymax = float('inf'), float('-inf')
zmin, zmax = float('inf'), float('-inf')
scalar_ranges = {v['name']: [float('inf'), float('-inf')] for v in variables}

for f in selected_gif:
    mesh = pv.read(str(f))
    if mesh.n_points > 0:
        pts = np.array(mesh.points)
        xmin, xmax = min(xmin, pts[:, 0].min()), max(xmax, pts[:, 0].max())
        ymin, ymax = min(ymin, pts[:, 1].min()), max(ymax, pts[:, 1].max())
        zmin, zmax = min(zmin, pts[:, 2].min()), max(zmax, pts[:, 2].max())
        
        for v in variables:
            data = np.array(mesh.point_data[v['field']])
            if v['name'] == 'velocity_mag':
                data = np.linalg.norm(data, axis=1)
            scalar_ranges[v['name']][0] = min(scalar_ranges[v['name']][0], np.percentile(data, 1))
            scalar_ranges[v['name']][1] = max(scalar_ranges[v['name']][1], np.percentile(data, 99))

print(f"Bounds: x=[{xmin:.4f},{xmax:.4f}] y=[{ymin:.4f},{ymax:.4f}] z=[{zmin:.4f},{zmax:.4f}]")
for v in variables:
    print(f"  {v['name']}: [{scalar_ranges[v['name']][0]:.4g}, {scalar_ranges[v['name']][1]:.4g}]")


def render_frame(mesh, var_info, scalar_range, fig, ax):
    """Render a single frame for a given variable."""
    ax.cla()
    pts = np.array(mesh.points)
    data = np.array(mesh.point_data[var_info['field']])
    if var_info['name'] == 'velocity_mag':
        data = np.linalg.norm(data, axis=1)
    
    # Subsample for performance
    max_pts = 5000
    if len(pts) > max_pts:
        idx = np.random.choice(len(pts), max_pts, replace=False)
        pts = pts[idx]
        data = data[idx]
    
    norm = plt.Normalize(vmin=scalar_range[0], vmax=scalar_range[1])
    cmap = plt.cm.get_cmap(var_info['cmap'])
    colors = cmap(norm(data))
    
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=2, alpha=0.75, depthshade=True)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel('X [m]', fontsize=9)
    ax.set_ylabel('Y [m]', fontsize=9)
    ax.set_zlabel('Z [m]', fontsize=9)
    ax.view_init(elev=30, azim=45)
    
    return norm, cmap


# Generate frames for each variable
for var_info in variables:
    vname = var_info['name']
    srange = scalar_ranges[vname]
    
    # --- PDF frames ---
    frames_dir = base_dir / f"spray_frames_{vname}"
    frames_dir.mkdir(exist_ok=True)
    print(f"\nGenerating PDF frames for {vname}...")
    
    for frame_idx, vtk_file in enumerate(selected_pdf):
        mesh = pv.read(str(vtk_file))
        ts = int(vtk_file.stem.split("_")[1])
        
        fig = plt.figure(figsize=(8, 6), dpi=120)
        ax = fig.add_subplot(111, projection='3d')
        
        if mesh.n_points > 0:
            norm, cmap = render_frame(mesh, var_info, srange, fig, ax)
            ax.set_title(f'3D Spray — {var_info["label"]}  |  Timestep: {ts}  |  Parcels: {mesh.n_points}',
                        fontsize=10, fontweight='bold')
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.55, pad=0.08)
            cbar.set_label(var_info['label'], fontsize=9)
        
        plt.tight_layout()
        plt.savefig(frames_dir / f"frame_{frame_idx:03d}.png", bbox_inches='tight', facecolor='white')
        plt.close()
    
    print(f"  Saved {n_frames_pdf} frames to {frames_dir}/")
    
    # --- GIF ---
    print(f"  Generating GIF for {vname}...")
    fig = plt.figure(figsize=(10, 8), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    gif_data = []
    for f in selected_gif:
        mesh = pv.read(str(f))
        gif_data.append((mesh, int(f.stem.split("_")[1])))
    
    def make_update(var_info_local, srange_local, gif_data_local):
        def update(frame_idx):
            mesh, ts = gif_data_local[frame_idx]
            ax.cla()
            if mesh.n_points > 0:
                render_frame(mesh, var_info_local, srange_local, fig, ax)
            ax.set_title(f'3D Spray — {var_info_local["label"]}  |  Timestep: {ts}  |  Parcels: {mesh.n_points}',
                        fontsize=11, fontweight='bold')
            return ax,
        return update
    
    update_fn = make_update(var_info, srange, gif_data)
    ani = FuncAnimation(fig, update_fn, frames=len(gif_data), interval=200, blit=False)
    
    sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(var_info['cmap']), 
                                norm=plt.Normalize(vmin=srange[0], vmax=srange[1]))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label(var_info['label'], fontsize=10)
    
    plt.tight_layout()
    gif_path = base_dir / f"spray_animation_{vname}.gif"
    ani.save(str(gif_path), writer=PillowWriter(fps=5))
    plt.close()
    print(f"  GIF saved: {gif_path} ({gif_path.stat().st_size/1024:.1f} KB)")

print("\nAll animations complete!")
