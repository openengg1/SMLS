"""
Export individual PNG frames for the LaTeX animate package.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import pyvista as pv

# Setup
vtk_dir = Path("/home/rmishra/projects/SMLS/suc/raw_data/VTK/lagrangian/sprayCloud/Lagrangian_VTK")
frames_dir = Path("/home/rmishra/projects/SMLS/presentation/spray_frames")
frames_dir.mkdir(exist_ok=True)

# Get all VTK files sorted
vtk_files = sorted(vtk_dir.glob("sprayCloud_*.vtk"), key=lambda f: int(f.stem.split("_")[1]))
print(f"Found {len(vtk_files)} VTK files")

# Use 20 frames for animate package (keeps PDF size reasonable)
n_frames = 20
step = max(1, len(vtk_files) // n_frames)
selected_files = vtk_files[::step][:n_frames]
print(f"Using {len(selected_files)} frames")

# Load all frame data
frames_data = []
origid_min, origid_max = float('inf'), float('-inf')
xmin, xmax = float('inf'), float('-inf')
ymin, ymax = float('inf'), float('-inf')
zmin, zmax = float('inf'), float('-inf')

for f in selected_files:
    mesh = pv.read(str(f))
    if mesh.n_points > 0:
        pts = np.array(mesh.points)
        oid = np.array(mesh.point_data['origId'])
        frames_data.append((pts, oid, int(f.stem.split("_")[1])))
        origid_min = min(origid_min, oid.min())
        origid_max = max(origid_max, oid.max())
        xmin, xmax = min(xmin, pts[:, 0].min()), max(xmax, pts[:, 0].max())
        ymin, ymax = min(ymin, pts[:, 1].min()), max(ymax, pts[:, 1].max())
        zmin, zmax = min(zmin, pts[:, 2].min()), max(zmax, pts[:, 2].max())
    else:
        frames_data.append((np.empty((0, 3)), np.array([]), int(f.stem.split("_")[1])))

print(f"origId range: {origid_min} - {origid_max}")

norm = plt.Normalize(vmin=origid_min, vmax=origid_max)
cmap = plt.cm.turbo

for frame_idx, (pts, oid, ts) in enumerate(frames_data):
    fig = plt.figure(figsize=(8, 6), dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    
    if len(pts) > 0:
        max_pts = 5000
        if len(pts) > max_pts:
            idx = np.random.choice(len(pts), max_pts, replace=False)
            pts = pts[idx]
            oid = oid[idx]
        
        colors = cmap(norm(oid))
        sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], 
                       c=colors, s=2, alpha=0.75, depthshade=True)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel('X [m]', fontsize=9)
    ax.set_ylabel('Y [m]', fontsize=9)
    ax.set_zlabel('Z [m]', fontsize=9)
    ax.set_title(f'3D Spray — Colored by origId  |  Timestep: {ts}  |  Parcels: {len(pts)}', 
                 fontsize=10, fontweight='bold')
    
    # Fixed isometric view
    ax.view_init(elev=30, azim=45)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.55, pad=0.08)
    cbar.set_label('origId', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(frames_dir / f"frame_{frame_idx:03d}.png", bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Frame {frame_idx+1}/{len(frames_data)}")

print(f"\nFrames saved to: {frames_dir}/")
