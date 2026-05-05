"""
Create a 3D animated GIF of the spray colored by origId using matplotlib.
No display/Xvfb needed.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import pyvista as pv

# Setup
vtk_dir = Path("/home/rmishra/projects/SMLS/suc/raw_data/VTK/lagrangian/sprayCloud/Lagrangian_VTK")
output_path = Path("/home/rmishra/projects/SMLS/presentation/spray_animation.gif")

# Get all VTK files sorted by timestep number
vtk_files = sorted(vtk_dir.glob("sprayCloud_*.vtk"), key=lambda f: int(f.stem.split("_")[1]))
print(f"Found {len(vtk_files)} VTK files")

# Sample frames
n_frames = 40
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
print(f"Bounds: x=[{xmin:.4f},{xmax:.4f}] y=[{ymin:.4f},{ymax:.4f}] z=[{zmin:.4f},{zmax:.4f}]")

# Create figure
fig = plt.figure(figsize=(10, 8), dpi=100)
ax = fig.add_subplot(111, projection='3d')

# Normalize origId for colormap
norm = plt.Normalize(vmin=origid_min, vmax=origid_max)
cmap = plt.cm.turbo

def update(frame_idx):
    ax.cla()
    pts, oid, ts = frames_data[frame_idx]
    
    if len(pts) > 0:
        # Subsample if too many points for performance
        max_pts = 5000
        if len(pts) > max_pts:
            idx = np.random.choice(len(pts), max_pts, replace=False)
            pts = pts[idx]
            oid = oid[idx]
        
        colors = cmap(norm(oid))
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], 
                   c=colors, s=1.5, alpha=0.7, depthshade=True)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel('X [m]', fontsize=9)
    ax.set_ylabel('Y [m]', fontsize=9)
    ax.set_zlabel('Z [m]', fontsize=9)
    ax.set_title(f'3D Spray Evolution — Colored by origId\nTimestep: {ts}  |  Parcels: {len(pts)}', 
                 fontsize=11, fontweight='bold')
    
    # Fixed isometric view
    ax.view_init(elev=30, azim=45)
    
    return ax,

print("Generating animation...")
ani = FuncAnimation(fig, update, frames=len(frames_data), interval=200, blit=False)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label('origId', fontsize=10)

plt.tight_layout()
ani.save(str(output_path), writer=PillowWriter(fps=5))
print(f"\nAnimation saved to: {output_path}")
print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
