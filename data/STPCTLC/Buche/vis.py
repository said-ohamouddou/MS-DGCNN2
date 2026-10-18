import numpy as np
import open3d as o3d

data = np.loadtxt("Jerome_Bu57_SEW36.xyz", skiprows=1)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(data[:, :3])

z = data[:, 2]
z = (z - z.min()) / (z.max() - z.min() + 1e-9)
pcd.colors = o3d.utility.Vector3dVector(np.c_[z, 1 - np.abs(2 * z - 1), 1 - z])

o3d.visualization.draw_geometries([pcd])
