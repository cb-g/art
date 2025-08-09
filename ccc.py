import numpy as np
import matplotlib.pyplot as plt

class Clifford_C_Cloud:
  def __init__(self,
    a=-1.5, b=1.7, c=1.8, d=0.6, p=0.5, q=0.4, r=1.1, s=0.9,
    num_points=10_000_000,
    noise_std=0.001,
  ):
    self.a = a
    self.b = b
    self.c = c
    self.d = d
    self.p = p
    self.q = q
    self.r = r
    self.s = s
    self.num_points = num_points
    self.noise_std = noise_std

  def generate(self):
    x = np.zeros(self.num_points, dtype=np.float32)
    y = np.zeros(self.num_points, dtype=np.float32)
    z = np.zeros(self.num_points, dtype=np.float32)

    for i in range(1, self.num_points):
      x0, y0, z0 = x[i-1], y[i-1], z[i-1]
      x[i] = (
        np.sin(self.a * y0)
        + self.c * np.cos(self.a * x0)
        + self.p * np.sin(self.a * z0)
        + np.random.normal(0, self.noise_std)
      )
      y[i] = (
        np.sin(self.b * x0)
        + self.d * np.cos(self.b * y0)
        + self.q * np.cos(self.b * z0)
        + np.random.normal(0, self.noise_std)
      )
      z[i] = (
        np.sin(self.r * x0)
        - np.cos(self.r * y0)
        + self.s * np.sin(self.r * z0)
        + np.random.normal(0, self.noise_std)
      )

    return x[1000:], y[1000:], z[1000:]

  def rotate_points(self, x, y, z, yaw=0, pitch=0, roll=0):
    """
    Rotate 3D points by yaw, pitch, roll (in radians).
    Yaw: rotation around Z axis
    Pitch: rotation around Y axis
    Roll: rotation around X axis
    """
    # Rotation matrices
    Rz = np.array([
      [np.cos(yaw), -np.sin(yaw), 0],
      [np.sin(yaw),  np.cos(yaw), 0],
      [0,            0,           1]
    ])
    Ry = np.array([
      [np.cos(pitch), 0, np.sin(pitch)],
      [0, 1, 0],
      [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rx = np.array([
      [1, 0, 0],
      [0, np.cos(roll), -np.sin(roll)],
      [0, np.sin(roll),  np.cos(roll)]
    ])

    R = Rz @ Ry @ Rx
    pts = np.vstack((x, y, z))
    rotated = R @ pts
    return rotated[0], rotated[1], rotated[2]

  def project_and_render(self,
    save_path_prefix='ccc', yaw=0, pitch=0, roll=0):

    x, y, z = self.generate()
    x, y, z = self.rotate_points(x, y, z, yaw, pitch, roll)

    # perspective projection
    fov = 100
    cam_offset = 60
    z_proj = z + cam_offset
    x_proj = fov * x / z_proj
    y_proj = fov * y / z_proj
    x_proj = -x_proj

    # painter’s algorithm
    sort_idx = np.argsort(z_proj)
    x_proj = x_proj[sort_idx]
    y_proj = y_proj[sort_idx]
    z_depth = z_proj[sort_idx]

    # normalize and style
    z_norm = (z_depth - z_depth.min()) / (z_depth.max() - z_depth.min())
    sizes = 0.15 + 1.2 * (1 - z_norm)**2
    alphas = 0.03 + 0.6 * (1 - z_norm)

    """
    https://matplotlib.org/stable/users/explain/colors/colormaps.html

    terrain_r, copper_r, PiYG,
    Spectral, BrBG, gist_stern,
    cubehelix, ocean, managua,
    etc.
    """
    colors = plt.cm.terrain_r(z_norm)
    colors[:, -1] = alphas

    fig = plt.figure(figsize=(10, 10), facecolor="#0F1116")
    ax = fig.add_subplot(111, facecolor="#0F1116")
    ax.scatter(x_proj, y_proj, s=sizes, c=colors, edgecolors='none')
    ax.axis("off")
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}.png", format="png", dpi=94) # keep below 1mb
    plt.savefig(f"{save_path_prefix}-m.png", format="png", dpi=300) # higher quality
    plt.close()

Clifford_C_Cloud().project_and_render(yaw=np.radians(90), pitch=np.radians(45), roll=np.radians(10))
