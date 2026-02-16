import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Paths
fun_file = 'FUN.NSGAIII.DTLZ2'
out_dir = 'notebooks/output'
out_file = os.path.join(out_dir, 'nsgaiii_dtlz2_front.png')

# Ensure output dir
os.makedirs(out_dir, exist_ok=True)

# Load front
if not os.path.exists(fun_file):
    raise SystemExit(f'File not found: {fun_file}')

try:
    data = np.loadtxt(fun_file)
except Exception:
    # try reading with flexible parsing
    data = np.genfromtxt(fun_file)

if data.size == 0:
    raise SystemExit('No data found in front file')

# If data is 1D, reshape
if data.ndim == 1:
    if data.size % 3 == 0:
        data = data.reshape(-1, 3)
    else:
        # fallback: plot first two columns
        data = data.reshape(-1, int(data.size))

# Determine dims
dims = data.shape[1]

plt.rcParams.update({'figure.max_open_warning': 0})

if dims >= 3:
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:,0], data[:,1], data[:,2], s=8, c='C0', alpha=0.8)
    ax.set_xlabel('f1')
    ax.set_ylabel('f2')
    ax.set_zlabel('f3')
    ax.set_title('NSGA-III DTLZ2 approximation front')
    plt.tight_layout()
    fig.savefig(out_file, dpi=200)
    print('Saved 3D scatter to', out_file)
else:
    # 2D plot
    plt.figure(figsize=(6,4))
    plt.scatter(data[:,0], data[:,1], s=8, c='C0', alpha=0.8)
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.title('NSGA-III DTLZ2 approximation front (2D)')
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    print('Saved 2D scatter to', out_file)
