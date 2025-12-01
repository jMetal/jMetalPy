import os
from typing import Iterable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import plotly.express as px
    import plotly.offline as pyoff
    _PLOTLY_AVAILABLE = True
except Exception:
    _PLOTLY_AVAILABLE = False


def save_plt_to_file(solutions: Iterable, filename: str, out_dir: str = "results", html_plotly: bool = False) -> str:
    """Save a visualization of a solution front to a PNG file.

    Parameters
    - solutions: iterable of solutions (objects with `.objectives`) or an array-like numeric front
    - filename: base filename (no extension) used to build output names
    - out_dir: directory where PNG/HTML will be written
    - html_plotly: if True and Plotly is available, write an interactive HTML parallel-coordinates page for multi-objective fronts

    Returns the path to the generated PNG file.
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        pass

    # Convert to numpy array: prefer reading .objectives when available
    try:
        arr = np.asarray([s.objectives for s in solutions], dtype=float)
    except Exception:
        try:
            arr = np.asarray(solutions, dtype=float)
        except Exception:
            raise ValueError("Could not parse solutions into numeric objectives")

    if arr.ndim == 1:
        if arr.size == 0:
            raise ValueError("Empty front provided")
        arr = arr.reshape(-1, 1)

    n_points, n_obj = arr.shape
    png_path = os.path.join(out_dir, f"{filename}_front.png")

    if n_obj == 2:
        plt.figure(figsize=(6, 4))
        plt.scatter(arr[:, 0], arr[:, 1], s=8, c="C0", alpha=0.8)
        plt.xlabel("f1")
        plt.ylabel("f2")
        plt.title(f"{filename} approximation front (2D)")
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close()

        # Optional interactive Plotly output for 2D
        if html_plotly and _PLOTLY_AVAILABLE:
            try:
                import pandas as pd

                df = pd.DataFrame(arr, columns=[f"f{i+1}" for i in range(n_obj)])
                figly = px.scatter(df, x=df.columns[0], y=df.columns[1], title=f"{filename} approximation front (2D)")
                html_path = os.path.join(out_dir, f"{filename}_front.html")
                pyoff.plot(figly, filename=html_path, auto_open=False)
            except Exception:
                pass

    elif n_obj == 3:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], s=8, c="C0", alpha=0.8)
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
        ax.set_zlabel("f3")
        ax.set_title(f"{filename} approximation front (3D)")
        plt.tight_layout()
        fig.savefig(png_path, dpi=200)
        plt.close(fig)

        # Optional interactive Plotly output for 3D
        if html_plotly and _PLOTLY_AVAILABLE:
            try:
                import pandas as pd

                df = pd.DataFrame(arr, columns=[f"f{i+1}" for i in range(n_obj)])
                figly = px.scatter_3d(df, x=df.columns[0], y=df.columns[1], z=df.columns[2], title=f"{filename} approximation front (3D)")
                html_path = os.path.join(out_dir, f"{filename}_front.html")
                pyoff.plot(figly, filename=html_path, auto_open=False)
            except Exception:
                pass

    else:
        # Parallel coordinates (matplotlib): small stacked plots
        fig, axes = plt.subplots(nrows=n_obj, ncols=1, figsize=(8, max(3, n_obj * 1.2)), sharex=True)
        if n_obj == 1:
            axes = [axes]
        for i in range(n_obj):
            ax = axes[i]
            ax.plot(arr[:, i], color="C0", alpha=0.6)
            ax.set_ylabel(f"f{i+1}")
        axes[-1].set_xlabel("solution index")
        fig.suptitle(f"{filename} approximation front (parallel coordinates)")
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close(fig)

        # Optional interactive Plotly output
        if html_plotly and _PLOTLY_AVAILABLE:
            try:
                import pandas as pd

                df = pd.DataFrame(arr, columns=[f"f{i+1}" for i in range(n_obj)])
                figly = px.parallel_coordinates(df, color=df.columns[0])
                html_path = os.path.join(out_dir, f"{filename}_front.html")
                pyoff.plot(figly, filename=html_path, auto_open=False)
            except Exception:
                # Ignore Plotly/Pandas errors
                pass

    return png_path
