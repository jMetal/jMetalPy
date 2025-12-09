import argparse
import os
import sys
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
        axis_labels = [f"f{i+1}" for i in range(n_obj)]
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], s=8, c="C0", alpha=0.8)
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_zlabel(axis_labels[2])
        ax.set_title(f"{filename} approximation front (3D)")
        plt.tight_layout()
        fig.savefig(png_path, dpi=200)
        plt.close(fig)

        # Optional interactive Plotly output for 3D
        if html_plotly and _PLOTLY_AVAILABLE:
            try:
                import pandas as pd

                df = pd.DataFrame(arr, columns=axis_labels)
                figly = px.scatter_3d(df, x=axis_labels[0], y=axis_labels[1], z=axis_labels[2], title=f"{filename} approximation front (3D)")
                figly.update_layout(scene=dict(xaxis_title=axis_labels[0], yaxis_title=axis_labels[1], zaxis_title=axis_labels[2]))
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


def _read_numeric_csv(csv_path: str) -> np.ndarray:
    """Load numeric columns from a CSV file into a NumPy array."""
    try:
        import pandas as pd  # type: ignore

        df = pd.read_csv(csv_path)
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError("No numeric columns found in CSV")
        arr = numeric_df.to_numpy()
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr
    except Exception:
        pass

    for skip_header in (0, 1):
        data = np.genfromtxt(csv_path, delimiter=",", skip_header=skip_header)
        if data.size == 0 or (isinstance(data, float) and np.isnan(data)):
            continue
        if data.ndim == 0:
            continue
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        data = np.asarray(data, dtype=float)
        if data.ndim > 1:
            data = data[~np.isnan(data).all(axis=1)]
        if data.size == 0 or np.isnan(data).all():
            continue
        return data

    raise ValueError(f"Could not read numeric data from {csv_path}")


def _cli() -> None:
    """Entry point for plotting approximation fronts from CSV files."""
    parser = argparse.ArgumentParser(description="Generate approximation front plots from a numeric CSV file.")
    parser.add_argument("csv_file", help="Path to a CSV file containing numeric columns to plot.")
    parser.add_argument("--out-dir", default="results", help="Directory where output files will be written (default: results).")
    parser.add_argument("--base-name", help="Base name for generated files (defaults to CSV filename without extension).")
    parser.add_argument("--html", action="store_true", help="Also generate an interactive Plotly HTML if available.")
    args = parser.parse_args()

    base_name = args.base_name or os.path.splitext(os.path.basename(args.csv_file))[0]

    try:
        front = _read_numeric_csv(args.csv_file)
    except Exception as ex:
        print(f"Error reading CSV '{args.csv_file}': {ex}", file=sys.stderr)
        sys.exit(1)

    try:
        png_path = save_plt_to_file(front, base_name, out_dir=args.out_dir, html_plotly=args.html)
    except Exception as ex:
        print(f"Error generating plot: {ex}", file=sys.stderr)
        sys.exit(1)

    print(f"PNG saved to: {png_path}")
    if args.html:
        if not _PLOTLY_AVAILABLE:
            print("Plotly not available; HTML file not generated.", file=sys.stderr)
        else:
            html_path = os.path.join(args.out_dir, f"{base_name}_front.html")
            if os.path.isfile(html_path):
                print(f"HTML saved to: {html_path}")


if __name__ == "__main__":
    _cli()
