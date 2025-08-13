#!/usr/bin/env python3
"""Interactive STL viewer with SVG export.

This script loads an STL mesh, displays it in an interactive
Matplotlib 3D window, and allows exporting the current view to SVG.
The user can toggle between full wireframe and surface-only wireframe
(back faces hidden) and between perspective and orthographic projection.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import matplotlib.pyplot as plt
import numpy as np
import trimesh


@dataclass
class ViewerState:
    """Hold mutable viewer state for callbacks."""
    wireframe: bool = False  # False -> surface-wireframe
    perspective: bool = True


def _set_axes_equal(ax: plt.Axes) -> None:
    """Set 3D plot axes to equal scale."""
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    center = limits.mean(axis=1)
    radius = (limits[:, 1] - limits[:, 0]).max() / 2
    ax.set_xlim3d(center[0] - radius, center[0] + radius)
    ax.set_ylim3d(center[1] - radius, center[1] + radius)
    ax.set_zlim3d(center[2] - radius, center[2] + radius)


class STLViewer:
    def __init__(self, mesh: trimesh.Trimesh, state: ViewerState, output: str):
        self.mesh = mesh
        self.state = state
        self.output = output

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_axis_off()
        self.ax.set_box_aspect([1, 1, 1])
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.draw()

    def draw(self) -> None:
        self.ax.cla()
        self.ax.set_axis_off()
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.set_proj_type("persp" if self.state.perspective else "ortho")

        if self.state.wireframe:
            segs = self.mesh.vertices[self.mesh.edges_unique]
            coll = Line3DCollection(segs, colors="k", linewidths=0.5)
            self.ax.add_collection3d(coll)
        else:
            tri = Poly3DCollection(
                self.mesh.triangles,
                facecolors="white",
                edgecolors="k",
                linewidths=0.5,
            )
            self.ax.add_collection3d(tri)
        bounds = self.mesh.bounds
        center = bounds.mean(axis=0)
        radius = (bounds[1] - bounds[0]).max() / 2
        self.ax.set_xlim(center[0] - radius, center[0] + radius)
        self.ax.set_ylim(center[1] - radius, center[1] + radius)
        self.ax.set_zlim(center[2] - radius, center[2] + radius)
        _set_axes_equal(self.ax)
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key == "m":
            self.state.wireframe = not self.state.wireframe
            mode = "wireframe" if self.state.wireframe else "surface-wireframe"
            print(f"Mode: {mode}")
            self.draw()
        elif event.key == "p":
            self.state.perspective = not self.state.perspective
            proj = "perspective" if self.state.perspective else "orthographic"
            print(f"Projection: {proj}")
            self.draw()
        elif event.key == "e":
            print(f"Exporting view to {self.output}")
            self.export_svg()

    def export_svg(self) -> None:
        export_svg_from_view(
            self.mesh,
            self.state,
            self.output,
            elev=self.ax.elev,
            azim=self.ax.azim,
        )

    def show(self) -> None:
        print("Controls: rotate-left, pan-right, zoom-scroll")
        print("Keys: m toggle wireframe, p toggle projection, e export SVG")
        plt.show()


def _visible_wireframe_segments(
    mesh: trimesh.Trimesh,
    cam_pos: np.ndarray,
    cam_dir: np.ndarray,
    obj2cam: np.ndarray,
    perspective: bool,
    samples: int = 20,
) -> np.ndarray:
    """Compute visible edge segments for the current view."""

    rmi = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

    edges = mesh.edges_unique
    if len(edges) == 0:
        return np.empty((0, 2, 3))

    # Map each unique edge to the faces that contain it
    edge_faces = [[] for _ in range(len(edges))]
    for f_idx, e_idxs in enumerate(mesh.faces_unique_edges):
        for e in e_idxs:
            edge_faces[e].append(f_idx)
    edges_faces = np.full((len(edges), 2), -1, dtype=int)
    for e_idx, flist in enumerate(edge_faces):
        edges_faces[e_idx, :len(flist)] = flist[:2]

    front_faces = (mesh.face_normals @ obj2cam) > 0
    mask = np.zeros(len(edges), dtype=bool)
    for i in range(2):
        valid = edges_faces[:, i] >= 0
        mask[valid] |= front_faces[edges_faces[valid, i]]
    edges = edges[mask]

    extent = mesh.scale * 10.0
    ts = np.linspace(0.0, 1.0, samples)

    segments: list[np.ndarray] = []
    for idx0, idx1 in edges:
        v0, v1 = mesh.vertices[[idx0, idx1]]
        pts = v0 + np.outer(ts, v1 - v0)
        if perspective:
            origins = np.repeat(cam_pos[None, :], samples, axis=0)
            dirs = pts - origins
            dists = np.linalg.norm(dirs, axis=1)
            dirs /= dists[:, None]
        else:
            dirs = np.repeat(cam_dir[None, :], samples, axis=0)
            origins = pts - dirs * extent
            dists = np.full(samples, extent)

        hits = rmi.intersects_first(origins, dirs)
        visible = np.isnan(hits) | (hits >= dists - 1e-6)
        if not visible.any():
            continue

        start = None
        for i, vis in enumerate(visible):
            if vis and start is None:
                start = i
            elif not vis and start is not None:
                seg = np.vstack((pts[start], pts[i - 1]))
                if not np.allclose(seg[0], seg[1]):
                    segments.append(seg)
                start = None
        if start is not None:
            seg = np.vstack((pts[start], pts[-1]))
            if not np.allclose(seg[0], seg[1]):
                segments.append(seg)

    return np.array(segments) if segments else np.empty((0, 2, 3))


def export_svg_from_view(
    mesh: trimesh.Trimesh,
    state: ViewerState,
    output: str,
    *,
    elev: float | None = None,
    azim: float | None = None,
) -> None:
    """Render the mesh from a specific view and export it to SVG."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()
    ax.set_proj_type("persp" if state.perspective else "ortho")
    if elev is not None or azim is not None:
        ax.view_init(
            elev=elev if elev is not None else ax.elev,
            azim=azim if azim is not None else ax.azim,
        )
    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    radius = (bounds[1] - bounds[0]).max() / 2
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    _set_axes_equal(ax)

    ax.get_proj()  # update camera parameters
    cam_pos = ax._get_camera_loc()
    obj2cam = ax._view_w
    cam_dir = -obj2cam

    if state.wireframe:
        segs = _visible_wireframe_segments(
            mesh, cam_pos, cam_dir, obj2cam, state.perspective
        )
        if len(segs):
            coll = Line3DCollection(segs, colors="k", linewidths=0.5)
            ax.add_collection3d(coll)
    else:
        mask = (mesh.face_normals @ obj2cam) > 0
        tri = Poly3DCollection(
            mesh.triangles[mask],
            facecolors="white",
            edgecolors="k",
            linewidths=0.5,
        )
        ax.add_collection3d(tri)
    fig.savefig(output, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"SVG saved to {output}")


def export_without_view(mesh: trimesh.Trimesh, state: ViewerState, output: str) -> None:
    """Export mesh without opening a GUI (useful for testing)."""
    export_svg_from_view(mesh, state, output)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Display an STL and export SVG views")
    parser.add_argument("stl", nargs="?", default="test/example.stl", help="STL file to open")
    parser.add_argument("-o", "--output", default="output.svg", help="SVG file to save")
    parser.add_argument("--wireframe", action="store_true", help="Start in wireframe mode")
    parser.add_argument(
        "--orthographic",
        action="store_true",
        help="Start in orthographic projection",
    )
    parser.add_argument(
        "--no-view",
        action="store_true",
        help="Export directly without opening interactive viewer",
    )
    args = parser.parse_args(argv)

    mesh = trimesh.load(args.stl)
    state = ViewerState(wireframe=args.wireframe, perspective=not args.orthographic)

    if args.no_view:
        export_without_view(mesh, state, args.output)
        return 0

    viewer = STLViewer(mesh, state, args.output)
    viewer.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())