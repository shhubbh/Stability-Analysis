import pyvista as pv
import numpy as np
import os

def is_degenerate_tetrahedron(points, cell):
    """
    Check if a tetrahedron is degenerate by computing its volume.
    """
    p0, p1, p2, p3 = points[cell]
    mat = np.array([
        [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]],
        [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]],
        [p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2]]
    ])
    volume = np.abs(np.linalg.det(mat)) / 6.0
    return volume < 1e-6  # Threshold for considering an element degenerate

def check_for_degenerate_elements(mesh):
    """
    Check for degenerate tetrahedrons in the mesh.
    """
    points = mesh.points
    cells = mesh.cells

    i = 0
    degenerate_count = 0
    while i < len(cells):
        n_points = cells[i]  # Number of points in this cell (should be 4 for tetrahedrons)
        if n_points == 4:
            cell = cells[i+1:i+1+n_points]  # Extract the cell point indices
            if is_degenerate_tetrahedron(points, cell):
                degenerate_count += 1
                print(f"Degenerate tetrahedron found at cell index {i//(n_points + 1)} with points {cell}.")
        i += n_points + 1  # Move to the next cell

    if degenerate_count == 0:
        print("No degenerate elements found in the mesh.")
    else:
        print(f"Total degenerate elements found: {degenerate_count}")

def main():
    # Load the VTK file
    mesh = pv.read(os.path.join(r"H:\Others\Backup from 1TB Anvita HDD\Orissa\Data\4_Tiringpahada\Slope_Stability_WS\code_ws\07-08-2024\3D_Volume_Meshes\Generating_Meshes\Output_Meshes\filtered_tiny_mesh.vtu"))

    # Check for degenerate elements
    check_for_degenerate_elements(mesh)

if __name__ == "__main__":
    main()
