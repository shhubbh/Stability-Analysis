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

def filter_degenerate_elements(mesh):
    """
    Filter out degenerate tetrahedrons from the mesh.
    """
    points = mesh.points
    cells = mesh.cells
    cell_types = mesh.celltypes

    new_cells = []
    new_cell_types = []
    i = 0
    while i < len(cells):
        n_points = cells[i]  # Number of points in this cell (should be 4 for tetrahedrons)
        if n_points == 4:
            cell = cells[i+1:i+1+n_points]  # Extract the cell point indices
            if not is_degenerate_tetrahedron(points, cell):
                new_cells.extend([n_points, *cell])
                new_cell_types.append(cell_types[i // (n_points + 1)])  # Append corresponding cell type
        i += n_points + 1  # Move to the next cell

    # Convert cells and cell_types to numpy arrays
    new_cells = np.array(new_cells, dtype=np.int32)
    new_cell_types = np.array(new_cell_types, dtype=np.uint8)

    # Create a new PyVista UnstructuredGrid with the filtered cells
    filtered_mesh = pv.UnstructuredGrid(new_cells, new_cell_types, points)
    
    return filtered_mesh

def main():
    # Load the VTK file
    mesh = pv.read(os.path.join((r"E:\Analytics\Slope_Stability_WS\code_ws\07-08-2024\Test_dems\test_6r.vtu")))

    # Filter out degenerate elements
    filtered_mesh = filter_degenerate_elements(mesh)

    plotter = pv.Plotter()
    
    # Plot the volume of the mesh with semi-transparent color
    plotter.add_mesh(mesh, color="red", opacity=1, show_edges=True, edge_color='black')
    
    # Plot the nodes as points
    plotter.add_points(mesh.points, color="blue", point_size=1, render_points_as_spheres=True)
    
    # Add coordinate axes
    plotter.add_axes()
    
    # Add title to the plot
    plotter.add_title("3D Mesh Visualization with Volume")
    
    # Show the plot
    plotter.show()

    # Save the filtered mesh as a new VTU file
    filtered_mesh.save(os.path.join(r"E:\Analytics\Slope_Stability_WS\code_ws\07-08-2024\Test_dems\mesh.vtu"))
    print("Filtered mesh saved in VTU format")

if __name__ == "__main__":
    main()
