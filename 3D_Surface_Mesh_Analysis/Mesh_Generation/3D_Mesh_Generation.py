import pyvista as pv
import numpy as np
import rasterio
from scipy.interpolate import griddata
import os

def create_surface_mesh(dem, transform):
    """
    Create a surface mesh from the DEM.
    """
    # Convert grid indices to coordinates
    x_coords, y_coords = np.meshgrid(np.arange(dem.shape[1]), np.arange(dem.shape[0]))
    x_coords = transform[0] * x_coords + transform[2]
    y_coords = transform[4] * y_coords + transform[5]
    z_flat = dem.ravel()

    # Create surface mesh
    grid = pv.StructuredGrid(x_coords, y_coords, z_flat.reshape(dem.shape))
    surface_mesh = grid.extract_surface().triangulate()

    return surface_mesh

def is_degenerate_triangle(points, cell):
    """
    Check if a triangle is degenerate by computing its area.
    """
    p0, p1, p2 = points[cell]
    mat = np.array([
        [p1[0] - p0[0], p1[1] - p0[1]],
        [p2[0] - p0[0], p2[1] - p0[1]]
    ])
    area = np.abs(np.linalg.det(mat)) / 2.0
    return area < 1e-6  # Threshold for considering an element degenerate

def filter_degenerate_elements_2d(mesh):
    """
    Filter out degenerate triangles from the 2D mesh.
    """
    points = mesh.points
    faces = mesh.faces.reshape((-1, 4))[:, 1:]  # Assuming triangular faces (VTK format)

    new_faces = []

    for face in faces:
        if not is_degenerate_triangle(points, face):
            new_faces.append(np.hstack([[3], face]))  # Add the number of points in a face first

    new_faces = np.array(new_faces, dtype=np.int32).flatten()

    filtered_mesh = pv.PolyData(points, new_faces)
    
    return filtered_mesh

import os

def main():
    # Load the DEM file and transform
    with rasterio.open(r"H:\Others\Backup from 1TB Anvita HDD\Orissa\Data\4_Tiringpahada\Slope_Stability_WS\code_ws\07-08-2024\Test_dems\Test8.tif") as src:
        dem = src.read(1)
        transform = src.transform

    # Interpolate NaN/negative values
    mask = np.isnan(dem) | (dem < 0)
    y, x = np.indices(dem.shape)
    dem[mask] = griddata(
        (x[~mask].ravel(), y[~mask].ravel()), dem[~mask].ravel(), (x[mask], y[mask]), method='nearest'
    )

    # Create surface mesh from DEM
    print("Creating surface mesh from DEM...")
    surface_mesh = create_surface_mesh(dem, transform)
    print(f"Surface mesh created with {surface_mesh.n_cells} cells")

    # Filter out degenerate elements
    print("Filtering out degenerate cells...")
    filtered_mesh = filter_degenerate_elements_2d(surface_mesh)
    print(f"Filtered mesh has {filtered_mesh.n_cells} cells")

    # Visualize the original and filtered mesh
    plotter = pv.Plotter()
    plotter.add_mesh(surface_mesh, color="red", opacity=0.5, show_edges=True, edge_color='black', label='Original Mesh')
    plotter.add_mesh(filtered_mesh, color="blue", opacity=0.5, show_edges=True, edge_color='black', label='Filtered Mesh')
    plotter.add_axes()
    plotter.add_title("2D Mesh Visualization")
    plotter.show()

    # Save the filtered mesh as a new VTU file
    output_directory = r"H:\Others\Backup from 1TB Anvita HDD\Orissa\Data\4_Tiringpahada\Slope_Stability_WS\code_ws\07-08-2024\2D_Meshes"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    output_file = os.path.join(output_directory, "testpit1_mesh.vtk")
    try:
        filtered_mesh.save(output_file)
        print(f"Filtered 2D mesh saved as {output_file}")
    except Exception as e:
        print(f"ERROR: Unable to save the file. {e}")

if __name__ == "__main__":
    main()
