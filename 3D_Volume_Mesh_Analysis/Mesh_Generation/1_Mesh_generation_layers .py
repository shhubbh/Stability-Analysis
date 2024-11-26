import numpy as np
import rasterio
from scipy.interpolate import griddata
import pyvista as pv
import os
from tqdm import tqdm

def create_unstructured_grid(points, cells, celltypes):
    """
    Create a PyVista UnstructuredGrid from points, cells, and cell types.
    """
    try:
        mesh = pv.UnstructuredGrid(cells, celltypes, points)
        return mesh
    except ValueError as e:
        print(f"Validation Error: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def trim_layers_with_dem(layers_points, dem_surface):
    """
    Trim layers at the DEM surface by setting points above the DEM's z-value.
    """
    trimmed_layers = []
    for points in layers_points:
        trimmed_points = points.copy()
        z_coords = trimmed_points[:, 2]
        mask = z_coords > dem_surface

        # Ensure all points above DEM are exactly on the DEM surface
        trimmed_points[mask, 2] = dem_surface[mask]
        trimmed_layers.append(trimmed_points)
    
    return trimmed_layers

def generate_tetrahedral_mesh(points, dem_shape, num_layers):
    """
    Generate tetrahedral cells to fill the entire volume, ensuring no hollow regions.
    """
    tetra_cells = []
    cell_types = []
    layer_colors = []

    # Loop through each pair of adjacent layers
    for layer in range(1, num_layers):
        top_cells = np.arange(len(points) // num_layers).reshape(dem_shape) + (layer - 1) * (len(points) // num_layers)
        bottom_cells = top_cells + (len(points) // num_layers)

        for i in range(dem_shape[0] - 1):
            for j in range(dem_shape[1] - 1):
                idx1 = top_cells[i, j]
                idx2 = top_cells[i, j + 1]
                idx3 = top_cells[i + 1, j]
                idx4 = top_cells[i + 1, j + 1]
                idx5 = bottom_cells[i, j]
                idx6 = bottom_cells[i, j + 1]
                idx7 = bottom_cells[i + 1, j]
                idx8 = bottom_cells[i + 1, j + 1]

                # Convert each hexahedral cell into 5 or 6 tetrahedral cells
                tetra_cells.extend([
                    [4, idx1, idx2, idx4, idx5],
                    [4, idx1, idx4, idx5, idx7],
                    [4, idx2, idx4, idx5, idx6],
                    [4, idx4, idx5, idx6, idx7],
                    [4, idx4, idx5, idx7, idx8]
                ])
                cell_types.extend([10] * 5)  # VTK_TETRA is type 10
                layer_colors.extend([layer] * 5)  # Assign layer color for each tetrahedron

    return np.array(tetra_cells, dtype=np.int32).flatten(), np.array(cell_types, dtype=np.uint8), layer_colors

def visualize_mesh_with_layer_colors(mesh, dem_surface_points, layers, layer_colors):
    """
    Visualize the DEM surface, mesh with different colors for each layer, using PyVista.
    """
    plotter = pv.Plotter()
    
    # Visualize the DEM surface
    dem_surface = pv.PolyData(dem_surface_points)
    plotter.add_mesh(dem_surface, color="brown", opacity=0.8, show_edges=True)
    
    # Visualize the mesh with cell colors based on layers
    mesh.cell_data["layer_colors"] = layer_colors
    plotter.add_mesh(mesh, scalars="layer_colors", show_edges=True, cmap="viridis")

    plotter.add_axes()
    plotter.add_title("3D Visualization of Solid Tetrahedral Mesh")
    plotter.show()

def save_mesh_with_crs(mesh, filename, crs):
    """
    Save the mesh in the specified format (either .vtu or .vtk) with embedded CRS information.
    """
    try:
        # Add CRS as field data in the mesh
        mesh.field_data["CRS"] = np.array([crs])

        # Save the mesh to the file
        mesh.save(filename)
        print(f"Mesh saved with CRS embedded as metadata: {filename}.")
    except Exception as e:
        print(f"Error saving mesh with CRS as {filename}: {e}")

def visualize_layers(layers_points):
    """
    Visualize each trimmed layer separately to diagnose any issues.
    """
    plotter = pv.Plotter()
    
    for i, layer_points in enumerate(layers_points):
        layer_mesh = pv.PolyData(layer_points)
        plotter.add_mesh(layer_mesh, opacity=0.5, color="yellow", show_edges=True, label=f"Layer {i}")
    
    plotter.add_axes()
    plotter.add_title("Layer-by-Layer Visualization")
    plotter.show()

def main():
    # Load the DEM and get CRS
    with rasterio.open(r"C:\Users\Sekhar\FEM_FOS\WaterFlow_Module\Inputs\water_Analysis_Dem.tif") as src:
        dem = src.read(1)
        transform = src.transform
        dem_shape = dem.shape
        crs = src.crs.to_wkt()  # Get CRS in WKT format

    mask = np.isnan(dem) | (dem < 0)
    print("Interpolating NaN/negative values...")
    y, x = np.indices(dem.shape)
    dem[mask] = griddata(
        (x[~mask].ravel(), y[~mask].ravel()), dem[~mask].ravel(), (x[mask], y[mask]), method='nearest'
    )

    print("Converting grid indices to coordinates...")
    x_coords, y_coords = np.meshgrid(np.arange(dem_shape[1]), np.arange(dem_shape[0]))
    x_coords = transform[0] * x_coords + transform[2]
    y_coords = transform[4] * y_coords + transform[5]
    x_flat = x_coords.ravel()
    y_flat = y_coords.ravel()
    z_flat = dem.ravel()

    # Store DEM surface points for visualization
    dem_surface_points = np.column_stack((x_flat, y_flat, z_flat))

    # Determine the number of layers and spacing
    num_layers = 10
    base_depth = np.min(z_flat) - 1  # Define the flat base plane
    depth_increment = (np.max(z_flat) - base_depth) / (num_layers - 1)

    layers_points = []  # Storing all layers' points for later use

    # Create flat layers parallel to the base plane
    print("Generating flat layers...")
    for layer in tqdm(range(num_layers), desc="Layers"):
        z_layer = np.full_like(z_flat, base_depth + layer * depth_increment)
        layer_points = np.column_stack((x_flat, y_flat, z_layer))
        layers_points.append(layer_points)

    # Trim layers where they are above the DEM surface
    dem_surface = z_flat
    trimmed_layers_points = trim_layers_with_dem(layers_points, dem_surface)

    # Visualize each trimmed layer to check for any issues
    print("Visualizing trimmed layers...")
    visualize_layers(trimmed_layers_points)

    # Combine points for mesh creation
    points = np.vstack(trimmed_layers_points)

    # Generate tetrahedral mesh
    print("Generating tetrahedral cells for the solid volume...")
    tetra_cells, cell_types, layer_colors = generate_tetrahedral_mesh(points, dem_shape, num_layers)

    print("Finalizing cells and cell types...")

    # Create the mesh
    print("Creating the tetrahedral mesh...")
    mesh = create_unstructured_grid(points, tetra_cells, cell_types)

    # Convert trimmed points to PyVista PolyData objects for visualization
    layers_polydata = [pv.PolyData(layer) for layer in trimmed_layers_points if not np.isnan(layer).all()]

    if mesh:
        print(f"Total number of cells in the mesh: {mesh.n_cells}")

        # Visualize the solid tetrahedral mesh
        print("Visualizing the solid tetrahedral mesh...")
        visualize_mesh_with_layer_colors(mesh, dem_surface_points, layers_polydata, layer_colors)

        # Save the final mesh with CRS information
        print("Saving the mesh as .vtu with CRS...")
        save_mesh_with_crs(mesh, os.path.join(r"E:\Analytics\Slope_Stability_WS\code_ws\07-08-2024\Test_dems\test_6r.vtu"), crs)
    else:
        print("Mesh creation failed.")

if __name__ == "__main__":
    main()
