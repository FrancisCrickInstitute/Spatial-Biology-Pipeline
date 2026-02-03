import sopa
import spatialdata
import argparse
import spatialdata as sd
import re
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import muspan as ms
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from shapely.geometry import mapping


# Strip the _ch_N suffix from column names
def remove_channel_suffix(name):
    return re.sub(r'_ch_\d+$', '', name)


def cluster_data(data, n_clusters=10, random_seed=42):
    # Perform k-means clustering
    print("\nPerforming k-means clustering...")

    # Standardize the data (important for clustering)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed)
    cluster_labels = kmeans.fit_predict(scaled_data)

    print(f"Clustering complete. Found {n_clusters} clusters.")
    print(f"Cluster distribution:")
    for i in range(n_clusters):
        count = np.sum(cluster_labels == i)
        print(f"  Cluster {i}: {count} cells ({100 * count / len(cluster_labels):.1f}%)")

    return cluster_labels


def get_colors_for_communities(n_communities):
    """Generate distinct colors for communities."""
    if n_communities <= 10:
        cmap = plt.cm.get_cmap('tab10')
    elif n_communities <= 20:
        cmap = plt.cm.get_cmap('tab20')
    else:
        cmap = plt.cm.get_cmap('hsv')

    colors = []
    for i in range(n_communities):
        rgba = cmap(i / max(n_communities - 1, 1))
        rgb = [int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)]
        colors.append(rgb)

    return colors


def run_muspan(spatial_data, cell_boundaries='stardist_boundaries', index_name='cell_id', output_dir='.',
               cell_colour='table: kmeans_cluster', comm_detect_res=0.3):
    # Set the index name (as we learned earlier)
    spatial_data.shapes[cell_boundaries].index.name = index_name

    # Create clean version without image_patches
    sdata_clean = sd.SpatialData(
        shapes={cell_boundaries: spatial_data.shapes[cell_boundaries]},
        tables=spatial_data.tables
    )

    # Convert to muspan domain
    muspan_domain = ms.io.spatialdata_to_domain(sdata_clean, import_shapes_as_points=True)

    print("\nVisualising cells...")
    ms.visualise.visualise(muspan_domain, color_by=cell_colour, marker_size=0.5, figure_kwargs=dict(figsize=(100, 100)))
    plt.savefig(os.path.join(output_dir, 'cells.pdf'))

    print("\nGenerating Delauney network...")
    ms.networks.generate_network(
        muspan_domain,
        network_name='Centroid Delaunay',
        network_type='Delaunay',
        max_edge_distance=1000
    )

    print("\nVisualising Delauney network...")
    ms.visualise.visualise_network(
        muspan_domain,
        network_name='Centroid Delaunay',
        edge_width=0.5,
        visualise_kwargs=dict(color_by=cell_colour, marker_size=0.05),
        figure_kwargs=dict(figsize=(100, 100))
    )
    plt.savefig(os.path.join(output_dir, 'delaunay_network.pdf'))

    print(f'\nDetecting communities at res {comm_detect_res}...')
    communities_res_1 = ms.networks.community_detection(
        muspan_domain,
        network_name='Centroid Delaunay',
        edge_weight_name=None,
        community_method='louvain',
        community_method_parameters=dict(resolution=comm_detect_res),
        community_label_name=f'Communities'
    )

    print(f'\nVisualising communities at res {comm_detect_res}...')
    ms.visualise.visualise_network(
        muspan_domain,
        network_name='Centroid Delaunay',
        edge_width=0.5,
        visualise_kwargs=dict(
            color_by='Communities',
            marker_size=0.05,
            scatter_kwargs=dict(linewidth=0.1, edgecolor='black')
        ),
        figure_kwargs=dict(figsize=(100, 100))
    )
    plt.savefig(os.path.join(output_dir, 'communities_network.pdf'))

    return muspan_domain


def export_to_qupath(domain, communities, clusters, output_path, cell_id='cell_id'):
    print("Exporting to QuPath GeoJSON format...")

    # Get cell IDs and community labels from the muspan domain
    cell_ids = domain.labels[cell_id]['labels']
    communities_labels = domain.labels[communities]['labels']
    cluster_labels_from_domain = domain.labels[clusters]['labels']

    # Create a mapping from cell_id to community labels
    cell_to_community = dict(zip(cell_ids, communities_labels))
    cell_to_cluster = dict(zip(cell_ids, cluster_labels_from_domain))

    # Get unique communities for coloring
    unique_communities = np.unique(communities_labels)

    community_colors = get_colors_for_communities(len(unique_communities))

    # Create GeoJSON features
    features = []

    print("Exporting cell boundaries with community labels...")
    boundaries = sdata.shapes['stardist_boundaries']
    for idx, row in boundaries.iterrows():
        # Get community assignments
        community = cell_to_community.get(idx, -1)
        cluster_id = cell_to_cluster.get(idx, -1)

        # Get color based on community (res 0.3 as primary classification)
        if 0 <= community < len(community_colors):
            color = community_colors[int(community)]
            classification_name = f"Community_{int(community)}"
        else:
            color = [128, 128, 128]  # Gray for unassigned
            classification_name = "Unassigned"

        # Build measurements dictionary
        measurements_dict = {
            "Community ID": float(community),
            "Cluster ID": float(cluster_id)
        }

        # Add intensity measurements if available
        if idx in intensity_df.index:
            for col in intensity_df.columns:
                if col != 'Cluster':
                    try:
                        val = intensity_df.loc[idx, col]
                        if pd.notna(val):
                            measurements_dict[f"Cell: {col} mean"] = float(val)
                    except (ValueError, TypeError, KeyError):
                        pass

        feature = {
            "type": "Feature",
            "id": str(idx),
            "geometry": mapping(row.geometry),
            "properties": {
                "objectType": "cell",
                "classification": {
                    "name": classification_name,
                    "color": color
                },
                "measurements": measurements_dict
            }
        }

        features.append(feature)

    # Create final GeoJSON
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    # Write to file
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)

    print(f"\n✓ Exported {len(features)} cells with community labels")
    print(f"\nOutput file: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', help='Path to input image')
    args = parser.parse_args()

    imagepath = args.input_file
    # imagepath = '../data/20251017_132551_2_p6Bnyk_EHP893_25_SPYREplus4_EHP893_25_Thymus_SPYREplus4_test1.tiff'

    print("Opening image")

    dataset = sopa.io.ome_tif(imagepath, as_image=False)

    print("Saving as Zarr...")

    zarr_path = '../data/20251017_132551_2_p6Bnyk_EHP893_25_SPYREplus4_EHP893_25_Thymus_SPYREplus4_test1.zarr'

    dataset.write(zarr_path, overwrite=True)

    print("Done")

    print("Loading Zarr...")

    dataset = spatialdata.read_zarr(zarr_path)  # we can read the data back

    image_name = list(dataset.images.keys())[0]

    print("Make image patches...")

    sopa.make_image_patches(dataset)

    print("Set backend to None (will use GPU)...")

    sopa.settings.parallelization_backend = None

    print("Get channel names...")

    channels = sopa.utils.get_channel_names(dataset)

    print(channels)

    unique_channels = [f"{ch}_ch_{i}" for i, ch in enumerate(channels)]
    print("Fixed channels:", unique_channels)

    for scale_name in dataset.images[image_name].children:
        scale_node = dataset.images[image_name][scale_name]
        # Update the dataset with new coordinates
        scale_node.ds = scale_node.ds.assign_coords(c=unique_channels)

    print("Run stardist...")

    sopa.segmentation.stardist(dataset, model_type='2D_versatile_fluo', channels=unique_channels[0])

    print("Aggregating...")

    sopa.aggregate(dataset)

    print("Done")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Load the data
    path_to_spatialData_file = '/nemo/stp/lm/working/barryd/hpc/projects/stps/ehp/2026.01/comet_lunaphore/data/20251017_132551_2_p6Bnyk_EHP893_25_SPYREplus4_EHP893_25_Thymus_SPYREplus4_test1.zarr'
    sdata = sd.read_zarr(path_to_spatialData_file)

    # Get the boundaries and measurements
    measurements = sdata.tables['table']

    # Convert AnnData to DataFrame
    intensity_df = pd.DataFrame(
        measurements.X,
        index=measurements.obs.index,
        columns=measurements.var.index
    )

    intensity_df.columns = [remove_channel_suffix(col) for col in intensity_df.columns]

    # Keep only first occurrence of duplicate columns
    intensity_df = intensity_df.loc[:, ~intensity_df.columns.duplicated(keep='first')]
    print(f"Columns after removing duplicates: {list(intensity_df.columns)}")
    print(f"Number of cells: {len(intensity_df)}")

    # ===== ADD CLUSTERING RESULTS TO SPATIALDATA =====
    cluster_labels = cluster_data(intensity_df)

    # Add cluster labels to the AnnData obs (as categorical for efficiency)
    sdata.tables['table'].obs['kmeans_cluster'] = pd.Categorical(cluster_labels)

    # Optionally, add the cluster labels as a string for better visualization
    sdata.tables['table'].obs['kmeans_cluster_label'] = pd.Categorical(
        [f'Cluster_{i}' for i in cluster_labels]
    )

    print("\n✓ Cluster labels added to sdata.tables['table'].obs['kmeans_cluster']")
    print("✓ Cluster labels added to sdata.tables['table'].obs['kmeans_cluster_label']")

    # Verify it was added
    print(f"\nUpdated obs columns: {list(sdata.tables['table'].obs.columns)}")

    example_domain = run_muspan(sdata)

    export_to_qupath(example_domain, communities='Communities', clusters='table: kmeans_cluster',
                     output_path='/nemo/stp/lm/working/barryd/hpc/projects/stps/lm/Test/muspan/qupath_communities_export.geojson')
