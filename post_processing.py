import numpy as np

def calculate_weighted_coordinates(probabilities, label_encoder, cell_centers, top_n=3):
    """
    Calculates refined coordinates based on a weighted average of the top N
    most probable grid cells.

    Args:
        probabilities (np.ndarray): A 2D array of prediction probabilities from a model,
                                    with shape (n_samples, n_classes).
        label_encoder (LabelEncoder): The fitted scikit-learn LabelEncoder instance
                                      used to map class indices back to cell_id strings.
        cell_centers (dict): A dictionary mapping cell_id strings to their
                             (latitude, longitude) center coordinates.
        top_n (int): The number of top probable cells to consider for the weighted average.

    Returns:
        np.ndarray: A 2D array of the calculated weighted coordinates (latitude, longitude)
                    for each sample.
    """
    # Get the indices of the top N probabilities for each sample.
    # np.argsort returns indices from lowest to highest, so we slice from the end.
    top_n_indices = np.argsort(probabilities, axis=1)[:, -top_n:]

    # Get the probability values for these top N indices
    top_n_probs = np.take_along_axis(probabilities, top_n_indices, axis=1)

    # Normalize the top N probabilities so that they sum to 1, to be used as weights
    # Adding a small epsilon to avoid division by zero if all probs are zero.
    prob_sums = top_n_probs.sum(axis=1, keepdims=True)
    normalized_weights = top_n_probs / (prob_sums + 1e-9)

    # Get the corresponding cell_id strings for the top N indices
    top_n_cell_ids = label_encoder.inverse_transform(top_n_indices.flatten()).reshape(top_n_indices.shape)

    # Get the center coordinates for each of the top N cell_ids
    # This will be a 3D array: (n_samples, top_n, 2) where the last dim is (lat, lon)
    top_n_coords = np.array(
        [[cell_centers.get(cell_id, (np.nan, np.nan)) for cell_id in sample_ids]
         for sample_ids in top_n_cell_ids]
    )

    # Separate latitudes and longitudes
    top_n_lats = top_n_coords[:, :, 0]
    top_n_lons = top_n_coords[:, :, 1]

    # Calculate the weighted average for latitudes and longitudes
    # The shape of weighted_lats will be (n_samples,)
    weighted_lats = np.sum(top_n_lats * normalized_weights, axis=1)
    weighted_lons = np.sum(top_n_lons * normalized_weights, axis=1)

    # Combine the weighted latitudes and longitudes back into a single array
    # The final shape will be (n_samples, 2)
    weighted_coordinates = np.vstack((weighted_lats, weighted_lons)).T

    return weighted_coordinates
