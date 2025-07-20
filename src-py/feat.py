import numpy as np

def extract_features_batch(points, local_len=4, keep_shape = True):
    # Ensure points is a numpy array of shape (batch_size, local_len*2)
    points = np.array(points, dtype=np.float32)
    batch_size = points.shape[0]
    points = points.reshape(batch_size, local_len, 2)  # Reshape to (batch_size, local_len, 2)
    
    # 1. Normalized Point Coordinates (npc)
    min_coords = np.min(points, axis=(1, 2), keepdims=True)
    max_coords = np.max(points, axis=(1, 2), keepdims=True)
    if keep_shape:
        npc = (points - min_coords) / (np.max(max_coords) - np.min(min_coords))
    else:
        npc = (points - min_coords) / (max_coords - min_coords)  # Normalization
    npc = npc.reshape(batch_size, -1)  # Shape (batch_size, local_len*2)

    # 2. Ratio of Chord Lengths (rcl)
    chord_lengths = np.linalg.norm(points[:, 1:] - points[:, :-1], axis=2)  # Shape (batch_size, local_len-1)
    rcl = chord_lengths / np.mean(chord_lengths, axis=1, keepdims=True)  # Ratios (batch_size, local_len-1)

    # 3. Coefficient of Variation of Chord Lengths (cvl)
    mean_length = np.mean(chord_lengths, axis=1, keepdims=True)  # Shape (batch_size, 1)
    std_length = np.std(chord_lengths, axis=1, keepdims=True)   # Shape (batch_size, 1)
    cvl = std_length / mean_length  # Coefficient of variation
    cvl[mean_length == 0] = 0  # Avoid division by zero

    # 4. Signed Angles of Neighbor Line Segments (san)
    angles = []
    for i in range(1, local_len - 1):  # (local_len - 2) angles for (local_len - 2) points
        angle = np.arctan2(points[:, i + 1, 1] - points[:, i, 1], points[:, i + 1, 0] - points[:, i, 0]) - \
                np.arctan2(points[:, i, 1] - points[:, i - 1, 1], points[:, i, 0] - points[:, i - 1, 0])
        angles.append(np.expand_dims(angle, axis=1))  # Shape (batch_size, 1)
    
    san = np.concatenate(angles, axis=1)  # Shape (batch_size, local_len - 2)

    # 5. Difference Between Adjacent Signed Angles (dsa)
    # 6. Absolute Differences Between Adjacent Angles (daa)
    dsa = []
    daa = []
    for i in range(0, local_len - 3):  # (local_len - 3) angle differences for (local_len - 2) points
        dsa.append((san[:, i+1] - san[:, i]).reshape(-1, 1)) 
        daa.append((np.abs(san[:, i]) - np.abs(san[:, i+1])).reshape(-1, 1))  
    dsa = np.concatenate(dsa, axis=1)
    daa = np.concatenate(daa, axis=1)

    # Combine all features into a single array
    features = np.concatenate((npc, rcl, cvl, san, dsa, daa), axis=1)  # Shape (batch_size, 6*local_len-8)

    return features