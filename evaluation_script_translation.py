import numpy as np
import scipy.io as sio
from scipy.spatial.distance import cdist
import os
from config import Gallery_ids_mat,Probe_ids_mat,Feature_folder
from tqdm import tqdm  # Import tqdm for the progress bar

# ---------------------------------------------------------
# 1. COMPUTE AVERAGE PRECISION (Python translation of compute_AP.m)
# ---------------------------------------------------------
def compute_AP(good_image, index):
    cmc = np.zeros(len(index))
    ngood = len(good_image)
    
    old_recall = 0.0
    old_precision = 1.0
    ap = 0.0
    intersect_size = 0
    j = 0
    good_now = 0
    
    for n in range(len(index)):
        flag = 0
        # Check if the current ranked index is in our list of "good" (correct) matches
        if index[n] in good_image:
            cmc[n:] = 1
            flag = 1
            good_now += 1
            
        if flag == 1:
            intersect_size += 1
            
        recall = intersect_size / ngood
        precision = intersect_size / (j + 1)
        ap += (recall - old_recall) * ((old_precision + precision) / 2)
        
        old_recall = recall
        old_precision = precision
        j += 1
        
        if good_now == ngood:
            break
            
    return ap, cmc

def Test():
    # ---------------------------------------------------------
    # 2. RUN EVALUATION (Python translation of test_face_identification.m)
    # ---------------------------------------------------------
    print("Loading feature matrices and ID pairs...")

    # 1. Load the ground truth IDs (You need to ensure these files are in your directory)
    # These are the files mentioned at the top of the MATLAB script.
    try:
        gallery_ids_mat = sio.loadmat(Gallery_ids_mat)
        probe_ids_mat = sio.loadmat(Probe_ids_mat)
        
        # Extract arrays (assuming the variable names inside match the file names)
        gallery_ids = gallery_ids_mat['gallery_ids'].flatten()
        probe_ids = probe_ids_mat['probe_ids'].flatten()
    except FileNotFoundError:
        print("WARNING: Could not find the ground-truth ID .mat files.")
        print("Please ensure 'gallery_match_img_ID_pairs.mat' and 'probe_img_ID_pairs.mat' are in this folder.")
        exit()

    # 2. Load the features we extracted with PyTorch
    feature_folder = Feature_folder
    gallery_data = sio.loadmat(os.path.join(feature_folder, 'gallery.mat'))
    probe_data = sio.loadmat(os.path.join(feature_folder, 'probe.mat'))
    distractor_data = sio.loadmat(os.path.join(feature_folder, 'distractor.mat'))

    gallery_feature_map = gallery_data['gallery_feature_map']
    probe_feature_map = probe_data['probe_feature_map']
    distractor_feature_map = distractor_data['distractor_feature_map']

    print("Constructing Gallery...")
    # Combine gallery and distractors (just like the MATLAB script)
    full_gallery_features = np.vstack((gallery_feature_map, distractor_feature_map))

    # Label distractors as -100
    distractor_ids = -100 * np.ones(distractor_feature_map.shape[0])
    full_gallery_ids = np.concatenate((gallery_ids, distractor_ids))

    print("Calculating Euclidean Distances (pdist2 equivalent)...")
    # cdist is the Python equivalent of MATLAB's pdist2
    dist = cdist(full_gallery_features, probe_feature_map, metric='euclidean')

    CMC_all = []
    ap_all = []

    print("Scoring probe images...")
    # Use tqdm here to wrap the range, which will automatically draw a progress bar
    for p in tqdm(range(probe_feature_map.shape[0]), desc="Scoring Probes", unit="probe"):
        probe_id = probe_ids[p]
        
        # Find all indices in the gallery that match the current probe ID
        good_index = np.where(full_gallery_ids == probe_id)[0]
        
        # Get the distances for this specific probe
        score = dist[:, p]
        
        # Sort scores in ascending order (lowest distance = best match)
        # argsort gives us the original indices in their new sorted order
        index = np.argsort(score)
        
        # Compute Average Precision and CMC for this probe
        ap, cmc = compute_AP(good_index, index)
        
        ap_all.append(ap)
        CMC_all.append(cmc)

    # ---------------------------------------------------------
    # 3. FINAL RESULTS
    # ---------------------------------------------------------
    # Convert to numpy arrays for easy math
    ap_all = np.array(ap_all)
    # Pad CMC arrays in case they have different lengths (though they shouldn't here)
    max_len = max(len(c) for c in CMC_all)
    CMC_padded = np.array([np.pad(c, (0, max_len - len(c)), 'edge') for c in CMC_all])

    # Mean Average Precision (mAP)
    mAP = np.mean(ap_all)

    # Rank-1, Rank-5, Rank-10 Accuracies
    mean_CMC = np.mean(CMC_padded, axis=0)

    print("\n" + "="*40)
    print("🏆 EVALUATION RESULTS 🏆")
    print("="*40)
    print(f"mAP (Mean Average Precision): {mAP * 100:.2f}%")
    print(f"Rank-1 Accuracy:  {mean_CMC[0] * 100:.2f}%")
    if len(mean_CMC) >= 5:
        print(f"Rank-5 Accuracy:  {mean_CMC[4] * 100:.2f}%")
    if len(mean_CMC) >= 10:
        print(f"Rank-10 Accuracy: {mean_CMC[9] * 100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    Test()