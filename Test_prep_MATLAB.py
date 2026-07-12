import torch
import os,re
from PIL import Image
from transformers import AutoImageProcessor,logging
import scipy.io as sio # This library lets Python save MATLAB .mat files!
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from config import BaseModelName,trainedModel,Gallery_ids_mat,Probe_ids_mat
# Import your custom model
from FaceMoE import SwinFaceMoE,MoEMLP
logging.set_verbosity_error()

def numeric_key(fname):
    return int(re.search(r'\d+', fname).group())

def Prep():
    # ---------------------------------------------------------
    # 1. SETUP & LOAD TRAINED WEIGHTS
    # ---------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Extracting on: {device.upper()}")

    # We don't care about NUM_IDENTITIES here because we are bypassing the classifier,
    # but we need to pass a number to initialize the model class without errors.
    model = SwinFaceMoE(num_face_identities=5139,model_name=BaseModelName)
    d_model = model.swin_backbone.config.hidden_size
    model.moe_layer = MoEMLP(d_model=d_model,hidden_dim=d_model*4,num_experts=8,k=2)
    # Load the weights you just trained!
    print("Loading trained MoE weights...")
    model.moe_layer.load_state_dict(torch.load(trainedModel, map_location=device))
    model.to(device)
    model.eval() # CRITICAL: Set to evaluation mode (turns off dropout/batchnorm)

    processor = AutoImageProcessor.from_pretrained(BaseModelName)

    # ---------------------------------------------------------
    # 2. FEATURE EXTRACTION FUNCTION
    # ---------------------------------------------------------
    gallery_pairs = sio.loadmat(Gallery_ids_mat)
    probe_pairs = sio.loadmat(Probe_ids_mat)

    
    
    def extract_features_from_folder(folder_path,ordered_filenames):

        """
        Reads all images in a folder, passes them through the FaceMoE model,
        and returns a massive matrix of their feature embeddings.
        """
        features = []
        
        # Sort the images so they align with the MATLAB ID pair lists (This sequence is wrong, We're now getting the sequence from the mat files provided)
        # image_files = sorted(
        #     [f for f in os.listdir(gallery_dir) if f.endswith(('.jpg', '.png'))],
        #     key=numeric_key
        # )
        
        print(f"Extracting features for {len(ordered_filenames)} images in {folder_path}...")
        
        with torch.no_grad(): # No training math needed!
            # Wrap the loop with tqdm to display a progress bar
            for img_name in tqdm(ordered_filenames, desc=f"Processing {os.path.basename(folder_path)}", unit="img"):
                img_path = os.path.join(folder_path, img_name)
                image = Image.open(img_path).convert("RGB")
                
                # Pre-process
                inputs = processor(images=image, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device)
                
                # --- THE MAGIC TRICK ---
                # Instead of calling model(pixel_values) which gives a classification guess,
                # we manually pass it through the frozen backbone and our MoE layer
                # to get the pure "Face Signature" embedding vector.
                outputs = model.swin_backbone(pixel_values=pixel_values)
                moe_output = model.moe_layer(outputs.last_hidden_state)

                # Average the tokens into a single 1D vector representing the whole face
                face_signature = moe_output.mean(dim=1)

                # This makes Euclidean distance highly accurate for facial recognition!
                face_signature = F.normalize(face_signature, p=2, dim=1)

                # Average the tokens into a single 1D vector representing the whole face
                face_signature = face_signature.cpu().numpy() 
                features.append(face_signature[0])
                
        return np.stack(features,axis=0)

    # ---------------------------------------------------------
    # 3. RUN EXTRACTION AND SAVE TO MATLAB FORMAT
    # ---------------------------------------------------------
    # NOTE: Update these paths to point to the actual TinyFace test folders
    gallery_dir = "./Dataset/tinyface/Testing_Set/Gallery_Match"
    probe_dir = "./Dataset/tinyface/Testing_Set/Probe"
    distractor_dir = "./Dataset/tinyface/Testing_Set/Gallery_Distractor"
    gallery_img_names = [str(x[0][0]).strip() for x in gallery_pairs['gallery_set']]  # adjust unwrapping to match what you confirmed works
    probe_img_names = [str(x[0][0]) for x in probe_pairs['probe_set']]  # check actual key name for probe file
    # Extract features
    gallery_features = extract_features_from_folder(gallery_dir,gallery_img_names)
    probe_features = extract_features_from_folder(probe_dir,probe_img_names)
    distractor_features = extract_features_from_folder(
        distractor_dir,
        sorted(os.listdir(distractor_dir)))

    # Create the output directory if it doesn't exist
    os.makedirs("./features", exist_ok=True)

    # Save as .mat files exactly as the MATLAB script requests them
    print("Saving matrices for MATLAB...")
    sio.savemat("./features/gallery.mat", {"gallery_feature_map": gallery_features})
    sio.savemat("./features/probe.mat", {"probe_feature_map": probe_features})
    sio.savemat("./features/distractor.mat", {"distractor_feature_map": distractor_features})

    print("Done! You can now run 'test_face_identification.m' in MATLAB!")

if __name__ == "__main__":
    Prep()