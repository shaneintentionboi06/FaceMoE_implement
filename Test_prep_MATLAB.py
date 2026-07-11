import torch
import os
from PIL import Image
from transformers import AutoImageProcessor
import scipy.io as sio # This library lets Python save MATLAB .mat files!

# Import your custom model
from FaceMoE import SwinFaceMoE

# ---------------------------------------------------------
# 1. SETUP & LOAD TRAINED WEIGHTS
# ---------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Extracting on: {device.upper()}")

# We don't care about NUM_IDENTITIES here because we are bypassing the classifier,
# but we need to pass a number to initialize the model class without errors.
model = SwinFaceMoE(num_face_identities=5139)

# Load the weights you just trained!
print("Loading trained MoE weights...")
model.moe_layer.load_state_dict(torch.load("facemoe_adapters.pth", map_location=device))
model.to(device)
model.eval() # CRITICAL: Set to evaluation mode (turns off dropout/batchnorm)

processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224")

# ---------------------------------------------------------
# 2. FEATURE EXTRACTION FUNCTION
# ---------------------------------------------------------
def extract_features_from_folder(folder_path):
    """
    Reads all images in a folder, passes them through the FaceMoE model,
    and returns a massive matrix of their feature embeddings.
    """
    features = []
    
    # Sort the images so they align with the MATLAB ID pair lists
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))])
    
    print(f"Extracting features for {len(image_files)} images in {folder_path}...")
    
    with torch.no_grad(): # No training math needed!
        for img_name in image_files:
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
            face_signature = moe_output.mean(dim=1).cpu().numpy() 
            features.append(face_signature[0])
            
    return features

# ---------------------------------------------------------
# 3. RUN EXTRACTION AND SAVE TO MATLAB FORMAT
# ---------------------------------------------------------
# NOTE: Update these paths to point to the actual TinyFace test folders
gallery_dir = "./dataset/tinyface/test/gallery"
probe_dir = "./dataset/tinyface/test/probe"
distractor_dir = "./dataset/tinyface/test/distractor"

# Extract features
gallery_features = extract_features_from_folder(gallery_dir)
probe_features = extract_features_from_folder(probe_dir)
distractor_features = extract_features_from_folder(distractor_dir)

# Create the output directory if it doesn't exist
os.makedirs("./features", exist_ok=True)

# Save as .mat files exactly as the MATLAB script requests them
print("Saving matrices for MATLAB...")
sio.savemat("./features/gallery.mat", {"gallery_feature_map": gallery_features})
sio.savemat("./features/probe.mat", {"probe_feature_map": probe_features})
sio.savemat("./features/distractor.mat", {"distractor_feature_map": distractor_features})

print("Done! You can now run 'test_face_identification.m' in MATLAB!")