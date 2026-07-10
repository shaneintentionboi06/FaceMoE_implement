import torch
from transformers import SwinModel, AutoImageProcessor,logging
from PIL import Image
from config import modelName

# 1. Setup the Device (The GPU Check)
# This automatically uses the GPU if you have an NVIDIA card installed properly, 
# otherwise it safely falls back to your regular processor (CPU).
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Hardware allocated: {device.upper()}")
logging.set_verbosity_error()

# 2. Load the Image Processor and the Model
print("Downloading model weights...")
model_name = modelName
processor = AutoImageProcessor.from_pretrained(model_name)
model = SwinModel.from_pretrained(model_name)

# --- THE CRITICAL STEP ---
# Move the entire massive model into the GPU's VRAM
model.to(device) 
# -------------------------

# 3. Create a Dummy Low-Resolution Image
image = Image.new('RGB', (16, 16), color='blue')

# 4. Pre-process the Image
inputs = processor(images=image, return_tensors="pt")

# --- THE CRITICAL STEP 2 ---
# Move the processed image tensor to the GPU so it is in the same memory as the model
inputs = {key: value.to(device) for key, value in inputs.items()}
# ---------------------------

# 5. The Forward Pass
print("Passing image through the Swin Transformer...")
with torch.no_grad():
    outputs = model(**inputs)

# 6. Extract the Features
last_hidden_states = outputs.last_hidden_state
print(f"Success! Output tensor shape: {last_hidden_states.shape}")