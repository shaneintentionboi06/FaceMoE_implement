import torch,time,os
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor,logging
from torch.cuda.amp import autocast, GradScaler
from config import BaseModelName ,DATA_DIR
from PIL import Image
# Import the model we just built in the other file!
from FaceMoE import SwinFaceMoE, MoEMLP
logging.set_verbosity_error()
# ---------------------------------------------------------
# 1. DATASET SETUP (The Placeholder)
# ---------------------------------------------------------
# For your actual internship, you will replace this with a script 
# that loads the real TinyFace dataset from your hard drive.

class TinyFaceDataset(Dataset):
    def __init__(self, root_dir):
        """
        Expects a folder structure where each person has their own sub-folder:
        root_dir/
        ├── person_0001/
        │   ├── img1.jpg
        │   └── img2.jpg
        ├── person_0002/
        │   └── img1.jpg
        """
        self.root_dir = root_dir
        # Swin expects 224x224 after pre-processing, even if the raw image is 16x16
        self.processor = AutoImageProcessor.from_pretrained(BaseModelName)
        
        self.image_paths = []
        self.labels = []
        
        print(f"Scanning directory: {root_dir}...")
        if os.path.exists(root_dir):
            self.classes = sorted(os.listdir(root_dir))
            for label_idx, class_name in enumerate(self.classes):
                class_dir = os.path.join(root_dir, class_name)
                if os.path.isdir(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.image_paths.append(os.path.join(class_dir, img_name))
                            self.labels.append(label_idx)
            print(f"Found {len(self.image_paths)} images across {len(self.classes)} identities.")
        else:
            print(f"WARNING: Directory '{root_dir}' not found! Please update the DATA_DIR path.")
            
    def __len__(self):
        # Prevent dataloader from crashing if folder is empty during initial setup
        return len(self.image_paths) if self.image_paths else 1

    def __getitem__(self, idx):
        # Fallback just in case the folder is empty during testing
        if not self.image_paths:
            return torch.randn(3, 224, 224), 0
            
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load the actual image from the hard drive
        image = Image.open(img_path).convert("RGB")
        
        # Pre-process it for Swin Transformer
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Extract the actual tensor (squeeze removes the batch dimension added by processor)
        image_tensor = inputs["pixel_values"].squeeze(0)
        
        return image_tensor, label

class DummyTinyFaceDataset(Dataset):
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        # Swin expects 224x224 after pre-processing, even if the raw image is 16x16
        self.processor = AutoImageProcessor.from_pretrained(BaseModelName)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Simulating an image tensor and a random face identity label (0 to 5138)
        dummy_image_tensor = torch.randn(3, 224, 224) 
        dummy_label = torch.randint(0, 5139, (1,)).item()
        return dummy_image_tensor, dummy_label

# ---------------------------------------------------------
# 2. HYPERPARAMETERS & SETUP
# ---------------------------------------------------------
def Main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting Training on: {device.upper()}")


    BATCH_SIZE = 32        # KEEP THIS LOW (2 or 4) for laptop GPUs!
    EPOCHS = 5

    # Create the dataset first so we know exactly how many identities we have
    dataset = TinyFaceDataset(root_dir=DATA_DIR)
    NUM_IDENTITIES = len(dataset.classes) if hasattr(dataset, 'classes') and dataset.classes else 5139

    # Initialize our custom model
    model = SwinFaceMoE(num_face_identities=NUM_IDENTITIES,model_name=BaseModelName)
    # model.moe_layer = MoEMLP(d_model=1024,hidden_dim=4096,num_experts=8,k=2)
    d_model = model.swin_backbone.config.hidden_size
    print("Dimensions of the Model:",d_model)
    model.moe_layer = MoEMLP(d_model=d_model,hidden_dim=d_model*4,num_experts=8,k=2)
    model.to(device)
    model.train() # Set to training mode

    # ---------------------------------------------------------
    # 3. TRAINING ENGINE
    # ---------------------------------------------------------
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Standard classification loss function
    criterion = nn.CrossEntropyLoss()

    # The Optimizer: Notice we ONLY pass the parameters that require gradients!
    # Because we froze the backbone, it only trains the MoE Layer and Classifier.
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # Modern PyTorch AMP Setup (Automatic Mixed Precision)
    use_amp = True if device == "cuda" else False
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---------------------------------------------------------
    # 4. THE TRAINING LOOP
    # ---------------------------------------------------------
    print("--- Beginning Training Loop ---")
    for epoch in range(EPOCHS):
        start_time = time.time()
        total_loss = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            # Move data to GPU
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # AUTOCAST: Automatically uses 16-bit math where safe to save memory
            with torch.autocast(device_type=device, enabled=use_amp):
                # 1. Forward Pass
                predictions = model(pixel_values=images)
                
                # 2. Calculate Loss
                loss = criterion(predictions, labels)
                
            # 3. Backward Pass (using the scaler for 16-bit math)
            scaler.scale(loss).backward()
            
            # 4. Update Weights
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch {batch_idx} | Loss: {loss.item():.4f}")
                
        epoch_time = time.time() - start_time
        print(f"==> End of Epoch {epoch+1} | Avg Loss: {total_loss/len(dataloader):.4f} | Time: {epoch_time:.2f}s\n")

    # ---------------------------------------------------------
    # 5. SAVE THE FINISHED MODEL
    # ---------------------------------------------------------
    torch.save(model.moe_layer.state_dict(), "facemoe_adapters.pth")
    print("Training Complete! MoE Weights saved to 'facemoe_adapters.pth'")
    
if __name__ == "__main__":
    Main()