import torch
from torch.utils.data import DataLoader
from FaceMoE import SwinFaceMoE
from TrainingLoop import TinyFaceDataset
def evaluate_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on: {device.upper()}")

    # 1. Setup Dataset (Make sure this points to your TEST folder, not TRAIN!)
    TEST_DIR = "./Dataset/tinyface/Testing_Set" 
    test_dataset = TinyFaceDataset(root_dir=TEST_DIR)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    NUM_IDENTITIES = len(test_dataset.classes)

    # 2. Load the Model and Your Custom Weights
    print("Loading architecture and trained weights...")
    model = SwinFaceMoE(num_face_identities=NUM_IDENTITIES)
    
    # Load the specific dictionary of weights we saved after training
    model.moe_layer.load_state_dict(torch.load("facemoe_adapters.pth"))
    model.to(device)
    
    # CRITICAL: Set the model to Evaluation Mode (turns off dropout, locks weights)
    model.eval()

    # 3. Metric Trackers
    total_images = 0
    rank1_correct = 0
    rank5_correct = 0

    print("--- Beginning Evaluation ---")
    
    # CRITICAL: Turn off gradient tracking to save massive amounts of memory and time
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            logits = model(pixel_values=images)
            
            # Get the Top 5 predictions for each image in the batch
            # topk returns the values and the indices (the guessed identities)
            _, top5_guesses = logits.topk(5, dim=1, largest=True, sorted=True)
            
            for i in range(labels.size(0)):
                true_label = labels[i]
                guesses = top5_guesses[i]
                
                # Check Rank-1: Is the first guess correct?
                if guesses[0] == true_label:
                    rank1_correct += 1
                    
                # Check Rank-5: Is the true label anywhere in the top 5 guesses?
                if true_label in guesses:
                    rank5_correct += 1
                    
                total_images += 1

    # 4. Calculate Final Percentages
    rank1_acc = (rank1_correct / total_images) * 100
    rank5_acc = (rank5_correct / total_images) * 100

    print("\n" + "="*30)
    print("🏆 FINAL EVALUATION METRICS 🏆")
    print("="*30)
    print(f"Total Images Tested: {total_images}")
    print(f"Rank-1 Accuracy:     {rank1_acc:.2f}%")
    print(f"Rank-5 Accuracy:     {rank5_acc:.2f}%")
    print("="*30)

if __name__ == "__main__":
    evaluate_model()