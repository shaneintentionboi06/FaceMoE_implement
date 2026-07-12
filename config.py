from pathlib import Path

# Update this path to wherever you downloaded and extracted the TinyFace images! 
ROOT_DIR = Path("./Dataset/tinyface/")
DATA_DIR = ROOT_DIR.joinpath("./Training_Set")

Feature_folder = Path("./features")
Gallery_ids_mat = ROOT_DIR.joinpath('./Face_Identification_Evaluation/gallery_match_img_ID_pairs.mat')
Probe_ids_mat = ROOT_DIR.joinpath('./Face_Identification_Evaluation/probe_img_ID_pairs.mat')

BaseModelName = "microsoft/swin-large-patch4-window12-384-in22k"
trainedModel = "facemoe_adapters_colab_large.pth"
