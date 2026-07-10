from FaceMoE import SwinFaceMoE

# Assuming TinyFace has 5139 identities to classify
model = SwinFaceMoE(num_face_identities=5139)
model.to("cuda")

# Now just pass your images in!
# logits = model(images)