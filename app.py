import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import io

# Load model class
import torch.nn as nn

class MultiLabelModel(nn.Module):
    def __init__(self):
        super(MultiLabelModel, self).__init__()
        self.base = models.efficientnet_b0(weights="IMAGENET1K_V1")
        self.base.classifier[1] = nn.Linear(self.base.classifier[1].in_features, 3)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.base(x)
        x = self.dropout(x)
        return torch.sigmoid(x)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiLabelModel().to(device)
model.load_state_dict(torch.load("thalassemia_model.pth", map_location=device))
model.eval()

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Streamlit UI
st.title("Thalassemia Classifier")
st.write("Upload a blood smear image:")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and preprocess the image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess the image for model input
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Perform prediction
        with torch.no_grad():
            output = model(input_tensor)
            prediction = (output > 0.5).float().cpu().numpy()[0]
            classes = ['major', 'minor', 'normal']
            results = {classes[i]: int(prediction[i]) for i in range(3)}

        # Display the prediction results
        st.write("### Prediction:")
        for cls, val in results.items():
            st.write(f"{cls.capitalize()}: {'✅' if val else '❌'}")

    except Exception as e:
        st.error(f"Error processing image: {e}")


