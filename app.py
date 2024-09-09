import streamlit as st
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch

# Load Hugging Face's Detr Object Detection model
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Function to analyze the image using Hugging Face's DETR model
def analyze_screenshots(screenshots):
    instructions = []
    for idx, image in enumerate(screenshots):
        # Prepare the image for the model
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        
        # Get the detected objects
        target_sizes = torch.tensor([image.size[::-1]])  # Model expects (height, width)
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]
        
        # Generate instructions based on detected objects
        objects = results['scores'] > 0.7  # Confidence threshold
        detected_labels = [model.config.id2label[obj] for obj in results['labels'][objects]]
        
        test_instructions = f"Test Case {idx+1}: \n"
        test_instructions += "Press the following icons or buttons to navigate: \n"
        
        for label, box in zip(detected_labels, results['boxes'][objects]):
            test_instructions += f"- Interact with '{label}' at location {box.tolist()}.\n"
        
        instructions.append(test_instructions)
    
    return instructions

# Streamlit app
st.title("Automated Testing Instruction Generator with Hugging Face")

# Input text box for optional context
context = st.text_input("Optional Context")

# Upload multiple images
uploaded_files = st.file_uploader("Upload screenshots", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Button to trigger the LLM
if st.button("Describe Testing Instructions"):
    if uploaded_files:
        screenshots = [Image.open(img) for img in uploaded_files]
        instructions = analyze_screenshots(screenshots)
        
        # Display the test cases
        for idx, test_case in enumerate(instructions):
            st.write(f"### Test Case {idx+1}")
            st.write(test_case)
    else:
        st.error("Please upload at least one screenshot.")
