import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import io
import os

def test_model_on_image(image_bytes, model_dir):
    """
    Tests a trained TrOCR model on a single image given its byte representation.

    Args:
        image_bytes (bytes): The byte representation of the image.
        model_dir (str): Path to the directory where the model and processor are saved.

    Returns:
        str: The predicted text.
    """
    # Load processor and model from the saved directory
    processor = TrOCRProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load image from bytes
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    # Generate predictions
    generated_ids = model.generate(pixel_values)
    generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)

    return generated_text

def main():
    """
    Main function to run the test.  Loads an image and the saved model,
    and then runs the test.
    """
    # Load a sample image.
    image_path = "C:/Users/SIDDHESHWAR DUBEY/Downloads/l.jpg"  # Replace with your image path
    try:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}.  Please make sure the path is correct.")
        return

    # Specify the directory where you saved the model
    model_dir = "C:/Users/SIDDHESHWAR DUBEY/Documents/CODES/Projects/Quantum/hnd/trocr_handwritten_quantum_inspired"  #  Make sure this matches where you saved

    # Check if the model directory exists
    if not os.path.exists(model_dir):
        print(f"Error: Model directory not found at {model_dir}.  Make sure the path is correct and the model is saved.")
        return

    # Test the model
    predicted_text = test_model_on_image(image_bytes, model_dir)
    print(f"Predicted Text: {predicted_text}")

if __name__ == "__main__":
    main()
