🖋 Quantum-Inspired Handwritten Text Recognition with TrOCR
This project implements a handwritten text recognition system using Microsoft's TrOCR model, enhanced with a custom quantum-inspired optimizer for potentially improved training dynamics.

It includes:

Training Script (trainQ.py) – Trains a TrOCR model on handwritten text data stored in .parquet format.

Testing Script (testQ.py) – Runs inference on a single handwritten image to extract text.

📂 Project Structure
.
├── trainQ.py   # Script for training the model
├── testQ.py    # Script for testing the trained model
├── DATA/       # Folder containing your .parquet training data
└── README.md   # Project documentation
🚀 Features
Cached Dataset Loading – Preloads and processes images into memory for faster training.

Quantum-Inspired Optimizer – A custom optimizer that simulates certain quantum behavior concepts for gradient updates.

Hugging Face Transformers Integration – Leverages TrOCRProcessor and VisionEncoderDecoderModel.

Easy Model Saving/Loading – Trained model and processor are stored for future inference.

🛠 Installation
Clone the repository:

git clone https://github.com/yourusername/quantum-trocr.git
cd quantum-trocr
Install dependencies:


pip install torch torchvision transformers pandas pillow tqdm
(Optional) Install GPU support for PyTorch:


pip install torch --index-url https://download.pytorch.org/whl/cu118
📊 Training
Prepare your dataset in Parquet format:

Each row must have:

image (dict with bytes key for image data)

text (ground truth transcription)

Run the training script:


python trainQ.py
Default model: microsoft/trocr-base-stage1

Output directory: trocr_handwritten_quantum_inspired

🧪 Testing
Update the image_path and model_dir variables in testQ.py with:

The path to your test image

The directory where your model is saved

Run:


python testQ.py
Example output:

Predicted Text: hello world
📦 Model Saving and Loading
After training, the model and processor are saved to the specified directory:

arduino

trocr_handwritten_quantum_inspired/
├── config.json
├── pytorch_model.bin
├── preprocessor_config.json
└── ...
These can be loaded in testing or in any Hugging Face compatible pipeline.

🧠 Quantum-Inspired Optimizer
The QuantumInspiredOptimizer modifies gradient updates by simulating momentum with a dynamic scaling factor, inspired by certain quantum probability adjustments.

Update formula:

exp_avg = β * exp_avg + (1 - β) * grad
param   = param - lr * (1 + γ * step) * exp_avg
📌 Requirements
Python 3.8+

PyTorch

Hugging Face Transformers

Pandas

Pillow

tqdm

📄 License
This project is licensed under the MIT License – see the LICENSE file for details.
