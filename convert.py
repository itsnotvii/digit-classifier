import torch
import torch.nn as nn

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

model = DigitCNN()
model.load_state_dict(torch.load('digit_model.pth', map_location='cpu'))
model.eval()
print("Model loaded successfully")

dummy_input = torch.zeros(1, 1, 28, 28)
torch.onnx.export(
    model,
    dummy_input,
    'digit_model.onnx',
    input_names=['input'],
    output_names=['output'],
    opset_version=18
)
print("Exported to digit_model.onnx")

import subprocess
result = subprocess.run([
    'tensorflowjs_converter',
    '--input_format=onnx',
    'digit_model.onnx',
    'tfjs_model'
], capture_output=True, text=True)

if result.returncode == 0:
    print("✓ Converted to TensorFlow.js format")
    print("Model saved to ./tfjs_model folder")
else:
    print("Error during conversion:")
    print(result.stderr)