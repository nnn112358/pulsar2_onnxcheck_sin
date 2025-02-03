import torch
import torch.nn as nn
import numpy as np
import onnxruntime
import tarfile
import os

class SinOperation(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Sin: Sine function
        # - Input: Real numbers from 0-255
        # - Output: Range [-1, 1]
        # - Feature: Periodic function (period: 2π)
        return torch.sin(x)

def generate_and_save_data():
    """Generate test data and save as npy files"""
    # Generate values from 0 to 255 with shape [1, 255]
    x = torch.arange(0, 255, dtype=torch.float32).reshape(1, 255)
    
    # List of temporary files
    temp_files = []
    
    try:
        # Generate 10 datasets
        for i in range(10):
            # Filename
            x_file = f'input_{i:02d}.npy'
            
            # Save data
            np.save(x_file, x.numpy())
            temp_files.append(x_file)
            
            #print(f"\nDataset {i:02d}:")
            #print(f"Input shape: {x.shape}")
            #print(f"Value range: [{x.min()}, {x.max()}]")
        
        # Create tar file
        with tarfile.open('input_data.tar', 'w') as tar:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    tar.add(temp_file)
        
        print("\nTar file creation complete: input_data.tar")
        
    finally:
        # Delete temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        print("Temporary files deleted")

def main():
    # Initialize model
    model = SinOperation()
    model.eval()
    
    # Generate test data (for demonstration)
    x = torch.arange(0, 255, dtype=torch.float32).reshape(1, 255)
    
    # ONNX export
    torch.onnx.export(
        model,
        x,
        "sin_model.onnx",
        input_names=['x'],
        output_names=['z'],
        opset_version=11
    )
    
    # Verify ONNX model
    session = onnxruntime.InferenceSession("sin_model.onnx", providers=['CPUExecutionProvider'])
    
    # Run inference
    input_numpy = x.numpy()
    onnx_output = session.run(['z'], {'x': input_numpy})[0]
    pytorch_output = model(x).detach().numpy()
    
    print("Sin Operation:")
    print(f"Input shape: {input_numpy.shape}")
    print(f"Output shape: {onnx_output.shape}")
    print(f"Output range: [{onnx_output.min():.4f}, {onnx_output.max():.4f}]")
    print("Match:", np.allclose(onnx_output, pytorch_output, rtol=1e-5))
    
    # Generate calibration data
    generate_and_save_data()

if __name__ == "__main__":
    main()
