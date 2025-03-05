import numpy as np
import axengine as axe

def run_onnx_model():
    # Load the ONNX model
    #session = onnxruntime.InferenceSession("sin_model.onnx", providers=['CPUExecutionProvider'])
    session = axe.InferenceSession("sin_model.axmodel")
    # Generate input data for inference
    input_data = np.arange(0, 255, dtype=np.float32).reshape(1, 255)
    
    # Run inference
    onnx_output = session.run(['z'], {'x': input_data})[0]
    
    # Print information
    print("ONNX Inference:")
    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {onnx_output.shape}")
    print(f"Output range: [{onnx_output.min():.4f}, {onnx_output.max():.4f}]")
    
    # Save results to CSV
    combined = np.vstack((input_data, onnx_output))
    np.savetxt('onnx_results.csv', combined.T, delimiter=',', 
               header='input,output', comments='')
    
    # Display a few sample values
    print("\n===== Sample Values =====")
    print("Input\t\tOutput")
    for i in range(0, 100, 10):
        print(f"{input_data[0,i]:.1f}\t\t{onnx_output[0,i]:.6f}")

if __name__ == "__main__":
    run_onnx_model()
