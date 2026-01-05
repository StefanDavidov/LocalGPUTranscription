import torch
import sys
import os

print("="*60)
print("GPU VERIFICATION DIAGNOSTIC")
print("="*60)

# 1. Check PyTorch CUDA
print(f"PyTorch Version: {torch.__version__}")
cuda_available = torch.cuda.is_available()
print(f"CUDA Available:  {cuda_available}")

if cuda_available:
    device_count = torch.cuda.device_count()
    print(f"GPU Count:       {device_count}")
    current_device = torch.cuda.current_device()
    print(f"Current Device:  {current_device}")
    gpu_name = torch.cuda.get_device_name(current_device)
    print(f"GPU Name:        {gpu_name}")
else:
    print("WARNING: CUDA is NOT available. PyTorch is running on CPU.")

print("-" * 60)

# 2. Check Faster-Whisper Device
print("Checking Faster-Whisper Model Load...")
try:
    from faster_whisper import WhisperModel
    
    # Try to load small model on CUDA
    # compute_type="int8" is what we use in the app
    model = WhisperModel("small", device="cuda", compute_type="int8")
    
    print(f"Model internal device: {model.model.device}")
    
    if "cuda" in str(model.model.device):
        print("\nSUCCESS: Faster-Whisper is successfully running on the NVIDIA GPU.")
    else:
        print("\nFAILURE: Model loaded, but it says it is on CPU.")
        
except Exception as e:
    print(f"\nERROR Loading Whisper on CUDA: {e}")
    print("This often means CUDA drivers are missing or incompatible, or VRAM is full.")

print("="*60)
input("Press Enter to close...")
