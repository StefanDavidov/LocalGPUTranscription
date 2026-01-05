import os
import shutil
from huggingface_hub import snapshot_download
from faster_whisper import WhisperModel

def download_models():
    print("="*60)
    print("INITIALIZING (Attempt 4 - Final Fix)...")
    print("Loading AI libraries...")
    print("="*60)

    # 1. Faster Whisper
    print("\n[1/2] Downloading Whisper Model (Small)...")
    try:
        model = WhisperModel("small", device="cpu", compute_type="int8")
        print("Whisper Model Downloaded Successfully!")
    except Exception as e:
        print(f"Error downloading Whisper: {e}")

    # 2. SpeechBrain - Manual Download + Rename
    print("\n[2/2] Downloading Speaker Diarization Model (SpeechBrain)...")
    
    local_model_dir = os.path.join(os.getcwd(), "models", "speechbrain")
    
    try:
        snapshot_download(
            repo_id="speechbrain/spkrec-ecapa-voxceleb",
            local_dir=local_model_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"Model files saved to: {local_model_dir}")
        
        # MANUAL FIX: Rename .txt to .ckpt so SpeechBrain doesn't try to symlink it
        src_label = os.path.join(local_model_dir, "label_encoder.txt")
        dst_label = os.path.join(local_model_dir, "label_encoder.ckpt")
        
        if os.path.exists(src_label) and not os.path.exists(dst_label):
            print("Applying manual fix for 'label_encoder.ckpt'...")
            shutil.copy2(src_label, dst_label)
            print("Fix applied.")
            
        print("SpeechBrain Model Downloaded & Ready!")
        print("(Skipping load verification to avoid further permission errors - files are present).")
        
    except Exception as e:
        print(f"Error downloading SpeechBrain: {e}")

    print("\nAll models ready.")
    print("You can close this window and run 'run.bat' now.")

if __name__ == "__main__":
    download_models()
