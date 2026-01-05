import os
from faster_whisper import WhisperModel

def download_models():
    print("="*60)
    print("MODEL PRELOADER")
    print("Downloading AI models for transcription...")
    print("="*60)

    # Define local model path
    local_model_path = os.path.join(os.getcwd(), "models", "whisper")
    os.makedirs(local_model_path, exist_ok=True)

    # Download Whisper Medium model (used by the app)
    print("\n[1/1] Downloading Whisper Model (Medium)...")
    try:
        model = WhisperModel("medium", device="cpu", compute_type="int8", download_root=local_model_path)
        print("Whisper Model Downloaded Successfully!")
    except Exception as e:
        print(f"Error downloading Whisper: {e}")

    print("\n" + "="*60)
    print("All models ready.")
    print("Note: Pyannote diarization model will be downloaded on first run.")
    print("You can close this window and run 'run.bat' now.")
    print("="*60)

if __name__ == "__main__":
    download_models()
