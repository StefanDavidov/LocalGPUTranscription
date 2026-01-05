import os
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

def download_models():
    print("="*60)
    print("OFFLINE MODEL PRELOADER")
    print("Downloading all models to local folder for portability...")
    print("="*60)

    # 0. CRITICAL: Disable Symlinks for Windows (Fixes Error 1314)
    # This forces HF to copy files instead of creating syslinks, which requires Admin/Dev mode
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

    # 1. Setup Local Cache Path
    # We will download EVERYTHING into 'models/huggingface_cache'
    # This structure allows us to just set HF_HOME environment variable later
    project_root = os.getcwd()
    models_dir = os.path.join(project_root, "models")
    hf_cache_dir = os.path.join(models_dir, "huggingface_cache")
    whisper_dir = os.path.join(models_dir, "whisper")
    
    os.makedirs(hf_cache_dir, exist_ok=True)
    os.makedirs(whisper_dir, exist_ok=True)
    
    # Set ENV var so snapshot_download uses this folder AND respects the symlink disable
    os.environ["HF_HOME"] = hf_cache_dir
    
    # 2. Download Whisper (Medium)
    print("\n[1/2] Downloading Whisper Model (Medium)...")
    try:
        # Note: Whisper has its own download_root param, so we keep it separate if we want
        # properly isolated whisper structure, or we can use the cache.
        # Sticking to separate folder for Whisper is cleaner for ctranslate2 models.
        model = WhisperModel("medium", device="cpu", compute_type="int8", download_root=whisper_dir)
        print("Whisper Model Downloaded Successfully!")
    except Exception as e:
        print(f"Error downloading Whisper: {e}")

    # 3. Download Pyannote Pipeline (and all dependencies)
    print("\n[2/2] Downloading Pyannote Diarization Pipeline...")
    try:
        load_dotenv()
        token = os.getenv("HUGGINGFACE_API_KEY")
        
        if not token:
            print("ERROR: HUGGINGFACE_API_KEY not found in .env. Cannot download Pyannote models.")
        else:
            print(f"Downloading to cache: {hf_cache_dir}")
            
            # Download the main pipeline
            # snapshot_download will automatically fetch dependencies (segmentation, embedding)
            # if we request the pipeline repo.
            
            # However, simply downloading the repo usually isn't enough for 'Pipeline.from_pretrained' 
            # to work offline unless the file structure matches exactly what huggingface_hub expects.
            # Using 'cache_dir' (via HF_HOME env var) is the key.
            
            # We trigger a download by "pretending" to load it, or just snapshotting the relevant repos.
            
            repos_to_fetch = [
                "pyannote/speaker-diarization-3.1",
                "pyannote/segmentation-3.0",
                "speechbrain/spkrec-ecapa-voxceleb"
            ]
            
            for repo in repos_to_fetch:
                print(f"Fetching {repo}...")
                snapshot_download(
                    repo_id=repo,
                    cache_dir=hf_cache_dir,
                    token=token
                )
                
            print("Pyannote models downloaded to local cache.")
            
    except Exception as e:
        print(f"Error downloading Pyannote: {e}")

    print("\n" + "="*60)
    print("DONE! The 'models' folder now contains everything.")
    print("You can zip the project, move it, and run it offline.")
    print("="*60)

if __name__ == "__main__":
    download_models()
