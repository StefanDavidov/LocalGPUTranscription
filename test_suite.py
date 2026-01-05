import os
import sys
import ffmpeg
import json
from transcribe import VideoTranscriber

def create_dummy_video(filename="test_video.mp4"):
    print(f"Generating synthetic video: {filename}...")
    try:
        # Generate 3 seconds of video (black) and audio (sine wave beep)
        video = ffmpeg.input('color=c=black:s=640x480:r=30', t=3, f='lavfi')
        audio = ffmpeg.input('sine=f=440:d=3', f='lavfi')
        
        out = ffmpeg.output(video, audio, filename, vcodec='libx264', acodec='aac', pix_fmt='yuv420p')
        out.run(quiet=True, overwrite_output=True)
        print("Video generated successfully.")
        return True
    except ffmpeg.Error as e:
        print(f"Failed to generate video via FFmpeg: {e.stderr.decode() if e.stderr else str(e)}")
        return False

def run_test():
    video_file = "test_video.mp4"
    
    # 1. Generate Content
    if not create_dummy_video(video_file):
        print("Skipping pipeline test due to generation failure.")
        return

    # 2. Run Pipeline
    print("\nInitializing Transcriber Pipeline...")
    try:
        # Use medium model, force cuda if possible
        transcriber = VideoTranscriber(model_size="medium", use_cuda=True)
        print("Transcriber initialized.")
        
        print(f"Processing {video_file}...")
        results = transcriber.process_video(video_file)
        
        print("\n=== TEST RESULTS ===")
        print(json.dumps(results, indent=2))
        
        if isinstance(results, list):
            print("\nSUCCESS: Pipeline finished without crashing.")
            if len(results) > 0:
                print("SUCCESS: Audio was detected and transcribed.")
            else:
                print("NOTE: Pipeline finished but transcript is empty (expected for sine wave).")
        else:
            print("FAILURE: Output format is incorrect.")

    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    if os.path.exists(video_file):
        try:
            os.remove(video_file)
            print("Cleanup complete.")
        except:
            pass

if __name__ == "__main__":
    run_test()
