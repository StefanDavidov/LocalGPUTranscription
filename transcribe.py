import os
from dotenv import load_dotenv

# Load env vars
load_dotenv()
# DISABLE SYMLINKS to fix Windows permission issues
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1" 

import ffmpeg
import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

class VideoTranscriber:
    def __init__(self, model_size="medium", use_cuda=True):
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        
        print(f"Loading Whisper Model: {model_size} on {self.device}...")
        
        # Define local model path
        local_model_path = os.path.join(os.getcwd(), "models", "whisper")
        os.makedirs(local_model_path, exist_ok=True)
        print(f"Model storage: {local_model_path}")
        
        self.whisper_model = WhisperModel(model_size, device=self.device, compute_type=self.compute_type, download_root=local_model_path)
        
        print("Loading Speaker Diarization Model (Pyannote)...")
        # Load token from env
        self.auth_token = os.getenv("HUGGINGFACE_API_KEY")
        
        if not self.auth_token:
            print("WARNING: HUGGINGFACE_API_KEY missing in .env. Diarization may fail.")
        
        try:
            # Fix for PyTorch 2.6+ security change causing "WeightsUnpickler error"
            # Pyannote checkpoints use TorchVersion, Specifications, Problem, Resolution which are not in the default safe list
            try:
                from pyannote.audio.core.task import Problem, Resolution, Specifications
                torch.serialization.add_safe_globals([torch.torch_version.TorchVersion, Problem, Resolution, Specifications])
            except Exception:
                pass # Ignore if this fails, might be old pytorch or other issue

            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.auth_token
            )
            if self.device == "cuda":
                self.diarization_pipeline.to(torch.device("cuda"))
        except Exception as e:
            print(f"Failed to load Pyannote pipeline: {e}")
            self.diarization_pipeline = None

    def extract_audio(self, video_path, output_wav="temp_audio.wav"):
        """
        Extracts mono 16kHz audio from video using FFmpeg.
        """
        try:
            if os.path.exists(output_wav):
                os.remove(output_wav)
            
            print(f"Extracting audio from {video_path}...")
            (
                ffmpeg
                .input(video_path)
                .output(output_wav, ac=1, ar=16000)
                .run(quiet=True, overwrite_output=True)
            )
            return output_wav
        except ffmpeg.Error as e:
            print("FFmpeg error:", e.stderr.decode() if e.stderr else str(e))
            raise

    def transcribe(self, audio_path, progress_callback=None):
        """
        Runs Whisper transcription.
        Returns list of dicts: {'start': 0.0, 'end': 1.0, 'text': 'foo'}
        """
        print("Transcribing audio...")
        # Enable VAD filter to prevent hallucinations in silence
        segments, info = self.whisper_model.transcribe(
            audio_path, 
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        total_duration = info.duration
        
        result_segments = []
        for segment in segments:
            result_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
            if progress_callback and total_duration > 0:
                # Calculate progress (0-100)
                # We allocate 80% to transcription (leaves 20% for diarization)
                percent = int((segment.end / total_duration) * 80)
                progress_callback(percent)
                
        return result_segments

    def diarize(self, audio_path, segments, num_speakers=None, progress_callback=None):
        """
        Runs Pyannote.audio pipeline and maps speakers to Whisper segments.
        """
        if not self.diarization_pipeline:
            print("Diarization pipeline not loaded. Skipping.")
            return segments

        print("Running Pyannote Diarization...")
        
        try:
            # Run pipeline
            # If num_speakers is provided, use it
            if num_speakers:
                diarization = self.diarization_pipeline(audio_path, num_speakers=num_speakers)
            else:
                diarization = self.diarization_pipeline(audio_path)
                
            # Convert Pyannote annotation to a list of turns
            # turn: (Segment(start, end), track, label)
            speaker_turns = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_turns.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
            
            print(f"Diarization complete. Found {len(speaker_turns)} speaker turns.")
            
            # Map speakers to Whisper segments
            # Strategy: For each Whisper segment, find which speaker overlaps the most
            for seg in segments:
                w_start = seg["start"]
                w_end = seg["end"]
                w_duration = w_end - w_start
                
                # If segment is too short, just assign closest?
                if w_duration <= 0:
                    continue
                    
                # Calculate overlap with each speaker
                speaker_overlaps = {}
                
                for turn in speaker_turns:
                    # Intersection of [w_start, w_end] and [t_start, t_end]
                    start_overlap = max(w_start, turn["start"])
                    end_overlap = min(w_end, turn["end"])
                    overlap_duration = max(0, end_overlap - start_overlap)
                    
                    if overlap_duration > 0:
                        spk = turn["speaker"]
                        if spk not in speaker_overlaps:
                            speaker_overlaps[spk] = 0
                        speaker_overlaps[spk] += overlap_duration
                
                # Find speaker with max overlap
                if speaker_overlaps:
                    best_speaker = max(speaker_overlaps, key=speaker_overlaps.get)
                    seg["speaker"] = best_speaker
                else:
                    seg["speaker"] = "Unknown"
                    
            return segments
            
        except Exception as e:
            print(f"Diarization failed: {e}")
            import traceback
            traceback.print_exc()
            return segments

    def process_video(self, video_path, num_speakers=None, progress_callback=None):
        # NOTE: We extract to .wav because AI models cannot read .mp4 video files directly.
        # They need pure audio data. This temporary file is deleted after processing.
        wav_path = self.extract_audio(video_path)
        
        segments = self.transcribe(wav_path, progress_callback)
        
        if progress_callback:
            progress_callback(80) # Transcription done, starting diarization
            
        final_data = self.diarize(wav_path, segments, num_speakers=num_speakers, progress_callback=progress_callback)
        
        if progress_callback:
            progress_callback(100)
        
        # Cleanup
        if os.path.exists(wav_path):
            os.remove(wav_path)
            
        return final_data
