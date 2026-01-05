import sys
import os

# SET OFFLINE CACHE PATH
# This ensures we use the models included in the zip, not the global user cache
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "models", "huggingface_cache")

import threading
import queue
import tempfile
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import faulthandler
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
import pygame

# CRITICAL FIX: Disable TQDM Monitor Thread
import tqdm
tqdm.tqdm.monitor_interval = 0

# ENABLE FAULTHANDLER FOR SEGFAULTS
crash_log_file = open("crash_dump.log", "w")
faulthandler.enable(file=crash_log_file)

# REDIRECT OUTPUT TO LOG FILE
log_file_path = "app.log"
log_file = open(log_file_path, 'w', buffering=1)
sys.stdout = log_file
sys.stderr = log_file

def log_debug(msg):
    try:
        print(f"[DEBUG] {msg}")
        sys.stdout.flush()
        log_file.flush()
    except:
        pass

def flush_log():
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        log_file.flush()
        os.fsync(log_file.fileno())
    except:
        pass

# Import our backend
# from transcribe import VideoTranscriber  <-- MOVED TO LAZY IMPORT
# from export_utils import export_to_pdf   <-- MOVED TO LAZY IMPORT

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Speaker colors for transcript display
SPEAKER_COLORS = ["#d32f2f", "#1976d2", "#388e3c", "#fbc02d", "#8e24aa", "#f57c00"]


class VideoPlayer(ctk.CTkFrame):
    """Custom video player widget using OpenCV and PIL with full controls and audio."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.video_path = None
        self.cap = None
        self.is_playing = False
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        self.frame_delay = 33  # ms between frames
        self.volume = 1.0  # 0.0 to 1.0
        self._seeking = False  # Prevent update conflicts during seek
        
        # Audio state
        self._audio_file = None  # Temporary audio file path
        self._audio_loaded = False
        self._audio_start_offset = 0  # Video time offset when audio started
        self._audio_paused = False  # Track if audio was specifically paused
        
        # Initialize pygame mixer for audio
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
            self._mixer_initialized = True
            log_debug("Pygame mixer initialized successfully")
        except Exception as e:
            self._mixer_initialized = False
            log_debug(f"Failed to initialize pygame mixer: {e}")
        
        self._photo_image = None
        self._update_job = None
        
        # Main layout
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Video display area
        self.canvas = ctk.CTkLabel(self, text="🎬 Load a video to begin", 
                                    font=("Segoe UI", 18), text_color="#666666")
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 5))
        
        # Click to play/pause
        self.canvas.bind("<Button-1>", lambda e: self._toggle_playback())
        
        # Controls frame
        controls_frame = ctk.CTkFrame(self, fg_color="transparent", height=80)
        controls_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(5, 10))
        controls_frame.grid_columnconfigure(2, weight=1)  # Slider expands
        
        # Play/Pause Button (new column 0)
        self.btn_play = ctk.CTkButton(controls_frame, text="▶", width=30, height=30,
                                       command=self._toggle_playback)
        self.btn_play.grid(row=0, column=0, padx=(0, 10), pady=5)
        
        # Time display (column 1)
        self.time_label = ctk.CTkLabel(controls_frame, text="00:00 / 00:00", 
                                        font=("Segoe UI", 11), width=100)
        self.time_label.grid(row=0, column=1, padx=(0, 10), pady=5)
        
        # Seek slider (column 2, expandable)
        self.seek_slider = ctk.CTkSlider(controls_frame, from_=0, to=100, 
                                          command=self._on_seek_slider)
        self.seek_slider.set(0)
        self.seek_slider.grid(row=0, column=2, sticky="ew", padx=5, pady=5)
        
        # Volume frame (column 3)
        volume_frame = ctk.CTkFrame(controls_frame, fg_color="transparent")
        volume_frame.grid(row=0, column=3, padx=(10, 0), pady=5)
        
        self.volume_icon = ctk.CTkLabel(volume_frame, text="🔊", font=("Segoe UI", 14), width=25)
        self.volume_icon.pack(side="left", padx=(0, 5))
        
        self.volume_slider = ctk.CTkSlider(volume_frame, from_=0, to=100, width=80,
                                            command=self._on_volume_change)
        self.volume_slider.set(100)
        self.volume_slider.pack(side="left")
        
    def load(self, video_path):
        """Load a video file and extract audio."""
        self.stop()
        self._cleanup_audio()
        
        if self.cap:
            self.cap.release()
            
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.frame_delay = max(1, int(1000 / self.fps))
        self.current_frame = 0
        
        # Update slider range
        self.seek_slider.configure(to=self.total_frames)
        
        # Extract audio for playback in background
        self._audio_thread = threading.Thread(target=self._extract_audio, args=(video_path,))
        self._audio_thread.daemon = True
        self._audio_thread.start()
        
        # Show first frame
        self._show_frame()
        self._update_time_display()
        log_debug(f"Video loaded: {video_path}, {self.total_frames} frames, {self.fps} fps")
        
    def _extract_audio(self, video_path):
        """Extract audio from video to a temporary file for playback."""
        if not self._mixer_initialized:
            return
            
        try:
            import subprocess
            
            # Create temp file for audio
            # Use unique name for each video to prevent conflicts
            import hashlib
            hash_name = hashlib.md5(video_path.encode()).hexdigest()[:8]
            # SWITCH TO WAV for accurate seeking
            self._audio_file = os.path.join(tempfile.gettempdir(), f"video_audio_{hash_name}.wav")
            
            # Check if current video path still matches (user typically didn't change it yet, but good practice)
            if self.video_path != video_path:
                return

            # Use ffmpeg to extract audio as WAV (uncompressed PCM)
            # This ensures sample-accurate seeking compared to VBR MP3
            cmd = [
                "ffmpeg", "-y",  # Overwrite
                "-i", video_path,
                "-vn",  # No video
                "-acodec", "pcm_s16le", # Uncompressed 16-bit PCM
                "-ar", "44100", # 44.1kHz
                "-ac", "2",     # Stereo
                self._audio_file
            ]
            
            log_debug(f"Starting audio extraction for {video_path}...")
            
            # Run ffmpeg (this blocks the thread, but not the UI now)
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            
            # Verify we're still on the same video before loading
            if self.video_path != video_path:
                log_debug("Video changed during extraction, discarding audio.")
                return

            if os.path.exists(self._audio_file):
                # We need to queue this back to the main thread ideally, but mixer operations 
                # are generally thread-safe for loading.
                pygame.mixer.music.load(self._audio_file)
                pygame.mixer.music.set_volume(self.volume)
                self._audio_loaded = True
                log_debug(f"Audio extracted and loaded: {self._audio_file}")
            else:
                log_debug(f"Audio extraction failed: {result.stderr}")
                self._audio_loaded = False
                
        except Exception as e:
            log_debug(f"Error extracting audio: {e}")
            self._audio_loaded = False
            
    def _cleanup_audio(self):
        """Clean up temporary audio file."""
        try:
            if self._audio_loaded:
                pygame.mixer.music.stop()
            if self._audio_file and os.path.exists(self._audio_file):
                try:
                    os.remove(self._audio_file)
                except:
                    pass
            self._audio_file = None
            self._audio_loaded = False
        except:
            pass
        
    def _show_frame(self):
        """Display the current frame."""
        if not self.cap:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.is_playing = False
            return
            
        self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to fit widget while maintaining aspect ratio
        canvas_width = self.canvas.winfo_width() - 20
        canvas_height = self.canvas.winfo_height() - 20
        
        if canvas_width > 100 and canvas_height > 100:
            h, w = frame_rgb.shape[:2]
            scale = min(canvas_width / w, canvas_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))
        
        # Convert to PIL Image and then to PhotoImage
        image = Image.fromarray(frame_rgb)
        self._photo_image = ImageTk.PhotoImage(image)
        
        # Update canvas
        self.canvas.configure(image=self._photo_image, text="")
        
        # Update UI if not seeking (to prevent feedback loop)
        if not self._seeking:
            self.seek_slider.set(self.current_frame)
            self._update_time_display()
        
    def _update_loop(self):
        """Main video playback loop with audio sync."""
        if not self.is_playing:
            return
            
        # Audio-driven sync
        if self._audio_loaded and pygame.mixer.music.get_busy():
            # Use mixer.music.get_pos() for hardware-clock sync
            # get_pos() returns ms played since last play()
            audio_pos_ms = pygame.mixer.music.get_pos()
            
            if audio_pos_ms >= 0:
                elapsed = audio_pos_ms / 1000.0
                expected_time = self._audio_start_offset + elapsed
                expected_frame = int(expected_time * self.fps)
            
            # If video is lagging behind audio
            if expected_frame > self.current_frame:
                # Catch up by skipping frames
                # If lag is large (> 5 frames), just jump to it
                if expected_frame - self.current_frame > 5:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, expected_frame)
                    # Check where we actually landed (keyframe snap)
                    actual_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    
                    # If we snapped to before our target, grab frames to catch up
                    # Limit to 50 frames to prevent freeze
                    if actual_frame < expected_frame:
                         frames_to_skip = min(expected_frame - actual_frame, 50)
                         for _ in range(frames_to_skip):
                             self.cap.grab()
                         actual_frame += frames_to_skip
                         
                    self.current_frame = actual_frame
                else:
                    # Small lag, fast forward
                    while self.current_frame < expected_frame:
                        self.cap.grab()
                        self.current_frame += 1
            
            # If video is ahead of audio
            elif expected_frame < self.current_frame:
                # Wait for audio to catch up
                delay_ms = int((self.current_frame - expected_frame) / self.fps * 1000)
                if delay_ms > 10:
                    # Reschedule loop with delay
                    self._update_job = self.after(min(delay_ms, 100), self._update_loop)
                    return

        self._show_frame()
        
        if self.is_playing and self.current_frame < self.total_frames:
            self._update_job = self.after(self.frame_delay, self._update_loop)
        else:
            self.is_playing = False
            self.btn_play.configure(text="▶")
            if self._audio_loaded:
                pygame.mixer.music.stop()
                
    def _on_seek_slider(self, value):
        """Handle seek slider movement."""
        if not self.cap:
            return
        self._seeking = True
        frame_num = int(float(value))
        frame_num = max(0, min(frame_num, self.total_frames - 1))
        # Video seek
        # Video seek
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        
        # PRECISION SEEK: Check where we actually landed
        actual_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        # If we landed significantly before target (keyframe snap), forward to target
        if actual_frame < frame_num:
            frames_to_skip = min(frame_num - actual_frame, 300) # Limit 300 frames (~10s)
            for _ in range(frames_to_skip):
                self.cap.grab()
        
        # IMPORTANT: _show_frame will update self.current_frame to the ACTUAL position
        self._show_frame()
        self._update_time_display()
        
        # Sync audio to actual video position
        if self._audio_loaded and self.is_playing:
            # Use actual current frame for audio sync to prevent drift
            seek_secs = self.current_frame / self.fps if self.fps > 0 else 0
            pygame.mixer.music.play(start=seek_secs)
            pygame.mixer.music.set_volume(self.volume)
            
            # Reset sync timers
            # get_pos() resets to 0 on play()
            self._audio_start_offset = seek_secs
            
        self._seeking = False
        
    def _on_volume_change(self, value):
        """Handle volume slider change."""
        self.volume = float(value) / 100.0
        # Update volume icon
        if self.volume == 0:
            self.volume_icon.configure(text="🔇")
        elif self.volume < 0.5:
            self.volume_icon.configure(text="🔉")
        else:
            self.volume_icon.configure(text="🔊")
        # Update audio volume
        if self._audio_loaded:
            pygame.mixer.music.set_volume(self.volume)
        
    def _update_time_display(self):
        """Update the time display label."""
        if not self.cap:
            return
        current_secs = self.current_frame / self.fps if self.fps > 0 else 0
        total_secs = self.total_frames / self.fps if self.fps > 0 else 0
        current_str = self._format_time(current_secs)
        total_str = self._format_time(total_secs)
        self.time_label.configure(text=f"{current_str} / {total_str}")
        
    def _format_time(self, seconds):
        """Format seconds as MM:SS or HH:MM:SS."""
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"
            
    def _toggle_playback(self):
        """Toggle video play/pause."""
        if not self.cap:
            return
            
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def play(self):
        """Start playback with audio."""
        if not self.cap:
            return
        self.is_playing = True
        self.btn_play.configure(text="⏸")
        
        # Start audio from current position
        if self._audio_loaded:
            current_secs = self.current_frame / self.fps if self.fps > 0 else 0
            pygame.mixer.music.play(start=current_secs)
            pygame.mixer.music.set_volume(self.volume)
            
            # Sync initialization
            self._audio_start_offset = current_secs
            
        self._update_loop()
        
    def pause(self):
        """Pause playback."""
        self.is_playing = False
        self.btn_play.configure(text="▶")
        if self._update_job:
            self.after_cancel(self._update_job)
            self._update_job = None
        # Pause audio
        if self._audio_loaded:
            pygame.mixer.music.pause()
            
    def stop(self):
        """Stop playback and reset position."""
        self.pause()
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame = 0
            self.seek_slider.set(0)
            self._update_time_display()
        # Stop audio
        if self._audio_loaded:
            pygame.mixer.music.stop()
            self._audio_paused = False
            
    def seek(self, seconds):
        """Seek to a specific time in seconds."""
        if not self.cap:
            return
        frame_num = int(seconds * self.fps)
        frame_num = max(0, min(frame_num, self.total_frames - 1))
        # Seek video
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        
        # PRECISION SEEK: Check where we actually landed
        actual_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        # If we landed significantly before target (keyframe snap), forward to target
        if actual_frame < frame_num:
            frames_to_skip = min(frame_num - actual_frame, 300) # Limit 300 frames (~10s)
            for _ in range(frames_to_skip):
                self.cap.grab()
        
        # Update current frame to ACTUAL position
        self._show_frame()
        self._update_time_display()
        
        # Sync audio to ACTUAL position
        if self._audio_loaded:
            actual_seconds = self.current_frame / self.fps if self.fps > 0 else 0
            
            if self.is_playing:
                pygame.mixer.music.play(start=actual_seconds)
                pygame.mixer.music.set_volume(self.volume)
                
                # Reset sync timers
                self._audio_start_offset = actual_seconds
                
            # Invalidate pause state so next play() restarts from new position
            self._audio_paused = False



class SpeakerRenameDialog(ctk.CTkToplevel):
    """Modal dialog for renaming speakers."""
    
    def __init__(self, parent, speakers, callback):
        super().__init__(parent)
        self.callback = callback
        
        self.title("Rename Speaker")
        self.geometry("400x250")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        
        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - 400) // 2
        y = parent.winfo_y() + (parent.winfo_height() - 250) // 2
        self.geometry(f"+{x}+{y}")
        
        # Main container frame
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Speaker selection
        ctk.CTkLabel(main_frame, text="Select Speaker:", font=("Segoe UI", 14)).pack(anchor="w", pady=(0, 5))
        self.speaker_var = ctk.StringVar(value=speakers[0] if speakers else "")
        self.speaker_menu = ctk.CTkOptionMenu(main_frame, values=speakers, variable=self.speaker_var, width=360)
        self.speaker_menu.pack(fill="x", pady=(0, 15))
        
        # New name entry
        ctk.CTkLabel(main_frame, text="New Name:", font=("Segoe UI", 14)).pack(anchor="w", pady=(0, 5))
        self.name_entry = ctk.CTkEntry(main_frame, width=360, placeholder_text="Enter new speaker name...")
        self.name_entry.pack(fill="x", pady=(0, 20))
        
        # Buttons frame - centered
        btn_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        btn_frame.pack(fill="x")
        
        # Inner frame for button centering
        btn_inner = ctk.CTkFrame(btn_frame, fg_color="transparent")
        btn_inner.pack(anchor="center")
        
        self.ok_button = ctk.CTkButton(btn_inner, text="✓ Confirm Rename", width=140, 
                                        fg_color="#2e7d32", hover_color="#1b5e20",
                                        command=self.on_ok)
        self.ok_button.pack(side="left", padx=(0, 10))
        
        self.cancel_button = ctk.CTkButton(btn_inner, text="✕ Cancel", width=100, 
                                            fg_color="#424242", hover_color="#616161",
                                            command=self.destroy)
        self.cancel_button.pack(side="left")
        
        # Bind Enter key to confirm and Escape to cancel
        self.name_entry.bind("<Return>", lambda e: self.on_ok())
        self.bind("<Escape>", lambda e: self.destroy())
        
        # Focus on entry field
        self.name_entry.focus()
        
        # Lift window to ensure it's visible
        self.lift()
        self.focus_force()
        
    def on_ok(self):
        speaker = self.speaker_var.get()
        new_name = self.name_entry.get().strip()
        if speaker and new_name:
            log_debug(f"Renaming speaker '{speaker}' to '{new_name}'")
            self.callback(speaker, new_name)
        self.destroy()


class TranscriptionApp(ctk.CTk):
    """Main application window for the video transcriber."""
    
    def __init__(self):
        super().__init__()
        
        self.title("Local AI Video Transcriber")
        self.geometry("1200x800")
        self.minsize(800, 600)
        
        # Application state
        self.transcript_data = []
        self.speaker_names = {}
        self.video_path = None
        
        # Threading state
        self.transcription_queue = queue.Queue()
        self.transcription_thread = None
        self.is_transcribing = False
        
        # Rendering state
        self.batch_size = 50
        self.current_render_index = 0
        
        # Following Mode State
        self.following_mode = False
        self._last_highlighted_index = -1
        
        self.init_ui()
        
        # Handle window close properly
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def init_ui(self):
        """Initialize the user interface."""
        
        # ===== TOOLBAR =====
        toolbar = ctk.CTkFrame(self, height=50)
        toolbar.pack(fill="x", padx=10, pady=(10, 5))
        toolbar.pack_propagate(False)
        
        self.btn_open = ctk.CTkButton(toolbar, text="📂 Open Video", width=120, command=self.open_file)
        self.btn_open.pack(side="left", padx=5, pady=10)
        
        self.btn_transcribe = ctk.CTkButton(toolbar, text="🎙️ Start Transcription", width=150, 
                                             command=self.start_transcription, state="disabled")
        self.btn_transcribe.pack(side="left", padx=5, pady=10)
        
        # Speaker Count Hint
        ctk.CTkLabel(toolbar, text="Speaker Count:").pack(side="left", padx=(10, 2))
        self.entry_speakers = ctk.CTkEntry(toolbar, width=50, placeholder_text="Auto")
        self.entry_speakers.pack(side="left", padx=2)
        
        # Play button removed (moved to video player)
        
        self.btn_rename = ctk.CTkButton(toolbar, text="✏️ Rename Speaker", width=140, 
                                         command=self.rename_speaker_dialog)
        self.btn_rename.pack(side="left", padx=5, pady=10)
        
        self.btn_export = ctk.CTkButton(toolbar, text="📄 Export PDF", width=120, command=self.export_pdf)
        self.btn_export.pack(side="left", padx=5, pady=10)
        
        # Following Mode Toggle
        self.btn_follow = ctk.CTkButton(toolbar, text="Follow Mode", width=100, 
                                         fg_color="#424242", hover_color="#616161",
                                         command=self.toggle_following_mode)
        self.btn_follow.pack(side="left", padx=5, pady=10)
        
        # ===== MAIN CONTENT (Video + Transcript) =====
        content_frame = ctk.CTkFrame(self, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Configure grid weights for responsive layout
        content_frame.grid_columnconfigure(0, weight=2)
        content_frame.grid_columnconfigure(1, weight=1)
        content_frame.grid_rowconfigure(0, weight=1)
        
        # Video player area
        self.video_player = VideoPlayer(content_frame, corner_radius=10)
        self.video_player.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=0)
        
        # Transcript area
        transcript_frame = ctk.CTkFrame(content_frame, corner_radius=10)
        transcript_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=0)
        transcript_frame.grid_rowconfigure(0, weight=1)
        transcript_frame.grid_columnconfigure(0, weight=1)
        
        self.transcript_box = ctk.CTkTextbox(transcript_frame, wrap="word", font=("Segoe UI", 13),
                                              state="disabled", cursor="arrow")
        self.transcript_box.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Bind scroll event for infinite loading AND disable following mode
        self.transcript_box.bind("<MouseWheel>", self.on_transcript_scroll)
        self.transcript_box.bind("<Button-1>", self.on_transcript_click)
        self.transcript_box.bind("<Motion>", self.on_transcript_hover)
        
        # ===== STATUS BAR =====
        status_frame = ctk.CTkFrame(self, height=40)
        status_frame.pack(fill="x", padx=10, pady=(5, 10))
        status_frame.pack_propagate(False)
        
        self.lbl_status = ctk.CTkLabel(status_frame, text="Ready", anchor="w")
        self.lbl_status.pack(side="left", padx=10, pady=5)
        
        self.progress_bar = ctk.CTkProgressBar(status_frame, width=300)
        self.progress_bar.pack(side="right", padx=10, pady=10)
        self.progress_bar.set(0)
        self.progress_bar.pack_forget()

        # Start scroll polling
        self.check_scroll_position()
        
    def open_file(self):
        """Open a video file."""
        file_path = filedialog.askopenfilename(
            title="Open Video",
            filetypes=[("Video Files", "*.mp4 *.mkv *.avi"), ("All Files", "*.*")]
        )
        if file_path:
            self.video_path = file_path
            self.lbl_status.configure(text=f"Loaded: {os.path.basename(file_path)}")
            
            try:
                self.video_player.load(file_path)
                self.btn_transcribe.configure(state="normal")
                log_debug(f"Video loaded: {file_path}")
            except Exception as e:
                log_debug(f"Error loading video: {e}")
                messagebox.showerror("Error", f"Could not load video: {e}")
            
    def start_transcription(self):
        """Start the transcription process in a background thread."""
        if not self.video_path or self.is_transcribing:
            return
            
        self.is_transcribing = True
        self.lbl_status.configure(text="Processing... (This may take a minute)")
        self.progress_bar.pack(side="right", padx=10, pady=10)
        self.progress_bar.set(0)
        self.btn_transcribe.configure(state="disabled")
        
        # Set loading message
        self.transcript_box.configure(state="normal")
        self.transcript_box.delete("1.0", "end")
        self.transcript_box.insert("1.0", "⏳ Transcribing... please wait.")
        self.transcript_box.configure(state="disabled")
        
        # Parse speaker count
        num_speakers = None
        try:
            val = self.entry_speakers.get().strip()
            if val:
                num_speakers = int(val)
        except ValueError:
            pass # Use auto

        # Start background thread
        self.transcription_thread = threading.Thread(
            target=self._transcription_worker,
            args=(self.video_path, num_speakers),
            daemon=True
        )
        self.transcription_thread.start()
        
        # Start polling for results
        self.after(100, self.poll_transcription)
        
    def _transcription_worker(self, video_path, num_speakers=None):
        """Background worker for transcription."""
        try:
            # Lazy Import Backend (Optimizes Startup Time)
            from transcribe import VideoTranscriber
            
            # Create transcriber and store reference to prevent GC during GUI session
            # CRITICAL: If transcriber is garbage collected while CUDA resources exist,
            # it causes a crash in the tkinter mainloop
            transcriber = VideoTranscriber(model_size="medium", use_cuda=True)
            
            def progress_callback(percent):
                self.transcription_queue.put(("progress", percent))
                
            results = transcriber.process_video(video_path, num_speakers=num_speakers, progress_callback=progress_callback)
            
            # Pass transcriber reference along with results to keep it alive
            self.transcription_queue.put(("finished", (results, transcriber)))
            
        except Exception as e:
            self.transcription_queue.put(("error", str(e)))
            
    def poll_transcription(self):
        """Poll the transcription queue for results."""
        try:
            while True:
                try:
                    msg_type, data = self.transcription_queue.get_nowait()
                    
                    if msg_type == "progress":
                        self.progress_bar.set(data / 100.0)
                        
                    elif msg_type == "finished":
                        self.on_transcription_finished(data)
                        return
                        
                    elif msg_type == "error":
                        self.on_transcription_error(data)
                        return
                        
                except queue.Empty:
                    break
                    
        except Exception as e:
            log_debug(f"Poll error: {e}")
            
        if self.is_transcribing:
            self.after(100, self.poll_transcription)
            
    def on_transcription_finished(self, data):
        """Handle transcription completion."""
        try:
            # Unpack results and transcriber reference
            # CRITICAL: Keep transcriber reference to prevent CUDA cleanup crash
            results, transcriber = data
            self._transcriber = transcriber  # Store to keep alive
            
            print("Processing finished.")
            flush_log()
            
            self.transcript_data = results
            self.is_transcribing = False
            
            # Update UI
            self.progress_bar.pack_forget()
            self.lbl_status.configure(text="Done")
            self.btn_transcribe.configure(state="normal")
            
            # Clean up thread reference
            self.transcription_thread = None
            
            # Render transcript
            self.render_transcript()
            
            print("[CALLBACK] Transcription completed successfully!")
            flush_log()
            
        except Exception as e:
            print(f"CRITICAL ERROR in callback: {e}")
            import traceback
            traceback.print_exc()
            flush_log()
            self.lbl_status.configure(text=f"UI Error: {e}")
            
    def on_transcription_error(self, err_msg):
        """Handle transcription error."""
        self.is_transcribing = False
        self.progress_bar.pack_forget()
        self.lbl_status.configure(text=f"Error: {err_msg}")
        self.btn_transcribe.configure(state="normal")
        
        self.transcript_box.configure(state="normal")
        self.transcript_box.delete("1.0", "end")
        self.transcript_box.insert("1.0", f"❌ Error: {err_msg}")
        self.transcript_box.configure(state="disabled")
        
    def render_transcript(self):
        """Render the transcript data to the text box."""
        print("[RENDER] render_transcript START")
        flush_log()
        
        try:
            # Capture current state to restore after refresh
            saved_yview = self.transcript_box.yview()
            saved_index = self.current_render_index
            
            self.transcript_box.configure(state="normal")
            self.transcript_box.delete("1.0", "end")
            self.current_render_index = 0
            
            # Configure text tags for styling
            # Timestamps styled to look clickable (cyan, underlined)
            self.transcript_box.tag_config("timestamp", foreground="#4fc3f7", underline=True)
            for i, color in enumerate(SPEAKER_COLORS):
                self.transcript_box.tag_config(f"speaker_{i}", foreground=color)
            self.transcript_box.tag_config("text", foreground="#ffffff")
            
            if len(self.transcript_data) == 0:
                self.transcript_box.insert("1.0", "No transcript data available.")
                self.transcript_box.configure(state="disabled")
                return
            
            # Render up to what was previously rendered (or at least one batch)
            # We temporarily adjust batch_size to render everything in one go for efficiency
            original_batch_size = self.batch_size
            target_amount = max(saved_index, original_batch_size)
            self.batch_size = target_amount
            
            self.append_batch()
            
            # Restore original batch size
            self.batch_size = original_batch_size
            
            self.transcript_box.configure(state="disabled")
            
            # Restore scroll position
            if saved_yview:
                self.transcript_box.yview_moveto(saved_yview[0])
            
            print("[RENDER] render_transcript END")
            flush_log()
            
        except Exception as e:
            print(f"[RENDER] EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            flush_log()
            
    def append_batch(self):
        """Append the next batch of transcript items."""
        if self.current_render_index >= len(self.transcript_data):
            return
            
        start = self.current_render_index
        end = min(start + self.batch_size, len(self.transcript_data))
        
        self.transcript_box.configure(state="normal")
        
        for item in self.transcript_data[start:end]:
            timestamp_str = self.format_time(item['start'])
            raw_speaker = item.get('speaker', 'Unknown')
            display_name = self.speaker_names.get(raw_speaker, raw_speaker)
            text_content = item.get('text', '')
            
            # Get speaker color index
            try:
                speaker_idx = int(raw_speaker.split(" ")[-1]) % len(SPEAKER_COLORS)
            except:
                speaker_idx = 0
                
            start_ms = int(item['start'] * 1000)
            
            # Insert timestamp (clickable - styled to look interactive)
            ts_tag = f"ts_{start_ms}"
            self.transcript_box.tag_config(ts_tag, foreground="#4fc3f7", underline=True)
            self.transcript_box.insert("end", f"[{timestamp_str}] ", ("timestamp", ts_tag))
            
            # Insert speaker name
            self.transcript_box.insert("end", f"{display_name}: ", f"speaker_{speaker_idx}")
            
            # Insert text
            self.transcript_box.insert("end", f"{text_content}\n\n", "text")
            
        self.current_render_index = end
        self.transcript_box.configure(state="disabled")
        
        print(f"[BATCH] Rendered items {start} to {end}")
        flush_log()
        
    def check_scroll_position(self):
        """Periodically check scroll position for infinite loading (handles scrollbar dragging)."""
        try:
            # If the widget exists and is mapped
            if self.transcript_box.winfo_exists():
                bottom_visible = self.transcript_box.yview()[1]
                # If we are near the bottom and have more data to render
                if bottom_visible > 0.95 and self.current_render_index < len(self.transcript_data):
                    self.append_batch()
                
                # Check again in 200ms
                self.after(200, self.check_scroll_position)
        except Exception:
            pass

    def on_transcript_scroll(self, event):
        """Handle scroll event for infinite loading and disable following mode."""
        try:
            # Disable following mode when user manually scrolls
            if self.following_mode:
                self.following_mode = False
                self.btn_follow.configure(fg_color="#424242")
                self._clear_following_highlight()
            
            bottom_visible = self.transcript_box.yview()[1]
            if bottom_visible > 0.9 and self.current_render_index < len(self.transcript_data):
                self.append_batch()
        except:
            pass
            
    def on_transcript_hover(self, event):
        """Handle hover over transcript to show clickable cursor and highlight timestamps."""
        try:
            index = self.transcript_box.index(f"@{event.x},{event.y}")
            tags = self.transcript_box.tag_names(index)
            
            # Find if hovering over a timestamp tag
            hovered_ts_tag = None
            for tag in tags:
                if tag.startswith("ts_"):
                    hovered_ts_tag = tag
                    break
            
            # Reset previously hovered tag if different
            if hasattr(self, '_last_hovered_tag') and self._last_hovered_tag:
                if self._last_hovered_tag != hovered_ts_tag:
                    # Reset to normal color
                    self.transcript_box.tag_config(self._last_hovered_tag, foreground="#4fc3f7")
            
            if hovered_ts_tag:
                # Highlight hovered timestamp with brighter color
                self.transcript_box.tag_config(hovered_ts_tag, foreground="#ffeb3b")
                self.transcript_box.configure(cursor="hand2")
                self._last_hovered_tag = hovered_ts_tag
            else:
                self.transcript_box.configure(cursor="arrow")
                self._last_hovered_tag = None
        except:
            pass
            
    def on_transcript_click(self, event):
        """Handle click on transcript to seek video."""
        try:
            index = self.transcript_box.index(f"@{event.x},{event.y}")
            tags = self.transcript_box.tag_names(index)
            
            for tag in tags:
                if tag.startswith("ts_"):
                    timestamp_ms = int(tag[3:])
                    self.video_player.seek(timestamp_ms / 1000)
                    log_debug(f"Seeking to {timestamp_ms}ms")
                    break
        except Exception as e:
            log_debug(f"Click handling error: {e}")
    
    def toggle_following_mode(self):
        """Toggle the following mode on/off."""
        self.following_mode = not self.following_mode
        
        if self.following_mode:
            # Enable - darker/highlighted button
            self.btn_follow.configure(fg_color="#1976d2", hover_color="#1565c0")
            # Start the following highlight loop
            self._update_following_highlight()
        else:
            # Disable - normal button
            self.btn_follow.configure(fg_color="#424242", hover_color="#616161")
            self._clear_following_highlight()
    
    def _update_following_highlight(self):
        """Update transcript highlight based on current video position."""
        if not self.following_mode:
            return
            
        try:
            # Get current video position in seconds
            if not self.video_player.cap:
                self.after(200, self._update_following_highlight)
                return
                
            current_time = self.video_player.current_frame / self.video_player.fps if self.video_player.fps > 0 else 0
            
            # Find the transcript segment that matches current time
            current_index = -1
            for i, item in enumerate(self.transcript_data):
                if item['start'] <= current_time < item.get('end', item['start'] + 10):
                    current_index = i
                    break
            
            # Only update if we moved to a different segment
            if current_index != self._last_highlighted_index and current_index >= 0:
                self._clear_following_highlight()
                self._last_highlighted_index = current_index
                
                # FORCE RENDER if segment hasn't been loaded yet
                if current_index >= self.current_render_index:
                    # We need to render up to (and including) this segment
                    segments_needed = current_index - self.current_render_index + 1
                    original_batch_size = self.batch_size
                    self.batch_size = segments_needed
                    self.append_batch()
                    self.batch_size = original_batch_size
                
                # Calculate the text position for this segment
                # We need to find the line in the textbox corresponding to this segment
                start_ms = int(self.transcript_data[current_index]['start'] * 1000)
                tag_name = f"ts_{start_ms}"
                
                # Configure highlight tag
                self.transcript_box.tag_config("following_highlight", background="#3a3a3a")
                
                # Find tag range and highlight the whole line
                try:
                    tag_ranges = self.transcript_box.tag_ranges(tag_name)
                    if tag_ranges:
                        # Get the line containing this timestamp
                        line_start = self.transcript_box.index(f"{tag_ranges[0]} linestart")
                        line_end = self.transcript_box.index(f"{tag_ranges[1]} lineend +1 line")
                        
                        self.transcript_box.tag_add("following_highlight", line_start, line_end)
                        
                        # Auto-scroll to CENTER the highlighted text
                        # Get the line number of the highlighted text
                        line_num = int(line_start.split('.')[0])
                        
                        # Get total number of lines in the widget
                        total_lines = int(self.transcript_box.index('end-1c').split('.')[0])
                        
                        # Calculate visible height as fraction of total
                        visible_fraction = self.transcript_box.yview()[1] - self.transcript_box.yview()[0]
                        
                        # Calculate scroll position to center the line
                        # We want the line to be in the middle of the visible area
                        line_fraction = line_num / max(total_lines, 1)
                        center_offset = visible_fraction / 2
                        scroll_position = max(0, line_fraction - center_offset)
                        
                        self.transcript_box.yview_moveto(scroll_position)
                except Exception as e:
                    log_debug(f"Highlight error: {e}")
                    pass
                    
        except Exception as e:
            log_debug(f"Following highlight error: {e}")
            
        # Continue polling
        if self.following_mode:
            self.after(200, self._update_following_highlight)
    
    def _clear_following_highlight(self):
        """Clear any existing following highlight."""
        try:
            self.transcript_box.tag_remove("following_highlight", "1.0", "end")
            self._last_highlighted_index = -1
        except:
            pass
            
            
    def rename_speaker_dialog(self):
        """Show dialog to rename a speaker."""
        if not self.transcript_data:
            return
            
        # Get raw speaker IDs from transcript data
        raw_speakers = sorted(list(set(x.get('speaker', 'Unknown') for x in self.transcript_data)))
        if not raw_speakers:
            return
            
        # Build display list mapping names back to raw IDs
        display_map = {}
        for raw in raw_speakers:
            # Get current display name if renamed, else raw ID
            display_name = self.speaker_names.get(raw, raw)
            
            # Handle collisions (e.g. if multiple speakers were renamed to 'Bob')
            if display_name in display_map:
                unique_name = f"{display_name} ({raw})"
                display_map[unique_name] = raw
            else:
                display_map[display_name] = raw
        
        # Sort by display name for the UI
        display_options = sorted(list(display_map.keys()))
            
        def on_rename(selected_option, new_name):
            # Map back to raw ID
            raw_speaker = display_map.get(selected_option)
            if raw_speaker:
                self.speaker_names[raw_speaker] = new_name
                self.render_transcript()
            
        SpeakerRenameDialog(self, display_options, on_rename)
        
    def export_pdf(self):
        """Export transcript to PDF."""
        if not self.transcript_data:
            messagebox.showwarning("Export", "No transcript data to export.")
            return

        # Lazy Import Export Utils
        from export_utils import export_to_pdf
            
        file_path = filedialog.asksaveasfilename(
            title="Save PDF",
            defaultextension=".pdf",
            filetypes=[("PDF Files", "*.pdf")],
            initialfile="transcript.pdf"
        )
        
        if file_path:
            try:
                export_to_pdf(file_path, self.transcript_data, self.speaker_names)
                self.lbl_status.configure(text=f"Saved to {file_path}")
                log_debug(f"PDF exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Could not save PDF: {e}")
                
    def format_time(self, seconds):
        """Format seconds as HH:MM:SS or MM:SS."""
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}" if h > 0 else f"{int(m):02d}:{int(s):02d}"
        
    def on_closing(self):
        """Handle window close - clean up resources properly."""
        try:
            log_debug("Application closing, cleaning up resources...")
            
            # Stop video playback
            if hasattr(self, 'video_player'):
                self.video_player.stop()
                if self.video_player.cap:
                    self.video_player.cap.release()
            
            # Stop audio explicitly
            try:
                if pygame.mixer.get_init():
                    pygame.mixer.music.stop()
                    pygame.mixer.quit()
            except:
                pass
                    
            # Wait for transcription thread if running
            self.is_transcribing = False
            
            # Helper to delete temp audio file
            if hasattr(self.video_player, '_cleanup_audio'):
                try:
                    self.video_player._cleanup_audio()
                except:
                    pass
            
            flush_log()
        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            self.destroy()
            # Force immediate exit to prevent hanging on threads
            import os
            os._exit(0)


if __name__ == "__main__":
    try:
        # Force garbage collection before starting GUI
        import gc
        gc.collect()
        
        # Hide console window
        try:
            import ctypes
            # SW_HIDE = 0
            ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)
        except Exception:
            pass
        
        app = TranscriptionApp()
        app.mainloop()
    except Exception as e:
        print(f"Application error: {e}")
        import traceback
        traceback.print_exc()
        flush_log()
    finally:
        # Ensure log files are flushed
        try:
            flush_log()
            log_file.close()
            crash_log_file.close()
        except:
            pass
