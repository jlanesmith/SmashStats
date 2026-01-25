#!/usr/bin/env python3
"""
SmashStats Live - Real-time analysis of Super Smash Bros video streams.
Uses a producer-consumer pattern with a frame buffer to handle 60fps input.
"""

import cv2
import threading
import time
import os
import argparse
import subprocess
import platform
from collections import deque
from pathlib import Path
from datetime import datetime

from find_frames import (
    load_and_scale_templates,
    FrameProcessor,
    State,
    DEFAULT_THRESHOLD
)
from analyze_game import (
    analyze_game_dir,
    print_game_result,
    save_result_to_db
)


class FrameBuffer:
    """Thread-safe circular buffer for frames with frame numbers."""

    def __init__(self, max_size=900):  # 15 seconds at 60fps
        self.max_size = max_size
        self.frames = {}  # Dict for O(1) lookup by frame number
        self.frame_order = deque(maxlen=max_size)  # Track order for cleanup
        self.lock = threading.Lock()
        self.frame_count = 0

    def put(self, frame):
        """Add a frame to the buffer."""
        with self.lock:
            frame_num = self.frame_count

            # If buffer is full, remove oldest frame
            if len(self.frame_order) >= self.max_size:
                oldest = self.frame_order[0]
                if oldest in self.frames:
                    del self.frames[oldest]

            # Add new frame
            self.frames[frame_num] = {
                'frame': frame.copy(),
                'timestamp': time.time()
            }
            self.frame_order.append(frame_num)
            self.frame_count += 1

    def get_frame_at(self, frame_num):
        """Get a specific frame by number, or None if not in buffer. O(1) lookup."""
        with self.lock:
            if frame_num in self.frames:
                return self.frames[frame_num]['frame']
            return None

    def get_latest(self):
        """Get the most recent frame and its number."""
        with self.lock:
            if self.frame_order:
                frame_num = self.frame_order[-1]
                return frame_num, self.frames[frame_num]['frame']
            return -1, None

    def pop_oldest(self):
        """Remove and return the oldest frame. Returns (frame_num, frame) or (-1, None)."""
        with self.lock:
            if self.frame_order:
                frame_num = self.frame_order.popleft()
                frame_data = self.frames.pop(frame_num, None)
                if frame_data:
                    return frame_num, frame_data['frame']
            return -1, None

    def get_current_frame_num(self):
        """Get the current frame count."""
        with self.lock:
            return self.frame_count - 1 if self.frame_count > 0 else -1

    def get_buffer_size(self):
        """Get current buffer occupancy."""
        with self.lock:
            return len(self.frames)

    def clear(self):
        """Clear all frames from the buffer."""
        with self.lock:
            self.frames.clear()
            self.frame_order.clear()


class LiveAnalyzer:
    """Real-time video stream analyzer for Super Smash Bros."""

    def __init__(self, template_dir, output_dir="live_captures",
                 threshold=DEFAULT_THRESHOLD, buffer_size=900):
        self.template_dir = template_dir
        self.output_dir = output_dir
        self.threshold = threshold

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Templates and processor will be loaded when stream starts (need video resolution)
        self.templates = None
        self.processor = None

        # Frame buffer
        self.frame_buffer = FrameBuffer(max_size=buffer_size)

        # Threading control
        self.running = False
        self.capture_thread = None
        self.process_thread = None

        # Video properties (set when stream starts)
        self.fps = 60
        self.target_height = 480
        self.target_width = 854
        self.debug_frames_dir = None

        # Performance tracking
        self.frames_processed = 0
        self.processing_times = deque(maxlen=100)

    def print_status(self):
        """Print current status for debugging."""
        if self.processor is None:
            print("[STATUS] Processor not initialized")
            return

        # Determine current stage description
        state = self.processor.state
        if state == State.LOOKING_FOR_CHARACTERS:
            stage = "Searching for CHARACTER SELECT screen"
        elif state == State.LOOKING_FOR_GAME:
            stage = "Searching for GAME END screen"
        elif state == State.WAITING_FOR_RESULTS:
            stage = "Waiting for results screen"
        elif state == State.LOOKING_FOR_RESULTS:
            found = [p.upper() for p, v in self.processor.found_players.items() if v]
            missing = [p.upper() for p, v in self.processor.found_players.items() if not v]
            stage = f"Searching for PLAYER RESULTS (found: {found}, missing: {missing})"
        else:
            stage = f"Unknown state: {state}"

        # Buffer info
        buffer_used = self.frame_buffer.get_buffer_size()
        buffer_max = self.frame_buffer.max_size
        buffer_pct = (buffer_used / buffer_max) * 100 if buffer_max > 0 else 0

        # Performance info
        avg_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0

        print(f"[STATUS] Game {self.processor.game_num} | {stage}")
        print(f"         Buffer: {buffer_used}/{buffer_max} ({buffer_pct:.1f}%) | "
              f"Processed: {self.frames_processed} frames | Avg: {avg_time:.1f}ms/frame")

    def capture_frames(self, source):
        """Capture thread - continuously reads frames into buffer."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Could not open video source: {source}")
            self.running = False
            return

        # Get video properties
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 60
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.target_width = int(video_width * (self.target_height / video_height))

        # Try to force 60fps for capture cards (may not work on all devices)
        cap.set(cv2.CAP_PROP_FPS, 60)

        # If reported FPS seems wrong, assume 60fps
        if self.fps < 30:
            print(f"Warning: Detected FPS ({self.fps}) seems low, assuming 60fps")
            self.fps = 60

        # Load and scale templates
        self.templates, _ = load_and_scale_templates(
            self.template_dir, self.target_height, debug_output_dir=self.output_dir
        )

        # Save scaled templates for debugging
        debug_templates_dir = os.path.join(self.output_dir, "debug_templates")
        os.makedirs(debug_templates_dir, exist_ok=True)
        print(f"\nSaving scaled templates to: {debug_templates_dir}")
        for name, t in self.templates.items():
            template_img = t['img_scaled']
            h, w = template_img.shape[:2] if len(template_img.shape) == 2 else template_img.shape[:2]
            cv2.imwrite(os.path.join(debug_templates_dir, f"{name}_template_scaled.png"), template_img)
            if t['mask_scaled'] is not None:
                cv2.imwrite(os.path.join(debug_templates_dir, f"{name}_mask_scaled.png"), t['mask_scaled'])
            print(f"  {name}: {w}x{h}")

        # Create debug frames directory
        self.debug_frames_dir = os.path.join(self.output_dir, "debug_frames")
        os.makedirs(self.debug_frames_dir, exist_ok=True)
        print(f"Debug frames will be saved to: {self.debug_frames_dir}")

        # Create frame processor with shared state machine logic
        # Use shorter wait for live mode (12 frames at ~2.4fps capture rate = ~5 seconds)
        self.processor = FrameProcessor(
            templates=self.templates,
            threshold=self.threshold,
            fps=self.fps,
            output_dir=self.output_dir,
            debug=False,
            on_game_complete=self._on_game_complete,
            verbose=True
        )
        # Override wait time for live mode (fewer frames due to skip rate)
        self.processor.WAIT_AFTER_GAME = 12

        print(f"\nStream opened: {video_width}x{video_height} @ {self.fps:.1f}fps (detected)")
        print(f"Processing at: {self.target_width}x{self.target_height}")
        print(f"Capture rate: 1/25 frames for characters/game, 1/2 frames for player results")
        print(f"Buffer size: {self.frame_buffer.max_size} frames max")

        frame_interval = 1.0 / self.fps if self.fps > 0 else 1.0 / 60
        last_frame_time = time.time()
        frame_count = 0
        is_live_source = isinstance(source, int)  # Device index = live source

        # Measure actual FPS
        fps_measure_start = time.time()
        fps_measure_frames = 0

        while self.running:
            ret, frame = cap.read()
            if not ret:
                # For live streams, retry on failure
                print("Warning: Failed to read frame, retrying...")
                time.sleep(0.01)
                continue

            # Dynamic skip based on processor state
            if self.processor.state == State.LOOKING_FOR_RESULTS:
                capture_skip = 2  # Check every 2 frames for player results
            else:
                capture_skip = 25  # Check every 25 frames for characters/game

            # Only add frames at the appropriate skip rate
            if frame_count % capture_skip == 0:
                self.frame_buffer.put(frame)
            frame_count += 1

            # Measure actual FPS (report every 5 seconds)
            fps_measure_frames += 1
            elapsed = time.time() - fps_measure_start
            if elapsed >= 5.0:
                actual_fps = fps_measure_frames / elapsed
                print(f"[CAPTURE] Actual capture rate: {actual_fps:.1f} fps")
                fps_measure_start = time.time()
                fps_measure_frames = 0

            # For file playback only, maintain proper frame rate
            # Live sources (capture cards) should not sleep
            if not is_live_source:
                current_time = time.time()
                elapsed = current_time - last_frame_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                last_frame_time = time.time()

        cap.release()
        print("Capture thread stopped")

    def _on_game_complete(self, game_dir, game_num):
        """Callback when a game is complete - analyze and save results."""
        self.analyze_and_save_game(game_dir, game_num)

    def process_frames(self):
        """Processing thread - analyzes frames using shared FrameProcessor."""
        last_status_time = time.time()
        STATUS_INTERVAL = 5.0  # Print status every 5 seconds

        print("\nProcessing started...")

        while self.running:
            buffer_size = self.frame_buffer.get_buffer_size()

            # Print status every 5 seconds
            current_time = time.time()
            if current_time - last_status_time >= STATUS_INTERVAL:
                self.print_status()
                print(f"         Buffer size: {buffer_size}")
                last_status_time = current_time

            # Wait for processor to be initialized
            if self.processor is None:
                time.sleep(0.01)
                continue

            # Check if there's a frame to process
            if buffer_size == 0:
                time.sleep(0.001)
                continue

            # Pop the oldest frame from buffer (removes it)
            frame_num, frame = self.frame_buffer.pop_oldest()
            if frame is None:
                continue

            # Track processing time
            start_time = time.perf_counter()

            # Downscale frame for processing
            frame_small = cv2.resize(frame, (self.target_width, self.target_height),
                                     interpolation=cv2.INTER_AREA)

            # Save debug frame with resolution info and timestamp
            if self.debug_frames_dir:
                state_name = self.processor.state.name if self.processor else "INIT"
                h, w = frame_small.shape[:2]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
                debug_filename = os.path.join(self.debug_frames_dir, f"frame_{timestamp}_{frame_num:06d}_{state_name}_{w}x{h}.png")
                cv2.imwrite(debug_filename, frame_small)

            # Process frame through shared state machine
            result = self.processor.process_frame(frame, frame_small, frame_num)

            # Track performance
            process_time = (time.perf_counter() - start_time) * 1000
            self.processing_times.append(process_time)
            self.frames_processed += 1

        print("Processing thread stopped")

    def analyze_and_save_game(self, game_dir, game_num):
        """Analyze the captured game and save results to database."""
        print(f"\nAnalyzing Game {game_num}...")

        # Use shared analysis function
        result = analyze_game_dir(game_dir, game_num=game_num, verbose=False)

        if result:
            # Print results using shared function
            print(f"\n{'='*50}")
            print(f"Game {game_num} Results")
            print(f"{'='*50}")
            print_game_result(result, game_num=game_num)

            # Save to database
            game_id = save_result_to_db(result)
            print(f"\nResults saved to database (game_id: {game_id})")

            # Print performance stats
            if self.processing_times:
                avg_time = sum(self.processing_times) / len(self.processing_times)
                max_time = max(self.processing_times)
                print(f"\n  Processing stats: avg {avg_time:.1f}ms, max {max_time:.1f}ms per frame")
                print(f"  Buffer usage: {self.frame_buffer.get_buffer_size()}/{self.frame_buffer.max_size}")

    def start(self, source):
        """Start capturing and processing from the given source."""
        self.running = True

        # Start capture thread
        self.capture_thread = threading.Thread(target=self.capture_frames, args=(source,), daemon=True)
        self.capture_thread.start()

        # Wait for buffer to start filling
        print("Waiting for stream...")
        timeout = 10
        start = time.time()
        while self.frame_buffer.get_current_frame_num() < 0 and self.running:
            if time.time() - start > timeout:
                print("Timeout waiting for frames")
                self.running = False
                return
            time.sleep(0.1)

        # Start processing thread
        self.process_thread = threading.Thread(target=self.process_frames, daemon=True)
        self.process_thread.start()

        print(f"\nLive analysis running. Press Ctrl+C to stop.\n")

    def stop(self):
        """Stop the analyzer gracefully."""
        print("\nStopping...")
        self.running = False

        # Finalize any pending player matches using shared logic
        if self.processor:
            self.processor.finalize()

        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        if self.process_thread:
            self.process_thread.join(timeout=2)

        print("Stopped.")
        print(f"Total frames processed: {self.frames_processed}")
        if self.processor:
            print(f"Total games analyzed: {self.processor.game_num}")

    def run(self, source):
        """Run the analyzer until interrupted."""
        try:
            self.start(source)
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Real-time Super Smash Bros video stream analyzer"
    )
    parser.add_argument(
        "source",
        help="Video source: file path, stream URL, or device index (0 for webcam/capture card)"
    )
    parser.add_argument(
        "-t", "--templates",
        default="templates",
        help="Directory containing template images (default: templates)"
    )
    parser.add_argument(
        "-o", "--output",
        default="live_captures",
        help="Output directory for captured frames (default: live_captures)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Match threshold 0-1, higher = stricter (default: {DEFAULT_THRESHOLD})"
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=900,
        help="Frame buffer size (default: 900, ~15s at 60fps)"
    )

    args = parser.parse_args()

    # Handle device index (e.g., "0" for first capture device)
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    # Resolve template path
    script_dir = Path(__file__).parent
    template_dir = args.templates
    if not os.path.isabs(template_dir):
        template_dir = script_dir / template_dir

    # Start caffeinate on macOS to keep capture card active when display sleeps
    caffeinate_proc = None
    if platform.system() == "Darwin":
        try:
            caffeinate_proc = subprocess.Popen(
                ["caffeinate", "-s"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print("Started caffeinate to keep capture card active during display sleep")
        except Exception as e:
            print(f"Warning: Could not start caffeinate: {e}")

    print("=" * 60)
    print("SMASH STATS LIVE")
    print("=" * 60)

    try:
        analyzer = LiveAnalyzer(
            template_dir=str(template_dir),
            output_dir=args.output,
            threshold=args.threshold,
            buffer_size=args.buffer_size
        )

        analyzer.run(source)
    finally:
        # Clean up caffeinate process
        if caffeinate_proc:
            caffeinate_proc.terminate()
            caffeinate_proc.wait()


if __name__ == "__main__":
    main()
