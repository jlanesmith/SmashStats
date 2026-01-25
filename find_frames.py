#!/usr/bin/env python3
"""
Find key frames in Super Smash Bros video recordings.
Detects character select, game end, and result screens using template matching.
Supports multiple games in a single video.
"""

import cv2
import numpy as np
import argparse
import os
import csv
import time
from pathlib import Path
from enum import Enum
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor


# Default template matching threshold (0-1, higher = stricter matching)
DEFAULT_THRESHOLD = 0.93


class State(Enum):
    LOOKING_FOR_CHARACTERS = 1
    LOOKING_FOR_GAME = 2
    WAITING_FOR_RESULTS = 3
    LOOKING_FOR_RESULTS = 4


class FrameProcessor:
    """
    Encapsulates the state machine and frame analysis logic for detecting
    Smash Bros game events. Can be used for both recorded video and live streams.
    """

    def __init__(self, templates, threshold=DEFAULT_THRESHOLD, fps=60, output_dir=None,
                 debug=False, on_game_complete=None, verbose=False):
        """
        Initialize the frame processor.

        Args:
            templates: Dictionary of loaded and scaled templates
            threshold: Matching threshold for templates (0-1)
            fps: Frames per second (used for timestamp calculation)
            output_dir: Directory to save captured frames (optional)
            debug: If True, track debug information
            on_game_complete: Callback function(game_dir, game_num) called when a game is complete
            verbose: If True, print frame-by-frame logs (for live mode)
        """
        self.templates = templates
        self.threshold = threshold
        self.fps = fps
        self.output_dir = output_dir
        self.debug = debug
        self.on_game_complete = on_game_complete
        self.verbose = verbose

        # State machine
        self.state = State.LOOKING_FOR_CHARACTERS
        self.found_players = {'p1': False, 'p2': False, 'p3': False, 'p4': False}

        # Player result tracking
        self.player_groups = {'p1': [], 'p2': [], 'p3': [], 'p4': []}
        self.active_group = {'p1': None, 'p2': None, 'p3': None, 'p4': None}
        self.player_done = {'p1': False, 'p2': False, 'p3': False, 'p4': False}
        self.p1_p2_wait_counter = None  # Counts frames after p1+p2 found, waiting for p3+p4

        # Game tracking
        self.game_num = 0
        self.game_directories = []
        self.current_game_dir = None
        self.wait_until_frame = 0
        self.game_result = None  # 'win' or 'loss'

        # Constants
        self.WAIT_AFTER_GAME = 300  # Frames to wait after game end before looking for results
        self.RESULTS_THRESHOLD = 0.80
        self.WIN_LOSS_THRESHOLD = DEFAULT_THRESHOLD
        self.MAX_DROP = 0.01
        self.MIN_CONSECUTIVE_FRAMES = 10
        self.P1_P2_EXTRA_WAIT = 10  # Extra frames to wait for p3/p4 after p1+p2 found

        # Debug tracking
        self.best_matches = {
            'characters': {'confidence': 0, 'frame_num': 0, 'frame': None, 'loc': None},
            'game': {'confidence': 0, 'frame_num': 0, 'frame': None, 'loc': None},
            'p1': {'confidence': 0, 'frame_num': 0, 'frame': None, 'loc': None},
            'p2': {'confidence': 0, 'frame_num': 0, 'frame': None, 'loc': None},
            'p3': {'confidence': 0, 'frame_num': 0, 'frame': None, 'loc': None},
            'p4': {'confidence': 0, 'frame_num': 0, 'frame': None, 'loc': None},
            'win': {'confidence': 0, 'frame_num': 0, 'frame': None, 'loc': None},
            'loss': {'confidence': 0, 'frame_num': 0, 'frame': None, 'loc': None},
        }
        self.debug_log = []

    def get_frame_skip(self):
        """Return the recommended frame skip based on current state."""
        if self.state == State.LOOKING_FOR_CHARACTERS:
            return 5
        elif self.state == State.LOOKING_FOR_GAME:
            return 25
        elif self.state == State.WAITING_FOR_RESULTS:
            return 1
        else:  # LOOKING_FOR_RESULTS
            return 1

    def _save_frame(self, frame, frame_small, name, frame_num):
        """Save both 480p and full resolution versions of a frame."""
        if self.current_game_dir is None:
            return None

        timestamp = frame_num / self.fps
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)

        base_filename = f"{name}_{frame_num:06d}_{minutes}m{seconds:02d}s.png"

        # Save low-res (480p) version
        path = os.path.join(self.current_game_dir, base_filename)
        cv2.imwrite(path, frame_small)

        # Save full resolution version if frame is different from frame_small
        if frame is not None and frame.shape != frame_small.shape:
            full_res_filename = f"{name}_{frame_num:06d}_{minutes}m{seconds:02d}s_full.png"
            full_res_path = os.path.join(self.current_game_dir, full_res_filename)
            cv2.imwrite(full_res_path, frame)
            print(f"  Saved: {base_filename} + {full_res_filename}")
        else:
            print(f"  Saved: {base_filename}")

        return path

    def _reset_for_new_game(self):
        """Reset state for finding the next game."""
        self.state = State.LOOKING_FOR_CHARACTERS
        self.best_matches['characters'] = {'confidence': 0, 'frame_num': 0, 'frame': None, 'loc': None}
        self.best_matches['win'] = {'confidence': 0, 'frame_num': 0, 'frame': None, 'loc': None}
        self.best_matches['loss'] = {'confidence': 0, 'frame_num': 0, 'frame': None, 'loc': None}
        self.game_result = None
        print(f"\nLooking for next game...")

    def _complete_game(self):
        """Complete the current game - save results and trigger callback."""
        # Save game result (win/loss) to file
        if self.game_result and self.current_game_dir:
            result_file = os.path.join(self.current_game_dir, "game_result.txt")
            with open(result_file, 'w') as f:
                f.write(self.game_result)
            print(f"Game result: {self.game_result.upper()}")

        # Add this game directory to our list
        if self.current_game_dir:
            self.game_directories.append(self.current_game_dir)
        print(f"Game {self.game_num} frames captured!")

        # Call the callback if provided
        if self.on_game_complete and self.current_game_dir:
            self.on_game_complete(self.current_game_dir, self.game_num)

        self._reset_for_new_game()

    def _select_best_frame_for_players(self, frame=None, frame_small=None):
        """Select and save the best frame for each player from their groups."""
        for player in ['p1', 'p2', 'p3', 'p4']:
            if not self.player_groups[player]:
                continue

            # Find group with highest peak confidence
            best_group = max(self.player_groups[player], key=lambda g: g['peak_confidence'])
            frames = best_group['frames']

            # Pick 3rd last frame (or first if <=2)
            if len(frames) <= 2:
                idx = 0
            else:
                idx = len(frames) - 3
            selected = frames[idx]

            print(f"{player.upper()} result found! Frame {selected['frame_num']}, "
                  f"Confidence: {selected['confidence']:.4f}, Peak: {best_group['peak_confidence']:.4f} "
                  f"(frame {idx+1} of {len(frames)}, best of {len(self.player_groups[player])} groups)")
            self._save_frame(selected.get('frame'), selected['frame_small'], player, selected['frame_num'])
            self.found_players[player] = True

    def process_frame(self, frame, frame_small, frame_num):
        """
        Process a single frame through the state machine.

        Args:
            frame: Full resolution frame (BGR), can be None for live mode
            frame_small: Downscaled frame (BGR) at processing resolution
            frame_num: The frame number

        Returns:
            dict: Event information including:
                - 'state': Current state after processing
                - 'event': Event type if any ('characters_found', 'game_end', 'results_complete', None)
                - 'game_num': Current game number
                - 'game_dir': Current game directory
        """
        frame_small_gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        event = None

        # Handle waiting state
        if self.state == State.WAITING_FOR_RESULTS:
            if frame_num >= self.wait_until_frame:
                self.state = State.LOOKING_FOR_RESULTS
                # Reset for player results
                for p in ['p1', 'p2', 'p3', 'p4']:
                    self.best_matches[p] = {'confidence': 0, 'frame_num': 0, 'frame': None, 'loc': None}
                    self.player_groups[p] = []
                    self.active_group[p] = None
                    self.player_done[p] = False
                print(f"Wait complete, now looking for player results...")
            return {'state': self.state, 'event': None, 'game_num': self.game_num, 'game_dir': self.current_game_dir}

        if self.state == State.LOOKING_FOR_CHARACTERS:
            t = self.templates['characters']
            matched, confidence, loc = match_template(
                frame_small_gray, t['img_scaled'], t['mask_scaled'], self.threshold
            )

            # Print confidence vs threshold for debugging
            match_indicator = "✓ MATCH" if matched else ""
            if self.verbose:
                print(f"  [CHAR] Frame {frame_num}: conf={confidence:.4f} thresh={self.threshold:.2f} {match_indicator}")

            # Track best match for debug
            if self.debug and confidence > self.best_matches['characters']['confidence']:
                self.best_matches['characters'] = {
                    'confidence': confidence,
                    'frame_num': frame_num,
                    'frame': frame_small.copy(),
                    'loc': loc
                }

            if self.debug:
                self.debug_log.append((frame_num, 'LOOKING_FOR_CHARACTERS', 'characters', confidence, matched))

            if matched:
                self.game_num += 1
                print(f"\n--- Game {self.game_num} ---")
                print(f"Characters found! Frame {frame_num}, Confidence: {confidence:.4f}")

                # Create new game directory
                if self.output_dir:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self.current_game_dir = os.path.join(self.output_dir, f"game_{self.game_num}_{timestamp}")
                    os.makedirs(self.current_game_dir, exist_ok=True)
                    self._save_frame(frame, frame_small, 'characters', frame_num)

                # Reset best match for game template
                self.best_matches['game'] = {'confidence': 0, 'frame_num': 0, 'frame': None, 'loc': None}

                self.state = State.LOOKING_FOR_GAME
                event = 'characters_found'
                print(f"Switching to game search...")

        elif self.state == State.LOOKING_FOR_GAME:
            t = self.templates['game']
            matched, confidence, loc = match_template(
                frame_small_gray, t['img_scaled'], t['mask_scaled'], self.threshold
            )

            # Print confidence vs threshold for debugging
            match_indicator = "✓ MATCH" if matched else ""
            if self.verbose:
                print(f"  [GAME] Frame {frame_num}: conf={confidence:.4f} thresh={self.threshold:.2f} {match_indicator}")

            # Track best match for debug
            if self.debug and confidence > self.best_matches['game']['confidence']:
                self.best_matches['game'] = {
                    'confidence': confidence,
                    'frame_num': frame_num,
                    'frame': frame_small.copy(),
                    'loc': loc
                }

            if self.debug:
                self.debug_log.append((frame_num, 'LOOKING_FOR_GAME', 'game', confidence, matched))

            if matched:
                print(f"Game end found! Frame {frame_num}, Confidence: {confidence:.4f}")
                self._save_frame(frame, frame_small, 'game', frame_num)

                # Wait before looking for results
                self.wait_until_frame = frame_num + self.WAIT_AFTER_GAME
                self.state = State.WAITING_FOR_RESULTS
                self.found_players = {'p1': False, 'p2': False, 'p3': False, 'p4': False}
                self.player_groups = {'p1': [], 'p2': [], 'p3': [], 'p4': []}
                self.active_group = {'p1': None, 'p2': None, 'p3': None, 'p4': None}
                self.player_done = {'p1': False, 'p2': False, 'p3': False, 'p4': False}
                self.p1_p2_wait_counter = None
                event = 'game_end'
                print(f"Waiting {self.WAIT_AFTER_GAME} frames before checking for results...")

        elif self.state == State.LOOKING_FOR_RESULTS:
            # Check for win/loss only after P2 has at least one frame over threshold
            p2_has_match = len(self.player_groups.get('p2', [])) > 0 or self.active_group.get('p2') is not None
            if self.game_result is None and p2_has_match:
                win_t = self.templates['win']
                loss_t = self.templates['loss']

                # Use color matching for win/loss templates
                win_matched, win_conf, win_loc = match_template_color(
                    frame_small, win_t['img_scaled'], win_t['mask_scaled'], self.WIN_LOSS_THRESHOLD
                )
                loss_matched, loss_conf, loss_loc = match_template_color(
                    frame_small, loss_t['img_scaled'], loss_t['mask_scaled'], self.WIN_LOSS_THRESHOLD
                )

                # Track best matches for debug
                if self.debug:
                    if win_conf > self.best_matches['win']['confidence']:
                        self.best_matches['win'] = {'confidence': win_conf, 'frame_num': frame_num, 'frame': frame_small.copy(), 'loc': win_loc}
                    if loss_conf > self.best_matches['loss']['confidence']:
                        self.best_matches['loss'] = {'confidence': loss_conf, 'frame_num': frame_num, 'frame': frame_small.copy(), 'loc': loss_loc}

                if win_matched:
                    self.game_result = 'win'
                    print(f"  [WIN DETECTED] Frame {frame_num}, Confidence: {win_conf:.4f} (loss conf: {loss_conf:.4f})")
                elif loss_matched:
                    self.game_result = 'loss'
                    print(f"  [LOSS DETECTED] Frame {frame_num}, Confidence: {loss_conf:.4f} (win conf: {win_conf:.4f})")

            # Check for all player results
            debug_parts = []

            # Determine which players still need matching
            players_to_match = [p for p in ['p1', 'p2', 'p3', 'p4'] if not self.player_done[p]]

            for player in ['p1', 'p2', 'p3', 'p4']:
                if self.player_done[player]:
                    debug_parts.append(f"{player}=DONE")
                    continue

                t = self.templates[player]
                matched, confidence, loc = match_template(
                    frame_small_gray, t['img_scaled'], t['mask_scaled'], self.RESULTS_THRESHOLD
                )

                # Track best match for debug
                if self.debug and confidence > self.best_matches[player]['confidence']:
                    self.best_matches[player] = {
                        'confidence': confidence,
                        'frame_num': frame_num,
                        'frame': frame_small.copy(),
                        'loc': loc
                    }

                if self.debug:
                    self.debug_log.append((frame_num, 'LOOKING_FOR_RESULTS', player, confidence, matched))

                if matched:
                    frame_data = {
                        'frame': frame,
                        'frame_small': frame_small.copy(),
                        'frame_num': frame_num,
                        'confidence': confidence
                    }

                    if self.active_group[player] is None:
                        # Start new group
                        self.active_group[player] = {
                            'frames': [frame_data],
                            'peak_confidence': confidence,
                            'last_confidence': confidence
                        }
                    else:
                        # Check if confidence dropped by more than MAX_DROP
                        if self.active_group[player]['last_confidence'] - confidence > self.MAX_DROP:
                            # End current group (incomplete), save it, start new one
                            self.player_groups[player].append(self.active_group[player])
                            self.active_group[player] = {
                                'frames': [frame_data],
                                'peak_confidence': confidence,
                                'last_confidence': confidence
                            }
                        else:
                            # Continue current group
                            self.active_group[player]['frames'].append(frame_data)
                            self.active_group[player]['last_confidence'] = confidence
                            if confidence > self.active_group[player]['peak_confidence']:
                                self.active_group[player]['peak_confidence'] = confidence

                            # Complete group if we have enough consecutive frames
                            if len(self.active_group[player]['frames']) >= self.MIN_CONSECUTIVE_FRAMES:
                                self.player_groups[player].append(self.active_group[player])
                                self.active_group[player] = None
                                self.player_done[player] = True
                                debug_parts.append(f"{player}={confidence:.3f}✓COMPLETE")
                                continue

                    num_groups = len(self.player_groups[player]) + (1 if self.active_group[player] else 0)
                    debug_parts.append(f"{player}={confidence:.3f}✓(g{num_groups})")
                else:
                    # Below threshold - end current group if active (incomplete)
                    if self.active_group[player] is not None:
                        self.player_groups[player].append(self.active_group[player])
                        self.active_group[player] = None
                    num_groups = len(self.player_groups[player])
                    debug_parts.append(f"{player}={confidence:.3f}(g{num_groups})")

            # Print debug info for all players
            if self.verbose:
                print(f"  [RESULTS] Frame {frame_num}: {' '.join(debug_parts)}")

            # Check completion conditions
            all_have_groups = all(len(self.player_groups[p]) > 0 for p in ['p1', 'p2', 'p3', 'p4'])
            p1_p2_have_groups = len(self.player_groups['p1']) > 0 and len(self.player_groups['p2']) > 0

            # Complete if all 4 players have groups
            if all_have_groups:
                self._select_best_frame_for_players()
                self._complete_game()
                event = 'results_complete'
            # Or if p1+p2 have groups and we've waited long enough for p3+p4
            elif p1_p2_have_groups:
                if self.p1_p2_wait_counter is None:
                    self.p1_p2_wait_counter = 0
                else:
                    self.p1_p2_wait_counter += 1

                if self.p1_p2_wait_counter >= self.P1_P2_EXTRA_WAIT:
                    if self.verbose:
                        print(f"  [RESULTS] Completing with p1+p2 only (waited {self.p1_p2_wait_counter} extra frames)")
                    self._select_best_frame_for_players()
                    self._complete_game()
                    event = 'results_complete'

        return {'state': self.state, 'event': event, 'game_num': self.game_num, 'game_dir': self.current_game_dir}

    def finalize(self):
        """
        Finalize any pending player matches (e.g., when video ends during result detection).

        Returns:
            bool: True if a game was finalized, False otherwise
        """
        if self.state == State.LOOKING_FOR_RESULTS and self.current_game_dir:
            # First, close any active groups
            for player in ['p1', 'p2', 'p3', 'p4']:
                if self.active_group[player] is not None:
                    self.player_groups[player].append(self.active_group[player])
                    self.active_group[player] = None

            # Check completion conditions
            all_have_groups = all(len(self.player_groups[p]) > 0 for p in ['p1', 'p2', 'p3', 'p4'])
            p1_p2_have_groups = len(self.player_groups['p1']) > 0 and len(self.player_groups['p2']) > 0

            if all_have_groups or p1_p2_have_groups:
                self._select_best_frame_for_players()
                self._complete_game()
                return True

        return False


def load_template_with_mask(template_path: str, use_color: bool = False):
    """
    Load template image and create mask from alpha channel if present.

    Args:
        template_path: Path to template image
        use_color: If True, keep BGR color channels; if False, convert to grayscale

    Returns:
        tuple: (template, mask) where template is BGR or grayscale, mask is None if no alpha
    """
    template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
    if template is None:
        raise FileNotFoundError(f"Could not load template: {template_path}")

    mask = None

    if len(template.shape) == 3 and template.shape[2] == 4:  # Has alpha channel
        alpha = template[:, :, 3]
        mask = (alpha > 127).astype(np.uint8) * 255
        template_bgr = template[:, :, :3]
    else:
        template_bgr = template if len(template.shape) == 3 else cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)

    if use_color:
        return template_bgr, mask
    else:
        template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
        return template_gray, mask


def scale_template_and_mask(template, mask, scale_factor):
    """Scale template and mask by the given factor. Works with both color and grayscale."""
    if len(template.shape) == 3:
        h, w, _ = template.shape
    else:
        h, w = template.shape
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)

    template_scaled = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)
    mask_scaled = None
    if mask is not None:
        mask_scaled = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return template_scaled, mask_scaled


def match_template(frame_gray, template_gray, mask=None, threshold=DEFAULT_THRESHOLD):
    """
    Perform template matching and return confidence score.

    Returns:
        tuple: (matched, confidence, location)
    """
    if mask is not None:
        result = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCORR_NORMED, mask=mask)
    else:
        result = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    return max_val >= threshold, max_val, max_loc


def match_template_color(frame_bgr, template_bgr, mask=None, threshold=DEFAULT_THRESHOLD):
    """
    Perform color-aware template matching and return confidence score.
    Uses TM_CCOEFF_NORMED for better color matching.

    Returns:
        tuple: (matched, confidence, location)
    """
    if mask is not None:
        result = cv2.matchTemplate(frame_bgr, template_bgr, cv2.TM_CCORR_NORMED, mask=mask)
    else:
        result = cv2.matchTemplate(frame_bgr, template_bgr, cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    return max_val >= threshold, max_val, max_loc


def load_and_scale_templates(template_dir, target_height=480, debug_output_dir=None):
    """
    Load all templates and scale them for the target processing resolution.

    Template reference resolution is auto-detected from the 'characters' template.

    Args:
        template_dir: Directory containing template images
        target_height: Target height for processing (default 480p)
        debug_output_dir: If provided, save scaled templates and masks here

    Returns:
        tuple: (templates dict, scale_factor)
    """
    templates = {}
    grayscale_templates = ['characters', 'game', 'p1', 'p2', 'p3', 'p4']
    color_templates = ['win', 'loss']  # These are color-sensitive

    # Load grayscale templates
    print(f"Loading templates from: {template_dir}")
    for name in grayscale_templates:
        path = os.path.join(template_dir, f"{name}_template.png")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Template not found: {path}")
        img, mask = load_template_with_mask(path, use_color=False)
        templates[name] = {'img': img, 'mask': mask, 'use_color': False}
        h, w = img.shape[:2]
        print(f"  Loaded {name}: {w}x{h} (mask={mask is not None}, grayscale)")

    # Load color templates (win/loss are color-sensitive)
    for name in color_templates:
        path = os.path.join(template_dir, f"{name}_template.png")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Template not found: {path}")
        img, mask = load_template_with_mask(path, use_color=True)
        templates[name] = {'img': img, 'mask': mask, 'use_color': True}
        h, w = img.shape[:2]
        print(f"  Loaded {name}: {w}x{h} (mask={mask is not None}, COLOR)")

    # Auto-detect template reference resolution from 'characters' template width
    # Characters template spans full screen width, so we can determine source resolution
    characters_h, characters_w = templates['characters']['img'].shape[:2]
    # Assume 16:9 aspect ratio: width/height = 16/9, so height = width * 9/16
    template_reference_height = int(characters_w * 9 / 16)
    print(f"\nAuto-detected template reference: {characters_w}x{template_reference_height} (from characters template {characters_w}x{characters_h})")

    # Scale factor for templates
    scale_factor = target_height / template_reference_height
    print(f"Scaling templates: {template_reference_height}p -> {target_height}p (scale={scale_factor:.3f})")

    # Scale templates
    for name, t in templates.items():
        t['img_scaled'], t['mask_scaled'] = scale_template_and_mask(
            t['img'], t['mask'], scale_factor
        )
        h, w = t['img_scaled'].shape[:2]
        print(f"  {name}: {w}x{h}")

    # Save debug images if output directory provided
    if debug_output_dir:
        debug_match_dir = os.path.join(debug_output_dir, "debug_matching")
        os.makedirs(debug_match_dir, exist_ok=True)
        print(f"\nSaving debug templates to: {debug_match_dir}")
        for name, t in templates.items():
            cv2.imwrite(os.path.join(debug_match_dir, f"{name}_template_scaled.png"), t['img_scaled'])
            if t['mask_scaled'] is not None:
                cv2.imwrite(os.path.join(debug_match_dir, f"{name}_mask_scaled.png"), t['mask_scaled'])
            print(f"  Saved {name}_template_scaled.png" + (f" + {name}_mask_scaled.png" if t['mask_scaled'] is not None else ""))

    return templates, scale_factor


def save_frame(frame, frame_small, output_dir, name, frame_num, fps):
    """Save both 480p and full resolution versions of a frame."""
    timestamp = frame_num / fps
    minutes = int(timestamp // 60)
    seconds = int(timestamp % 60)

    base_filename = f"{name}_{frame_num:06d}_{minutes}m{seconds:02d}s.png"

    # Save low-res (480p) version
    path = os.path.join(output_dir, base_filename)
    cv2.imwrite(path, frame_small)

    # Save full resolution version
    full_res_filename = f"{name}_{frame_num:06d}_{minutes}m{seconds:02d}s_full.png"
    full_res_path = os.path.join(output_dir, full_res_filename)
    cv2.imwrite(full_res_path, frame)

    print(f"  Saved: {base_filename} + {full_res_filename}")

    return path


def find_frames(video_path: str, template_dir: str, output_dir: str,
                threshold: float = DEFAULT_THRESHOLD, debug: bool = False,
                on_game_complete: callable = None):
    """
    Analyze video to find character select, game end, and result screens.
    Supports multiple games in a single video.

    Args:
        video_path: Path to the input video file
        template_dir: Directory containing template images
        output_dir: Directory to save matched frames
        threshold: Matching threshold (0-1), higher = stricter matching
        debug: If True, save debug CSV and best match frames
        on_game_complete: Optional callback function called after each game's
                         results are captured. Receives (game_dir, game_num) args.

    Returns:
        list: List of game directory paths for each game found
    """
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Target 480p for matching
    target_height = 480
    target_width = int(video_width * (target_height / video_height))

    # Load and scale templates
    templates, template_scale_factor = load_and_scale_templates(template_dir, target_height, debug_output_dir=output_dir)

    # Create frame processor with shared state machine logic
    processor = FrameProcessor(
        templates=templates,
        threshold=threshold,
        fps=fps,
        output_dir=output_dir,
        debug=debug,
        on_game_complete=on_game_complete
    )

    frame_num = 0
    last_progress_report = 0
    start_time = time.perf_counter()

    print(f"\nStarting frame detection...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\nReached end of video")
            break

        # Determine frame skip based on processor state
        frame_skip = processor.get_frame_skip()

        # Skip frames to reduce processing load
        if frame_num % frame_skip != 0:
            frame_num += 1
            continue

        # Progress reporting every 5%
        progress = (frame_num / total_frames) * 100
        if progress >= last_progress_report + 5.0:
            milestone = int(progress // 5) * 5
            print(f"Progress: {milestone}% (Frame {frame_num}/{total_frames}, Games found: {processor.game_num})")
            last_progress_report = milestone

        # Downscale frame to 480p for faster matching
        frame_small = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

        # Process frame through state machine
        processor.process_frame(frame, frame_small, frame_num)

        frame_num += 1

    cap.release()
    total_time = time.perf_counter() - start_time

    # Finalize any pending player matches (video ended during result detection)
    processor.finalize()

    # Save debug info if enabled
    if debug:
        debug_dir = os.path.join(output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)

        # Save debug CSV with all confidence scores
        debug_csv_path = os.path.join(debug_dir, "debug_matches.csv")
        with open(debug_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_num', 'state', 'template', 'confidence', 'matched'])
            for row in processor.debug_log:
                writer.writerow(row)
        print(f"\nDebug CSV saved: {debug_csv_path}")

        # Save best match summary
        summary_csv_path = os.path.join(debug_dir, "best_matches.csv")
        with open(summary_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['template', 'best_confidence', 'frame_num', 'threshold', 'would_match'])
            for template_name, data in processor.best_matches.items():
                would_match = "Yes" if data['confidence'] >= threshold else "No"
                writer.writerow([
                    template_name,
                    f"{data['confidence']:.4f}",
                    data['frame_num'],
                    threshold,
                    would_match
                ])
        print(f"Best matches summary saved: {summary_csv_path}")

        # Save best match frames with template overlay
        for template_name, data in processor.best_matches.items():
            if data['frame'] is not None and data['loc'] is not None:
                frame_copy = data['frame'].copy()
                t = templates[template_name]
                template_img = t['img_scaled']
                loc = data['loc']

                # Get template dimensions
                if len(template_img.shape) == 3:
                    th, tw, _ = template_img.shape
                else:
                    th, tw = template_img.shape

                # Draw rectangle showing where template matched
                x, y = loc
                cv2.rectangle(frame_copy, (x, y), (x + tw, y + th), (0, 255, 0), 2)

                # Overlay template with transparency
                if len(template_img.shape) == 2:
                    template_bgr = cv2.cvtColor(template_img, cv2.COLOR_GRAY2BGR)
                else:
                    template_bgr = template_img

                # Blend template onto frame at match location
                alpha = 0.5
                roi = frame_copy[y:y+th, x:x+tw]
                if roi.shape[0] == template_bgr.shape[0] and roi.shape[1] == template_bgr.shape[1]:
                    blended = cv2.addWeighted(roi, 1 - alpha, template_bgr, alpha, 0)
                    frame_copy[y:y+th, x:x+tw] = blended

                # Add text label
                label = f"{template_name}: {data['confidence']:.4f}"
                cv2.putText(frame_copy, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                frame_path = os.path.join(debug_dir, f"best_{template_name}_conf{data['confidence']:.4f}_frame{data['frame_num']}.png")
                cv2.imwrite(frame_path, frame_copy)
                print(f"  Saved best {template_name} frame with overlay: confidence={data['confidence']:.4f}")

    # Print summary
    print("\n" + "-" * 50)
    print("FRAME DETECTION COMPLETE")
    print("-" * 50)
    print(f"Total frames processed: {frame_num}")
    print(f"Total processing time: {total_time:.2f}s")
    print(f"Games found: {len(processor.game_directories)}")

    # Print best match summary
    if debug:
        print("\nBest matches per template:")
        for template_name, data in processor.best_matches.items():
            status = "MATCHED" if data['confidence'] >= threshold else "NOT MATCHED"
            print(f"  {template_name}: {data['confidence']:.4f} (threshold: {threshold}) - {status}")

    return processor.game_directories
