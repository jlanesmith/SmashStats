#!/usr/bin/env python3
"""
Analyze Super Smash Bros game frames to extract match information.
Reads character names from the character selection screen.
"""

import cv2
import numpy as np
import argparse
import csv
import os
import math
import difflib
from pathlib import Path
from datetime import datetime
import re
import sys

try:
    import pytesseract
except ImportError:
    print("Error: pytesseract is required. Install with: pip install pytesseract")
    print("Also ensure tesseract-ocr is installed on your system:")
    print("  macOS: brew install tesseract")
    print("  Ubuntu: sudo apt-get install tesseract-ocr")
    exit(1)


# Shared CSV field names for game results
CSV_FIELDNAMES = [
    'datetime',
    'p1_character', 'p2_character', 'p3_character', 'p4_character',
    'p1_kos', 'p2_kos', 'p3_kos', 'p4_kos',
    'p1_falls', 'p2_falls', 'p3_falls', 'p4_falls',
    'p1_damage', 'p2_damage', 'p3_damage', 'p4_damage',
    'win',
    'opponent'
]


def load_character_list():
    """Load the list of valid character names from character_list.txt."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    char_list_path = os.path.join(script_dir, "character_list.txt")

    if not os.path.exists(char_list_path):
        return []

    with open(char_list_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


# Cache the character list
_CHARACTER_LIST = None


def get_character_list():
    """Get the cached character list, loading it if necessary."""
    global _CHARACTER_LIST
    if _CHARACTER_LIST is None:
        _CHARACTER_LIST = load_character_list()
    return _CHARACTER_LIST


def normalize_for_matching(text):
    """Normalize text for fuzzy matching - lowercase, remove punctuation."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text


def find_closest_character(ocr_text, min_similarity=0.6):
    """
    Find the closest matching character name from the character list.

    Args:
        ocr_text: The OCR-detected text
        min_similarity: Minimum similarity ratio (0-1) to accept a match

    Returns:
        The closest matching character name, or the original text if no good match
    """
    if not ocr_text:
        return ocr_text

    character_list = get_character_list()
    if not character_list:
        return ocr_text

    ocr_normalized = normalize_for_matching(ocr_text)

    # Build a mapping of normalized names to original names
    normalized_to_original = {}
    for char in character_list:
        normalized_to_original[normalize_for_matching(char)] = char

    # Find closest match using difflib
    matches = difflib.get_close_matches(
        ocr_normalized,
        normalized_to_original.keys(),
        n=1,
        cutoff=min_similarity
    )

    if matches:
        return normalized_to_original[matches[0]]

    return ocr_text


def count_kos(result_image_path: str, player_num: int, save_debug_images: bool = True):
    """
    Count KOs from a player's result screen by detecting icon graphics.

    Args:
        result_image_path: Path to the player result screenshot (480p version)
        player_num: Player number (1-4)
        save_debug_images: If True, save the extracted region for debugging

    Returns:
        int: Number of KOs
    """
    # Try to use full resolution image for better accuracy
    full_res_path = result_image_path.replace('.png', '_full.png')
    if os.path.exists(full_res_path):
        img = cv2.imread(full_res_path)
        using_full_res = True
    else:
        img = cv2.imread(result_image_path)
        using_full_res = False

    if img is None:
        raise FileNotFoundError(f"Could not load image: {result_image_path}")

    height, width = img.shape[:2]

    # Base dimensions for 480p
    base_height = 480
    scale = height / base_height

    # KO icon region for each player - base coordinates for 480p
    # P1: top-left at (19, 257), size 150x22, each icon ~22x22
    ko_regions_480p = {
        1: {'x': 19, 'y': 257, 'width': 150, 'height': 22},
        2: {'x': 233, 'y': 257, 'width': 150, 'height': 22},
        3: {'x': 446, 'y': 257, 'width': 150, 'height': 22},
        4: {'x': 659, 'y': 257, 'width': 150, 'height': 22},
    }

    base_region = ko_regions_480p[player_num]
    x_start = int(base_region['x'] * scale)
    y_start = int(base_region['y'] * scale)
    box_width = int(base_region['width'] * scale)
    box_height = int(base_region['height'] * scale)
    icon_width = int(22 * scale)

    # Extract the KO region
    ko_region = img[y_start:y_start + box_height, x_start:x_start + box_width]

    # Create debug directory if needed
    if save_debug_images:
        debug_dir = os.path.dirname(result_image_path)
        debug_subdir = os.path.join(debug_dir, "debug_regions")
        os.makedirs(debug_subdir, exist_ok=True)

        res_suffix = f"_{height}p" if using_full_res else "_480p"
        cv2.imwrite(os.path.join(debug_subdir, f"p{player_num}_ko_region{res_suffix}.png"), ko_region)

        # Save bounding box overlay
        img_with_box = img.copy()
        cv2.rectangle(img_with_box, (x_start, y_start), (x_start + box_width, y_start + box_height), (0, 255, 0), 2)
        cv2.putText(img_with_box, f"P{player_num} KOs", (x_start, y_start - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (0, 255, 0), int(1 * scale))
        cv2.imwrite(os.path.join(debug_subdir, f"p{player_num}_ko_bbox{res_suffix}.png"), img_with_box)

    # Convert to grayscale
    gray = cv2.cvtColor(ko_region, cv2.COLOR_BGR2GRAY)

    # Count icons by checking each icon-width segment for significant content
    max_icons = box_width // icon_width  # Maximum possible icons
    ko_count = 0

    for i in range(max_icons):
        icon_x_start = i * icon_width
        icon_x_end = min((i + 1) * icon_width, box_width)

        icon_segment = gray[:, icon_x_start:icon_x_end]

        # Check if this segment has significant content (not mostly empty)
        # Calculate the standard deviation - icons will have more variation than empty space
        std_dev = np.std(icon_segment)

        # Threshold for detecting an icon (adjust if needed)
        if std_dev > 20:
            ko_count += 1
        else:
            # Once we hit an empty segment, stop counting
            break

    return ko_count


def count_falls(result_image_path: str, player_num: int, save_debug_images: bool = True):
    """
    Count falls from a player's result screen by detecting icon graphics.
    Falls region is 50 pixels below the KOs region.

    Args:
        result_image_path: Path to the player result screenshot (480p version)
        player_num: Player number (1-4)
        save_debug_images: If True, save the extracted region for debugging

    Returns:
        int: Number of falls
    """
    # Try to use full resolution image for better accuracy
    full_res_path = result_image_path.replace('.png', '_full.png')
    if os.path.exists(full_res_path):
        img = cv2.imread(full_res_path)
        using_full_res = True
    else:
        img = cv2.imread(result_image_path)
        using_full_res = False

    if img is None:
        raise FileNotFoundError(f"Could not load image: {result_image_path}")

    height, width = img.shape[:2]

    # Base dimensions for 480p
    base_height = 480
    scale = height / base_height

    # Falls icon region - same as KOs but 50 pixels lower (at 480p)
    falls_regions_480p = {
        1: {'x': 19, 'y': 307, 'width': 150, 'height': 22},
        2: {'x': 233, 'y': 307, 'width': 150, 'height': 22},
        3: {'x': 446, 'y': 307, 'width': 150, 'height': 22},
        4: {'x': 659, 'y': 307, 'width': 150, 'height': 22},
    }

    base_region = falls_regions_480p[player_num]
    x_start = int(base_region['x'] * scale)
    y_start = int(base_region['y'] * scale)
    box_width = int(base_region['width'] * scale)
    box_height = int(base_region['height'] * scale)
    icon_width = int(22 * scale)

    # Extract the falls region
    falls_region = img[y_start:y_start + box_height, x_start:x_start + box_width]

    # Create debug directory and save region
    if save_debug_images:
        debug_dir = os.path.dirname(result_image_path)
        debug_subdir = os.path.join(debug_dir, "debug_regions")
        os.makedirs(debug_subdir, exist_ok=True)

        res_suffix = f"_{height}p" if using_full_res else "_480p"
        cv2.imwrite(os.path.join(debug_subdir, f"p{player_num}_falls_region{res_suffix}.png"), falls_region)

        # Save bounding box overlay
        img_with_box = img.copy()
        cv2.rectangle(img_with_box, (x_start, y_start), (x_start + box_width, y_start + box_height), (0, 0, 255), 2)
        cv2.putText(img_with_box, f"P{player_num} Falls", (x_start, y_start - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (0, 0, 255), int(1 * scale))
        cv2.imwrite(os.path.join(debug_subdir, f"p{player_num}_falls_bbox{res_suffix}.png"), img_with_box)

    # Convert to grayscale
    gray = cv2.cvtColor(falls_region, cv2.COLOR_BGR2GRAY)

    # Count icons by checking each icon-width segment for significant content
    max_icons = box_width // icon_width
    falls_count = 0

    for i in range(max_icons):
        icon_x_start = i * icon_width
        icon_x_end = min((i + 1) * icon_width, box_width)

        icon_segment = gray[:, icon_x_start:icon_x_end]

        # Check if this segment has significant content
        std_dev = np.std(icon_segment)

        # Threshold for detecting an icon
        if std_dev > 20:
            falls_count += 1
        else:
            # Once we hit an empty segment, stop counting
            break

    return falls_count


def extract_damage(result_image_path: str, player_num: int, save_debug_images: bool = True):
    """
    Extract damage percentage from a player's result screen using OCR.

    Args:
        result_image_path: Path to the player result screenshot (480p version)
        player_num: Player number (1-4)
        save_debug_images: If True, save the extracted region for debugging

    Returns:
        int: Damage percentage (or None if not detected)
    """
    # Try to use full resolution image for better OCR accuracy
    full_res_path = result_image_path.replace('.png', '_full.png')
    if os.path.exists(full_res_path):
        img = cv2.imread(full_res_path)
        using_full_res = True
    else:
        img = cv2.imread(result_image_path)
        using_full_res = False

    if img is None:
        raise FileNotFoundError(f"Could not load image: {result_image_path}")

    height, width = img.shape[:2]

    # Base dimensions for 480p
    base_height = 480
    scale = height / base_height

    # Damage region - right-aligned, so we specify top-right corner (at 480p)
    # P1: top-right at (205, 404), size 90x22
    # The x values are spaced ~213 pixels apart for each player
    damage_regions_480p = {
        1: {'x_right': 205, 'y': 404, 'width': 90, 'height': 22},
        2: {'x_right': 418, 'y': 404, 'width': 90, 'height': 22},
        3: {'x_right': 631, 'y': 404, 'width': 90, 'height': 22},
        4: {'x_right': 844, 'y': 404, 'width': 90, 'height': 22},
    }

    base_region = damage_regions_480p[player_num]
    x_end = int(base_region['x_right'] * scale)
    y_start = int(base_region['y'] * scale)
    box_width = int(base_region['width'] * scale)
    box_height = int(base_region['height'] * scale)
    x_start = x_end - box_width

    # Extract the damage region
    damage_region = img[y_start:y_start + box_height, x_start:x_end]

    # Create debug directory if needed
    if save_debug_images:
        debug_dir = os.path.dirname(result_image_path)
        debug_subdir = os.path.join(debug_dir, "debug_regions")
        os.makedirs(debug_subdir, exist_ok=True)

        res_suffix = f"_{height}p" if using_full_res else "_480p"
        cv2.imwrite(os.path.join(debug_subdir, f"p{player_num}_damage_region{res_suffix}.png"), damage_region)

        # Save bounding box overlay
        img_with_box = img.copy()
        cv2.rectangle(img_with_box, (x_start, y_start), (x_end, y_start + box_height), (255, 255, 0), 2)
        cv2.putText(img_with_box, f"P{player_num} Damage", (x_start, y_start - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (255, 255, 0), int(1 * scale))
        cv2.imwrite(os.path.join(debug_subdir, f"p{player_num}_damage_bbox{res_suffix}.png"), img_with_box)

    # Convert to grayscale
    gray = cv2.cvtColor(damage_region, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    if save_debug_images:
        cv2.imwrite(os.path.join(debug_subdir, f"p{player_num}_damage_thresh{res_suffix}.png"), thresh)

    # Perform OCR - PSM 7 for single line of text
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789%'
    text = pytesseract.image_to_string(thresh, config=custom_config)

    # Clean up and extract the number
    text = text.strip()
    # Remove % and any other non-digit characters
    damage_str = re.sub(r'[^0-9]', '', text)

    damage = None
    if damage_str:
        try:
            damage = int(damage_str)
        except ValueError:
            pass

    return damage


def extract_opponent_name(characters_image_path: str, save_debug_images: bool = True):
    """
    Extract opponent name from the characters frame.

    The opponent name appears in P3's and P4's columns on the characters screen.
    P3 shows "<name>1" and P4 shows "<name>2".
    We check P3 first - if it ends in "1", use that name (minus the "1").
    Otherwise, check P4 - if it ends in "2", use that name (minus the "2").

    Args:
        characters_image_path: Path to the characters screenshot (480p version)
        save_debug_images: If True, save the extracted regions for debugging

    Returns:
        str: Opponent name (or empty string if not detected)
    """
    # Base coordinates for 1080p (1920x1080) - from characters frame
    # P3: top-left at (1040, 143), size 270x60
    # P4: 481 pixels to the right of P3
    base_height = 1080
    base_coords = {
        'p3': {'x': 1040, 'y': 151, 'width': 240, 'height': 37},
        'p4': {'x': 1521, 'y': 151, 'width': 240, 'height': 37},
    }

    # Try to use full resolution image for better OCR accuracy
    full_res_path = characters_image_path.replace('.png', '_full.png')
    if os.path.exists(full_res_path):
        img = cv2.imread(full_res_path)
        using_full_res = True
    else:
        img = cv2.imread(characters_image_path)
        using_full_res = False

    if img is None:
        return ""

    height, width = img.shape[:2]
    scale = height / base_height

    def extract_name_from_region(player_key: str):
        """Helper to extract opponent name from a region."""
        coords = base_coords[player_key]
        x_start = int(coords['x'] * scale)
        y_start = int(coords['y'] * scale)
        box_width = int(coords['width'] * scale)
        box_height = int(coords['height'] * scale)

        x_end = min(x_start + box_width, width)
        y_end = min(y_start + box_height, height)

        # Extract the opponent name region
        name_region = img[y_start:y_end, x_start:x_end]

        if name_region.size == 0:
            return {
                'text': None, 'bbox': None,
                'gray': None, 'gray_upscaled': None, 'denoised': None,
                'thresh': None, 'region': None
            }

        # Convert to grayscale
        gray = cv2.cvtColor(name_region, cv2.COLOR_BGR2GRAY)

        # Upscale to target height for optimal OCR accuracy
        # At 1080p, region is ~37px tall. At 480p, region is ~16px tall.
        # Target ~100px height for good Tesseract accuracy (works well with 30-100px text)
        TARGET_OCR_HEIGHT = 100
        current_height = gray.shape[0]
        upscale_factor = max(1.0, TARGET_OCR_HEIGHT / current_height)
        gray_upscaled = cv2.resize(gray, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC)

        # Rotate to correct for angled text (goes up 8px over 160px = ~2.86° clockwise)
        angle = math.degrees(math.atan(8 / 160))  # ~2.86 degrees
        h, w = gray_upscaled.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)  # negative = clockwise
        gray_upscaled = cv2.warpAffine(gray_upscaled, rotation_matrix, (w, h),
                                        borderMode=cv2.BORDER_REPLICATE)

        # Apply bilateral filter to reduce noise while keeping edges sharp
        denoised = cv2.bilateralFilter(gray_upscaled, 9, 75, 75)

        # Use Otsu thresholding
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Determine if we need to invert (text should be black on white for OCR)
        white_pixels = np.sum(thresh == 255)
        black_pixels = np.sum(thresh == 0)
        if black_pixels > white_pixels:
            thresh = cv2.bitwise_not(thresh)

        # Remove small noise with morphological opening
        kernel_open = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)

        # Erode slightly to thin the text (makes OCR more accurate)
        # Dilate the white background to thin black text
        kernel_thin = np.ones((2, 2), np.uint8)
        thresh = cv2.dilate(thresh, kernel_thin, iterations=1)

        # Add white padding around image - tesseract needs margin to recognize edge characters
        padding = 20
        thresh = cv2.copyMakeBorder(thresh, padding, padding, padding, padding,
                                    cv2.BORDER_CONSTANT, value=255)

        # Perform OCR - PSM 7 for single line of text
        custom_config = r'--oem 3 --psm 7'
        text = pytesseract.image_to_string(thresh, config=custom_config).strip()

        bbox = (x_start, y_start, x_end, y_end)
        return {
            'text': text,
            'bbox': bbox,
            'gray': gray,
            'gray_upscaled': gray_upscaled,
            'denoised': denoised,
            'thresh': thresh,
            'region': name_region
        }

    # Extract from P3 and P4 regions
    p3_data = extract_name_from_region('p3')
    p4_data = extract_name_from_region('p4')

    # Helper to fix common OCR misreads for trailing player numbers
    def fix_trailing_number(text, expected_char, common_misreads):
        """Replace common OCR misreads at end of text with expected character."""
        if text and text[-1] in common_misreads:
            return text[:-1] + expected_char
        return text

    # Fix common "1" misreads for P3 (!, l, I, |)
    p3_text = fix_trailing_number(p3_data['text'], '1', '!lI|')
    # Fix common "2" misreads for P4 (Z, z, ?)
    p4_text = fix_trailing_number(p4_data['text'], '2', 'Zz?')

    # Determine selection result
    selected_from = None
    final_name = ""
    selection_reason = ""

    if p3_text and p3_text.endswith('1'):
        selected_from = "P3"
        final_name = p3_text[:-1].strip()
        selection_reason = f"P3 text ends with '1': '{p3_data['text']}' -> '{p3_text}' -> '{final_name}'"
    elif p4_text and p4_text.endswith('2'):
        selected_from = "P4"
        final_name = p4_text[:-1].strip()
        selection_reason = f"P4 text ends with '2': '{p4_data['text']}' -> '{p4_text}' -> '{final_name}'"
    else:
        selection_reason = f"Neither P3 ('{p3_text}') ends with '1' nor P4 ('{p4_text}') ends with '2'"

    # Save debug images
    if save_debug_images:
        debug_dir = os.path.dirname(characters_image_path)
        debug_subdir = os.path.join(debug_dir, "debug_regions")
        os.makedirs(debug_subdir, exist_ok=True)

        res_suffix = f"_{height}p" if using_full_res else "_480p"

        # Save P3 debug images
        if p3_data['region'] is not None:
            cv2.imwrite(os.path.join(debug_subdir, f"game_p3_opponent_region{res_suffix}.png"), p3_data['region'])
            cv2.imwrite(os.path.join(debug_subdir, f"game_p3_opponent_gray{res_suffix}.png"), p3_data['gray'])
            cv2.imwrite(os.path.join(debug_subdir, f"game_p3_opponent_gray_upscaled{res_suffix}.png"), p3_data['gray_upscaled'])
            cv2.imwrite(os.path.join(debug_subdir, f"game_p3_opponent_thresh{res_suffix}.png"), p3_data['thresh'])

        # Save P4 debug images
        if p4_data['region'] is not None:
            cv2.imwrite(os.path.join(debug_subdir, f"game_p4_opponent_region{res_suffix}.png"), p4_data['region'])
            cv2.imwrite(os.path.join(debug_subdir, f"game_p4_opponent_gray{res_suffix}.png"), p4_data['gray'])
            cv2.imwrite(os.path.join(debug_subdir, f"game_p4_opponent_gray_upscaled{res_suffix}.png"), p4_data['gray_upscaled'])
            cv2.imwrite(os.path.join(debug_subdir, f"game_p4_opponent_thresh{res_suffix}.png"), p4_data['thresh'])

        # Save with bounding boxes on full image
        img_with_boxes = img.copy()
        if p3_data['bbox']:
            x_start, y_start, x_end, y_end = p3_data['bbox']
            cv2.rectangle(img_with_boxes, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            cv2.putText(img_with_boxes, f"P3: {p3_data['text'] or '?'}", (x_start, y_start - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (0, 255, 0), max(1, int(scale)))
        if p4_data['bbox']:
            x_start, y_start, x_end, y_end = p4_data['bbox']
            cv2.rectangle(img_with_boxes, (x_start, y_start), (x_end, y_end), (255, 0, 255), 2)
            cv2.putText(img_with_boxes, f"P4: {p4_data['text'] or '?'}", (x_start, y_start - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (255, 0, 255), max(1, int(scale)))
        cv2.imwrite(os.path.join(debug_subdir, f"game_opponent_bbox{res_suffix}.png"), img_with_boxes)

        # Save summary text file
        summary_path = os.path.join(debug_subdir, "opponent_name_debug.txt")
        with open(summary_path, 'w') as f:
            f.write("=== OPPONENT NAME DETECTION DEBUG (GAME FRAME) ===\n\n")
            f.write(f"Source Image: {characters_image_path}\n")
            f.write(f"Resolution: {width}x{height} (full_res={using_full_res})\n")
            f.write(f"Scale factor: {scale:.3f} (base: {base_height}p)\n\n")
            f.write("--- Base Coordinates (1080p) ---\n")
            f.write(f"  P3: x={base_coords['p3']['x']}, y={base_coords['p3']['y']}, w={base_coords['p3']['width']}, h={base_coords['p3']['height']}\n")
            f.write(f"  P4: x={base_coords['p4']['x']}, y={base_coords['p4']['y']}, w={base_coords['p4']['width']}, h={base_coords['p4']['height']}\n\n")
            f.write("--- P3 Extraction ---\n")
            f.write(f"  Bounding Box: {p3_data['bbox']}\n")
            f.write(f"  OCR Text: '{p3_data['text']}'\n")
            f.write(f"  Fixed Text: '{p3_text}'\n")
            f.write(f"  Ends with '1': {p3_text.endswith('1') if p3_text else False}\n\n")
            f.write("--- P4 Extraction ---\n")
            f.write(f"  Bounding Box: {p4_data['bbox']}\n")
            f.write(f"  OCR Text: '{p4_data['text']}'\n")
            f.write(f"  Fixed Text: '{p4_text}'\n")
            f.write(f"  Ends with '2': {p4_text.endswith('2') if p4_text else False}\n\n")
            f.write("--- Selection Logic ---\n")
            f.write(f"  {selection_reason}\n")
            f.write(f"  Selected From: {selected_from or 'NONE'}\n")
            f.write(f"  Final Name: '{final_name}'\n")

    return final_name


def extract_character_names(characters_image_path: str, save_debug_images: bool = True,
                            threshold_value: int = None, invert_threshold: bool = False):
    """
    Extract character names from the character selection screen.

    Args:
        characters_image_path: Path to the characters screenshot (480p version)
        save_debug_images: If True, save the extracted regions for debugging
        threshold_value: Manual threshold value (0-255). If None, uses automatic Otsu thresholding
        invert_threshold: If True, invert the thresholded image (white text on black -> black text on white)

    Returns:
        dict: Mapping of player number to character name
    """
    # Try to use full resolution image for better OCR accuracy
    full_res_path = characters_image_path.replace('.png', '_full.png')
    if os.path.exists(full_res_path):
        img = cv2.imread(full_res_path)
        using_full_res = True
    else:
        img = cv2.imread(characters_image_path)
        using_full_res = False

    if img is None:
        raise FileNotFoundError(f"Could not load image: {characters_image_path}")

    height, width = img.shape[:2]

    # Base dimensions for 480p
    base_height = 480
    scale = height / base_height

    # Character names region - base coordinates for 480p
    # Names are in the top 66 pixels at 480p
    base_name_height = 66
    base_skip_left = 35  # Skip "P#" label

    # Scale to current resolution
    name_height = int(base_name_height * scale)
    skip_left = int(base_skip_left * scale)
    name_width = width // 4  # 4 players, equal width

    # Create debug directory if needed
    if save_debug_images:
        debug_dir = os.path.dirname(characters_image_path)
        debug_subdir = os.path.join(debug_dir, "debug_regions")
        os.makedirs(debug_subdir, exist_ok=True)

        res_suffix = f"_{height}p" if using_full_res else "_480p"
        print(f"Saving debug images to: {debug_subdir} (resolution: {height}p)")
        print("-" * 50)

        # Create a copy for bounding box overlay
        img_with_boxes = img.copy()

    characters = {}

    for player_num in range(1, 5):
        # Calculate the region for this player
        x_start = (player_num - 1) * name_width
        x_end = player_num * name_width
        y_start = 0
        y_end = name_height

        # Skip the first pixels from the left where "P#" is displayed
        x_start += skip_left

        # Draw bounding box on overlay image
        if save_debug_images:
            color = [(0, 255, 0), (0, 255, 255), (255, 0, 0), (255, 0, 255)][player_num - 1]
            cv2.rectangle(img_with_boxes, (x_start, y_start), (x_end, y_end), color, 2)
            # Add player label
            cv2.putText(img_with_boxes, f"P{player_num}", (x_start + 5, y_start + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, color, int(1 * scale))

        # Extract the region
        region = img[y_start:y_end, x_start:x_end]

        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to improve OCR accuracy
        if threshold_value is not None:
            thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)[1]
        else:
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Optionally invert the threshold result
        if invert_threshold:
            thresh = cv2.bitwise_not(thresh)

        # Save region for debugging
        if save_debug_images:
            cv2.imwrite(os.path.join(debug_subdir, f"p{player_num}_char_region{res_suffix}.png"), region)
            cv2.imwrite(os.path.join(debug_subdir, f"p{player_num}_char_thresh{res_suffix}.png"), thresh)
            print(f"Saved debug images for P{player_num}")

        # Perform OCR
        # PSM 6: Assume a single uniform block of text (handles multi-line better than PSM 7)
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(thresh, config=custom_config)

        # Clean up the text
        text = text.strip()

        # Remove "P1", "P2", "P3", "P4" and other common noise
        text = re.sub(r'\bP[1-4]\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace (combines multi-line into single)
        text = text.strip()

        # Remove any remaining special characters but keep letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.strip()

        # Find closest matching character name from the known list
        text = find_closest_character(text)

        player_key = f"p{player_num}"
        characters[player_key] = text

    # Save the full image with all bounding boxes
    if save_debug_images:
        cv2.imwrite(os.path.join(debug_subdir, f"characters_bbox_overlay{res_suffix}.png"), img_with_boxes)
        print(f"Saved bounding box overlay: characters_bbox_overlay{res_suffix}.png")

    return characters


def analyze_game_dir(game_dir: str, game_num: int = None, verbose: bool = True):
    """
    Analyze a completed game's captured frames.

    Args:
        game_dir: Path to the game directory containing captured frames
        game_num: Game number for display purposes (optional)
        verbose: If True, print progress messages

    Returns:
        dict: Game results including characters, stats, and winner, or None if failed
    """
    if verbose and game_num:
        print(f"\n{'='*50}")
        print(f"Analyzing Game {game_num}...")
        print(f"{'='*50}")

    # Find characters image
    characters_files = list(Path(game_dir).glob("characters_*.png"))
    if not characters_files:
        if verbose:
            print(f"Warning: No characters file found in {game_dir}")
        return None

    characters_path = str(characters_files[0])

    # Extract character names
    characters = extract_character_names(characters_path, save_debug_images=True)

    # Extract stats for each player
    kos = {}
    falls = {}
    damage = {}

    for player_num in range(1, 5):
        player_key = f"p{player_num}"
        player_files = list(Path(game_dir).glob(f"{player_key}_*.png"))

        if not player_files:
            if verbose:
                print(f"Warning: No {player_key} file found")
            # Use None (N/A) when data isn't available for any player
            kos[player_key] = None
            falls[player_key] = None
            damage[player_key] = None
            continue

        player_path = str(player_files[0])
        kos[player_key] = count_kos(player_path, player_num, save_debug_images=True)
        falls[player_key] = count_falls(player_path, player_num, save_debug_images=True)
        extracted_damage = extract_damage(player_path, player_num, save_debug_images=True)
        # Use None (N/A) for any player when damage couldn't be extracted
        damage[player_key] = extracted_damage

    # Read game result from file (detected via win/loss template matching)
    game_result_file = os.path.join(game_dir, "game_result.txt")
    if os.path.exists(game_result_file):
        with open(game_result_file, 'r') as f:
            game_result = f.read().strip()
        team1_won = (game_result == 'win')
        win = "Yes" if team1_won else "No"
        if verbose:
            print(f"Game result from template matching: {game_result.upper()}")
    else:
        # Fallback to counting falls if game_result.txt doesn't exist
        if verbose:
            print("Warning: game_result.txt not found, falling back to falls-based detection")
        team1_falls = falls.get('p1', 0) + falls.get('p2', 0)
        team2_falls = falls.get('p3', 0) + falls.get('p4', 0)
        team1_won = team1_falls < team2_falls
        win = "Yes" if team1_won else "No"

    # Extract opponent name from characters frame
    opponent = ""
    # Filter out full resolution files for the 480p path lookup
    characters_files_480p = [f for f in characters_files if '_full.png' not in str(f)]
    if characters_files_480p:
        opponent = extract_opponent_name(str(characters_files_480p[0]), save_debug_images=True) or ""

    result = {
        'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'p1_character': characters.get('p1', ''),
        'p2_character': characters.get('p2', ''),
        'p3_character': characters.get('p3', ''),
        'p4_character': characters.get('p4', ''),
        'p1_kos': kos.get('p1', 0),
        'p2_kos': kos.get('p2', 0),
        'p3_kos': kos.get('p3', 0),
        'p4_kos': kos.get('p4', 0),
        'p1_falls': falls.get('p1', 0),
        'p2_falls': falls.get('p2', 0),
        'p3_falls': falls.get('p3', 0),
        'p4_falls': falls.get('p4', 0),
        'p1_damage': damage.get('p1', 0),
        'p2_damage': damage.get('p2', 0),
        'p3_damage': damage.get('p3', 0),
        'p4_damage': damage.get('p4', 0),
        'win': win,
        'opponent': opponent
    }

    return result


def print_game_result(result: dict, game_num: int = None):
    """Print formatted game results to console."""
    if game_num:
        print(f"\nGame {game_num} Results:")
    else:
        print(f"\nGame Results:")

    def format_stat(value):
        """Format a stat value, showing N/A for None."""
        return "N/A" if value is None else value

    print(f"  P1: {result['p1_character']} - KOs: {format_stat(result['p1_kos'])}, Falls: {format_stat(result['p1_falls'])}, Damage: {format_stat(result['p1_damage'])}%")
    print(f"  P2: {result['p2_character']} - KOs: {format_stat(result['p2_kos'])}, Falls: {format_stat(result['p2_falls'])}, Damage: {format_stat(result['p2_damage'])}%")
    print(f"  P3: {result['p3_character']} - KOs: {format_stat(result['p3_kos'])}, Falls: {format_stat(result['p3_falls'])}, Damage: {format_stat(result['p3_damage'])}%")
    print(f"  P4: {result['p4_character']} - KOs: {format_stat(result['p4_kos'])}, Falls: {format_stat(result['p4_falls'])}, Damage: {format_stat(result['p4_damage'])}%")

    # Calculate team falls, treating None as 0 for the calculation
    team1_falls = (result['p1_falls'] or 0) + (result['p2_falls'] or 0)
    team2_falls = (result['p3_falls'] or 0) + (result['p4_falls'] or 0)
    print(f"  Team 1 Falls: {team1_falls}, Team 2 Falls: {team2_falls}")
    print(f"  Win: {result['win']}")
    print(f"  Opponent: {result['opponent']}")


def save_result_to_csv(result: dict, csv_path: str):
    """Append a single game result to CSV file."""
    file_exists = os.path.exists(csv_path)

    # Convert None values to "N/A" for CSV output
    csv_result = {}
    for key, value in result.items():
        if value is None:
            csv_result[key] = "N/A"
        else:
            csv_result[key] = value

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(csv_result)


def save_results_to_csv(results: list, csv_path: str):
    """Save multiple game results to CSV file."""
    if not results:
        return

    file_exists = os.path.exists(csv_path)

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        for result in results:
            # Convert None values to "N/A" for CSV output
            csv_result = {}
            for key, value in result.items():
                if value is None:
                    csv_result[key] = "N/A"
                else:
                    csv_result[key] = value
            writer.writerow(csv_result)

    print(f"\nResults saved to: {csv_path}")


def save_result_to_db(result: dict) -> int:
    """
    Save a game result to the database.

    Args:
        result: Dictionary with game data

    Returns:
        int: The ID of the inserted game
    """
    # Import database module (lazy import to avoid circular dependencies)
    sys.path.insert(0, str(Path(__file__).parent / "webapp"))
    from database import save_game_result, init_db

    # Ensure database is initialized
    init_db()

    return save_game_result(result)
