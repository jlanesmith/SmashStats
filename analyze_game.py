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
from pathlib import Path
from datetime import datetime
import re

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


def extract_opponent_name(p3_image_path: str, p4_image_path: str, team1_won: bool = False, save_debug_images: bool = True):
    """
    Extract opponent name from P3 and P4 result screens.

    The opponent name appears as "<name>1" in P3's column and "<name>2" in P4's column.
    We check P3 first - if it ends in "1", use that name (minus the "1").
    Otherwise, check P4 - if it ends in "2", use that name (minus the "2").
    Otherwise, return empty string.

    Args:
        p3_image_path: Path to P3's result screenshot (480p version)
        p4_image_path: Path to P4's result screenshot (480p version)
        team1_won: If True, P1+P2 won (bounding box moves up); if False, P3+P4 won
        save_debug_images: If True, save the extracted regions for debugging

    Returns:
        str: Opponent name (or empty string if not detected)
    """
    # Base coordinates for 480p (when P3+P4 win)
    # P3 column: x starts at 474, P4 column: x starts at 687 (474 + 213 column width)
    # Size 95x34
    base_x_p3 = 474
    base_x_p4 = 687  # P4 is in the 4th column (rightmost)
    base_y_start = 78
    base_box_width = 95
    base_box_height = 34
    base_height = 480

    # If P1+P2 win, the bounding box moves up by 16 pixels at 720p
    if team1_won:
        base_y_offset = 16 * (480 / 720)
        base_y_start = base_y_start - base_y_offset

    def extract_from_image(image_path: str, player_label: str, base_x_start: int):
        """Helper to extract opponent name from a single image."""
        # Try to use full resolution image for better OCR accuracy
        full_res_path = image_path.replace('.png', '_full.png')
        if os.path.exists(full_res_path):
            img = cv2.imread(full_res_path)
            using_full_res = True
        else:
            img = cv2.imread(image_path)
            using_full_res = False

        if img is None:
            return {
                'text': None, 'raw_text': None, 'img': None, 'bbox': None,
                'height': None, 'full_res': False, 'gray': None, 'thresh': None, 'region': None
            }

        height, width = img.shape[:2]
        scale = height / base_height

        x_start = int(base_x_start * scale)
        y_start = int(base_y_start * scale)
        box_width = int(base_box_width * scale)
        box_height = int(base_box_height * scale)

        x_end = min(x_start + box_width, width)
        y_end = min(y_start + box_height, height)

        # Extract the opponent name region
        name_region = img[y_start:y_end, x_start:x_end]

        # Convert to grayscale
        gray = cv2.cvtColor(name_region, cv2.COLOR_BGR2GRAY)

        # Upscale for better OCR accuracy (2x)
        upscale_factor = 2
        gray_upscaled = cv2.resize(gray, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC)

        # Apply bilateral filter to reduce noise while keeping edges sharp
        denoised = cv2.bilateralFilter(gray_upscaled, 9, 75, 75)

        # Use Otsu thresholding (works better for this case)
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Determine if we need to invert (text should be black on white for OCR)
        # If the image is mostly black, invert it
        white_pixels = np.sum(thresh == 255)
        black_pixels = np.sum(thresh == 0)
        if black_pixels > white_pixels:
            thresh = cv2.bitwise_not(thresh)

        # Remove small noise with morphological opening (removes small white dots)
        kernel_open = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)

        # Close small gaps in text with morphological closing
        kernel_close = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)

        # Perform OCR - PSM 7 for single line of text
        custom_config = r'--oem 3 --psm 7'
        raw_text = pytesseract.image_to_string(thresh, config=custom_config)

        # Clean up the text
        text = raw_text.strip()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = text.strip()

        # Return extracted data
        bbox = (x_start, y_start, x_end, y_end)
        return {
            'text': text,
            'raw_text': raw_text.strip(),
            'img': img,
            'bbox': bbox,
            'height': height,
            'full_res': using_full_res,
            'gray': gray,
            'gray_upscaled': gray_upscaled,
            'denoised': denoised,
            'thresh': thresh,
            'region': name_region
        }

    # Extract from P3 (3rd column)
    p3_data = extract_from_image(p3_image_path, "P3", base_x_p3)

    # Extract from P4 (4th column - rightmost)
    p4_data = extract_from_image(p4_image_path, "P4", base_x_p4)

    # Save debug images
    if save_debug_images:
        debug_dir = os.path.dirname(p3_image_path)
        debug_subdir = os.path.join(debug_dir, "debug_regions")
        os.makedirs(debug_subdir, exist_ok=True)

        # Save P3 debug images
        if p3_data['img'] is not None and p3_data['bbox'] is not None:
            res_suffix = f"_{p3_data['height']}p" if p3_data['full_res'] else "_480p"
            x_start, y_start, x_end, y_end = p3_data['bbox']

            # Save cropped region (color)
            cv2.imwrite(os.path.join(debug_subdir, f"p3_opponent_region{res_suffix}.png"), p3_data['region'])

            # Save grayscale (original size)
            cv2.imwrite(os.path.join(debug_subdir, f"p3_opponent_gray{res_suffix}.png"), p3_data['gray'])

            # Save grayscale upscaled (2x)
            cv2.imwrite(os.path.join(debug_subdir, f"p3_opponent_gray_upscaled{res_suffix}.png"), p3_data['gray_upscaled'])

            # Save denoised (after bilateral filter)
            cv2.imwrite(os.path.join(debug_subdir, f"p3_opponent_denoised{res_suffix}.png"), p3_data['denoised'])

            # Save thresholded (what OCR sees)
            cv2.imwrite(os.path.join(debug_subdir, f"p3_opponent_thresh{res_suffix}.png"), p3_data['thresh'])

            # Save with bounding box
            p3_with_box = p3_data['img'].copy()
            cv2.rectangle(p3_with_box, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            cv2.putText(p3_with_box, f"P3: {p3_data['text'] or '?'}", (x_start, y_start - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * (p3_data['height'] / 480), (0, 255, 0), int(1 * (p3_data['height'] / 480)))
            cv2.imwrite(os.path.join(debug_subdir, f"p3_opponent_bbox{res_suffix}.png"), p3_with_box)

        # Save P4 debug images
        if p4_data['img'] is not None and p4_data['bbox'] is not None:
            res_suffix = f"_{p4_data['height']}p" if p4_data['full_res'] else "_480p"
            x_start, y_start, x_end, y_end = p4_data['bbox']

            # Save cropped region (color)
            cv2.imwrite(os.path.join(debug_subdir, f"p4_opponent_region{res_suffix}.png"), p4_data['region'])

            # Save grayscale (original size)
            cv2.imwrite(os.path.join(debug_subdir, f"p4_opponent_gray{res_suffix}.png"), p4_data['gray'])

            # Save grayscale upscaled (2x)
            cv2.imwrite(os.path.join(debug_subdir, f"p4_opponent_gray_upscaled{res_suffix}.png"), p4_data['gray_upscaled'])

            # Save denoised (after bilateral filter)
            cv2.imwrite(os.path.join(debug_subdir, f"p4_opponent_denoised{res_suffix}.png"), p4_data['denoised'])

            # Save thresholded (what OCR sees)
            cv2.imwrite(os.path.join(debug_subdir, f"p4_opponent_thresh{res_suffix}.png"), p4_data['thresh'])

            # Save with bounding box
            p4_with_box = p4_data['img'].copy()
            cv2.rectangle(p4_with_box, (x_start, y_start), (x_end, y_end), (255, 0, 255), 2)
            cv2.putText(p4_with_box, f"P4: {p4_data['text'] or '?'}", (x_start, y_start - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * (p4_data['height'] / 480), (255, 0, 255), int(1 * (p4_data['height'] / 480)))
            cv2.imwrite(os.path.join(debug_subdir, f"p4_opponent_bbox{res_suffix}.png"), p4_with_box)

        # Determine selection logic result
        selected_from = None
        final_name = ""
        selection_reason = ""

        if p3_data['text'] and p3_data['text'].endswith('1'):
            selected_from = "P3"
            final_name = p3_data['text'][:-1].strip()
            selection_reason = f"P3 text ends with '1': '{p3_data['text']}' -> '{final_name}'"
        elif p4_data['text'] and p4_data['text'].endswith('2'):
            selected_from = "P4"
            final_name = p4_data['text'][:-1].strip()
            selection_reason = f"P4 text ends with '2': '{p4_data['text']}' -> '{final_name}'"
        else:
            selection_reason = f"Neither P3 ('{p3_data['text']}') ends with '1' nor P4 ('{p4_data['text']}') ends with '2'"

        # Save summary text file
        summary_path = os.path.join(debug_subdir, "opponent_name_debug.txt")
        with open(summary_path, 'w') as f:
            f.write("=== OPPONENT NAME DETECTION DEBUG ===\n\n")
            f.write(f"Team 1 Won: {team1_won}\n")
            f.write(f"Y Offset Applied: {16 * (480 / 720) if team1_won else 0:.1f} pixels (at 480p)\n\n")
            f.write("--- Preprocessing Pipeline ---\n")
            f.write("  1. Convert to grayscale\n")
            f.write("  2. Upscale 2x with cubic interpolation\n")
            f.write("  3. Bilateral filter (d=9, sigmaColor=75, sigmaSpace=75)\n")
            f.write("  4. Otsu thresholding\n")
            f.write("  5. Auto-invert if mostly black (for black text on white)\n")
            f.write("  6. Morphological opening (2x2) - remove noise\n")
            f.write("  7. Morphological closing (2x2) - fill gaps\n\n")
            f.write("--- P3 Extraction ---\n")
            f.write(f"  Image: {p3_image_path}\n")
            f.write(f"  Resolution: {p3_data['height']}p (full_res={p3_data['full_res']})\n")
            f.write(f"  Bounding Box: {p3_data['bbox']}\n")
            f.write(f"  Raw OCR Output: '{p3_data['raw_text']}'\n")
            f.write(f"  Cleaned Text: '{p3_data['text']}'\n")
            f.write(f"  Ends with '1': {p3_data['text'].endswith('1') if p3_data['text'] else False}\n\n")
            f.write("--- P4 Extraction ---\n")
            f.write(f"  Image: {p4_image_path}\n")
            f.write(f"  Resolution: {p4_data['height']}p (full_res={p4_data['full_res']})\n")
            f.write(f"  Bounding Box: {p4_data['bbox']}\n")
            f.write(f"  Raw OCR Output: '{p4_data['raw_text']}'\n")
            f.write(f"  Cleaned Text: '{p4_data['text']}'\n")
            f.write(f"  Ends with '2': {p4_data['text'].endswith('2') if p4_data['text'] else False}\n\n")
            f.write("--- Selection Logic ---\n")
            f.write(f"  {selection_reason}\n")
            f.write(f"  Selected From: {selected_from or 'NONE'}\n")
            f.write(f"  Final Name: '{final_name}'\n")

    # Apply selection logic
    # If P3's text ends in "1", use it (minus the "1")
    if p3_data['text'] and p3_data['text'].endswith('1'):
        return p3_data['text'][:-1].strip()

    # Otherwise, if P4's text ends in "2", use it (minus the "2")
    if p4_data['text'] and p4_data['text'].endswith('2'):
        return p4_data['text'][:-1].strip()

    # Otherwise, return empty string
    return ""


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
            kos[player_key] = 0
            falls[player_key] = 0
            damage[player_key] = 0
            continue

        player_path = str(player_files[0])
        kos[player_key] = count_kos(player_path, player_num, save_debug_images=True)
        falls[player_key] = count_falls(player_path, player_num, save_debug_images=True)
        damage[player_key] = extract_damage(player_path, player_num, save_debug_images=True) or 0

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

    # Extract opponent name from P3 and P4 result screens
    opponent = ""
    p3_files = list(Path(game_dir).glob("p3_*.png"))
    p4_files = list(Path(game_dir).glob("p4_*.png"))
    # Filter out full resolution files for the 480p path lookup
    p3_files = [f for f in p3_files if '_full.png' not in str(f)]
    p4_files = [f for f in p4_files if '_full.png' not in str(f)]
    if p3_files and p4_files:
        opponent = extract_opponent_name(str(p3_files[0]), str(p4_files[0]), team1_won=team1_won, save_debug_images=True) or ""

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

    print(f"  P1: {result['p1_character']} - KOs: {result['p1_kos']}, Falls: {result['p1_falls']}, Damage: {result['p1_damage']}%")
    print(f"  P2: {result['p2_character']} - KOs: {result['p2_kos']}, Falls: {result['p2_falls']}, Damage: {result['p2_damage']}%")
    print(f"  P3: {result['p3_character']} - KOs: {result['p3_kos']}, Falls: {result['p3_falls']}, Damage: {result['p3_damage']}%")
    print(f"  P4: {result['p4_character']} - KOs: {result['p4_kos']}, Falls: {result['p4_falls']}, Damage: {result['p4_damage']}%")

    team1_falls = result['p1_falls'] + result['p2_falls']
    team2_falls = result['p3_falls'] + result['p4_falls']
    print(f"  Team 1 Falls: {team1_falls}, Team 2 Falls: {team2_falls}")
    print(f"  Win: {result['win']}")
    print(f"  Opponent: {result['opponent']}")


def save_result_to_csv(result: dict, csv_path: str):
    """Append a single game result to CSV file."""
    file_exists = os.path.exists(csv_path)

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


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
            writer.writerow(result)

    print(f"\nResults saved to: {csv_path}")
