#!/usr/bin/env python3
"""
SmashStats - Analyze Super Smash Bros video recordings.
Finds key frames and extracts match statistics.
"""

import argparse
import os
from pathlib import Path

from find_frames import find_frames as find_frames_func
from analyze_game import (
    analyze_game_dir,
    print_game_result,
    save_result_to_csv
)


def process_video(video_path: str, template_dir: str, output_dir: str,
                  threshold: float = 0.95, csv_path: str = "data.csv",
                  debug: bool = False):
    """
    Process a video to find frames and analyze match results.

    Args:
        video_path: Path to the input video file
        template_dir: Directory containing template images
        output_dir: Directory to save matched frames
        threshold: Template matching threshold (0-1)
        csv_path: Path to CSV file for saving results
        debug: If True, save debug CSV and best match frames

    Returns:
        list: List of game results
    """
    print("=" * 60)
    print("SMASH STATS - Video Analysis")
    print("=" * 60)
    print()

    # Track results as games are analyzed
    all_results = []

    def on_game_complete(game_dir: str, game_num: int):
        """Callback to analyze each game immediately after capture."""
        print()
        print("-" * 60)
        print(f"Analyzing Game {game_num}...")
        print("-" * 60)

        result = analyze_game_dir(game_dir, game_num=game_num)
        if result:
            print_game_result(result, game_num=game_num)
            all_results.append(result)

            # Save this game's result to CSV
            save_result_to_csv(result, csv_path)
            print(f"Result saved to: {csv_path}")

        print()
        print("-" * 60)
        print("Continuing video analysis...")
        print("-" * 60)

    # Find frames and analyze each game as it's found
    print("Finding key frames and analyzing games...")
    print("-" * 60)

    game_directories = find_frames_func(
        video_path=video_path,
        template_dir=template_dir,
        output_dir=output_dir,
        threshold=threshold,
        debug=debug,
        on_game_complete=on_game_complete
    )

    # Print final summary
    print()
    print("=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Games analyzed: {len(all_results)}")

    if all_results:
        wins = sum(1 for r in all_results if r['win'] == 'Yes')
        losses = len(all_results) - wins
        print(f"Record: {wins}W - {losses}L")
        print(f"Results saved to: {csv_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Super Smash Bros video recordings"
    )
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument(
        "-t", "--templates",
        default="templates",
        help="Directory containing template images (default: templates)"
    )
    parser.add_argument(
        "-o", "--output",
        default="game_captures",
        help="Output directory for matched frames (default: game_captures)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Template match threshold 0-1, higher = stricter (default: 0.95)"
    )
    parser.add_argument(
        "--csv",
        default="data.csv",
        help="Path to CSV file for saving results (default: data.csv)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode: save CSV with all confidence scores and best match frames"
    )

    args = parser.parse_args()

    # Resolve template path
    script_dir = Path(__file__).parent
    template_dir = args.templates
    if not os.path.isabs(template_dir):
        template_dir = script_dir / template_dir

    results = process_video(
        video_path=args.video,
        template_dir=str(template_dir),
        output_dir=args.output,
        threshold=args.threshold,
        csv_path=args.csv,
        debug=args.debug
    )

    return results


if __name__ == "__main__":
    main()
