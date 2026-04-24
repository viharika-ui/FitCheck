import argparse
import shutil
import subprocess
from pathlib import Path



def run_schp_inference(
    image_path: Path,
    schp_root: Path,
    checkpoint_path: Path,
    output_dir: Path,
    schp_python: Path,
) -> Path:
    schp_root = schp_root.resolve()
    checkpoint_path = checkpoint_path.resolve()
    schp_python = schp_python.resolve()
    output_dir = output_dir.resolve()

    input_dir = output_dir / "_input"
    parsing_dir = output_dir / "parsing"
    input_dir.mkdir(parents=True, exist_ok=True)
    parsing_dir.mkdir(parents=True, exist_ok=True)

    input_image_path = input_dir / image_path.name
    shutil.copy2(image_path, input_image_path)

    extractor = schp_root / "simple_extractor.py"
    command = [
        str(schp_python),
        str(extractor),
        "--dataset",
        "lip",
        "--model-restore",
        str(checkpoint_path),
        "--gpu",
        "0",
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(parsing_dir),
    ]

    subprocess.run(command, cwd=str(schp_root), check=True)

    parsing_path = parsing_dir / f"{image_path.stem}.png"
    if not parsing_path.exists():
        raise FileNotFoundError(f"Parsing output not found at: {parsing_path}")
    return parsing_path


def segment_person_with_schp(
    person_image: Path,
    schp_root: Path,
    checkpoint_path: Path,
    output_dir: Path,
    schp_python: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    if not person_image.exists():
        raise FileNotFoundError(f"Could not read image: {person_image}")

    return run_schp_inference(person_image, schp_root, checkpoint_path, output_dir, schp_python)


def parse_args():
    parser = argparse.ArgumentParser(description="Run SCHP segmentation and save the parsing mask")
    parser.add_argument("--image", default="test.jpg", help="Path to person image")
    parser.add_argument(
        "--schp-root",
        default="../Self-Correction-Human-Parsing",
        help="Path to Self-Correction-Human-Parsing repo",
    )
    parser.add_argument(
        "--checkpoint",
        default="../Self-Correction-Human-Parsing/checkpoints/exp-schp-201908261155-lip.pth",
        help="Path to pretrained SCHP LIP checkpoint (.pth)",
    )
    parser.add_argument(
        "--schp-python",
        default="../Self-Correction-Human-Parsing/venv/Scripts/python.exe",
        help="Python interpreter used to run SCHP simple_extractor.py",
    )
    parser.add_argument("--output-dir", default="output", help="Directory to store outputs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    person_image = Path(args.image)
    schp_root = Path(args.schp_root)
    checkpoint = Path(args.checkpoint)
    schp_python = Path(args.schp_python)
    output_dir = Path(args.output_dir)

    parsing_path = segment_person_with_schp(person_image, schp_root, checkpoint, output_dir, schp_python)
    print("Segmentation completed:")
    print(f"- parsing_map: {parsing_path.resolve()}")