import base64
import json
import warnings
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

import pyarrow.parquet as pq
from PIL import Image


def extract_images_from_parquet(
    parquet_path: Path, dest_dir: Path, num_images: int, seed: Optional[int] = None
) -> List[Tuple[str, str]]:
    """
    Extract random images and their metadata from a parquet file and save them to disk.

    Args:
        parquet_path: Path to the parquet file
        dest_dir: Directory to save images and metadata
        num_images: Number of images to extract
        columns: Specific columns to include in metadata
        seed: Random seed for sampling

    Returns:
        List of tuples containing (image_path, metadata_path)
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # read parquet file, sample random image rows
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    sample_df = df.sample(n=min(num_images, len(df)), random_state=seed)
    image_col = next((col for col in sample_df.columns if "image" in col.lower()), None)
    metadata_cols = [c for c in sample_df.columns if c != image_col]

    saved_files = []
    parquet_prefix = parquet_path.stem
    for idx, row in sample_df.iterrows():
        try:
            img_data = row[image_col]
            if isinstance(img_data, dict):
                key = next((k for k in img_data if "bytes" in k.lower() or "image" in k.lower()), None)
                img_data = img_data[key]

            try:
                img = Image.open(BytesIO(img_data))
            except Exception:
                img_data = base64.b64decode(img_data)
                img = Image.open(BytesIO(img_data))

            base_filename = f"{parquet_prefix}__image_{idx}"
            image_format = img.format.lower() if img.format else "png"
            img_filename = f"{base_filename}.{image_format}"
            img_path = dest_dir / img_filename
            img.save(img_path)

            metadata = {
                "source_parquet": str(parquet_path),
                "original_index": str(idx),
                "image_format": image_format,
                "image_size": img.size,
                "image_mode": img.mode,
            }

            for col in metadata_cols:
                # Convert any non-serializable types to strings
                try:
                    json.dumps({col: row[col]})
                    metadata[col] = row[col]
                except (TypeError, OverflowError):
                    metadata[col] = str(row[col])

            metadata_filename = f"{base_filename}.json"
            metadata_path = dest_dir / metadata_filename
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            saved_files.append(str(img_path))

        except Exception as e:
            warnings.warn(f"Failed to extract/save image {idx}: {e}")
            continue

    return saved_files
