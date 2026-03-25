"""
CLI entry: sample rows from a projected-metadata CSV and save figure panels.

Delegates to utils.visuresults.visualize_plantnet_samples; override paths via main(...).
"""
from pathlib import Path
from typing import Optional, Union

from utils.visuresults import visualize_plantnet_samples

__all__ = ["main", "visualize_plantnet_samples"]


def main(
    csv_path: Union[str, Path] = "/data/data2/plant_street_view/images_kika/images_projected_metadata_kika.csv",
    projected_views_dir: Union[str, Path] = "/data/data2/plant_street_view/images_kika/projected_views",
    output_dir: Optional[Union[str, Path]] = None,
    num_samples: int = 100,
    crop_name: Optional[str] = "Gossypium",
    score_threshold: float = 0.02,
    depth: bool = True,
    target_id: Optional[Union[str, int]] = None,
    seed: Optional[int] = None,
) -> None:
    """
    Build visualizations from a projected-views metadata CSV.

    If ``output_dir`` is None, writes under ``<plant_street_view>/results/visualization_random_samples``.
    """
    _root = Path(__file__).resolve().parent
    if output_dir is None:
        output_dir = _root / "results/visualization_random_samples"
    
    
    visualize_plantnet_samples(
        csv_path=csv_path,
        images_dir=projected_views_dir,
        output_dir=output_dir,
        num_samples=num_samples,
        seed=seed,
        crop_name=crop_name,
        score_threshold=score_threshold,
        depth=depth,
        target_id=target_id,
    )


if __name__ == "__main__":
    main()
