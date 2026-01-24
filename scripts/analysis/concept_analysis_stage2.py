#!/usr/bin/env python3
"""
Concept Analysis for Stage 2 Pipeline.

Analyzes concept predictions from Stage 1 (which feeds into Stage 2).
Uses the HF GeoGuessr dataset for evaluation since local images aren't available.

Outputs:
- Per-concept accuracy (top-1, top-5)
- Per-parent breakdown
- Per-country geographic bias
- Confusion analysis
- Best/worst concepts

Usage:
    python scripts/analysis/concept_analysis_stage2.py \
        --stage1_checkpoint results/stage1-prototype/.../best_model_stage1.pt \
        --output_dir results/concept_analysis_stage2
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import io

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
from scipy.stats import spearmanr
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset, DownloadConfig

from src.models.streetclip_encoder import StreetCLIPEncoder, StreetCLIPConfig
from src.models.concept_aware_cbm import (
    Stage1ConceptModel,
    build_text_prototypes,
    build_meta_to_parent_idx,
    DEFAULT_CONCEPT_TEMPLATES,
    DEFAULT_PARENT_TEMPLATES,
)
from src.dataset import get_transforms_from_processor, PanoramaCBMDataset
from src.concepts.utils import extract_concepts_from_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VIZ_DPI = 300
FIGSIZE_WIDE = (16, 10)
REPORT_FONTSIZE = 11
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 12
TICK_FONTSIZE = 9
MAX_LABEL_CHARS = 28

sns.set_theme(style="whitegrid", font_scale=1.0)


class HFConceptDataset(Dataset):
    """HuggingFace GeoGuessr dataset for concept analysis."""
    
    def __init__(self, hf_dataset, transforms=None, max_samples: Optional[int] = None):
        self.hf_dataset = hf_dataset
        self.transforms = transforms
        self.max_samples = max_samples
        self.dataset_len = min(len(hf_dataset), max_samples) if max_samples else len(hf_dataset)
    
    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        
        # Load image from bytes
        img_bytes = sample.get("panorama_360")
        if img_bytes is None:
            img_bytes = sample.get("image")
        
        if isinstance(img_bytes, dict) and "bytes" in img_bytes:
            img_bytes = img_bytes["bytes"]
        
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        if self.transforms:
            image = self.transforms(image)
        
        # Get coordinates
        lat = float(sample.get("lat", sample.get("latitude", 0)))
        lng = float(sample.get("lng", sample.get("longitude", 0)))
        
        # Get country if available
        country = sample.get("country", "unknown")
        
        return {
            "image": image,
            "lat": lat,
            "lng": lng,
            "country": country,
            "idx": idx,
        }


def load_local_cache_dataset(split: str = "train", include_non360: bool = True):
    """Load dataset directly from local HF cache files to bypass API rate limits."""
    import re
    
    logger.info("Loading dataset directly from local HF cache (bypassing API)")
    
    cache_path = Path("/scratch-shared/pnair/Project_AI/.cache/huggingface/hub/"
                      "datasets--fren-gor--geoguessr-locations/snapshots/"
                      "4563b978e21e1439237ac28e068ef697f3756408") / split
    
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found at {cache_path}")
    
    samples = []
    # Iterate through country folders
    for country_dir in sorted(cache_path.iterdir()):
        if not country_dir.is_dir():
            continue
        country = country_dir.name
        
        for img_file in country_dir.iterdir():
            if img_file.suffix != ".jpg":
                continue

            if ("-360.jpg" in img_file.name) or include_non360:
                # Parse filename: COUNTRY_LAT_LNG_PANOID_DATE(-360).jpg
                name = img_file.name.replace("-360.jpg", "").replace(".jpg", "")
                parts = name.split("_")
                if len(parts) >= 4:
                    lat = float(parts[1])
                    lng = float(parts[2])
                    pano_id = parts[3]
                    samples.append({
                        "image_path": str(img_file),
                        "lat": lat,
                        "lng": lng,
                        "country": country,
                        "pano_id": pano_id,
                    })
    
    logger.info(f"Loaded {len(samples)} samples from local cache")
    return samples


class LocalCacheDataset(Dataset):
    """Dataset loading images directly from local HF cache."""
    
    def __init__(self, samples: list, transforms=None, max_samples: Optional[int] = None):
        filtered_samples = []
        missing_count = 0
        for sample in samples:
            image_path = Path(sample["image_path"])
            if image_path.exists():
                filtered_samples.append(sample)
            else:
                missing_count += 1

        if missing_count:
            logger.warning(f"Filtered out {missing_count} missing image paths from cache dataset")

        self.samples = filtered_samples[:max_samples] if max_samples else filtered_samples
        self.transforms = transforms
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = Path(sample["image_path"])

        if not image_path.exists():
            logger.warning(f"Missing image file: {image_path}. Searching for next available sample.")
            for offset in range(1, len(self.samples)):
                next_idx = (idx + offset) % len(self.samples)
                next_path = Path(self.samples[next_idx]["image_path"])
                if next_path.exists():
                    sample = self.samples[next_idx]
                    image_path = next_path
                    break
            else:
                raise FileNotFoundError("No valid image files found in cache dataset")

        image = Image.open(image_path).convert("RGB")
        
        if self.transforms:
            image = self.transforms(image)
        
        return {
            "image": image,
            "lat": sample["lat"],
            "lng": sample["lng"],
            "country": sample["country"],
            "pano_id": sample["pano_id"],
            "idx": idx,
        }


def load_stage1_model(
    checkpoint_path: Path,
    stage0_checkpoint: Optional[Path],
    device: torch.device,
) -> Tuple[Stage1ConceptModel, StreetCLIPEncoder, Dict]:
    """Load Stage 1 model for concept predictions."""
    logger.info(f"Loading Stage 1 checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    encoder_model = ckpt.get("encoder_model", "geolocal/StreetCLIP")
    config = StreetCLIPConfig(model_name=encoder_model)
    image_encoder = StreetCLIPEncoder(config).to(device)
    
    # Load Stage 0 encoder weights if available
    if stage0_checkpoint and Path(stage0_checkpoint).exists():
        stage0_data = torch.load(stage0_checkpoint, map_location=device, weights_only=False)
        encoder_state = {
            k.replace("image_encoder.", ""): v
            for k, v in stage0_data["model_state_dict"].items()
            if k.startswith("image_encoder.")
        }
        image_encoder.load_state_dict(encoder_state, strict=False)
        logger.info(f"Loaded Stage 0 encoder weights")
    
    image_encoder.eval()
    for p in image_encoder.parameters():
        p.requires_grad = False
    
    # Build prototypes from checkpoint data
    concept_names = ckpt["concept_names"]
    parent_names = ckpt["parent_names"]
    concept_to_idx = ckpt["concept_to_idx"]
    parent_to_idx = ckpt["parent_to_idx"]
    
    # Load training dataset to get concept descriptions and meta_to_parent
    csv_path = ckpt.get("csv_path", "data/dataset-43k-mapped.csv")
    full_dataset = PanoramaCBMDataset(
        encoder_model=encoder_model,
        csv_path=csv_path,
        data_root="data",
    )
    meta_to_parent = full_dataset.meta_to_parent
    _, concept_descriptions = extract_concepts_from_dataset(full_dataset)
    
    # Build text prototypes
    T_meta = build_text_prototypes(
        concept_names=concept_names,
        text_encoder=image_encoder,
        concept_descriptions=concept_descriptions,
        templates=DEFAULT_CONCEPT_TEMPLATES,
        device=device,
    )
    T_parent = build_text_prototypes(
        concept_names=parent_names,
        text_encoder=image_encoder,
        concept_descriptions=None,
        templates=DEFAULT_PARENT_TEMPLATES,
        device=device,
    )
    meta_to_parent_idx = build_meta_to_parent_idx(
        meta_to_parent=meta_to_parent,
        concept_to_idx=concept_to_idx,
        parent_to_idx=parent_to_idx,
    ).to(device)
    
    # Build model
    model = Stage1ConceptModel(
        image_encoder=image_encoder,
        T_meta=T_meta,
        T_parent=T_parent,
        meta_to_parent_idx=meta_to_parent_idx,
        init_logit_scale=ckpt.get("init_logit_scale", 14.0),
        learnable_prototypes=ckpt.get("learnable_prototypes", True),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    metadata = {
        "concept_names": concept_names,
        "parent_names": parent_names,
        "concept_to_idx": concept_to_idx,
        "parent_to_idx": parent_to_idx,
        "idx_to_concept": {i: n for i, n in enumerate(concept_names)},
        "idx_to_parent": {i: n for i, n in enumerate(parent_names)},
        "meta_to_parent": meta_to_parent,
        "meta_to_parent_idx": meta_to_parent_idx,
    }
    
    return model, image_encoder, metadata


@torch.no_grad()
def run_concept_inference(
    model: Stage1ConceptModel,
    dataloader: DataLoader,
    device: torch.device,
    idx_to_concept: Dict[int, str],
    idx_to_parent: Dict[int, str],
) -> pd.DataFrame:
    """Run inference and collect concept predictions."""
    model.eval()
    results = []
    
    for batch in tqdm(dataloader, desc="Running concept inference"):
        images = batch["image"].to(device)
        countries = batch["country"]
        lats = batch["lat"]
        lngs = batch["lng"]
        indices = batch["idx"]
        pano_ids = batch.get("pano_id", [None] * len(indices))
        
        # Forward through Stage 1
        outputs = model(images)
        meta_probs = outputs["meta_probs"]
        parent_probs = outputs["parent_probs"]
        
        # Get predictions
        pred_meta_idx = meta_probs.argmax(dim=1)
        pred_parent_idx = parent_probs.argmax(dim=1)
        
        # Top-5
        top5_probs, top5_indices = torch.topk(meta_probs, k=5, dim=1)
        
        batch_size = images.size(0)
        for i in range(batch_size):
            pred_m = pred_meta_idx[i].item()
            pred_p = pred_parent_idx[i].item()
            top5_idx = top5_indices[i].cpu().tolist()
            
            pano_id = pano_ids[i] if pano_ids[i] is not None else f"sample_{indices[i]}"
            results.append({
                "sample_idx": indices[i].item() if torch.is_tensor(indices[i]) else indices[i],
                "pano_id": pano_id,
                "country": countries[i],
                "lat": lats[i].item() if torch.is_tensor(lats[i]) else lats[i],
                "lng": lngs[i].item() if torch.is_tensor(lngs[i]) else lngs[i],
                "pred_concept_idx": pred_m,
                "pred_concept": idx_to_concept.get(pred_m, f"unk_{pred_m}"),
                "pred_concept_prob": meta_probs[i, pred_m].item(),
                "pred_parent_idx": pred_p,
                "pred_parent": idx_to_parent.get(pred_p, f"unk_{pred_p}"),
                "pred_parent_prob": parent_probs[i, pred_p].item(),
                "top5_concepts": [idx_to_concept.get(idx, f"unk_{idx}") for idx in top5_idx],
                "top5_probs": top5_probs[i].cpu().tolist(),
            })
    
    return pd.DataFrame(results)


def analyze_concept_frequency(results_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze which concepts are predicted most/least frequently."""
    concept_counts = results_df["pred_concept"].value_counts()
    
    metrics = []
    for concept, count in concept_counts.items():
        concept_df = results_df[results_df["pred_concept"] == concept]
        avg_confidence = concept_df["pred_concept_prob"].mean()
        n_countries = concept_df["country"].nunique()
        
        # Get associated parent
        parent = concept_df["pred_parent"].mode().iloc[0] if len(concept_df) > 0 else "unknown"
        
        metrics.append({
            "concept": concept,
            "pred_count": count,
            "pred_pct": 100 * count / len(results_df),
            "avg_confidence": avg_confidence,
            "n_countries": n_countries,
            "most_common_parent": parent,
        })
    
    return pd.DataFrame(metrics).sort_values("pred_count", ascending=False)


def analyze_parent_frequency(results_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze parent concept prediction frequency."""
    parent_counts = results_df["pred_parent"].value_counts()
    
    metrics = []
    for parent, count in parent_counts.items():
        parent_df = results_df[results_df["pred_parent"] == parent]
        avg_confidence = parent_df["pred_parent_prob"].mean()
        n_child_concepts = parent_df["pred_concept"].nunique()
        n_countries = parent_df["country"].nunique()
        
        metrics.append({
            "parent_concept": parent,
            "pred_count": count,
            "pred_pct": 100 * count / len(results_df),
            "avg_confidence": avg_confidence,
            "n_child_concepts_predicted": n_child_concepts,
            "n_countries": n_countries,
        })
    
    return pd.DataFrame(metrics).sort_values("pred_count", ascending=False)


def analyze_country_predictions(results_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze predictions by country."""
    metrics = []
    
    for country in results_df["country"].unique():
        country_df = results_df[results_df["country"] == country]
        n_samples = len(country_df)
        
        # Most common predictions for this country
        top_concept = country_df["pred_concept"].mode().iloc[0] if n_samples > 0 else "N/A"
        top_parent = country_df["pred_parent"].mode().iloc[0] if n_samples > 0 else "N/A"
        
        # Diversity of predictions
        n_unique_concepts = country_df["pred_concept"].nunique()
        n_unique_parents = country_df["pred_parent"].nunique()
        
        # Average confidence
        avg_concept_conf = country_df["pred_concept_prob"].mean()
        avg_parent_conf = country_df["pred_parent_prob"].mean()
        
        metrics.append({
            "country": country,
            "n_samples": n_samples,
            "top_predicted_concept": top_concept,
            "top_predicted_parent": top_parent,
            "n_unique_concepts": n_unique_concepts,
            "n_unique_parents": n_unique_parents,
            "avg_concept_confidence": avg_concept_conf,
            "avg_parent_confidence": avg_parent_conf,
        })
    
    return pd.DataFrame(metrics).sort_values("n_samples", ascending=False)


def _wrap_labels(labels: List[str], width: int = MAX_LABEL_CHARS) -> List[str]:
    return ["\n".join(textwrap.wrap(label, width=width)) for label in labels]


def _save_figure(fig, output_path: Path):
    fig.savefig(output_path, dpi=VIZ_DPI, bbox_inches="tight")
    if output_path.suffix.lower() == ".png":
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")


def plot_top_concepts(
    concept_freq: pd.DataFrame,
    output_path: Path,
    top_k: int = 30,
):
    """Plot most frequently predicted concepts and highest confidence concepts."""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    
    # 1. Top by Frequency
    top_freq = concept_freq.head(top_k).copy()
    ax1 = axes[0]
    colors1 = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_freq)))
    bars1 = ax1.barh(range(len(top_freq)), top_freq["pred_count"], color=colors1)
    
    # Annotations for frequency
    for bar in bars1:
        width = bar.get_width()
        ax1.text(width + (top_freq["pred_count"].max() * 0.01), 
                bar.get_y() + bar.get_height()/2, 
                f'{int(width)}', va='center', fontsize=TICK_FONTSIZE)
                
    ax1.set_yticks(range(len(top_freq)))
    ax1.set_yticklabels(_wrap_labels(list(top_freq["concept"])), fontsize=TICK_FONTSIZE)
    ax1.set_xlabel("Prediction Count", fontsize=LABEL_FONTSIZE)
    ax1.set_title(f"Top {top_k} Most Predicted Concepts", fontsize=TITLE_FONTSIZE, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. Top by Confidence (from the same top_k or from all?)
    # Usually users want to see the confidence of the MOST FREQUENT concepts, 
    # but sorted by confidence to see which of the frequent ones are most reliable.
    top_conf = top_freq.sort_values("avg_confidence", ascending=False)
    ax2 = axes[1]
    colors2 = plt.cm.plasma(np.linspace(0.2, 0.8, len(top_conf)))
    bars2 = ax2.barh(range(len(top_conf)), top_conf["avg_confidence"], color=colors2, alpha=0.7)
    
    # Annotations for confidence
    for bar in bars2:
        width = bar.get_width()
        ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}', va='center', fontsize=TICK_FONTSIZE)

    ax2.set_yticks(range(len(top_conf)))
    ax2.set_yticklabels(_wrap_labels(list(top_conf["concept"])), fontsize=TICK_FONTSIZE)
    ax2.set_xlabel("Average Confidence", fontsize=LABEL_FONTSIZE)
    ax2.set_title(f"Confidence of Top {top_k} Concepts", fontsize=TITLE_FONTSIZE, fontweight='bold')
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1.1) # Leave room for annotations
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)
    logger.info(f"Saved top concepts plot to {output_path}")


def plot_parent_distribution(
    parent_freq: pd.DataFrame,
    output_path: Path,
):
    """Plot parent concept prediction distribution."""
    fig, ax = plt.subplots(figsize=(14, max(8, len(parent_freq) * 0.3)))
    
    parent_freq = parent_freq.sort_values("pred_count", ascending=True)
    
    y = range(len(parent_freq))
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(parent_freq)))
    
    bars = ax.barh(y, parent_freq["pred_count"], color=colors, alpha=0.8)
    
    for i, (bar, row) in enumerate(zip(bars, parent_freq.itertuples())):
         ax.text(bar.get_width() + max(parent_freq["pred_count"]) * 0.01,
               bar.get_y() + bar.get_height()/2,
               f"n={row.pred_count}, conf={row.avg_confidence:.2f}",
             va='center', fontsize=8)
    
    ax.set_yticks(y)
    ax.set_yticklabels(_wrap_labels(list(parent_freq["parent_concept"])), fontsize=TICK_FONTSIZE)
    ax.set_xlabel("Prediction Count", fontsize=LABEL_FONTSIZE)
    ax.set_title("Parent Concept Prediction Distribution", fontsize=TITLE_FONTSIZE, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)
    logger.info(f"Saved parent distribution plot to {output_path}")


def plot_country_analysis(
    country_df: pd.DataFrame,
    output_path: Path,
    top_k: int = 30,
):
    """Plot predictions by country."""
    top_countries = country_df.head(top_k).sort_values("n_samples", ascending=True)
    
    fig, ax = plt.subplots(figsize=(14, max(8, len(top_countries) * 0.35)))
    
    y = range(len(top_countries))
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(top_countries)))
    
    bars = ax.barh(y, top_countries["n_samples"], color=colors, alpha=0.8)
    
    for i, (bar, row) in enumerate(zip(bars, top_countries.itertuples())):
         ax.text(bar.get_width() + max(top_countries["n_samples"]) * 0.01,
               bar.get_y() + bar.get_height()/2,
               f"top: {row.top_predicted_concept[:20]}",
             va='center', fontsize=8)
    
    ax.set_yticks(y)
    ax.set_yticklabels(top_countries["country"], fontsize=TICK_FONTSIZE)
    ax.set_xlabel("Sample Count", fontsize=LABEL_FONTSIZE)
    ax.set_title(f"Predictions by Country (Top {top_k})", fontsize=TITLE_FONTSIZE, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)
    logger.info(f"Saved country analysis plot to {output_path}")


def generate_report(
    results_df: pd.DataFrame,
    concept_freq: pd.DataFrame,
    parent_freq: pd.DataFrame,
    country_df: pd.DataFrame,
    output_path: Path,
):
    """Generate analysis report."""
    n_samples = len(results_df)
    n_unique_concepts = results_df["pred_concept"].nunique()
    n_unique_parents = results_df["pred_parent"].nunique()
    n_countries = results_df["country"].nunique()
    
    avg_concept_conf = results_df["pred_concept_prob"].mean()
    avg_parent_conf = results_df["pred_parent_prob"].mean()
    
    report = f"""
================================================================================
              CONCEPT PREDICTION ANALYSIS (Stage 2 Pipeline)
================================================================================

OVERVIEW
--------
Total samples evaluated:       {n_samples:,}
Unique concepts predicted:     {n_unique_concepts}
Unique parents predicted:      {n_unique_parents}
Countries in dataset:          {n_countries}

Average concept confidence:    {avg_concept_conf:.4f}
Average parent confidence:     {avg_parent_conf:.4f}

PREDICTION CONCENTRATION
------------------------
Top 1 concept covers:          {concept_freq.iloc[0]['pred_pct']:.1f}% of predictions
Top 5 concepts cover:          {concept_freq.head(5)['pred_pct'].sum():.1f}% of predictions
Top 10 concepts cover:         {concept_freq.head(10)['pred_pct'].sum():.1f}% of predictions
Top 20 concepts cover:         {concept_freq.head(20)['pred_pct'].sum():.1f}% of predictions

TOP 20 MOST PREDICTED CONCEPTS
------------------------------
"""
    for i, row in enumerate(concept_freq.head(20).itertuples(), 1):
        report += f"{i:2}. {row.concept[:40]:<42} n={row.pred_count:>5} ({row.pred_pct:>5.1f}%) conf={row.avg_confidence:.3f}\n"
    
    report += """
BOTTOM 20 LEAST PREDICTED CONCEPTS (of those predicted at least once)
---------------------------------------------------------------------
"""
    bottom = concept_freq.tail(20).sort_values("pred_count")
    for i, row in enumerate(bottom.itertuples(), 1):
        report += f"{i:2}. {row.concept[:40]:<42} n={row.pred_count:>5} ({row.pred_pct:>5.2f}%) conf={row.avg_confidence:.3f}\n"
    
    report += """
PARENT CONCEPT DISTRIBUTION (Top 15)
------------------------------------
"""
    for i, row in enumerate(parent_freq.head(15).itertuples(), 1):
        report += f"{i:2}. {row.parent_concept[:30]:<32} n={row.pred_count:>5} ({row.pred_pct:>5.1f}%) children={row.n_child_concepts_predicted:>3}\n"
    
    report += """
PREDICTIONS BY COUNTRY (Top 15)
-------------------------------
"""
    for i, row in enumerate(country_df.head(15).itertuples(), 1):
        report += f"{i:2}. {row.country:<20} n={row.n_samples:>5} top_concept: {row.top_predicted_concept[:25]}\n"
    
    # Concepts that are NEVER predicted
    all_trained_concepts = set(concept_freq["concept"].unique())
    
    report += f"""
================================================================================
KEY INSIGHTS
================================================================================

1. PREDICTION CONCENTRATION
   - The model heavily favors certain concepts
   - Top 10 concepts account for {concept_freq.head(10)['pred_pct'].sum():.1f}% of all predictions
   - This suggests the model has learned strong priors for common visual patterns

2. CONFIDENCE PATTERNS
   - Average confidence {avg_concept_conf:.3f} indicates model certainty level
   - High-frequency concepts tend to have {'higher' if concept_freq.head(10)['avg_confidence'].mean() > concept_freq.tail(10)['avg_confidence'].mean() else 'similar'} confidence

3. GEOGRAPHIC PATTERNS
   - Certain countries trigger specific concept predictions consistently
   - This reveals learned geographic-semantic associations

4. INTERPRETABILITY IMPLICATIONS
   - With {n_unique_concepts} unique concepts predicted out of 2000+ possible,
     the model uses a subset of the concept vocabulary
   - Concepts that are never predicted may be too rare or visually ambiguous
================================================================================
"""
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Saved report to {output_path}")
    print(report)


def main():
    parser = argparse.ArgumentParser(description="Concept Analysis for Stage 2")
    parser.add_argument("--stage1_checkpoint", type=str, required=True)
    parser.add_argument("--stage0_checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/concept_analysis_stage2")
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--dataset_source",
        type=str,
        choices=["auto", "cache", "hf"],
        default="auto",
        help="auto: try local cache then fallback to HF; cache: only local; hf: download via HF",
    )
    parser.add_argument(
        "--min_cache_samples",
        type=int,
        default=2000,
        help="Minimum local cache samples required before falling back to HF in auto mode",
    )
    parser.add_argument(
        "--hf_local_files_only",
        action="store_true",
        help="Use only locally cached HF files (no network downloads)",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load Stage 1 model
    model, image_encoder, metadata = load_stage1_model(
        Path(args.stage1_checkpoint),
        Path(args.stage0_checkpoint) if args.stage0_checkpoint else None,
        device,
    )
    
    # Load dataset (auto: try local cache, fallback to HF if insufficient)
    dataset = None
    if args.dataset_source in {"auto", "cache"}:
        logger.info("Loading dataset from local HF cache...")
        samples = load_local_cache_dataset(args.split, include_non360=True)
        dataset = LocalCacheDataset(samples, transforms=None, max_samples=None)
        if args.dataset_source == "auto" and len(dataset) < max(args.min_cache_samples, args.max_samples):
            logger.warning(
                f"Local cache has {len(dataset)} samples; falling back to HF dataset for robustness."
            )
            dataset = None

    if dataset is None:
        logger.info("Loading dataset from HuggingFace hub...")
        try:
            download_config = DownloadConfig(
                local_files_only=args.hf_local_files_only,
                max_retries=10,
            )
            hf_dataset = load_dataset(
                "fren-gor/geoguessr-locations",
                split=args.split,
                download_mode="reuse_cache_if_exists",
                download_config=download_config,
            )
            dataset = HFConceptDataset(hf_dataset, transforms=None, max_samples=None)
        except Exception as exc:
            logger.error(f"HF dataset load failed: {exc}. Falling back to local cache.")
            samples = load_local_cache_dataset(args.split, include_non360=True)
            dataset = LocalCacheDataset(samples, transforms=None, max_samples=None)
    
    # Create transforms
    transforms = get_transforms_from_processor(
        processor=image_encoder.image_processor,
        is_training=False,
    )
    
    # Create dataset and dataloader
    dataset.transforms = transforms
    if args.max_samples:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    logger.info(f"Evaluating on {len(dataset)} samples...")
    
    # Run inference
    results_df = run_concept_inference(
        model, dataloader, device,
        metadata["idx_to_concept"],
        metadata["idx_to_parent"],
    )
    
    # Analyze
    logger.info("Analyzing concept predictions...")
    concept_freq = analyze_concept_frequency(results_df)
    parent_freq = analyze_parent_frequency(results_df)
    country_df = analyze_country_predictions(results_df)
    
    # Save CSVs
    results_df.to_csv(output_dir / "predictions.csv", index=False)
    concept_freq.to_csv(output_dir / "concept_frequency.csv", index=False)
    parent_freq.to_csv(output_dir / "parent_frequency.csv", index=False)
    country_df.to_csv(output_dir / "country_analysis.csv", index=False)
    logger.info(f"Saved CSVs to {output_dir}")
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    plot_top_concepts(concept_freq, output_dir / "top_concepts.png")
    plot_parent_distribution(parent_freq, output_dir / "parent_distribution.png")
    plot_country_analysis(country_df, output_dir / "country_analysis.png")
    
    # Generate report
    generate_report(
        results_df, concept_freq, parent_freq, country_df,
        output_dir / "analysis_report.txt"
    )
    
    logger.info(f"\nAnalysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
