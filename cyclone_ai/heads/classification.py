"""
Current-State Classification Head — Stage 8.

Classifies the cyclone's current intensity category (IMD scale),
developmental stage, and structural cloud pattern.

Takes concatenation of global features (Stage 4) + temporal features (Stage 7).
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ClassificationHead(nn.Module):
    """
    Current-State Classification Head (Stage 8).

    Three parallel classifiers:
    - Category: IMD intensity scale (D, DD, CS, SCS, VSCS, ESCS, SuCS)
    - Stage: developmental stage (forming, intensifying, mature, weakening, dissipating)
    - Pattern: structural cloud pattern (CDO, banding, shear, annular, eye_pattern, etc.)

    Input: concatenation of global [B, 768] + temporal [B, 512] = [B, 1280].

    Args:
        global_dim: Global feature dimension from encoder.
        temporal_dim: Temporal feature dimension from temporal block.
        categories: List of IMD category labels.
        stages: List of developmental stage labels.
        patterns: List of structural pattern labels.
        label_smoothing: Label smoothing factor for cross-entropy.
    """

    # Default label sets
    DEFAULT_CATEGORIES = ["D", "DD", "CS", "SCS", "VSCS", "ESCS", "SuCS"]
    DEFAULT_STAGES = ["forming", "intensifying", "mature", "weakening", "dissipating"]
    DEFAULT_PATTERNS = ["CDO", "banding", "shear", "annular", "eye_pattern", "ragged_eye", "pinhole_eye"]

    def __init__(
        self,
        global_dim: int = 768,
        temporal_dim: int = 512,
        categories: list[str] | None = None,
        stages: list[str] | None = None,
        patterns: list[str] | None = None,
        label_smoothing: float = 0.1,
    ):
        super().__init__()

        self.categories = categories or self.DEFAULT_CATEGORIES
        self.stages = stages or self.DEFAULT_STAGES
        self.patterns = patterns or self.DEFAULT_PATTERNS
        self.label_smoothing = label_smoothing

        input_dim = global_dim + temporal_dim

        # Category classifier
        self.category_head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, len(self.categories)),
        )

        # Stage classifier
        self.stage_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, len(self.stages)),
        )

        # Pattern classifier
        self.pattern_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, len(self.patterns)),
        )

    def forward(
        self,
        global_features: torch.Tensor,
        temporal_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            global_features: [B, global_dim] from SharedEncoder.
            temporal_features: [B, temporal_dim] from TemporalBlock.

        Returns:
            Dict with:
                'category_logits': [B, num_categories]
                'category_probs': [B, num_categories] softmax probabilities
                'stage_logits': [B, num_stages]
                'stage_probs': [B, num_stages]
                'pattern_logits': [B, num_patterns]
                'pattern_probs': [B, num_patterns]
        """
        combined = torch.cat([global_features, temporal_features], dim=-1)

        cat_logits = self.category_head(combined)
        stage_logits = self.stage_head(combined)
        pattern_logits = self.pattern_head(combined)

        return {
            "category_logits": cat_logits,
            "category_probs": F.softmax(cat_logits, dim=-1),
            "stage_logits": stage_logits,
            "stage_probs": F.softmax(stage_logits, dim=-1),
            "pattern_logits": pattern_logits,
            "pattern_probs": F.softmax(pattern_logits, dim=-1),
        }

    def compute_loss(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Compute classification losses with label smoothing.

        Args:
            predictions: Output from forward().
            targets: Dict with 'category' [B], 'stage' [B], 'pattern' [B]
                     as integer class indices.
        """
        losses = {}

        losses["category"] = F.cross_entropy(
            predictions["category_logits"],
            targets["category"],
            label_smoothing=self.label_smoothing,
        )

        losses["stage"] = F.cross_entropy(
            predictions["stage_logits"],
            targets["stage"],
            label_smoothing=self.label_smoothing,
        )

        losses["pattern"] = F.cross_entropy(
            predictions["pattern_logits"],
            targets["pattern"],
            label_smoothing=self.label_smoothing,
        )

        losses["total"] = losses["category"] + losses["stage"] + losses["pattern"]
        return losses

    def to_contract(self, predictions: dict[str, torch.Tensor]) -> dict:
        """Convert to contract-compliant current_state dict."""
        cat_idx = int(predictions["category_probs"][0].argmax().cpu())
        stage_idx = int(predictions["stage_probs"][0].argmax().cpu())
        pattern_idx = int(predictions["pattern_probs"][0].argmax().cpu())

        return {
            "category": self.categories[cat_idx],
            "stage": self.stages[stage_idx],
            "pattern": self.patterns[pattern_idx],
        }
