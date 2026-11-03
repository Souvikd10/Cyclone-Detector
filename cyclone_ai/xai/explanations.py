"""
Human-Readable Explanation Tags — Stage 13.

Rule-based generator that produces meteorologically meaningful
explanation tags from structural, classification, and uncertainty outputs.
"""

import logging

logger = logging.getLogger(__name__)


class ExplanationGenerator:
    """
    Generate human-readable explanation tags from model outputs.

    Maps combinations of structural features, classification results,
    and uncertainty estimates to plain-language tags that help
    meteorologists and end-users understand the model's assessment.

    Usage:
        gen = ExplanationGenerator()
        tags = gen.generate(
            structure={"eye_present": True, "eye_confidence": 0.9, ...},
            classification={"category": "VSCS", "stage": "mature", ...},
            ri={"probability": 0.7, ...},
            uncertainty={"novelty_score": 1.2, ...},
            env_scalars={"WIND_SHEAR_mean": 5.0, ...},
        )
        # → ["well-defined eye", "strong eyewall", "symmetric convection"]
    """

    def generate(
        self,
        structure: dict,
        classification: dict | None = None,
        ri: dict | None = None,
        uncertainty: dict | None = None,
        env_scalars: dict | None = None,
        forecast: dict | None = None,
    ) -> list[str]:
        """
        Generate explanation tags.

        Args:
            structure: Structure head contract dict.
            classification: Classification head contract dict.
            ri: RI contract dict.
            uncertainty: Uncertainty contract dict.
            env_scalars: Environmental scalar features.
            forecast: Forecast contract dict.

        Returns:
            List of human-readable explanation tag strings.
        """
        tags = []

        # ── Eye-related tags ──
        if structure.get("eye_present", False):
            eye_conf = structure.get("eye_confidence", 0)
            eye_diam = structure.get("eye_diameter_km", 0)

            if eye_conf > 0.8:
                tags.append("well-defined eye")
            elif eye_conf > 0.5:
                tags.append("developing eye")

            if eye_diam > 0 and eye_diam < 20:
                tags.append("pinhole eye structure")
            elif eye_diam > 60:
                tags.append("large eye diameter")

            # Strong eyewall implied by high eye confidence + high symmetry
            if eye_conf > 0.7 and structure.get("symmetry", 0) > 0.7:
                tags.append("strong eyewall convection")

        # ── Symmetry and circularity ──
        symmetry = structure.get("symmetry", 0.5)
        circularity = structure.get("circularity", 0.5)

        if symmetry > 0.8:
            tags.append("symmetric convection")
        elif symmetry < 0.4:
            tags.append("asymmetric structure")

        if circularity > 0.8:
            tags.append("circular organization")

        # ── Classification-based tags ──
        if classification:
            category = classification.get("category", "")
            stage = classification.get("stage", "")
            pattern = classification.get("pattern", "")

            if category in ("ESCS", "SuCS"):
                tags.append("peak intensity")
            elif category in ("D", "DD"):
                tags.append("developing system")

            if stage == "intensifying":
                tags.append("intensifying trend")
            elif stage == "weakening":
                tags.append("weakening trend")
            elif stage == "forming":
                tags.append("genesis stage")

            if pattern == "CDO":
                tags.append("central dense overcast dominant")
            elif pattern == "annular":
                tags.append("annular hurricane structure")
            elif pattern == "shear":
                tags.append("shear-induced asymmetry")

        # ── RI-related tags ──
        if ri:
            ri_prob = ri.get("probability", 0)
            if ri_prob > 0.7:
                tags.append("rapid intensification likely")
            elif ri_prob > 0.5:
                tags.append("rapid intensification possible")

        # ── Environmental tags ──
        if env_scalars:
            shear = env_scalars.get("WIND_SHEAR_mean", 10)
            sst = env_scalars.get("SST_mean", 28)

            if shear < 5:
                tags.append("low wind shear environment")
            elif shear > 15:
                tags.append("high wind shear environment")

            if sst > 30:
                tags.append("very warm SST")
            elif sst < 26.5:
                tags.append("marginal SST conditions")

        # ── Uncertainty tags ──
        if uncertainty:
            novelty = uncertainty.get("novelty_score", 1.0)
            if novelty > 2.5:
                tags.append("unusual storm characteristics")

            if uncertainty.get("human_review_flag", False):
                tags.append("flagged for expert review")

        # ── Forecast tags ──
        if forecast:
            track = forecast.get("track", [])
            for tp in track:
                if self._is_approaching_land(tp.get("lat", 0), tp.get("lon", 0)):
                    tags.append("approaching landfall")
                    break

        # Ensure at least one tag
        if not tags:
            tags.append("organized convection")

        # Deduplicate while preserving order
        seen = set()
        unique_tags = []
        for tag in tags:
            if tag not in seen:
                seen.add(tag)
                unique_tags.append(tag)

        return unique_tags

    @staticmethod
    def _is_approaching_land(lat: float, lon: float) -> bool:
        """Simplified check if a forecast position is near land."""
        # India east coast
        if 10 <= lat <= 22 and 80 <= lon <= 88:
            return True
        # Bangladesh/Myanmar
        if 18 <= lat <= 25 and 88 <= lon <= 95:
            return True
        return False
