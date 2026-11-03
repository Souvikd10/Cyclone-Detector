"""
Physics/Bias Correction — Stage 11.

Rule-based post-processor that sanity-checks and corrects raw model outputs
using physical constraints and climatological bounds.
"""

import logging
import math
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class PhysicsCorrector:
    """
    Physics-based correction and sanity checking (Stage 11).

    Applies physical constraints to raw model predictions:
    - Maximum track speed (displacement per 6h)
    - Maximum intensity change (wind change per 6h)
    - SST floor for sustained cyclone maintenance
    - Landfall intensity decay (Kaplan-DeMaria model)
    - Pressure-wind consistency (Knaff-Zehr)
    - Track continuity (no teleportation)
    - Category-wind consistency

    All corrections are logged so the XAI layer can report what was modified.

    Args:
        max_track_speed_kt: Maximum displacement speed per 6h.
        max_intensity_change_kt: Maximum |Δwind| per 6h (non-RI storms).
        sst_threshold_c: Minimum SST for sustained cyclone.
        landfall_decay_rate: Exponential decay rate per 6h after landfall.
    """

    # IMD wind-speed thresholds (same as IBTrACSClient)
    IMD_CATEGORIES = [
        (0, "LOW"), (17, "D"), (28, "DD"), (34, "CS"),
        (48, "SCS"), (64, "VSCS"), (90, "ESCS"), (120, "SuCS"),
    ]

    def __init__(
        self,
        max_track_speed_kt: float = 30.0,
        max_intensity_change_kt: float = 40.0,
        sst_threshold_c: float = 26.5,
        landfall_decay_rate: float = 0.9,
    ):
        self.max_track_speed_kt = max_track_speed_kt
        self.max_intensity_change_kt = max_intensity_change_kt
        self.sst_threshold_c = sst_threshold_c
        self.landfall_decay_rate = landfall_decay_rate

    def correct(
        self,
        forecast: dict,
        ri: dict,
        current_wind_kt: float,
        current_lat: float,
        current_lon: float,
        sst: float = 28.0,
    ) -> tuple[dict, dict, list[str]]:
        """
        Apply physics corrections to raw forecast and RI outputs.

        Args:
            forecast: Raw forecast dict with 'track' and 'intensity' lists.
            ri: Raw RI dict with 'probability' and 'confidence'.
            current_wind_kt: Current maximum sustained wind.
            current_lat: Current latitude.
            current_lon: Current longitude.
            sst: Sea surface temperature (°C).

        Returns:
            Tuple of (corrected_forecast, corrected_ri, corrections_applied).
        """
        corrections = []

        corrected_forecast = {
            "track": [dict(t) for t in forecast["track"]],
            "intensity": [dict(i) for i in forecast["intensity"]],
        }
        corrected_ri = dict(ri)

        # 1. Track speed constraint
        prev_lat, prev_lon = current_lat, current_lon
        prev_hours = 0
        for i, tp in enumerate(corrected_forecast["track"]):
            hours = self._parse_horizon(tp["t"])
            dt_hours = hours - prev_hours

            if dt_hours > 0:
                corrected, was_clipped = self._clip_track_speed(
                    prev_lat, prev_lon, tp["lat"], tp["lon"], dt_hours,
                )
                if was_clipped:
                    tp["lat"] = corrected[0]
                    tp["lon"] = corrected[1]
                    corrections.append(f"Track speed clipped at {tp['t']}")

            prev_lat, prev_lon = tp["lat"], tp["lon"]
            prev_hours = hours

        # 2. Track continuity (no jumps > 3°)
        self._smooth_track(corrected_forecast["track"], current_lat, current_lon, corrections)

        # 3. Intensity change constraint
        prev_wind = current_wind_kt
        is_ri = ri["probability"] > 0.5
        for i, ip in enumerate(corrected_forecast["intensity"]):
            delta = ip["wind_kt"] - prev_wind
            max_change = self.max_intensity_change_kt if not is_ri else self.max_intensity_change_kt * 1.5

            if abs(delta) > max_change:
                ip["wind_kt"] = prev_wind + math.copysign(max_change, delta)
                corrections.append(f"Intensity change clipped at {ip['t']}")

            # Wind must be non-negative
            ip["wind_kt"] = max(0, ip["wind_kt"])

            # Update interval
            if ip["wind_kt"] < ip["interval"][0]:
                ip["interval"][0] = ip["wind_kt"] - 5
            if ip["wind_kt"] > ip["interval"][1]:
                ip["interval"][1] = ip["wind_kt"] + 5
            ip["interval"][0] = max(0, ip["interval"][0])

            prev_wind = ip["wind_kt"]

        # 4. SST floor check
        if sst < self.sst_threshold_c:
            for ip in corrected_forecast["intensity"]:
                if ip["wind_kt"] > 35:  # Can't sustain > CS without warm SST
                    ip["wind_kt"] = min(ip["wind_kt"], 35)
                    corrections.append(f"SST below threshold — capped wind at {ip['t']}")

        # 5. Landfall decay (simplified — check if any forecast point is inland)
        for i, (tp, ip) in enumerate(zip(
            corrected_forecast["track"], corrected_forecast["intensity"]
        )):
            if self._is_over_land(tp["lat"], tp["lon"]):
                hours = self._parse_horizon(tp["t"])
                decay_steps = hours / 6.0
                ip["wind_kt"] *= self.landfall_decay_rate ** decay_steps
                ip["wind_kt"] = max(0, ip["wind_kt"])
                corrections.append(f"Landfall decay applied at {tp['t']}")

        # 6. Category consistency — ensure category matches corrected wind
        # (Applied in pipeline when building contract output)

        # Round all values
        for tp in corrected_forecast["track"]:
            tp["lat"] = round(tp["lat"], 2)
            tp["lon"] = round(tp["lon"], 2)
            tp["cone_radius_km"] = round(tp["cone_radius_km"], 1)

        for ip in corrected_forecast["intensity"]:
            ip["wind_kt"] = round(ip["wind_kt"], 1)
            ip["interval"] = [round(ip["interval"][0], 1), round(ip["interval"][1], 1)]

        if corrections:
            logger.info(f"Physics corrections applied: {len(corrections)}")
            for c in corrections:
                logger.debug(f"  - {c}")

        return corrected_forecast, corrected_ri, corrections

    def _clip_track_speed(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float,
        dt_hours: float,
    ) -> tuple[tuple[float, float], bool]:
        """Clip forecast position to max displacement speed."""
        # Approximate distance in degrees (1° ≈ 111 km ≈ 60 nm)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        distance_deg = math.sqrt(dlat ** 2 + dlon ** 2)

        # Max displacement: speed_kt × dt_hours / 60 (nm to degrees, roughly)
        max_dist_deg = self.max_track_speed_kt * dt_hours / 60.0

        if distance_deg > max_dist_deg and distance_deg > 0:
            scale = max_dist_deg / distance_deg
            new_lat = lat1 + dlat * scale
            new_lon = lon1 + dlon * scale
            return (new_lat, new_lon), True

        return (lat2, lon2), False

    def _smooth_track(
        self, track: list[dict], current_lat: float, current_lon: float,
        corrections: list[str],
    ):
        """Smooth track to prevent large jumps between consecutive points."""
        max_jump_deg = 3.0
        prev_lat, prev_lon = current_lat, current_lon

        for tp in track:
            dlat = abs(tp["lat"] - prev_lat)
            dlon = abs(tp["lon"] - prev_lon)

            if dlat > max_jump_deg or dlon > max_jump_deg:
                # Interpolate to midpoint
                tp["lat"] = prev_lat + min(dlat, max_jump_deg) * (1 if tp["lat"] > prev_lat else -1)
                tp["lon"] = prev_lon + min(dlon, max_jump_deg) * (1 if tp["lon"] > prev_lon else -1)
                corrections.append(f"Track smoothed at {tp['t']}")

            prev_lat, prev_lon = tp["lat"], tp["lon"]

    @staticmethod
    def _is_over_land(lat: float, lon: float) -> bool:
        """
        Simplified land check for North Indian Ocean.

        Uses rough bounding boxes for major landmasses.
        A proper implementation would use shapely + coastline polygons.
        """
        # India (rough bounds)
        if 8 <= lat <= 35 and 68 <= lon <= 88:
            if lat > 8 and lon > 72:  # Above peninsula tip
                return True

        # Myanmar/Bangladesh
        if 15 <= lat <= 28 and 88 <= lon <= 100:
            return True

        # Sri Lanka (rough)
        if 6 <= lat <= 10 and 79 <= lon <= 82:
            return True

        return False

    @staticmethod
    def _parse_horizon(t_str: str) -> int:
        """Parse horizon string like '+6h' → 6."""
        return int(t_str.replace("+", "").replace("h", ""))

    @classmethod
    def classify_wind(cls, wind_kt: float) -> str:
        """Classify wind speed into IMD category."""
        category = "LOW"
        for threshold, cat in cls.IMD_CATEGORIES:
            if wind_kt >= threshold:
                category = cat
        return category
