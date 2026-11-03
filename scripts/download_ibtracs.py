"""
IBTrACS download script — one-shot data pull.

Downloads the North Indian Ocean best track data from NCEI.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cyclone_ai.data.ibtracs_client import IBTrACSClient


def main():
    print("=== IBTrACS Data Download ===\n")

    client = IBTrACSClient(config_path="configs/data_sources.yaml")
    csv_path = client.download(force="--force" in sys.argv)
    print(f"CSV path: {csv_path}")

    # Load and show summary
    df = client.load()
    print(f"\nTotal records: {len(df)}")

    if "SID" in df.columns:
        print(f"Unique storms: {df['SID'].nunique()}")

    # Show target storms
    target_storms = [
        ("2020121N10087", "AMPHAN"),
        ("2019121N08085", "FANI"),
        ("2023126N10091", "MOCHA"),
    ]

    print("\n--- Target Storms ---")
    for sid, name in target_storms:
        try:
            track = client.get_storm(sid)
            print(
                f"\n{name} ({sid}):"
                f"\n  Track points: {len(track.points)}"
                f"\n  Peak wind: {track.peak_wind_kt:.0f} kt"
                f"\n  Min pressure: {track.min_pressure_hpa:.0f} hPa"
                f"\n  Genesis: {track.genesis_point.lat:.1f}°N, {track.genesis_point.lon:.1f}°E"
            )
        except ValueError as e:
            print(f"\n{name}: {e}")


if __name__ == "__main__":
    main()
