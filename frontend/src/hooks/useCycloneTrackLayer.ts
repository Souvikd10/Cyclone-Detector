/**
 * useCycloneTrackLayer — produces all GeoJSON needed for the
 * NHC/ECMWF-style track visualization (matching the reference design):
 *
 *  historicalLineFC   → solid teal connecting line (past obs)
 *  historicalFC       → teal dots along past track
 *  forecastLineFC     → dashed orange connecting line (future)
 *  forecastFC         → orange dots for each forecast step
 *  forecastCirclesFC  → growing dashed-orange CIRCLES per forecast step
 *  currentPosFC       → single large dot at live eye (rendered as HTML marker)
 */
"use client";
import { useMemo } from "react";
import type { Prediction } from "@/types/prediction";
import { generateHistoricalTrack, createCircleCoords } from "@/lib/geoUtils";

export interface CycloneTrackLayerData {
  historicalLineFC: GeoJSON.FeatureCollection;
  historicalFC: GeoJSON.FeatureCollection;
  forecastLineFC: GeoJSON.FeatureCollection;
  forecastFC: GeoJSON.FeatureCollection;
  /** Per-step dashed uncertainty circles — the key visual from the reference */
  forecastCirclesFC: GeoJSON.FeatureCollection;
}

const EMPTY_FC: GeoJSON.FeatureCollection = { type: "FeatureCollection", features: [] };

/** Starting uncertainty radius at +6h (km) */
const BASE_RADIUS_KM = 55;
/** How much the radius grows per 6-hour forecast step */
const RADIUS_GROWTH_KM = 32;

function windToCategory(w: number): string {
  if (w >= 120) return "SuCS";
  if (w >= 90) return "ESCS";
  if (w >= 64) return "VSCS";
  if (w >= 48) return "SCS";
  if (w >= 34) return "CS";
  if (w >= 28) return "DD";
  return "D";
}

export function useCycloneTrackLayer(prediction: Prediction | null): CycloneTrackLayerData {
  return useMemo(() => {
    if (!prediction?.current_state) return {
      historicalLineFC: EMPTY_FC,
      historicalFC: EMPTY_FC,
      forecastLineFC: EMPTY_FC,
      forecastFC: EMPTY_FC,
      forecastCirclesFC: EMPTY_FC,
    };

    const cs = prediction.current_state;
    const rawForecast = prediction.forecast?.track ?? [];

    // ── Historical track (genesis → live eye) ──
    const histRaw = generateHistoricalTrack(cs.lat, cs.lon, cs.max_wind_knots, 14);

    // Drop last point (= live eye, rendered separately as HTML marker)
    const histPoints = histRaw.slice(0, -1);

    const historicalFeatures: GeoJSON.Feature[] = histPoints.map((p, i) => ({
      type: "Feature",
      geometry: { type: "Point", coordinates: [p.lon, p.lat] },
      properties: {
        category: p.category ?? windToCategory(p.wind_speed ?? 30),
        wind_speed: p.wind_speed ?? 30,
        timeLabel: `-${(histPoints.length - i) * 6}h`,
        stepIdx: i,
        totalSteps: histPoints.length,
      },
    }));

    // Full historical line includes live eye as final point
    const histLineCoords = [
      ...histPoints.map((p) => [p.lon, p.lat]),
      [cs.lon, cs.lat],
    ];
    const historicalLineFC: GeoJSON.FeatureCollection = {
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          geometry: { type: "LineString", coordinates: histLineCoords },
          properties: { type: "historical" },
        },
      ],
    };

    // ── Forecast track (live eye → landfall → inland) ──
    const forecastFeatures: GeoJSON.Feature[] = rawForecast.map((pt, idx) => ({
      type: "Feature",
      geometry: { type: "Point", coordinates: [pt.lon, pt.lat] },
      properties: {
        category: pt.category ?? windToCategory(pt.wind_speed ?? 70),
        wind_speed: pt.wind_speed ?? 70,
        timeLabel: `+${(idx + 1) * 6}h`,
        stepIdx: idx,
      },
    }));

    const forecastLineCoords: number[][] = [
      [cs.lon, cs.lat],
      ...rawForecast.map((p) => [p.lon, p.lat]),
    ];
    const forecastLineFC: GeoJSON.FeatureCollection = forecastLineCoords.length >= 2
      ? {
          type: "FeatureCollection",
          features: [
            {
              type: "Feature",
              geometry: { type: "LineString", coordinates: forecastLineCoords },
              properties: { type: "forecast" },
            },
          ],
        }
      : EMPTY_FC;

    // ── Forecast uncertainty circles (one per step, growing radius) ──
    const circleFeatures: GeoJSON.Feature[] = rawForecast.map((pt, idx) => {
      const radiusKm = BASE_RADIUS_KM + idx * RADIUS_GROWTH_KM;
      const ring = createCircleCoords(pt.lat, pt.lon, radiusKm, 48);
      return {
        type: "Feature",
        geometry: { type: "Polygon", coordinates: [ring] },
        properties: {
          stepIdx: idx,
          radiusKm,
          timeLabel: `+${(idx + 1) * 6}h`,
        },
      };
    });

    return {
      historicalLineFC,
      historicalFC: { type: "FeatureCollection", features: historicalFeatures },
      forecastLineFC,
      forecastFC: { type: "FeatureCollection", features: forecastFeatures },
      forecastCirclesFC: { type: "FeatureCollection", features: circleFeatures },
    };
  }, [prediction]);
}
