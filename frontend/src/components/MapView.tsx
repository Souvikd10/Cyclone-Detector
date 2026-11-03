"use client";

import React, { useEffect, useRef, useCallback, useState } from "react";
import * as maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import {
  Layers,
  Eye,
  Route,
  Hexagon,
  Target as TargetIcon,
  ShieldAlert,
  Wind,
  Globe,
  Moon,
  Sun,
  LocateFixed,
  RefreshCw,
} from "lucide-react";
import { useDashboardStore } from "@/store/useDashboardStore";
import { getCategoryColor, getCategoryLabel, formatDateTime, CATEGORY_COLORS } from "@/lib/utils";
import { generateSafetyZonesGeoJSON } from "@/lib/safetyZones";
import { buildEyeDiameterPolygon, generateHistoricalTrack } from "@/lib/geoUtils";
import { useCycloneTrackLayer } from "@/hooks/useCycloneTrackLayer";
import type { Prediction } from "@/types/prediction";

interface MapViewProps {
  prediction: Prediction | null;
}

// ── Satellite (Zoom Earth) Base Style ──
const SATELLITE_STYLE: maplibregl.StyleSpecification = {
  version: 8,
  sources: {
    "esri-imagery": {
      type: "raster",
      tiles: [
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      ],
      tileSize: 256,
      attribution: "&copy; Esri, Maxar, Earthstar Geographics, NOAA, NASA",
      maxzoom: 18,
    },
    "esri-boundaries-labels": {
      type: "raster",
      tiles: [
        "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
      ],
      tileSize: 256,
      attribution: "",
      maxzoom: 18,
    },
  },
  layers: [
    { id: "esri-imagery-layer", type: "raster", source: "esri-imagery", minzoom: 0, maxzoom: 18 },
    { id: "esri-boundaries-labels-layer", type: "raster", source: "esri-boundaries-labels", minzoom: 0, maxzoom: 18 },
  ],
};

const DARK_STYLE: maplibregl.StyleSpecification = {
  version: 8,
  sources: {
    "canvas-dark": {
      type: "raster",
      tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"],
      tileSize: 256,
      attribution: "&copy; Esri, Garmin, GEBCO, NOAA",
    },
  },
  layers: [{ id: "canvas-dark-layer", type: "raster", source: "canvas-dark", minzoom: 0, maxzoom: 16 }],
};

const LIGHT_STYLE: maplibregl.StyleSpecification = {
  version: 8,
  sources: {
    "canvas-light": {
      type: "raster",
      tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"],
      tileSize: 256,
      attribution: "&copy; Esri, Garmin, GEBCO, NOAA",
    },
  },
  layers: [{ id: "canvas-light-layer", type: "raster", source: "canvas-light", minzoom: 0, maxzoom: 16 }],
};

const EMPTY_FC: GeoJSON.FeatureCollection = { type: "FeatureCollection", features: [] };

// MapLibre `match` expression for IMD category colors
// Order: [layer, feature-property, val, color, val, color..., fallback]
const CATEGORY_MATCH_EXPR: maplibregl.ExpressionSpecification = [
  "match",
  ["get", "category"],
  "SuCS", CATEGORY_COLORS["SuCS"],
  "ESCS", CATEGORY_COLORS["ESCS"],
  "VSCS", CATEGORY_COLORS["VSCS"],
  "SCS",  CATEGORY_COLORS["SCS"],
  "CS",   CATEGORY_COLORS["CS"],
  "DD",   CATEGORY_COLORS["DD"],
  "D",    CATEGORY_COLORS["D"],
  "#94a3b8",
];

function ensureMapSources(map: maplibregl.Map) {
  const sources = [
    "safety-zones",
    "forecast-cone-fill",   // Layer 2 — forecast cone polygon
    "historical-line",      // Layer 3a — past track solid line
    "forecast-line",        // Layer 3b — future track dashed line
    "historical-dots",      // Layer 4a — past obs category dots
    "forecast-dots",        // Layer 4b — future obs dots (reduced opacity)
    "current-pos",          // Layer 5  — current eye large distinct dot
    "impact-zone",          // red impact circle
    "eye-diameter",         // AI eye diameter circle
  ];
  sources.forEach((src) => {
    if (!map.getSource(src)) {
      map.addSource(src, { type: "geojson", data: EMPTY_FC });
    }
  });
}

function ensureMapLayers(map: maplibregl.Map) {
  if (!map.isStyleLoaded()) return;
  ensureMapSources(map);

  // ── Layer 1: Safety Zones (bottom) ──
  if (!map.getLayer("safety-zones-fill")) {
    map.addLayer({
      id: "safety-zones-fill",
      type: "fill",
      source: "safety-zones",
      paint: {
        "fill-color": ["match", ["get", "zone"], "red", "#ef4444", "yellow", "#f59e0b", "green", "#10b981", "#ef4444"],
        "fill-opacity": ["match", ["get", "zone"], "red", 0.30, "yellow", 0.20, "green", 0.12, 0.18],
      },
    });
  }
  if (!map.getLayer("safety-zones-line")) {
    map.addLayer({
      id: "safety-zones-line",
      type: "line",
      source: "safety-zones",
      paint: {
        "line-color": ["match", ["get", "zone"], "red", "#dc2626", "yellow", "#d97706", "green", "#059669", "#dc2626"],
        "line-width": 2.2,
        "line-opacity": 0.90,
      },
    });
  }

  // ── Layer 2: Forecast Cone (white translucent) ──
  if (!map.getLayer("forecast-cone-fill-layer")) {
    map.addLayer({
      id: "forecast-cone-fill-layer",
      type: "fill",
      source: "forecast-cone-fill",
      paint: {
        "fill-color": "#ffffff",
        "fill-opacity": 0.18,
      },
    });
  }
  if (!map.getLayer("forecast-cone-outline")) {
    map.addLayer({
      id: "forecast-cone-outline",
      type: "line",
      source: "forecast-cone-fill",
      paint: {
        "line-color": "rgba(255, 255, 255, 0.70)",
        "line-width": 1.8,
        "line-dasharray": [4, 2],
      },
    });
  }

  // ── Layer 2b: Impact Zone ──
  if (!map.getLayer("impact-zone-fill")) {
    map.addLayer({
      id: "impact-zone-fill",
      type: "fill",
      source: "impact-zone",
      paint: { "fill-color": "#ef4444", "fill-opacity": 0.28 },
    });
  }
  if (!map.getLayer("impact-zone-line")) {
    map.addLayer({
      id: "impact-zone-line",
      type: "line",
      source: "impact-zone",
      paint: { "line-color": "#b91c1c", "line-width": 2.8, "line-opacity": 0.92 },
    });
  }

  // ── Layer 2c: Eye Diameter Circle ──
  if (!map.getLayer("eye-diameter-fill")) {
    map.addLayer({
      id: "eye-diameter-fill",
      type: "fill",
      source: "eye-diameter",
      paint: { "fill-color": "#06b6d4", "fill-opacity": 0.22 },
    });
  }
  if (!map.getLayer("eye-diameter-line")) {
    map.addLayer({
      id: "eye-diameter-line",
      type: "line",
      source: "eye-diameter",
      paint: {
        "line-color": "#06b6d4",
        "line-width": 2.2,
        "line-opacity": 0.92,
        "line-dasharray": [3, 2],
      },
    });
  }

  // ── Layer 3a: Historical Track (solid, dark casing + orange inner) ──
  if (!map.getLayer("historical-line-casing")) {
    map.addLayer({
      id: "historical-line-casing",
      type: "line",
      source: "historical-line",
      paint: { "line-color": "#020617", "line-width": 7.5, "line-opacity": 0.95 },
      layout: { "line-cap": "round", "line-join": "round" },
    });
  }
  if (!map.getLayer("historical-line-inner")) {
    map.addLayer({
      id: "historical-line-inner",
      type: "line",
      source: "historical-line",
      paint: { "line-color": "#f97316", "line-width": 4.0, "line-opacity": 1.0 },
      layout: { "line-cap": "round", "line-join": "round" },
    });
  }

  // ── Layer 3b: Forecast Track (dashed, dark casing + purple inner) ──
  if (!map.getLayer("forecast-line-casing")) {
    map.addLayer({
      id: "forecast-line-casing",
      type: "line",
      source: "forecast-line",
      paint: { "line-color": "#020617", "line-width": 6.5, "line-opacity": 0.95 },
      layout: { "line-cap": "round", "line-join": "round" },
    });
  }
  if (!map.getLayer("forecast-line-inner")) {
    map.addLayer({
      id: "forecast-line-inner",
      type: "line",
      source: "forecast-line",
      paint: {
        "line-color": "#c084fc",
        "line-width": 3.5,
        "line-opacity": 1.0,
        "line-dasharray": [2.5, 1.8],
      },
      layout: { "line-cap": "round", "line-join": "round" },
    });
  }

  // ── Layer 4a: Historical Observation Dots (category-colored, full opacity) ──
  if (!map.getLayer("historical-dots-circle")) {
    map.addLayer({
      id: "historical-dots-circle",
      type: "circle",
      source: "historical-dots",
      paint: {
        "circle-radius": 5.5,
        "circle-color": CATEGORY_MATCH_EXPR,
        "circle-stroke-width": 2.0,
        "circle-stroke-color": "#ffffff",
        "circle-opacity": 1.0,
      },
    });
  }

  // ── Layer 4b: Forecast Dots (category-colored, reduced opacity → signals "predicted") ──
  if (!map.getLayer("forecast-dots-circle")) {
    map.addLayer({
      id: "forecast-dots-circle",
      type: "circle",
      source: "forecast-dots",
      paint: {
        "circle-radius": 5.5,
        "circle-color": CATEGORY_MATCH_EXPR,
        "circle-stroke-width": 1.8,
        "circle-stroke-color": "rgba(255, 255, 255, 0.70)",
        "circle-opacity": 0.72,
      },
    });
  }

  // ── Layer 5: Current Eye Position — large white-outlined, glowing dot ──
  if (!map.getLayer("current-pos-glow")) {
    map.addLayer({
      id: "current-pos-glow",
      type: "circle",
      source: "current-pos",
      paint: {
        "circle-radius": 18,
        "circle-color": CATEGORY_MATCH_EXPR,
        "circle-opacity": 0.22,
        "circle-stroke-width": 0,
      },
    });
  }
  if (!map.getLayer("current-pos-circle")) {
    map.addLayer({
      id: "current-pos-circle",
      type: "circle",
      source: "current-pos",
      paint: {
        "circle-radius": 9,
        "circle-color": CATEGORY_MATCH_EXPR,
        "circle-stroke-width": 3.0,
        "circle-stroke-color": "#ffffff",
        "circle-opacity": 1.0,
      },
    });
  }
}

export default function MapView({ prediction }: MapViewProps) {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const eyeMarkerRef = useRef<maplibregl.Marker | null>(null);
  const landfallMarkerRef = useRef<maplibregl.Marker | null>(null);
  const userLocationMarkerRef = useRef<maplibregl.Marker | null>(null);
  const [legendExpanded, setLegendExpanded] = useState(true);
  const [mapBaseMode, setMapBaseMode] = useState<"satellite" | "dark" | "light">("satellite");

  const { layers, toggleLayer, setLayers, userLocation } = useDashboardStore();

  // ── Hook: produce all GeoJSON for the track layers ──
  const trackData = useCycloneTrackLayer(prediction);

  // ── Synchronize Layer Visibilities from Zustand store ──
  const syncLayerVisibility = useCallback((map: maplibregl.Map) => {
    if (!map.isStyleLoaded()) return;

    const setVis = (layerId: string, visible: boolean) => {
      if (map.getLayer(layerId)) {
        map.setLayoutProperty(layerId, "visibility", visible ? "visible" : "none");
      }
    };

    // Safety zones
    setVis("safety-zones-fill", layers.safeZones);
    setVis("safety-zones-line", layers.safeZones);

    // Forecast cone
    setVis("forecast-cone-fill-layer", layers.cone);
    setVis("forecast-cone-outline", layers.cone);

    // Impact zone & eye diameter
    setVis("impact-zone-fill", layers.impactZone);
    setVis("impact-zone-line", layers.impactZone);
    setVis("eye-diameter-fill", layers.waypoints);
    setVis("eye-diameter-line", layers.waypoints);

    // Historical + forecast track
    setVis("historical-line-casing", layers.track);
    setVis("historical-line-inner", layers.track);
    setVis("forecast-line-casing", layers.track);
    setVis("forecast-line-inner", layers.track);

    // Waypoints (historical dots + forecast dots + current pos)
    setVis("historical-dots-circle", layers.waypoints || layers.track);
    setVis("forecast-dots-circle", layers.waypoints || layers.track);
    setVis("current-pos-glow", layers.track);
    setVis("current-pos-circle", layers.track);
  }, [layers]);

  // ── Auto-Center & Fit Viewport to Entire Cyclone Route ──
  const fitCycloneBounds = useCallback(() => {
    const map = mapRef.current;
    if (!map || !prediction?.current_state) return;

    const cs = prediction.current_state;
    const rawForecast = prediction.forecast?.track || [];
    const lf = prediction.impact?.landfall_location;

    // Immediate center on live eye
    map.flyTo({ center: [cs.lon, cs.lat], zoom: 5.8, essential: true, duration: 500 });

    const bounds = new maplibregl.LngLatBounds();
    bounds.extend([cs.lon, cs.lat]);

    // Include all historical track origin points
    generateHistoricalTrack(cs.lat, cs.lon, cs.max_wind_knots, 14).forEach((p) =>
      bounds.extend([p.lon, p.lat])
    );

    rawForecast.forEach((p) => bounds.extend([p.lon, p.lat]));
    if (lf) bounds.extend([lf.lon, lf.lat]);

    if (!bounds.isEmpty()) {
      setTimeout(() => {
        try {
          map.fitBounds(bounds, {
            padding: { top: 90, bottom: 90, left: 130, right: 130 },
            maxZoom: 6.8,
            duration: 900,
            essential: true,
          });
        } catch (e) {
          console.warn("[Map] fitBounds failed", e);
        }
      }, 200);
    }
  }, [prediction]);

  // ── Push all GeoJSON into MapLibre sources ──
  const updateMapData = useCallback(() => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded()) return;

    ensureMapLayers(map);
    syncLayerVisibility(map);

    // Remove old HTML markers
    eyeMarkerRef.current?.remove();
    eyeMarkerRef.current = null;
    landfallMarkerRef.current?.remove();
    landfallMarkerRef.current = null;

    if (!prediction) {
      const clearSrc = (id: string) =>
        (map.getSource(id) as maplibregl.GeoJSONSource)?.setData(EMPTY_FC);
      [
        "safety-zones", "forecast-cone-fill", "historical-line", "forecast-line",
        "historical-dots", "forecast-dots", "current-pos", "impact-zone", "eye-diameter",
      ].forEach(clearSrc);
      return;
    }

    const cs = prediction.current_state;
    const lf = prediction.impact?.landfall_location;
    const structure = prediction.structure;

    // ── Push track layer GeoJSON (from hook) ──
    (map.getSource("historical-line") as maplibregl.GeoJSONSource)?.setData(trackData.historicalLineFC);
    (map.getSource("historical-dots") as maplibregl.GeoJSONSource)?.setData(trackData.historicalFC);
    (map.getSource("forecast-line") as maplibregl.GeoJSONSource)?.setData(trackData.forecastLineFC);
    (map.getSource("forecast-dots") as maplibregl.GeoJSONSource)?.setData(trackData.forecastFC);
    (map.getSource("forecast-cone-fill") as maplibregl.GeoJSONSource)?.setData(trackData.coneFillFC);
    (map.getSource("current-pos") as maplibregl.GeoJSONSource)?.setData(trackData.currentPosFC);

    // ── Safety Zones ──
    const safetyZonesGeoJSON = generateSafetyZonesGeoJSON(prediction);
    (map.getSource("safety-zones") as maplibregl.GeoJSONSource)?.setData(safetyZonesGeoJSON);

    // ── Impact Zone ──
    const impactGJ = prediction.impact?.impact_zone_geojson;
    if (impactGJ) {
      (map.getSource("impact-zone") as maplibregl.GeoJSONSource)?.setData({
        type: "FeatureCollection",
        features: [impactGJ as unknown as GeoJSON.Feature],
      });
    }

    // ── Eye Diameter Circle ──
    if (cs && structure?.eye_detected) {
      const eyeDiameterKm = structure.eye_radius_km ? structure.eye_radius_km * 2 : 45;
      const eyeCircle = buildEyeDiameterPolygon(cs.lat, cs.lon, eyeDiameterKm, structure.eyewall_completeness ?? 0.88);
      (map.getSource("eye-diameter") as maplibregl.GeoJSONSource)?.setData({
        type: "FeatureCollection",
        features: [eyeCircle],
      });
    } else {
      (map.getSource("eye-diameter") as maplibregl.GeoJSONSource)?.setData(EMPTY_FC);
    }

    // ── Rotating Hurricane Swirl Eye Marker (HTML marker at live eye) ──
    if (cs) {
      const catColor = getCategoryColor(cs.category);
      const el = document.createElement("div");
      el.innerHTML = `
        <div style="position:relative;width:54px;height:54px;display:flex;align-items:center;justify-content:center;cursor:pointer;z-index:70;">
          <div style="position:absolute;width:58px;height:58px;border-radius:50%;border:2.5px solid ${catColor};opacity:0.8;animation:ping 2s cubic-bezier(0,0,0.2,1) infinite;"></div>
          <div style="position:relative;width:42px;height:42px;display:flex;align-items:center;justify-content:center;filter:drop-shadow(0 0 12px ${catColor}) drop-shadow(0 0 6px rgba(0,0,0,0.9));">
            <svg viewBox="0 0 100 100" style="width:42px;height:42px;animation:spin 8s linear infinite;">
              <path fill="${catColor}" d="M50,14 C44,14 34,20 28,30 C22,40 22,50 28,60 C22,55 16,45 16,34 C16,14 32,4 50,4 C62,4 72,10 78,18 C72,15 62,14 50,14 Z"/>
              <path fill="${catColor}" d="M50,86 C56,86 66,80 72,70 C78,60 78,50 72,40 C78,45 84,55 84,66 C84,86 68,96 50,96 C38,96 28,90 22,82 C28,85 38,86 50,86 Z"/>
              <circle cx="50" cy="50" r="13" fill="#ffffff" stroke="${catColor}" stroke-width="4"/>
            </svg>
          </div>
          <div style="position:absolute;bottom:-13px;background:rgba(2,6,23,0.96);border:2px solid ${catColor};color:#fff;font-size:11px;font-weight:800;font-family:monospace;padding:2px 7px;border-radius:9999px;white-space:nowrap;box-shadow:0 4px 14px rgba(0,0,0,0.9);">
            ${cs.max_wind_knots} kt • ${cs.category}
          </div>
        </div>`;

      const marker = new maplibregl.Marker({ element: el })
        .setLngLat([cs.lon, cs.lat])
        .setPopup(
          new maplibregl.Popup({ offset: 32 }).setHTML(`
            <div style="padding:6px;min-width:210px;font-family:system-ui,sans-serif;">
              <div style="display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid rgba(255,255,255,0.15);padding-bottom:4px;margin-bottom:6px;">
                <span style="font-weight:800;font-size:0.9rem;color:${catColor};">🌀 ${cs.category} EYE</span>
                <span style="font-size:0.65rem;font-weight:700;padding:2px 6px;border-radius:4px;background:${catColor}30;color:${catColor};border:1px solid ${catColor}60;">LIVE</span>
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:0.75rem;color:#f8fafc;margin-bottom:6px;">
                <div><b>Wind:</b> <span style="color:#38bdf8;font-weight:700;">${cs.max_wind_knots} kt</span></div>
                <div><b>Pressure:</b> <span style="color:#fbbf24;font-weight:700;">${cs.min_pressure_hpa} hPa</span></div>
                <div><b>Speed:</b> ${cs.motion_speed_knots ?? 14} kt</div>
                <div><b>Heading:</b> ${cs.motion_direction_deg ?? 340}°</div>
              </div>
              <div style="font-size:0.72rem;color:#94a3b8;padding-top:4px;border-top:1px solid rgba(255,255,255,0.1);">
                Coords: <b>${cs.lat.toFixed(2)}°N, ${cs.lon.toFixed(2)}°E</b>
              </div>
              ${structure?.eye_detected
                ? `<div style="font-size:0.70rem;color:#06b6d4;font-weight:600;margin-top:4px;padding:3px 6px;background:rgba(6,182,212,0.15);border-radius:4px;">
                    👁️ Eye Ø: ${structure.eye_radius_km ? Math.round(structure.eye_radius_km * 2) : 45} km (${Math.round((structure.eyewall_completeness ?? 0.88) * 100)}% eyewall)
                  </div>` : ""}
            </div>`)
        )
        .addTo(map);

      eyeMarkerRef.current = marker;
    }

    // ── Landfall Strike Target ──
    const landfallEta = prediction.impact?.landfall_eta;
    if (lf) {
      const lfEl = document.createElement("div");
      lfEl.innerHTML = `
        <div style="position:relative;width:48px;height:48px;display:flex;align-items:center;justify-content:center;cursor:pointer;z-index:65;">
          <div style="position:absolute;width:50px;height:50px;border-radius:50%;background:rgba(239,68,68,0.30);border:2.5px solid #ef4444;animation:ping 1.6s cubic-bezier(0,0,0.2,1) infinite;"></div>
          <div style="position:relative;width:24px;height:24px;border-radius:4px;background:#ef4444;border:2.5px solid #fff;transform:rotate(45deg);box-shadow:0 0 18px rgba(239,68,68,0.9),0 0 6px rgba(0,0,0,0.9);display:flex;align-items:center;justify-content:center;">
            <div style="width:7px;height:7px;background:#fff;border-radius:50%;"></div>
          </div>
          <div style="position:absolute;bottom:-11px;background:rgba(185,28,28,0.95);border:1.5px solid #fff;color:#fff;font-size:9px;font-weight:800;padding:1px 6px;border-radius:4px;white-space:nowrap;box-shadow:0 2px 8px rgba(0,0,0,0.8);">LANDFALL</div>
        </div>`;

      const lfMarker = new maplibregl.Marker({ element: lfEl })
        .setLngLat([lf.lon, lf.lat])
        .setPopup(
          new maplibregl.Popup({ offset: 28 }).setHTML(`
            <div style="padding:6px;min-width:220px;font-family:system-ui,sans-serif;">
              <div style="font-weight:800;font-size:0.9rem;color:#ef4444;border-bottom:1px solid rgba(239,68,68,0.3);padding-bottom:3px;margin-bottom:5px;">🎯 PREDICTED LANDFALL</div>
              <div style="font-size:0.78rem;color:#f8fafc;margin-bottom:3px;"><b>Location:</b> ${lf.name ?? "Coastal Zone"} (${lf.country ?? "IN"})</div>
              ${landfallEta ? `<div style="font-size:0.74rem;color:#fca5a5;margin-bottom:4px;"><b>ETA:</b> ${formatDateTime(landfallEta)}</div>` : ""}
              <div style="font-size:0.72rem;color:#94a3b8;"><b>Coords:</b> ${lf.lat.toFixed(2)}°N, ${lf.lon.toFixed(2)}°E</div>
              <div style="font-size:0.70rem;font-weight:700;color:#f87171;margin-top:5px;padding:4px;background:rgba(239,68,68,0.18);border-radius:4px;border-left:3px solid #ef4444;">🚨 Catastrophic storm surge & coastal winds expected.</div>
            </div>`)
        )
        .addTo(map);

      landfallMarkerRef.current = lfMarker;
    }

    // ── Auto-frame the full trajectory ──
    fitCycloneBounds();
  }, [prediction, trackData, syncLayerVisibility, fitCycloneBounds]);

  // ── Initialize MapLibre Map ──
  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return;

    const initialStyle = mapBaseMode === "satellite" ? SATELLITE_STYLE : mapBaseMode === "dark" ? DARK_STYLE : LIGHT_STYLE;
    const center: [number, number] = prediction?.current_state
      ? [prediction.current_state.lon, prediction.current_state.lat]
      : [85.0, 16.0];

    const map = new maplibregl.Map({
      container: mapContainerRef.current,
      style: initialStyle,
      center,
      zoom: 5.2,
      minZoom: 3,
      maxZoom: 15,
      attributionControl: {},
    });

    map.addControl(new maplibregl.NavigationControl(), "bottom-right");

    map.on("load", () => {
      ensureMapLayers(map);
      updateMapData();
      map.resize();

      // Safety Zone hover tooltip
      map.on("mousemove", "safety-zones-fill", (e) => {
        if (!e.features?.length) return;
        map.getCanvas().style.cursor = "pointer";
        const props = e.features[0].properties;
        if (!props) return;
        new maplibregl.Popup({ closeButton: false, closeOnClick: false, offset: 12 })
          .setLngLat(e.lngLat)
          .setHTML(`
            <div style="padding:4px;min-width:220px;font-family:inherit;">
              <div style="font-weight:700;font-size:0.85rem;color:${props.color ?? "#fff"};margin-bottom:3px;">${props.title}</div>
              <div style="font-size:0.72rem;color:#cbd5e1;margin-bottom:2px;"><b>Wind:</b> ${props.windSpeedRange}</div>
              <div style="font-size:0.72rem;color:#94a3b8;margin-bottom:4px;">${props.damageDescription}</div>
              <div style="font-size:0.7rem;font-weight:600;color:#f8fafc;padding:4px;border-radius:4px;background:rgba(0,0,0,0.4);border-left:3px solid ${props.color};">⚠️ ${props.safetyAdvice}</div>
            </div>`)
          .addTo(map);
      });
      map.on("mouseleave", "safety-zones-fill", () => { map.getCanvas().style.cursor = ""; });

      // Waypoint click popups
      const waypointClickHandler = (e: maplibregl.MapMouseEvent & { features?: maplibregl.MapGeoJSONFeature[] }) => {
        if (!e.features?.length) return;
        const props = e.features[0].properties;
        const coords = (e.features[0].geometry as GeoJSON.Point).coordinates;
        new maplibregl.Popup({ offset: 14 })
          .setLngLat([coords[0], coords[1]])
          .setHTML(`
            <div style="padding:4px;min-width:185px;font-family:system-ui,sans-serif;">
              <div style="font-weight:700;font-size:0.86rem;color:#38bdf8;margin-bottom:3px;">📍 ${props?.timeLabel ?? "Track Point"}</div>
              <div style="font-size:0.76rem;color:#f8fafc;"><b>Category:</b> ${props?.category} (${getCategoryLabel(props?.category ?? "CS")})</div>
              <div style="font-size:0.76rem;color:#f8fafc;"><b>Wind:</b> ${props?.wind_speed} kt (${Math.round((props?.wind_speed ?? 0) * 1.852)} km/h)</div>
              <div style="font-size:0.72rem;color:#94a3b8;margin-top:3px;">${coords[1].toFixed(2)}°N, ${coords[0].toFixed(2)}°E</div>
            </div>`)
          .addTo(map);
      };
      map.on("click", "historical-dots-circle", waypointClickHandler);
      map.on("click", "forecast-dots-circle", waypointClickHandler);
      map.on("click", "current-pos-circle", waypointClickHandler);
    });

    const resizeObserver = new ResizeObserver(() => map.resize());
    if (mapContainerRef.current) resizeObserver.observe(mapContainerRef.current);

    mapRef.current = map;
    return () => {
      resizeObserver.disconnect();
      map.remove();
      mapRef.current = null;
    };
  }, [mapBaseMode]);

  // Re-push data on prediction/trackData change
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    if (map.isStyleLoaded()) {
      updateMapData();
    } else {
      map.once("load", updateMapData);
    }
  }, [updateMapData]);

  // Sync layer visibility on store change
  useEffect(() => {
    const map = mapRef.current;
    if (map) syncLayerVisibility(map);
  }, [layers, syncLayerVisibility]);

  // ── User Location Marker ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    userLocationMarkerRef.current?.remove();
    userLocationMarkerRef.current = null;
    if (!userLocation) return;

    const renderUserMarker = () => {
      userLocationMarkerRef.current?.remove();
      const el = document.createElement("div");
      el.innerHTML = `
        <div style="position:relative;width:34px;height:34px;display:flex;align-items:center;justify-content:center;cursor:pointer;">
          <div style="position:absolute;width:34px;height:34px;border-radius:50%;background:rgba(6,182,212,0.3);border:1.5px solid #06b6d4;animation:ping 2s cubic-bezier(0,0,0.2,1) infinite;"></div>
          <div style="position:relative;width:14px;height:14px;border-radius:50%;background:#06b6d4;border:2.5px solid #fff;box-shadow:0 0 14px rgba(6,182,212,0.9);"></div>
        </div>`;
      const marker = new maplibregl.Marker({ element: el })
        .setLngLat([userLocation.lon, userLocation.lat])
        .setPopup(
          new maplibregl.Popup({ offset: 18 }).setHTML(`
            <div style="padding:2px;min-width:175px;">
              <div style="font-weight:700;font-size:0.82rem;color:#06b6d4;margin-bottom:2px;">📍 Your Location</div>
              <div style="font-weight:600;font-size:0.85rem;color:#f8fafc;">${userLocation.city ?? "Current Position"}${userLocation.state ? ", " + userLocation.state : ""}</div>
              <div style="font-size:0.72rem;color:#94a3b8;margin-top:2px;">${userLocation.lat.toFixed(3)}°N, ${userLocation.lon.toFixed(3)}°E</div>
            </div>`)
        )
        .addTo(map);
      userLocationMarkerRef.current = marker;
    };

    if (map.isStyleLoaded()) renderUserMarker();
    else map.once("load", renderUserMarker);
  }, [userLocation]);

  // ── Switch Base Map ──
  const switchMapBase = (mode: "satellite" | "dark" | "light") => {
    const map = mapRef.current;
    if (!map) return;
    setMapBaseMode(mode);
    map.setStyle(mode === "satellite" ? SATELLITE_STYLE : mode === "dark" ? DARK_STYLE : LIGHT_STYLE);
    map.once("style.load", () => {
      ensureMapLayers(map);
      updateMapData();
    });
  };

  const enableAllLayers = () => {
    setLayers({ track: true, cone: true, impactZone: true, safeZones: true, waypoints: true });
    fitCycloneBounds();
  };

  return (
    <div className="relative map-container w-full h-full min-h-[500px] rounded-2xl overflow-hidden shadow-2xl border border-white/10">
      <div ref={mapContainerRef} className="w-full h-full" style={{ minHeight: 500 }} />

      {/* ── Top-Right HUD ── */}
      {prediction?.current_state && (
        <div className="absolute top-3 right-3 z-10 hidden sm:flex items-center gap-2 bg-slate-950/90 backdrop-blur-md border border-white/20 px-3.5 py-1.5 rounded-full shadow-2xl">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-orange-400 animate-pulse"></span>
            <span className="text-[0.74rem] font-bold text-white tracking-wide">
              {prediction.current_state.lat.toFixed(1)}°N, {prediction.current_state.lon.toFixed(1)}°E
            </span>
          </div>
          <span className="text-white/30">|</span>
          <div className="flex items-center gap-1 text-[0.74rem] text-cyan-300 font-semibold">
            <Wind className="w-3.5 h-3.5 text-cyan-400" />
            <span>{prediction.current_state.max_wind_knots} kt</span>
          </div>
          {prediction.impact?.landfall_location && (
            <>
              <span className="text-white/30">|</span>
              <div className="flex items-center gap-1 text-[0.74rem] text-red-400 font-bold">
                <TargetIcon className="w-3.5 h-3.5 text-red-500" />
                <span>
                  {prediction.impact.landfall_location.lat.toFixed(1)}°N,{" "}
                  {prediction.impact.landfall_location.lon.toFixed(1)}°E
                </span>
              </div>
            </>
          )}
          <span className="text-white/30">|</span>
          <button
            onClick={fitCycloneBounds}
            className="flex items-center gap-1 text-[0.70rem] font-bold text-cyan-300 hover:text-white bg-cyan-500/20 hover:bg-cyan-500/30 px-2.5 py-0.5 rounded-full border border-cyan-400/40 transition-all"
          >
            <LocateFixed className="w-3 h-3 text-cyan-400" />
            <span>Center</span>
          </button>
        </div>
      )}

      {/* ── Map Style + Layers (Top-Left) ── */}
      <div className="absolute top-3 left-3 z-10 flex flex-col gap-2">
        <div className="glass-card p-1.5 flex items-center gap-1 bg-slate-950/90 backdrop-blur-md border border-white/20 rounded-xl shadow-xl">
          {(["satellite", "dark", "light"] as const).map((mode) => (
            <button
              key={mode}
              onClick={() => switchMapBase(mode)}
              className={`flex items-center gap-1 px-2.5 py-1 rounded-lg text-[0.68rem] font-bold transition-all ${
                mapBaseMode === mode
                  ? mode === "satellite" ? "bg-blue-600 text-white shadow-md"
                    : mode === "dark" ? "bg-slate-700 text-white shadow-md"
                    : "bg-slate-200 text-slate-900 shadow-md"
                  : "text-slate-400 hover:text-white hover:bg-white/10"
              }`}
            >
              {mode === "satellite" ? <Globe className="w-3.5 h-3.5" /> : mode === "dark" ? <Moon className="w-3.5 h-3.5" /> : <Sun className="w-3.5 h-3.5" />}
              <span className="capitalize">{mode}</span>
            </button>
          ))}
        </div>

        {/* ── Layer Controls ── */}
        <div className="glass-card p-2 flex flex-col gap-1 shadow-2xl backdrop-blur-md bg-slate-950/90 border border-white/20 rounded-xl min-w-[170px]">
          <div className="flex items-center justify-between px-1.5 mb-1 text-[0.64rem] text-cyan-300 uppercase tracking-wider font-bold border-b border-white/10 pb-1">
            <div className="flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-cyan-400" />
              <span>Map Layers</span>
            </div>
            <button onClick={enableAllLayers} className="text-[0.62rem] text-cyan-400 hover:text-cyan-300 flex items-center gap-0.5">
              <RefreshCw className="w-2.5 h-2.5" /><span>All</span>
            </button>
          </div>
          <LayerToggle icon={<Route className="w-3.5 h-3.5" />} label="Cyclone Track" active={layers.track} color="#f97316" onToggle={() => toggleLayer("track")} />
          <LayerToggle icon={<Eye className="w-3.5 h-3.5" />} label="Waypoints & Eye" active={layers.waypoints} color="#8b5cf6" onToggle={() => toggleLayer("waypoints")} />
          <LayerToggle icon={<Hexagon className="w-3.5 h-3.5" />} label="Uncertainty Cone" active={layers.cone} color="#e2e8f0" onToggle={() => toggleLayer("cone")} />
          <LayerToggle icon={<TargetIcon className="w-3.5 h-3.5" />} label="Landfall Strike" active={layers.impactZone} color="#ef4444" onToggle={() => toggleLayer("impactZone")} />
          <LayerToggle icon={<ShieldAlert className="w-3.5 h-3.5" />} label="Risk Zones" active={layers.safeZones} color="#10b981" onToggle={() => toggleLayer("safeZones")} />
        </div>
      </div>

      {/* ── Bottom-Left Legend ── */}
      <div className="absolute bottom-4 left-4 z-10">
        <div className="glass-card p-3 shadow-2xl backdrop-blur-md bg-slate-950/90 border border-white/20 rounded-xl min-w-[250px]">
          <div className="flex items-center justify-between cursor-pointer" onClick={() => setLegendExpanded(!legendExpanded)}>
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-orange-500 animate-pulse"></span>
              <span className="text-[0.74rem] font-bold text-white uppercase tracking-wider">Track Legend</span>
            </div>
            <button className="text-[0.66rem] font-semibold text-cyan-400 hover:underline">{legendExpanded ? "Hide" : "Show"}</button>
          </div>

          {legendExpanded && (
            <div className="mt-2.5 flex flex-col gap-2 text-[0.7rem] text-slate-200 border-t border-white/10 pt-2.5 animate-fade-in">
              {/* Track type indicators */}
              <div className="flex items-center justify-between text-[0.66rem]">
                <div className="flex items-center gap-1.5">
                  <span className="w-7 h-1.5 rounded-full bg-orange-500"></span>
                  <span className="text-slate-300">Past Track (observed)</span>
                </div>
              </div>
              <div className="flex items-center justify-between text-[0.66rem]">
                <div className="flex items-center gap-1.5">
                  <span className="w-7 h-1 rounded-full bg-purple-400" style={{ backgroundImage: "repeating-linear-gradient(to right, #c084fc 0, #c084fc 5px, transparent 5px, transparent 9px)" }}></span>
                  <span className="text-slate-300">Forecast Track</span>
                </div>
              </div>

              {/* Intensity category palette */}
              <div className="pt-1">
                <div className="text-[0.62rem] text-slate-400 mb-1 uppercase tracking-wider">IMD Intensity Scale</div>
                <div className="flex items-center gap-1 flex-wrap">
                  {(["D","DD","CS","SCS","VSCS","ESCS","SuCS"] as const).map((cat) => (
                    <span
                      key={cat}
                      className="px-1.5 py-0.5 rounded text-[0.62rem] font-bold font-mono text-white"
                      style={{ backgroundColor: CATEGORY_COLORS[cat] ?? "#94a3b8" }}
                      title={getCategoryLabel(cat)}
                    >
                      {cat}
                    </span>
                  ))}
                </div>
              </div>

              {/* Markers row */}
              <div className="flex items-center justify-between pt-2 border-t border-white/10 text-[0.66rem] text-slate-300">
                <div className="flex items-center gap-1.5">
                  <span className="text-sm">🌀</span>
                  <span>Live Eye</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-3 h-3 rounded bg-white/35 border border-white/70"></span>
                  <span>Cone</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rotate-45 bg-red-500 border border-white shadow-[0_0_6px_#ef4444]"></span>
                  <span>Landfall</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function LayerToggle({
  icon, label, active, color, onToggle,
}: {
  icon: React.ReactNode;
  label: string;
  active: boolean;
  color: string;
  onToggle: () => void;
}) {
  return (
    <button
      onClick={onToggle}
      className={`flex items-center justify-between px-2.5 py-1.5 rounded-lg text-[0.72rem] font-medium transition-all w-full ${
        active
          ? "bg-white/15 text-white shadow-md border border-white/20 font-semibold"
          : "text-slate-400 hover:text-slate-200 hover:bg-white/5 opacity-70"
      }`}
    >
      <div className="flex items-center gap-2 truncate">
        <span className="shrink-0" style={active ? { color } : {}}>{icon}</span>
        <span className="truncate">{label}</span>
      </div>
      <span
        className="w-2 h-2 rounded-full shrink-0 ml-1.5 transition-all"
        style={{
          backgroundColor: active ? color : "rgba(255,255,255,0.2)",
          boxShadow: active ? `0 0 8px ${color}80` : "none",
        }}
      />
    </button>
  );
}
