"use client";

import React, { useState, useEffect } from "react";
import {
  MapPin,
  Navigation,
  CheckCircle2,
  AlertCircle,
  X,
  Compass,
  Radio,
  Search,
  ChevronRight,
  Loader2,
  ShieldAlert,
} from "lucide-react";
import { useDashboardStore, type UserLocation } from "@/store/useDashboardStore";

// Offline coastal hubs fallback database for instant matching
const COASTAL_PRESETS: Array<{
  name: string;
  state: string;
  lat: number;
  lon: number;
  basin: string;
}> = [
  { name: "Chennai", state: "Tamil Nadu", lat: 13.0827, lon: 80.2707, basin: "Bay of Bengal" },
  { name: "Puri / Bhubaneswar", state: "Odisha", lat: 19.8135, lon: 85.8312, basin: "Bay of Bengal" },
  { name: "Visakhapatnam", state: "Andhra Pradesh", lat: 17.6868, lon: 83.2185, basin: "Bay of Bengal" },
  { name: "Kolkata / Sundarbans", state: "West Bengal", lat: 22.5726, lon: 88.3639, basin: "Bay of Bengal" },
  { name: "Digha", state: "West Bengal", lat: 21.6266, lon: 87.5074, basin: "Bay of Bengal" },
  { name: "Paradeep", state: "Odisha", lat: 20.3165, lon: 86.6114, basin: "Bay of Bengal" },
  { name: "Machilipatnam", state: "Andhra Pradesh", lat: 16.1875, lon: 81.1389, basin: "Bay of Bengal" },
  { name: "Mumbai", state: "Maharashtra", lat: 19.076, lon: 72.8777, basin: "Arabian Sea" },
  { name: "Kochi", state: "Kerala", lat: 9.9312, lon: 76.2673, basin: "Arabian Sea" },
  { name: "Panaji / Goa", state: "Goa", lat: 15.4909, lon: 73.8278, basin: "Arabian Sea" },
];

function findNearestPreset(lat: number, lon: number) {
  let best = COASTAL_PRESETS[0];
  let minDistance = Infinity;

  for (const preset of COASTAL_PRESETS) {
    const dlat = preset.lat - lat;
    const dlon = preset.lon - lon;
    const distSq = dlat * dlat + dlon * dlon;
    if (distSq < minDistance) {
      minDistance = distSq;
      best = preset;
    }
  }
  return best;
}

export default function LocationModal() {
  const { userLocation, setUserLocation, locationModalOpen, setLocationModalOpen } =
    useDashboardStore();

  const [isDetecting, setIsDetecting] = useState(false);
  const [detectionError, setDetectionError] = useState<string | null>(null);
  const [detectionSuccess, setDetectionSuccess] = useState<string | null>(null);

  // Manual inputs
  const [showManual, setShowManual] = useState(false);
  const [manualCity, setManualCity] = useState("");
  const [manualLat, setManualLat] = useState("");
  const [manualLon, setManualLon] = useState("");

  useEffect(() => {
    if (locationModalOpen) {
      setDetectionError(null);
      setDetectionSuccess(null);
    }
  }, [locationModalOpen]);

  if (!locationModalOpen) return null;

  // ── GPS Geolocation Handler ──
  const handleDetectLocation = () => {
    if (!navigator.geolocation) {
      setDetectionError("Geolocation is not supported by your browser.");
      return;
    }

    setIsDetecting(true);
    setDetectionError(null);
    setDetectionSuccess(null);

    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        const { latitude, longitude, accuracy } = pos.coords;
        let detectedCity = "";
        let detectedState = "";
        let detectedCountry = "India";

        // Attempt reverse geocoding via Nominatim
        try {
          const controller = new AbortController();
          const timeoutId = setTimeout(() => controller.abort(), 3500);

          const res = await fetch(
            `https://nominatim.openstreetmap.org/reverse?format=json&lat=${latitude}&lon=${longitude}&zoom=10`,
            { signal: controller.signal }
          );
          clearTimeout(timeoutId);

          if (res.ok) {
            const data = await res.json();
            const addr = data.address || {};
            detectedCity =
              addr.city ||
              addr.town ||
              addr.village ||
              addr.suburb ||
              addr.state_district ||
              addr.county ||
              "";
            detectedState = addr.state || "";
            detectedCountry = addr.country || "India";
          }
        } catch {
          // If offline or blocked, match with nearest known coastal hub
          const nearest = findNearestPreset(latitude, longitude);
          detectedCity = `${nearest.name} Region`;
          detectedState = nearest.state;
        }

        if (!detectedCity) {
          const nearest = findNearestPreset(latitude, longitude);
          detectedCity = nearest.name;
          detectedState = nearest.state;
        }

        const newLoc: UserLocation = {
          lat: Number(latitude.toFixed(4)),
          lon: Number(longitude.toFixed(4)),
          city: detectedCity,
          state: detectedState,
          country: detectedCountry,
          source: "gps",
          timestamp: Date.now(),
        };

        setUserLocation(newLoc);
        setIsDetecting(false);
        setDetectionSuccess(
          `Verified exact location: ${detectedCity}${detectedState ? ", " + detectedState : ""} (±${Math.round(accuracy)}m)`
        );

        // Auto close after 1.5 seconds on success
        setTimeout(() => {
          setLocationModalOpen(false);
        }, 1500);
      },
      (err) => {
        setIsDetecting(false);
        let msg = "Could not retrieve GPS location.";
        if (err.code === err.PERMISSION_DENIED) {
          msg = "Location permission denied. Please select your nearest coastal hub below.";
        } else if (err.code === err.POSITION_UNAVAILABLE) {
          msg = "Location unavailable. Please select your region below.";
        } else if (err.code === err.TIMEOUT) {
          msg = "GPS request timed out. Please choose a preset below.";
        }
        setDetectionError(msg);
      },
      {
        enableHighAccuracy: true,
        timeout: 10000,
        maximumAge: 30000,
      }
    );
  };

  // ── Preset Selection Handler ──
  const handleSelectPreset = (preset: (typeof COASTAL_PRESETS)[0]) => {
    const newLoc: UserLocation = {
      lat: preset.lat,
      lon: preset.lon,
      city: preset.name,
      state: preset.state,
      country: "India",
      source: "preset",
      timestamp: Date.now(),
    };
    setUserLocation(newLoc);
    setDetectionSuccess(`Selected coastal hub: ${preset.name}, ${preset.state}`);
    setTimeout(() => {
      setLocationModalOpen(false);
    }, 600);
  };

  // ── Manual Coordinate / City Submission ──
  const handleManualSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const latNum = parseFloat(manualLat);
    const lonNum = parseFloat(manualLon);

    if (isNaN(latNum) || isNaN(lonNum)) {
      setDetectionError("Please enter valid decimal coordinates (e.g. Lat: 13.08, Lon: 80.27).");
      return;
    }

    const newLoc: UserLocation = {
      lat: Number(latNum.toFixed(4)),
      lon: Number(lonNum.toFixed(4)),
      city: manualCity.trim() || "Custom Location",
      country: "India",
      source: "manual",
      timestamp: Date.now(),
    };
    setUserLocation(newLoc);
    setLocationModalOpen(false);
  };

  const handleDismiss = () => {
    setLocationModalOpen(false);
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-4 bg-black/60 backdrop-blur-md animate-fade-in"
      role="dialog"
      aria-modal="true"
      aria-labelledby="location-modal-title"
    >
      <div className="relative w-full max-w-xl max-h-[90vh] overflow-y-auto custom-scrollbar rounded-2xl border border-[var(--bg-glass-border)] bg-[var(--bg-card)] shadow-2xl p-5 sm:p-6 text-[var(--text-primary)]">
        {/* Close button */}
        <button
          onClick={handleDismiss}
          className="absolute top-4 right-4 p-1.5 rounded-lg text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-secondary)] transition-all"
          title="Dismiss for now"
          aria-label="Close"
        >
          <X className="w-5 h-5" />
        </button>

        {/* ── Top Header with Radar Animation ── */}
        <div className="flex items-start gap-4 mb-5">
          <div className="relative flex-shrink-0">
            <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center shadow-lg shadow-blue-500/20 animate-pulse-glow">
              <MapPin className="w-6 h-6 text-white animate-bounce" />
            </div>
            <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-emerald-400 rounded-full border-2 border-[var(--bg-card)] flex items-center justify-center">
              <Radio className="w-2.5 h-2.5 text-[#0f172a] animate-ping" />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-[0.65rem] font-bold tracking-wider uppercase px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20">
                Location Verification
              </span>
              {userLocation && (
                <span className="text-[0.65rem] font-medium text-emerald-400 flex items-center gap-1">
                  <CheckCircle2 className="w-3 h-3" /> Active
                </span>
              )}
            </div>
            <h2 id="location-modal-title" className="text-lg sm:text-xl font-bold mt-1">
              Enable Exact Location Services
            </h2>
            <p className="text-xs text-[var(--text-muted)] mt-0.5">
              Allows the AI Agent to detect approaching cyclones, calculate precise distances, and provide localized safety alerts for your area.
            </p>
          </div>
        </div>

        {/* ── Value Points ── */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 mb-5">
          <div className="p-2.5 rounded-xl bg-[var(--bg-secondary)] border border-[var(--bg-glass-border)]">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-blue-400">
              <Compass className="w-3.5 h-3.5" />
              <span>Real-Time Proximity</span>
            </div>
            <p className="text-[0.68rem] text-[var(--text-muted)] mt-1">
              Live distance (km) and bearing to active cyclone eye.
            </p>
          </div>

          <div className="p-2.5 rounded-xl bg-[var(--bg-secondary)] border border-[var(--bg-glass-border)]">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-amber-400">
              <ShieldAlert className="w-3.5 h-3.5" />
              <span>Hazard Danger Zone</span>
            </div>
            <p className="text-[0.68rem] text-[var(--text-muted)] mt-1">
              Immediate Danger (&lt;150 km) or Warning Zone alerts.
            </p>
          </div>

          <div className="p-2.5 rounded-xl bg-[var(--bg-secondary)] border border-[var(--bg-glass-border)]">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-emerald-400">
              <Radio className="w-3.5 h-3.5" />
              <span>Localized AI Voice</span>
            </div>
            <p className="text-[0.68rem] text-[var(--text-muted)] mt-1">
              AI answers in EN, HI, BN tuned strictly to your city.
            </p>
          </div>
        </div>

        {/* ── Status Feedback Banners ── */}
        {detectionSuccess && (
          <div className="mb-4 p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs flex items-center gap-2 animate-fade-in">
            <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
            <span className="font-medium">{detectionSuccess}</span>
          </div>
        )}

        {detectionError && (
          <div className="mb-4 p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-start gap-2 animate-fade-in">
            <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
            <div>
              <span className="font-semibold">Notice: </span>
              {detectionError}
            </div>
          </div>
        )}

        {/* ── Primary Action: Auto-Detect GPS ── */}
        <div className="mb-5">
          <button
            onClick={handleDetectLocation}
            disabled={isDetecting}
            className="w-full py-3 px-4 rounded-xl font-semibold text-sm bg-gradient-to-r from-blue-600 via-cyan-600 to-teal-500 text-white shadow-lg shadow-blue-500/25 hover:shadow-cyan-500/35 hover:scale-[1.01] active:scale-[0.99] transition-all flex items-center justify-center gap-2 cursor-pointer disabled:opacity-75 disabled:cursor-wait"
          >
            {isDetecting ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Acquiring High-Accuracy GPS Position...</span>
              </>
            ) : (
              <>
                <Navigation className="w-4 h-4" />
                <span>Detect My Exact GPS Location</span>
              </>
            )}
          </button>
        </div>

        {/* ── Divider ── */}
        <div className="relative flex items-center justify-center my-4">
          <div className="border-t border-[var(--bg-glass-border)] w-full" />
          <span className="bg-[var(--bg-card)] px-3 text-[0.7rem] uppercase tracking-wider text-[var(--text-muted)] font-semibold absolute">
            Or select high-risk coastal hub
          </span>
        </div>

        {/* ── Quick Coastal Presets ── */}
        <div className="mb-4">
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {COASTAL_PRESETS.map((preset) => {
              const isSelected =
                userLocation?.city?.toLowerCase() === preset.name.toLowerCase() ||
                (userLocation &&
                  Math.abs(userLocation.lat - preset.lat) < 0.2 &&
                  Math.abs(userLocation.lon - preset.lon) < 0.2);

              return (
                <button
                  key={preset.name}
                  onClick={() => handleSelectPreset(preset)}
                  className={`p-2.5 rounded-xl text-left border transition-all flex flex-col justify-between ${
                    isSelected
                      ? "border-blue-500 bg-blue-500/10 text-[var(--text-primary)] shadow-sm"
                      : "border-[var(--bg-glass-border)] bg-[var(--bg-secondary)] hover:border-[var(--accent-blue)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                  }`}
                >
                  <div className="flex items-center justify-between w-full">
                    <span className="text-xs font-semibold">{preset.name}</span>
                    {isSelected && <CheckCircle2 className="w-3.5 h-3.5 text-blue-400" />}
                  </div>
                  <div className="flex items-center justify-between text-[0.65rem] text-[var(--text-muted)] mt-1">
                    <span>{preset.state}</span>
                    <span>{preset.basin.replace("Bay of Bengal", "BoB")}</span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* ── Manual Coordinate Input Toggle ── */}
        <div className="pt-2 border-t border-[var(--bg-glass-border)]">
          <button
            onClick={() => setShowManual(!showManual)}
            className="text-xs font-medium text-[var(--text-muted)] hover:text-[var(--accent-blue)] flex items-center gap-1.5 transition-colors"
          >
            <ChevronRight
              className={`w-3.5 h-3.5 transition-transform ${showManual ? "rotate-90" : ""}`}
            />
            <span>Enter custom coordinates or locality</span>
          </button>

          {showManual && (
            <form onSubmit={handleManualSubmit} className="mt-3 grid grid-cols-1 sm:grid-cols-3 gap-2 animate-fade-in">
              <div>
                <label className="text-[0.68rem] text-[var(--text-muted)] block mb-1">
                  City / Area Name
                </label>
                <input
                  type="text"
                  placeholder="e.g. Gopalpur"
                  value={manualCity}
                  onChange={(e) => setManualCity(e.target.value)}
                  className="w-full px-2.5 py-1.5 rounded-lg bg-[var(--bg-secondary)] border border-[var(--bg-glass-border)] text-xs text-[var(--text-primary)] focus:outline-none focus:border-blue-400"
                />
              </div>
              <div>
                <label className="text-[0.68rem] text-[var(--text-muted)] block mb-1">
                  Latitude (°N)
                </label>
                <input
                  type="number"
                  step="any"
                  placeholder="19.26"
                  value={manualLat}
                  onChange={(e) => setManualLat(e.target.value)}
                  className="w-full px-2.5 py-1.5 rounded-lg bg-[var(--bg-secondary)] border border-[var(--bg-glass-border)] text-xs text-[var(--text-primary)] focus:outline-none focus:border-blue-400"
                />
              </div>
              <div>
                <label className="text-[0.68rem] text-[var(--text-muted)] block mb-1">
                  Longitude (°E)
                </label>
                <div className="flex gap-1.5">
                  <input
                    type="number"
                    step="any"
                    placeholder="84.86"
                    value={manualLon}
                    onChange={(e) => setManualLon(e.target.value)}
                    className="w-full px-2.5 py-1.5 rounded-lg bg-[var(--bg-secondary)] border border-[var(--bg-glass-border)] text-xs text-[var(--text-primary)] focus:outline-none focus:border-blue-400"
                  />
                  <button
                    type="submit"
                    className="px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold whitespace-nowrap"
                  >
                    Set
                  </button>
                </div>
              </div>
            </form>
          )}
        </div>

        {/* ── Footer / Dismiss ── */}
        <div className="mt-5 flex items-center justify-between pt-3 border-t border-[var(--bg-glass-border)]">
          <p className="text-[0.68rem] text-[var(--text-muted)]">
            {userLocation ? (
              <span>
                Current: <strong className="text-[var(--text-primary)]">{userLocation.city}</strong> ({userLocation.lat.toFixed(2)}°N, {userLocation.lon.toFixed(2)}°E)
              </span>
            ) : (
              <span>No location set yet.</span>
            )}
          </p>
          <button
            onClick={handleDismiss}
            className="px-3.5 py-1.5 rounded-lg text-xs font-medium text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-secondary)] transition-all"
          >
            {userLocation ? "Done" : "Skip for now"}
          </button>
        </div>
      </div>
    </div>
  );
}
