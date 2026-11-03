/**
 * Zustand store for dashboard UI state.
 * Manages: selected storm, active map layers, replay mode, sidebar visibility.
 */

import { create } from "zustand";

export type DashboardMode = "live" | "replay";

export interface MapLayers {
  track: boolean;
  cone: boolean;
  impactZone: boolean;
  safeZones: boolean;
  satellite: boolean;
  waypoints: boolean;
}

export interface UserLocation {
  lat: number;
  lon: number;
  city?: string;
  state?: string;
  country?: string;
  source?: "gps" | "preset" | "manual";
  timestamp?: number;
}

interface DashboardState {
  // ── User Location ──
  userLocation: UserLocation | null;
  setUserLocation: (location: UserLocation | null) => void;
  locationModalOpen: boolean;
  setLocationModalOpen: (open: boolean) => void;

  // ── Storm Selection ──
  selectedStormId: string | null;
  setSelectedStormId: (id: string | null) => void;

  // ── Dashboard Mode ──
  mode: DashboardMode;
  setMode: (mode: DashboardMode) => void;

  // ── Map Layer Toggles ──
  layers: MapLayers;
  toggleLayer: (layer: keyof MapLayers) => void;
  setLayers: (layers: Partial<MapLayers>) => void;

  // ── Historical Replay ──
  replayIndex: number;
  replayLength: number;
  replayPlaying: boolean;
  replaySpeed: number; // 1, 2, or 4
  setReplayIndex: (index: number) => void;
  setReplayLength: (length: number) => void;
  setReplayPlaying: (playing: boolean) => void;
  setReplaySpeed: (speed: number) => void;
  stepReplay: (direction: 1 | -1) => void;

  // ── UI Panels ──
  sidebarOpen: boolean;
  setSidebarOpen: (open: boolean) => void;
  alertPanelOpen: boolean;
  setAlertPanelOpen: (open: boolean) => void;
  feedbackModalOpen: boolean;
  setFeedbackModalOpen: (open: boolean) => void;
  xmlModalOpen: boolean;
  setXmlModalOpen: (open: boolean) => void;

  // ── Theme (Light / Dark) ──
  theme: "light" | "dark";
  setTheme: (theme: "light" | "dark") => void;
  toggleTheme: () => void;
}

function getInitialUserLocation(): UserLocation | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = localStorage.getItem("cyclone_user_location");
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export const useDashboardStore = create<DashboardState>((set, get) => ({
  // ── User Location ──
  userLocation: getInitialUserLocation(),
  setUserLocation: (location) => {
    if (typeof window !== "undefined") {
      try {
        if (location) {
          localStorage.setItem("cyclone_user_location", JSON.stringify(location));
        } else {
          localStorage.removeItem("cyclone_user_location");
        }
      } catch (e) {
        console.warn("Failed saving user location to localStorage", e);
      }
    }
    set({ userLocation: location });
  },
  locationModalOpen: false,
  setLocationModalOpen: (open) => set({ locationModalOpen: open }),

  // ── Storm Selection ──
  selectedStormId: null,
  setSelectedStormId: (id) => set({ selectedStormId: id, replayIndex: 0, replayPlaying: false }),


  // ── Dashboard Mode ──
  mode: "live",
  setMode: (mode) => set({ mode, replayIndex: 0, replayPlaying: false }),

  // ── Map Layer Toggles ──
  layers: {
    track: true,
    cone: true,
    impactZone: true,
    safeZones: true,
    satellite: false,
    waypoints: true,
  },
  toggleLayer: (layer) =>
    set((s) => ({ layers: { ...s.layers, [layer]: !s.layers[layer] } })),
  setLayers: (partial) =>
    set((s) => ({ layers: { ...s.layers, ...partial } })),

  // ── Historical Replay ──
  replayIndex: 0,
  replayLength: 0,
  replayPlaying: false,
  replaySpeed: 1,
  setReplayIndex: (index) => set({ replayIndex: index }),
  setReplayLength: (length) => set({ replayLength: length }),
  setReplayPlaying: (playing) => set({ replayPlaying: playing }),
  setReplaySpeed: (speed) => set({ replaySpeed: speed }),
  stepReplay: (direction) =>
    set((s) => ({
      replayIndex: Math.max(0, Math.min(s.replayLength - 1, s.replayIndex + direction)),
    })),

  // ── UI Panels ──
  sidebarOpen: true,
  setSidebarOpen: (open) => set({ sidebarOpen: open }),
  alertPanelOpen: false,
  setAlertPanelOpen: (open) => set({ alertPanelOpen: open }),
  feedbackModalOpen: false,
  setFeedbackModalOpen: (open) => set({ feedbackModalOpen: open }),
  xmlModalOpen: false,
  setXmlModalOpen: (open) => set({ xmlModalOpen: open }),

  // ── Theme (Light / Dark) ──
  theme: "light",
  setTheme: (theme) => {
    if (typeof document !== "undefined") {
      document.documentElement.setAttribute("data-theme", theme);
      document.documentElement.classList.toggle("dark", theme === "dark");
    }
    set({ theme });
  },
  toggleTheme: () => {
    const next = get().theme === "light" ? "dark" : "light";
    if (typeof document !== "undefined") {
      document.documentElement.setAttribute("data-theme", next);
      document.documentElement.classList.toggle("dark", next === "dark");
    }
    set({ theme: next });
  },
}));
