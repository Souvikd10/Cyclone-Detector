export interface TrackPoint {
  lat: number;
  lon: number;
  timestamp: string;
  wind_speed?: number;
  pressure?: number;
  category?: string;
}

export interface IntensityPoint {
  timestamp: string;
  wind_speed: number;
  pressure: number;
  category?: string;
}

export interface ForecastData {
  track: TrackPoint[];
  intensity: IntensityPoint[];
}

export interface DetectionData {
  detected: boolean;
  confidence: number;
  bounding_box?: {
    min_lat: number;
    max_lat: number;
    min_lon: number;
    max_lon: number;
  };
  center?: { lat: number; lon: number };
}

export interface StructureData {
  eye_detected: boolean;
  eye_radius_km?: number;
  eyewall_completeness?: number;
  rain_bands?: number;
  symmetry_score?: number;
}

export interface CurrentStateData {
  lat: number;
  lon: number;
  max_wind_knots: number;
  min_pressure_hpa: number;
  category: string;
  dvorak_ci?: number;
  motion_speed_knots?: number;
  motion_direction_deg?: number;
}

export interface RIData {
  probability: number;
  expected_increase_knots?: number;
  timeframe_hours: number;
}

export interface UncertaintyData {
  cone_geojson?: {
    type: string;
    geometry: {
      type: string;
      coordinates: number[][][];
    };
    properties: Record<string, unknown>;
  };
  track_spread_km?: number;
  intensity_spread_knots?: number;
  ensemble_members?: number;
}

export interface XAIData {
  feature_importance?: Record<string, number>;
  attention_map_url?: string;
  reasoning?: string;
}

export interface ImpactData {
  landfall_location?: {
    lat: number;
    lon: number;
    name?: string;
    country?: string;
  };
  landfall_eta?: string;
  impact_zone_geojson?: {
    type: string;
    geometry: {
      type: string;
      coordinates: number[][][];
    };
    properties: Record<string, unknown>;
  };
}

export interface Prediction {
  id: string;
  storm_id: string;
  timestamp: string;
  detection?: DetectionData;
  structure?: StructureData;
  current_state?: CurrentStateData;
  forecast?: ForecastData;
  ri?: RIData;
  uncertainty?: UncertaintyData;
  xai?: XAIData;
  impact?: ImpactData;
  model_run_id?: string;
  created_at: string;
}

export interface PredictionHistoryResponse {
  storm_id: string;
  count: number;
  predictions: Prediction[];
}
