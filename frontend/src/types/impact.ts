export interface LandfallLocation {
  lat: number;
  lon: number;
  name?: string;
  country?: string;
  state?: string;
  distance_km?: number;
}

export interface ImpactZone {
  type: string;
  geometry: {
    type: string;
    coordinates: number[][][];
  };
  properties: Record<string, unknown>;
}

export interface ImpactResponse {
  storm_id: string;
  prediction_id: string;
  landfall_location?: LandfallLocation;
  landfall_eta?: string;
  impact_zone?: ImpactZone;
  wind_radius_km?: number;
  storm_surge_risk?: string;
  population_exposed?: number;
  computed_at: string;
}
