export interface Storm {
  id: string;
  name: string;
  basin: string;
  status: "active" | "historical";
  category: string;
  current_lat: number;
  current_lon: number;
  max_wind_speed: number;
  min_pressure: number;
  formed_at: string;
  last_updated: string;
  created_at: string;
}

export interface StormListResponse {
  storms: Storm[];
  count: number;
}
