export interface FeedbackCreate {
  prediction_id: string;
  action: "accept" | "override";
  corrected_value?: Record<string, unknown>;
  forecaster_note?: string;
  forecaster_id?: string;
}

export interface FeedbackResponse {
  id: string;
  prediction_id: string;
  storm_id: string;
  action: string;
  corrected_value?: Record<string, unknown>;
  forecaster_note?: string;
  forecaster_id?: string;
  created_at: string;
  retrain_triggered: boolean;
}

export interface FeedbackStats {
  storm_id: string;
  total: number;
  accepts: number;
  overrides: number;
  override_rate: number;
}
