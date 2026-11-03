export interface AlertInfo {
  storm_id: string;
  alert_id: string;
  status: string;
  msg_type: string;
  severity: string;
  urgency: string;
  certainty: string;
  headline: string;
  description: string;
  effective: string;
  expires?: string;
  sender: string;
}

export interface AlertResponse {
  info: AlertInfo;
  cap_xml: string;
}
