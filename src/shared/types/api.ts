// API response types

export interface ApiResponse<T> {
  data?: T;
  error?: string;
}

export interface HealthResponse {
  status: string;
  timestamp: string;
}

export interface ConfigResponse {
  version: string;
  dataDir: string;
}
