const defaultApiBaseUrl =
  process.env.REACT_APP_API_BASE_URL ||
  "http://localhost:5000";

export const apiBaseUrl =
  defaultApiBaseUrl.replace(/\/+$/, "");

export const authBaseUrl =
  `${apiBaseUrl}/user`;

export const baseURL =
  `${apiBaseUrl}/sign-kit`;

export const detectionBaseUrl =
  process.env.REACT_APP_DETECTION_API_URL ||
  "http://127.0.0.1:8000";