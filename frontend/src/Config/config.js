const defaultApiBaseUrl =
  process.env.REACT_APP_API_BASE_URL ||
  "https://sign-language-recognization-backend.onrender.com";

export const apiBaseUrl =
  defaultApiBaseUrl.replace(/\/+$/, "");

export const authBaseUrl =
  `${apiBaseUrl}/user`;

export const baseURL =
  `${apiBaseUrl}/sign-kit`;

export const detectionBaseUrl =
  process.env.REACT_APP_DETECTION_API_URL ||
  "https://sign-language-recognization-ml.onrender.com";