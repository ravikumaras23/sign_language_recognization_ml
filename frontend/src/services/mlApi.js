const API_BASE_URL = "http://127.0.0.1:8000";

async function apiRequest(
  endpoint,
  options = {}
) {
  const response = await fetch(
    `${API_BASE_URL}${endpoint}`,
    {
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {}),
      },
      ...options,
    }
  );

  let data;

  try {
    data = await response.json();
  } catch {
    data = {
      detail: "Invalid response from ML server",
    };
  }

  if (!response.ok) {
    throw new Error(
      data.detail ||
      `API request failed: ${response.status}`
    );
  }

  return data;
}

export async function startDetection(
  language = "ISL"
) {
  return apiRequest(
    "/start_detection",
    {
      method: "POST",
      body: JSON.stringify({
        language,
      }),
    }
  );
}

export async function stopDetection() {
  return apiRequest(
    "/stop_detection",
    {
      method: "POST",
    }
  );
}

export async function getHealth() {
  return apiRequest("/health");
}

export async function processLandmarks(
  landmarks
) {
  return apiRequest(
    "/process_frame",
    {
      method: "POST",
      body: JSON.stringify({
        landmarks,
      }),
    }
  );
}