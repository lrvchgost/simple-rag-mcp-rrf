// SDK клиента AuthService: логин и обновление токенов.

const REFRESH_STORAGE_KEY = "auth.refresh.session";

export async function login(baseUrl, username, password) {
  const resp = await fetch(`${baseUrl}/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!resp.ok) {
    throw new Error(`login failed: ${resp.status}`);
  }
  const data = await resp.json();
  localStorage.setItem(REFRESH_STORAGE_KEY, data.refresh_token);
  return data.access_token;
}

export async function refresh(baseUrl) {
  const token = localStorage.getItem(REFRESH_STORAGE_KEY);
  const resp = await fetch(`${baseUrl}/refresh`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
  });
  if (resp.status === 401) {
    localStorage.removeItem(REFRESH_STORAGE_KEY);
    throw new Error("session expired");
  }
  return (await resp.json()).access_token;
}
