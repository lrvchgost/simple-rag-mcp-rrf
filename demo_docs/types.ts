// Типы контрактов AuthService, используются всеми TS-клиентами.

export interface LoginRequest {
  username: string;
  password: string;
}

export interface TokenPair {
  access_token: string; // JWT RS256, живёт 1 час
  refresh_token: string; // опак-токен, живёт 30 дней
  session_id: string;
}

export interface AuditEvent {
  kind: "login.success" | "login.failure" | "logout" | "refresh";
  user_id: string;
  ip: string;
  ts: number; // unix-время события
}
