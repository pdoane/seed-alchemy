// User management types

export interface User {
  username: string;
  hasPassword: boolean;
}

export interface UserCredentials {
  passwordHash?: string; // bcrypt hash, undefined if no password
}
