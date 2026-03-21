import { create } from "zustand";
import type { AppMode } from "../../shared/types/state";
import type { User } from "../../shared/types/users";
import { api } from "../api/client";

export type { AppMode };

interface AppState {
  mode: AppMode;
  setMode: (mode: AppMode) => void;

  // User management
  currentUser: string;
  users: User[];
  setCurrentUser: (username: string) => void;
  setUsers: (users: User[]) => void;
  loadUsers: () => Promise<void>;
}

export const useAppStore = create<AppState>((set) => ({
  mode: "image",
  setMode: (mode) => set({ mode }),

  // User management
  currentUser: "default",
  users: [],
  setCurrentUser: (username) => set({ currentUser: username }),
  setUsers: (users) => set({ users }),
  loadUsers: async () => {
    try {
      const users = await api.getUsers();
      set({ users });
    } catch (error) {
      console.error("Failed to load users:", error);
    }
  },
}));
