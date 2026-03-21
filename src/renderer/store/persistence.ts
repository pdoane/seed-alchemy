import type { ProgressEvent } from "../../shared/types/progress";
import { api } from "../api/client";
import { useAppStore } from "./appStore";
import { useImageStore } from "./imageStore";
import { useDocumentStore } from "./documentStore";

let saveTimeout: ReturnType<typeof setTimeout> | null = null;
const SAVE_DEBOUNCE_MS = 150;

// Load user state from server (params, ui, generation session)
async function loadUserState(): Promise<void> {
  try {
    const state = await api.getState();
    if (state) {
      if (state.mode) {
        useAppStore.getState().setMode(state.mode);
      }
      useImageStore.getState().restoreUserState(state.params, state.ui);
      if (state.documentUi) {
        useDocumentStore.getState().restoreUserState(state.documentUi);
      }
    }
  } catch (error) {
    console.error("Failed to load user state:", error);
  }

  // Restore generation session state (for page refresh during generation)
  try {
    const session = await api.getGenerationSession();
    if (session.generation) {
      useImageStore.getState().restoreGenerationSession(session.generation);
    }
  } catch (error) {
    console.error("Failed to load session state:", error);
  }
}

// Save current state to server (debounced)
function saveUserState(): void {
  if (saveTimeout) {
    clearTimeout(saveTimeout);
  }

  saveTimeout = setTimeout(async () => {
    try {
      const appState = useAppStore.getState();
      const imageState = useImageStore.getState();
      const documentState = useDocumentStore.getState();

      await api.saveState({
        mode: appState.mode,
        params: imageState.params,
        ui: imageState.ui,
        documentUi: documentState.documentUi,
      });
    } catch (error) {
      console.error("Failed to save state:", error);
    }
  }, SAVE_DEBOUNCE_MS);
}

// Initialize session and load user state
async function initializeSession(): Promise<void> {
  try {
    const session = await api.getAuthSession();
    if (session.authenticated && session.username) {
      useAppStore.getState().setCurrentUser(session.username);
    } else {
      await api.login("default");
      useAppStore.getState().setCurrentUser("default");
    }
  } catch (error) {
    console.error("Failed to initialize session:", error);
    await api.login("default");
    useAppStore.getState().setCurrentUser("default");
  }

  await loadUserState();
}

// Initialize app - called once at startup
export function initializeApp(): () => void {
  // Start async initialization
  initializeSession().then(() => {
    useAppStore.getState().loadUsers();
    useImageStore.getState().loadImages();
    useImageStore.getState().loadCheckpoints();
    useImageStore.getState().loadLoras();
  });

  // Subscribe to SSE progress events
  const unsubscribeProgress = api.subscribeProgress((event: ProgressEvent) => {
    useImageStore.getState().handleProgressEvent(event);
    useDocumentStore.getState().handleProgressEvent(event);
  });

  // Subscribe to store changes for auto-save
  const unsubApp = useAppStore.subscribe(() => {
    saveUserState();
  });

  const unsubImage = useImageStore.subscribe((state, prevState) => {
    if (state.params !== prevState.params || state.ui !== prevState.ui) {
      saveUserState();
    }
  });

  const unsubDocument = useDocumentStore.subscribe((state, prevState) => {
    if (state.documentUi !== prevState.documentUi) {
      saveUserState();
    }
  });

  return () => {
    unsubscribeProgress();
    unsubApp();
    unsubImage();
    unsubDocument();
    if (saveTimeout) {
      clearTimeout(saveTimeout);
    }
  };
}

// Switch to a different user - clears state and loads new user's data
export async function switchUser(username: string): Promise<void> {
  await api.login(username);
  useAppStore.getState().setCurrentUser(username);

  useImageStore.getState().resetForUserSwitch();
  useDocumentStore.getState().resetForUserSwitch();

  await loadUserState();
  await useImageStore.getState().loadImages();
  await useDocumentStore.getState().loadDocuments();
}
