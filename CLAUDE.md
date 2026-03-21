# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
npm run dev              # Vite dev server (frontend)
npm run dev:server       # Fastify dev server (backend)
npm run electron:build   # Build Electron app (DMG/installer)
npm test                 # Run tests
npm run test:watch       # Run tests in watch mode
npm run typecheck        # TypeScript check all configs
npm run lint             # ESLint
npm run format           # Prettier format
npm run clean            # Remove dist/release
```

Before completing work: `npm run typecheck && npm run lint && npm run format && npm test`

**Fix pre-existing issues:** When you encounter warnings, errors, or code quality issues—even if they existed before your changes—fix them rather than ignoring them. This includes test warnings, linter errors, and type issues.

## Architecture

**Three-process Electron app:**
- `src/renderer/` - React frontend (Vite-built)
- `src/server/` - Fastify API server (spawned by Electron)
- `src/electron/` - Electron main process
- `src/shared/` - Shared types between frontend/server

**State:** Zustand store in `src/renderer/store/`. All persistent data stored in `~/SeedAlchemy`.

**Styling:** Radix Themes (`@radix-ui/themes`) with dark mode. Use Radix color variables (`var(--gray-1)`) not Tailwind colors. Tailwind for layout utilities only. Never use `variant="ghost"` on buttons that toggle between variants based on state - use `soft` for inactive states instead.

**Four app modes:** Image (single image + parameters), Canvas (infinite node canvas), Gallery (fullscreen slideshow), Models (model manager).

## Code Style

- Use `//` comments, not `/** */` JSDoc-style comments
- Keep comments concise and on single lines when possible
- Comments should describe what the code does, not document changes or history

## Testing

Write tests for new functionality, especially for store actions and utility functions. Test files are colocated with source files using `.test.ts` or `.test.tsx` suffix. Use Vitest for testing.

- Use top-level imports, not dynamic `await import()` - if cycles exist, remove them

## Documentation

- `docs/overview.md` - Project overview, API routes, technical details, gotchas
- `docs/tasks/backlog.md` - Remaining work items (remove completed sections, keep it focused on what's next)
- `docs/decisions/` - ADRs for significant technical decisions
