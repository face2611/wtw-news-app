# Meridian Frontend

This is the Nuxt 3 frontend application for the [Meridian project](https://github.com/iliane5/meridian) (your personal AI intelligence agency). It provides the web interface for viewing generated intelligence briefs and managing sources (admin).

Built with:

- [Nuxt 3](https://nuxt.com/) (Vue 3)
- [Tailwind CSS](https://tailwindcss.com/) (with Radix UI colors)
- [TypeScript](https://www.typescriptlang.org/)

## Key Features

- Displays daily intelligence briefs with rich formatting (`/briefs/[slug]`).
- Interactive Table of Contents for easy navigation within briefs.
- Subscription form for updates (`/`).
- Consumes the Meridian API (via Nitro server routes in `/server/api` and potentially external workers).

## Setup

Make sure you have [Node.js](https://nodejs.org/) (v22+ recommended) and [pnpm](https://pnpm.io/) installed.

From the _root_ of the Meridian monorepo:

```bash
# Install all workspace dependencies
pnpm install
```

Or, if you're only working within this app (less common in a monorepo):

```bash
cd apps/frontend
pnpm install
```

You'll also need to ensure the necessary environment variables are configured (likely in a `.env` file in the root or this directory, depending on your setup) – particularly for the database connection (`DATABASE_URL`) and any external API endpoints (`WORKER_API`). See the [main project README](https://github.com/iliane5/meridian#setup) for full setup details.

## Development Server

Start the Nuxt development server (usually on `http://localhost:3000`):

```bash
# From the root directory
pnpm --filter @meridian/frontend dev

# Or from the apps/frontend directory
pnpm dev
```

## Production Build

Build the application for production:

```bash
# From the root directory
pnpm --filter @meridian/frontend build

# Or from the apps/frontend directory
pnpm build
```

Locally preview the production build:

```bash
# From the root directory
pnpm --filter @meridian/frontend preview

# Or from the apps/frontend directory
pnpm preview
```

## Deployment

This application is typically deployed using [Cloudflare Pages](https://pages.cloudflare.com/).

Check out the [Nuxt deployment documentation](https://nuxt.com/docs/getting-started/deployment) for general deployment information.
