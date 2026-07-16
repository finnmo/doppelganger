# Doppelgänger Analytics — self-contained image.
# Drop your messaging export ZIP(s) into ./data and run `docker compose up`.
FROM node:22-slim AS build

WORKDIR /app

# Toolchain for native modules (better-sqlite3) in case no prebuilt binary
# matches this platform. Only present in the build stage.
RUN apt-get update \
  && apt-get install -y --no-install-recommends python3 make g++ \
  && rm -rf /var/lib/apt/lists/*

# Install dependencies first for layer caching. --ignore-scripts skips the
# postinstall hook (which needs the source tree, copied below) — so the
# better-sqlite3 binary must be built explicitly afterwards.
# BuildKit cache mount keeps npm's cache off the tiny overlay filesystem
# (avoids ENOSPC on Docker Desktop VMs with a full disk).
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --ignore-scripts && npm rebuild better-sqlite3

# Run in the dashboard directory itself: `npm ci --prefix` can wrongly
# resolve the parent package-lock.json and fail with bogus sync errors.
COPY dashboard/package.json dashboard/package-lock.json ./dashboard/
WORKDIR /app/dashboard
# Do not use --ignore-scripts: optional native deps (e.g. lightningcss) need
# their platform packages linked for Next/Tailwind builds.
RUN --mount=type=cache,target=/root/.npm \
    npm ci
WORKDIR /app

# Build the backend CLI/processors.
COPY tsconfig.json ./
COPY src ./src
RUN npx tsc

# Build the dashboard (production Next.js build).
COPY dashboard ./dashboard
RUN npm run build --prefix dashboard

FROM node:22-slim

ENV NODE_ENV=production
WORKDIR /app

COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/dist ./dist
COPY --from=build /app/package.json ./package.json
COPY --from=build /app/dashboard ./dashboard
COPY docker/entrypoint.mjs ./docker/entrypoint.mjs

# /export  — mount your messaging export folder (read-only)
# DB + generated metrics persist in volumes (see docker-compose.yml)
VOLUME ["/root/.doppelgaenger-analytics", "/app/dash-data"]

EXPOSE 3000

CMD ["node", "docker/entrypoint.mjs"]
