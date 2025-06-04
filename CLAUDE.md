# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Krist is a cryptocurrency node implementation designed for Minecraft servers, written in TypeScript/Node.js. It provides an API-based system for addresses, blocks, transactions, and a domain-like naming system.

## Development Commands

```bash
# Install dependencies (must use pnpm)
pnpm install

# Build TypeScript to JavaScript
pnpm run build

# Run in development mode with hot reload
pnpm run dev

# Run production build
pnpm start

# Run tests
pnpm test

# Run tests with coverage
pnpm run test:coverage

# Lint code
pnpm run lint

# Generate API documentation
pnpm run docs
```

## Architecture Overview

### Application Structure
The application follows a layered architecture:
- **Routes** (`src/webserver/routes/`) → **Controllers** (`src/controllers/`) → **Krist Core** (`src/krist/`) → **Database/Redis**
- WebSocket connections are managed separately through `src/websockets/`

### Database Architecture
- **Primary Database**: MariaDB via Sequelize v7 with decorators
  - Models: Address, Block, Name, Transaction, AuthLog (defined in `src/database/schemas.ts`)
  - Migrations in `src/database/migrations/`
- **Cache Layer**: Redis for runtime state, rate limiting, and WebSocket tokens

### Key Architectural Patterns
1. **Error Handling**: Custom `KristError` base class with domain-specific error types in `src/errors/`
2. **WebSockets**: Token-based authentication with event broadcasting for real-time updates
3. **API Design**: RESTful routes with version support (legacy and v2)
4. **Idempotency**: Request deduplication using `Idempotency-Key` headers
5. **Rate Limiting**: Redis-based per-endpoint limits

### Important Implementation Details
- All HTTP responses return status 200 for legacy compatibility (errors use `ok: false` in response body)
- Commit messages follow Conventional Commits format
- The application must run behind a reverse proxy (not exposed directly)
- Environment variables are required for database connections and configuration

### Testing Approach
- Tests use Mocha with TypeScript support
- Test files follow `*.test.ts` pattern
- Separate test configuration in `tsconfig.test.json`
- Database fixtures and seeds in `test/fixtures.ts` and `test/seed.ts`