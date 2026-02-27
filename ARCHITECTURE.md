# Architecture

This repository follows a layered architectural pattern for a CLI-first high-assurance verification framework for safety-critical software systems.

## Directory Structure

```
.
├── cli/                           # CLI layer - public entrypoint
├── backend/                       # Backend services
│   ├── api/                       # HTTP boundary (FastAPI routing)
│   ├── services/                  # Business logic and orchestration
│   └── data/                      # Database access and persistence
├── ARCHITECTURE.md                # This file
├── README.md                      # Project overview
└── LICENSE                        # License
```

## Layer Responsibilities

### CLI Layer (`cli/`)
- Command-line interface entrypoint
- User-facing commands
- Routes user requests to backend services

### Backend

#### API Layer (`backend/api/`)
- HTTP boundary using FastAPI
- Request/response routing and validation
- Serves as the interface between CLI and core backend logic

#### Services Layer (`backend/services/`)
- Business logic and orchestration
- Policy enforcement
- Actor–checker coordination
- Verification workflow execution
- Audit logging
- Deterministic execution tracking

#### Data Layer (`backend/data/`)
- Database access patterns (repositories, database adapters)
- Persistence management
- Exclusively handles database interaction
- No general utilities; strictly data-focused

