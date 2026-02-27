# API Guide

## Purpose

The `backend/api/` layer serves as the HTTP boundary for the HPEMA framework. It defines all routes, request/response schemas, and coordinates with the services layer for business logic.

## Current State

The API layer is in bootstrap state. FastAPI application is initialized with basic configuration in `backend/main.py`. No routes are currently defined.

## Running the Server

Start the development server:

```bash
uvicorn backend.main:app --reload
```

The server will run on `http://localhost:8000` by default.

## API Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

These are available even with no routes defined, and will be updated as new endpoints are added.

## Architecture

The API layer is responsible for:
- HTTP request/response routing
- Request validation and schema definitions
- Response serialization
- Coordination with the services layer

Business logic resides in the `services/` layer. Database operations are handled by the `data/` layer.

## Adding Routes

New routers will be added to `backend/api/routers/` as endpoints are required. Each router module will be imported and included in the FastAPI application instance.
