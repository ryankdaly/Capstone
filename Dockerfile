# HPEMA — multi-stage build
# Stage 1: install Dafny via .NET SDK
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS dafny-builder
RUN dotnet tool install --global Dafny && \
    cp -r /root/.dotnet/tools /dafny-tools

# Stage 2: Python runtime + Dafny
FROM python:3.13-slim

# Install .NET runtime (required to run Dafny) and Z3 (required by Dafny verifier)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libicu-dev \
    z3 \
    && rm -rf /var/lib/apt/lists/*

ENV DOTNET_ROOT=/usr/share/dotnet
RUN curl -sSL https://dot.net/v1/dotnet-install.sh | bash -s -- --runtime dotnet --channel 8.0 --install-dir $DOTNET_ROOT
ENV PATH="$DOTNET_ROOT:$PATH"

# Copy Dafny tools from builder stage
COPY --from=dafny-builder /dafny-tools /root/.dotnet/tools
ENV PATH="/root/.dotnet/tools:$PATH"

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
