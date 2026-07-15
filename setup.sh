#!/usr/bin/env bash
# CEMG — one-command setup
# Usage:  bash setup.sh

set -e
echo "=== CEMG Setup ==="

# 1. Python deps
echo "→ Installing Python dependencies…"
pip install -r requirements.txt --quiet

# 2. Copy .env if missing
if [ ! -f .env ]; then
    cp .env.example .env
    echo "→ Created .env from .env.example  ← EDIT THIS with your keys"
else
    echo "→ .env already exists"
fi

# 3. Start Neo4j via Docker (skip if already running)
if command -v docker &>/dev/null; then
    if ! docker ps --format '{{.Names}}' | grep -q "cemg-neo4j"; then
        echo "→ Starting Neo4j container…"
        docker run -d \
            --name cemg-neo4j \
            -p 7474:7474 -p 7687:7687 \
            -e NEO4J_AUTH=neo4j/cemg_password \
            -v "$(pwd)/data/neo4j:/data" \
            neo4j:5.20
        echo "→ Waiting 10s for Neo4j to start…"
        sleep 10
    else
        echo "→ Neo4j container already running"
    fi
else
    echo "⚠  Docker not found — start Neo4j manually and update .env"
fi

# 4. Create data dirs
mkdir -p data/neo4j data

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env — add ANTHROPIC_API_KEY or OPENAI_API_KEY"
echo "  2. Update NEO4J_PASSWORD in .env to match Docker above (cemg_password)"
echo "  3. Run the demo:    python demo/run_demo.py"
echo "  4. Run the API:     uvicorn cemg.api:app --reload --port 8100"
echo "  5. API docs:        http://localhost:8100/docs"
