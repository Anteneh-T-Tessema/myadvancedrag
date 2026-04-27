#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo ""
echo "🚀 Advanced RAG Studio"
echo "══════════════════════════════════════════"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
  echo "❌ python3 not found"; exit 1
fi

# Check/install deps
echo "📦 Checking dependencies..."
pip3 install flask flask-cors sentence-transformers rank-bm25 numpy ollama --quiet 2>&1 | grep -v "^$" | tail -5

echo ""
echo "▶  Starting API on http://localhost:7891"
echo "🌐 Open: http://localhost:7891/static/index.html"
echo ""
echo "Press Ctrl+C to stop."
echo ""

python3 api/server.py
