#!/bin/sh
if [ -z "$GOOGLE_API_KEY" ]; then
  echo "ERROR: GOOGLE_API_KEY is not set!"
  echo "Please run the container with -e GOOGLE_API_KEY=your_key"
  exit 1
fi

if [ -z "$DISPLAY" ]; then
  echo "ERROR: DISPLAY is not set!"
  echo "Please run the container with -e DISPLAY=:0"
  exit 1
fi

echo "Environment OK. Starting..."
cd /jemdzem
uv run uvicorn jemdzem.backend:app

