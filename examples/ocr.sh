#! /usr/bin/env bash
# Simple helper script for testing the ``/ocr`` endpoint.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
IMAGE_PATH="${SCRIPT_DIR}/hello_world.png"

curl -s -X POST -F "file=@${IMAGE_PATH}" -H "X-API-Key: tym_razem_to_musi_poleciec" http://localhost:8000/ocr | python -m json.tool
