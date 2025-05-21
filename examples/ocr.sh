#! /usr/bin/env bash

curl -s -X POST -F "file=@hello_world.png" -H "X-API-Key: tym_razem_to_musi_poleciec" http://localhost:8000/ocr | python -m json.tool
