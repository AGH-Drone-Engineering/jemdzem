"""Shared Gemini client used across modules."""

from google import genai


client = genai.Client()
