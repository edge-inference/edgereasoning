#! /usr/bin/env bash
# Start the OpenAI-compatible API (new shell)
python -m vllm.entrypoints.openai.api_server \
  --model microsoft/DialoGPT-small\
  --port 8000 \
  --max-num-seqs 32 \
  --collect-detailed-traces all \
  --show-hidden-metrics-for-version=0.9 \
  --otlp-traces-endpoint http://localhost:4317
