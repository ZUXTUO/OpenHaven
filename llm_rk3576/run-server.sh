#!/bin/bash
CHAT_MODEL="/home/olsc/rkllm/models/Qwen2.5-1.5B/qwen2.5_1.5b_w8a8_v123.rkllm"
EMBED_MODEL="/home/olsc/rkllm/models/Embedding_Gemma-300M/embedding_gemma_rk3576_w8a8.rkllm"

sudo ./llm_server --chat_model "$CHAT_MODEL" --embed_model "$EMBED_MODEL" --port 8080 --max_tokens 512 --max_context 2048 --mock_embed