#!/bin/bash
g++ -std=c++17 -O2 -pthread -o llm_server src/llm_server.cpp \
    -lrkllmrt -lm