#!/bin/bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 &
streamlit run ui.py --server.port 7860 --server.address 0.0.0.0
