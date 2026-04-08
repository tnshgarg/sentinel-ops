#!/bin/bash
export API_PORT=8000
uvicorn server.app:app --host 0.0.0.0 --port $API_PORT &
streamlit run ui.py --server.port 7860 --server.address 0.0.0.0
