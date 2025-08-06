@echo off
cd /d "C:\Users\Nouri\Documents\GitHub\Een\website"
echo Starting local server on http://localhost:8003
echo Open http://localhost:8003/meta-optimal-landing.html in your browser
echo Press Ctrl+C to stop the server
python -m http.server 8003
pause