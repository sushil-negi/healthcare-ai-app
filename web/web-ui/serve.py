#!/usr/bin/env python3
"""
Simple web server to serve the Healthcare AI chat interface
"""

import http.server
import os
import socketserver
import webbrowser
from pathlib import Path

# Configuration
PORT = 3001
DIRECTORY = Path(__file__).parent


class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # Add CORS headers to allow requests to the API
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        # Handle preflight requests
        self.send_response(200)
        self.end_headers()


def start_server():
    """Start the web server"""
    try:
        with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
            print(f"🌐 Healthcare AI Chat Interface")
            print(f"📍 Serving at: http://localhost:{PORT}")
            print(f"📄 Open: http://localhost:{PORT}/healthcare-chat.html")
            print(f"🔗 API: http://localhost:8082")
            print(f"")
            print(f"⚠️  Make sure your Healthcare AI API is running on port 8082")
            print(
                f"   kubectl port-forward -n healthcare-ai-staging service/healthcare-ai-v2-service 8082:80"
            )
            print(f"")
            print(f"Press Ctrl+C to stop the server")

            # Try to open browser automatically
            try:
                webbrowser.open(f"http://localhost:{PORT}/healthcare-chat.html")
            except:
                pass

            httpd.serve_forever()
    except KeyboardInterrupt:
        print(f"\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")


if __name__ == "__main__":
    start_server()
