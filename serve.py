import http.server
import socketserver
import os
import webbrowser
from threading import Timer

PORT = 8000

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

def open_browser():
    webbrowser.open(f'http://localhost:{PORT}')

if __name__ == "__main__":
    # Change to web directory
    web_dir = 'web'
    if os.path.exists(web_dir):
        os.chdir(web_dir)
        print(f"Serving from: {os.getcwd()}")
    else:
        print("Web directory not found. Run create_web_metadata.py first.")
        exit(1)
    
    # Start server
    with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
        print(f"Server running at http://localhost:{PORT}")
        print("Press Ctrl+C to stop")
        
        # Open browser after a short delay
        Timer(1.0, open_browser).start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")