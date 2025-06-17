#!/usr/bin/env python3
"""
Fixed Dashboard Demo - Corrected port management and server handling
"""

import http.server
import logging
import socketserver
import threading
import time
import webbrowser
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DashboardDemo:
    def __init__(self):
        self.demo_dir = Path(__file__).parent
        self.servers = {}

    def find_available_port(self, start_port=8888):
        """Find an available port starting from start_port"""
        import socket

        for port in range(start_port, start_port + 10):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("localhost", port))
                return port
            except OSError:
                continue
        return None

    def start_server(self, name, port=None):
        """Start a server on an available port"""
        if port is None:
            port = self.find_available_port(8888)

        if port is None:
            logger.error(f"❌ No available ports for {name}")
            return None

        try:
            demo_dir = self.demo_dir  # Capture the demo_dir in local scope

            class Handler(http.server.SimpleHTTPRequestHandler):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, directory=str(demo_dir), **kwargs)

                def log_message(self, format, *args):
                    # Suppress HTTP logs
                    pass

            httpd = socketserver.TCPServer(("localhost", port), Handler)

            def serve():
                httpd.serve_forever()

            thread = threading.Thread(target=serve, daemon=True)
            thread.start()

            self.servers[name] = {"port": port, "server": httpd, "thread": thread}

            logger.info(f"✅ {name} server started on http://localhost:{port}")
            return port

        except Exception as e:
            logger.error(f"❌ Failed to start {name} server: {e}")
            return None

    def start_all_dashboards(self):
        """Start all dashboard servers"""
        logger.info("🚀 Starting dashboard servers...")

        # Start main dashboard server
        port = self.start_server("MLOps Dashboard")
        if port:
            self.unified_url = f"http://localhost:{port}/unified_dashboard.html"

        return port is not None

    def show_dashboard_urls(self):
        """Display dashboard URLs"""
        if hasattr(self, "unified_url"):
            print(f"\n🌐 Dashboard URL:")
            print(f"🚀 Unified Dashboard: {self.unified_url}")
        else:
            print(f"\n❌ No dashboards available")

    def open_dashboards(self):
        """Open dashboards in browser"""
        if hasattr(self, "unified_url"):
            try:
                logger.info("🌐 Opening unified dashboard in browser...")
                webbrowser.open(self.unified_url)
                logger.info("✅ Unified dashboard opened in browser")
            except Exception as e:
                logger.error(f"❌ Failed to open browser: {e}")

    def run_demo(self):
        """Run the dashboard demo"""
        print("\n" + "=" * 60)
        print("📊 MLOPS DASHBOARD DEMONSTRATION")
        print("=" * 60)

        print(f"\n🎯 This demo showcases:")
        print(f"• 🚀 Unified Dashboard - All-in-one tabbed interface")
        print(f"  📊 Platform Tab - Multi-tenant overview & architecture")
        print(f"  📈 Monitoring Tab - Drift detection & alerts")
        print(f"  ⚡ Real-time Tab - Live metrics & operations")

        # Start servers
        if self.start_all_dashboards():
            self.show_dashboard_urls()

            response = input(f"\nOpen dashboards in browser? (y/n): ")
            if response.lower() == "y":
                self.open_dashboards()

            print(f"\n✨ Dashboard demo ready!")
            print(f"💡 Explore the interactive features:")
            print(f"   • A/B testing visualization")
            print(f"   • Real-time metric updates")
            print(f"   • Model performance charts")
            print(f"   • Drift detection alerts")

            print(f"\nPress Ctrl+C to stop servers...")

            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print(f"\n🛑 Stopping dashboard servers...")
                self.cleanup()
        else:
            print(f"\n❌ Failed to start dashboard servers")

    def cleanup(self):
        """Stop all servers"""
        for name, server_info in self.servers.items():
            try:
                server_info["server"].shutdown()
                logger.info(f"✅ Stopped {name}")
            except:
                pass


def main():
    """Run the dashboard demo"""
    demo = DashboardDemo()
    demo.run_demo()


if __name__ == "__main__":
    main()
