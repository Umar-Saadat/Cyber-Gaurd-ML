# app.py - Main Entrypoint for CyberGuard ML Modern Web Application
# Launches the modern HTML5 / CSS3 / JavaScript SOC Dashboard server.
import sys
import webbrowser
from server import app

if __name__ == "__main__":
    port = 5000
    host = "127.0.0.1"
    url = f"http://{host}:{port}"
    
    print("\n" + "=" * 65)
    print("  🛡️  CYBERGUARD ML — MODERN SOC DASHBOARD")
    print(f"  🚀 Launching web server at: {url}")
    print("  💡 Frontend: HTML5, CSS3, JavaScript (ES6+)")
    print("  💡 Backend: Flask REST API + Scikit-Learn Pipelines")
    print("=" * 65 + "\n")
    
    # Auto-open browser
    try:
        webbrowser.open_new_tab(url)
    except Exception:
        pass
        
    app.run(host=host, port=port, debug=False)
