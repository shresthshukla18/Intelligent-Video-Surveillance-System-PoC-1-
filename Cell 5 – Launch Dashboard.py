# cell 5

import os
import time
import subprocess
import re

print("Starting Streamlit server...")
# 1. Run Streamlit quietly in the background
subprocess.Popen([
    "streamlit", "run", "dashboard.py",
    "--server.port", "8501",
    "--server.headless", "true"
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

time.sleep(5) # Give Streamlit a few seconds to boot up

# 2. Download cloudflared if it isn't already there
if not os.path.exists("cloudflared-linux-amd64"):
    print("Downloading Cloudflared...")
    os.system("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64")
    os.system("chmod +x cloudflared-linux-amd64")

print("Starting Cloudflare Tunnel...")
# 3. Run Cloudflare in the background and save its messy logs to a text file
os.system("nohup ./cloudflared-linux-amd64 tunnel --url http://localhost:8501 > tunnel_logs.txt 2>&1 &")

time.sleep(8) # Wait for the tunnel to negotiate with the server

# 4. Search the messy logs for the clean URL and print it
print("\n" + "="*50)
print(" YOUR DASHBOARD IS READY!")
print("="*50)

found_url = False
with open("tunnel_logs.txt", "r") as f:
    for line in f:
        if "trycloudflare.com" in line:
            # Extract just the URL from the log line
            match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
            if match:
                print(f"\n Click this link to open: {match.group(0)}\n")
                found_url = True
                break

if not found_url:
    print("\n Still waiting for the URL... Try running this specific block again in a few seconds:")
    print("!grep -o 'https://.*\.trycloudflare\.com' tunnel_logs.txt")
