"""
 CapsuleRAG - Modular Architecture

REFACTORED: Original 2100+ line server.py split into focused modules.

 New Structure:
  main.py (278 lines) - FastAPI app
  modules/ - 7 focused modules

 To run: python main.py

Original saved as: server_old_monolithic.py
"""

print(" CapsuleRAG has been modularized!")
print(" Use: python main.py")

if __name__ == "__main__":
    import subprocess
    import sys
    subprocess.run([sys.executable, "main.py"])
