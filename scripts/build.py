#!/usr/bin/env python3
import subprocess
import sys

def run_command(cmd, description):
    print(f"🔧 {description}...")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ {description} failed")
        sys.exit(1)
    print(f"✅ {description} completed")

def main():
    print("🚀 Building SmartML Auto package...")
    
    run_command("rm -rf dist/ build/ *.egg-info/", "Cleaning previous builds")
    run_command("pip3 install --upgrade build twine", "Installing build tools")
    run_command("python3 -m build", "Building package")
    run_command("twine check dist/*", "Verifying package")
    
    print("\n🎉 Build completed successfully!")

if __name__ == "__main__":
    main()
