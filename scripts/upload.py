#!/usr/bin/env python3
import subprocess
import sys

def main():
    print("📤 Uploading SmartML Auto to PyPI...")
    print("⚠️  Make sure you have a PyPI account and are logged in!")
    
    response = input("🔥 Ready to make smartml-auto globally available? (yes/no): ")
    if response.lower() != 'yes':
        print("❌ Upload cancelled")
        sys.exit(0)
        
    print("🚀 Uploading to PyPI...")
    result = subprocess.run("python3 -m twine upload dist/*", shell=True)
    
    if result.returncode == 0:
        print("\n🎉 SUCCESS! Your package is now LIVE on PyPI!")
        print("🌟 Anyone can now install it with:")
        print("   pip install smartml-auto")
        print("📊 Check your package at: https://pypi.org/project/smartml-auto/")
    else:
        print("❌ Upload failed - check your PyPI credentials")

if __name__ == "__main__":
    main()
