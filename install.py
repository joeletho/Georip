import os
import platform
import subprocess
import sys

def run_script(script_path):
    """ Run the appropriate shell script (install.sh or install.bat) """
    if platform.system() == "Windows":
        subprocess.run([script_path], shell=True)
    else:
        subprocess.run(["bash", script_path])

def main():
    setup_dir = os.path.join(os.getcwd(), "setup")

    if not os.path.isdir(setup_dir):
        print(f"Error: The 'setup' directory does not exist at {setup_dir}.")
        sys.exit(1)

    requirements_path = os.path.join(setup_dir, "requirements.txt")
    if not os.path.isfile(requirements_path):
        print(f"Error: The 'requirements.txt' file does not exist at {requirements_path}.")
        sys.exit(1)

    current_os = platform.system()

    if current_os == "Windows":
        print("Detected Windows OS. Running install.bat...")
        install_script = os.path.join(setup_dir, "install.bat")
        run_script(install_script)
    elif current_os == "Darwin" or current_os == "Linux":
        print("Detected macOS/Linux OS. Running install.sh...")
        install_script = os.path.join(setup_dir, "install.sh")
        run_script(install_script)
    else:
        print(f"Unsupported OS: {current_os}. This script only supports Windows, macOS, and Linux.")
        sys.exit(1)

if __name__ == "__main__":
    main()
