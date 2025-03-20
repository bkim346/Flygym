import os
import sys
import phi

def patch_phi_utils():
    # Locate the installed phi package directory
    phi_dir = os.path.dirname(phi.__file__)
    utils_file = os.path.join(phi_dir, "utils.py")

    # Confirm the file exists
    if not os.path.isfile(utils_file):
        print(f"[ERROR] utils.py not found at: {utils_file}")
        sys.exit(1)

    print(f"[INFO] Located utils.py at: {utils_file}")

    # Read the contents of the file
    with open(utils_file, "r", encoding="utf-8") as file:
        content = file.read()

    # Check if the patch is already applied
    if "inspect.getfullargspec" in content:
        print("[INFO] Patch already applied. No changes made.")
        return

    # Replace getargspec with getfullargspec
    patched_content = content.replace("inspect.getargspec", "inspect.getfullargspec")

    # Write the patched content back
    with open(utils_file, "w", encoding="utf-8") as file:
        file.write(patched_content)

    print("[SUCCESS] Patched inspect.getargspec -> inspect.getfullargspec")

if __name__ == "__main__":
    patch_phi_utils()
