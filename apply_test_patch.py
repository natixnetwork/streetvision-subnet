#!/usr/bin/env python3
"""
Apply/Remove Test Authentication Bypass
=======================================

This script temporarily modifies validator_proxy.py to add authentication bypass
for testing purposes. Can be applied and reverted easily.
"""

import os
import shutil
from pathlib import Path


def apply_test_bypass():
    """Apply the test authentication bypass to validator_proxy.py"""
    
    proxy_file = Path("neurons/validator_proxy.py")
    backup_file = Path("neurons/validator_proxy.py.backup")
    
    if not proxy_file.exists():
        print("❌ neurons/validator_proxy.py not found!")
        return False
    
    # Create backup
    shutil.copy2(proxy_file, backup_file)
    print(f"✅ Created backup: {backup_file}")
    
    # Read current content
    with open(proxy_file, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "test_bypass" in content:
        print("⚠️  Test bypass already applied!")
        return True
    
    # Find the authenticate_token method and add bypass
    lines = content.split('\n')
    new_lines = []
    
    for i, line in enumerate(lines):
        new_lines.append(line)
        
        # Add bypass after "public_key_bytes = base64.b64decode(public_key_bytes)"
        if "public_key_bytes = base64.b64decode(public_key_bytes)" in line:
            # Add the bypass code with proper indentation
            new_lines.extend([
                "",
                "        # TEST BYPASS: Allow test_bypass token for local testing",
                "        if public_key_bytes == b\"test_bypass\":",
                "            bt.logging.info(\"Using test authentication bypass\")",
                "            return public_key_bytes"
            ])
    
    # Write modified content
    with open(proxy_file, 'w') as f:
        f.write('\n'.join(new_lines))
    
    print("✅ Applied test authentication bypass")
    print("⚠️  Remember to revert after testing!")
    return True


def remove_test_bypass():
    """Remove the test authentication bypass from validator_proxy.py"""
    
    proxy_file = Path("neurons/validator_proxy.py")
    backup_file = Path("neurons/validator_proxy.py.backup")
    
    if not backup_file.exists():
        print("❌ No backup file found! Cannot safely revert.")
        return False
    
    # Restore from backup
    shutil.copy2(backup_file, proxy_file)
    os.remove(backup_file)
    
    print("✅ Reverted to original validator_proxy.py")
    print("✅ Removed backup file")
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply/remove test authentication bypass")
    parser.add_argument("action", choices=["apply", "remove"], 
                       help="Apply or remove the test bypass")
    
    args = parser.parse_args()
    
    if args.action == "apply":
        success = apply_test_bypass()
    else:
        success = remove_test_bypass()
    
    if success:
        print(f"\n✅ {args.action.title()} completed successfully!")
    else:
        print(f"\n❌ {args.action.title()} failed!")


if __name__ == "__main__":
    main()