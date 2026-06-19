#!/bin/bash
set -e
read -s -p "Password: " PASS
echo
for d in */; do
    name=${d%/}
    echo "Encrypting $name..."
    if 7z a -p"$PASS" -mhe=on "${name}.7z" "$name" >/dev/null; then
        rm -rf "$name"
        echo "  -> ${name}.7z (deleted ${name}/)"
    else
        echo "  FAILED: $name (kept directory)" >&2
    fi
done
echo "Done."

# 7z x name.7z
