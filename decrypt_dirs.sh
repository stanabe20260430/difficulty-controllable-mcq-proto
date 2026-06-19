#!/bin/bash
set -e
read -s -p "Password: " PASS
echo
for f in *.7z; do
    [ -e "$f" ] || continue
    name=${f%.7z}
    echo "Decrypting $name..."
    if 7z x -p"$PASS" "$f" >/dev/null; then
        rm -f "$f"
        echo "  -> ${name}/ (deleted ${f})"
    else
        echo "  FAILED: $f (kept archive)" >&2
    fi
done
echo "Done."
