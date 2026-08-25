#!/bin/sh
# haste one-command install (Linux/macOS):
#   curl -fsSL https://raw.githubusercontent.com/NodeNestor/haste/master/install.sh | sh
set -e
repo="NodeNestor/haste"
case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) asset="haste-macos-aarch64" ;;
  Linux-x86_64) asset="haste-linux-x86_64" ;;
  *) echo "haste: no prebuilt binary for $(uname -s)/$(uname -m) — build with cargo" >&2; exit 1 ;;
esac
dir="${HOME}/.local/bin"
mkdir -p "$dir"
url=$(curl -fsSL -H "User-Agent: haste-install" "https://api.github.com/repos/$repo/releases/latest" \
  | grep -o "\"browser_download_url\": *\"[^\"]*$asset\"" | grep -o "https://[^\"]*")
[ -n "$url" ] || { echo "haste: latest release has no asset $asset" >&2; exit 1; }
curl -fsSL -H "User-Agent: haste-install" -o "$dir/haste" "$url"
chmod +x "$dir/haste"
echo "haste installed to $dir/haste"
# swift, the fleet manager, ships beside haste (best-effort)
sasset=$(echo "$asset" | sed 's/^haste/swift/')
surl=$(curl -fsSL -H "User-Agent: haste-install" "https://api.github.com/repos/$repo/releases/latest" \
  | grep -o "\"browser_download_url\": *\"[^\"]*$sasset\"" | grep -o "https://[^\"]*") || true
if [ -n "$surl" ]; then
  curl -fsSL -H "User-Agent: haste-install" -o "$dir/swift" "$surl"
  chmod +x "$dir/swift"
  echo "swift installed to $dir/swift (heads-up: may shadow Apple's swift on macOS PATH)"
fi
case ":$PATH:" in
  *":$dir:"*) ;;
  *) echo "add $dir to your PATH" ;;
esac
