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
case ":$PATH:" in
  *":$dir:"*) ;;
  *) echo "add $dir to your PATH" ;;
esac
