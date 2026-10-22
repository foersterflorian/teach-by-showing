#!/bin/sh
# Install the stand's start script for the stand account (DECISIONS.md §26).
# Run once, and again after the scripts change, from the repository root:
#
#   sudo sh tools/stand-setup/install.sh stand
#
# It copies the two scripts to /opt/kmi-stand/ (root-owned, readable by
# everyone) and the autostart entry into the account's ~/.config/autostart/.
# It creates no account and changes no group: the stand account must never
# be given Docker, which would let it read every home on this machine.
set -eu

account="${1:?usage: install.sh ACCOUNT}"
here="$(dirname "$0")"
home="$(getent passwd "$account" | cut -d: -f6)"
[ -n "$home" ] || { echo "install.sh: no account $account" >&2; exit 1; }

install -d -m 755 /opt/kmi-stand
install -m 644 "$here/../kiosk.py" "$here/../stand.py" /opt/kmi-stand/

install -d -m 755 -o "$account" -g "$account" "$home/.config" "$home/.config/autostart"
install -m 644 -o "$account" -g "$account" "$here/kmi-stand.desktop" "$home/.config/autostart/"

echo "installed: /opt/kmi-stand/{kiosk.py,stand.py}, $home/.config/autostart/kmi-stand.desktop"
