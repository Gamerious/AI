#!/usr/bin/env bash
# Installs the NEXUS control SSH key into authorized_keys ON THE POD.
# Self-verifying: refuses to run if there is no GPU (i.e. you're on your laptop).
set -e

if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "!!! NOT ON THE POD (no GPU found)."
  echo "!!! Run this in the RunPod WEB TERMINAL (browser), not on your laptop."
  exit 1
fi

echo "On pod:"; nvidia-smi -L

mkdir -p ~/.ssh
KEY="$(dirname "$0")/pod_key.pub"
if grep -qF "runpod-nexus" ~/.ssh/authorized_keys 2>/dev/null; then
  echo "key already present"
else
  cat "$KEY" >> ~/.ssh/authorized_keys
  echo "key appended"
fi
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
echo "KEY_OK_ON_POD"
