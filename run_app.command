#!/bin/zsh

cd "$(dirname "$0")" || exit 1

if [ -x /opt/anaconda3/bin/python3 ]; then
  PYTHON=/opt/anaconda3/bin/python3
else
  PYTHON=$(command -v python3)
fi

if [ -z "$PYTHON" ]; then
  echo "Python 3 was not found."
  printf "Press Enter to close this window..."
  read -r _
  exit 1
fi

"$PYTHON" "HCI-images-deep learning/test_2.py"
status=$?

if [ "$status" -ne 0 ]; then
  echo
  echo "The app stopped with an error."
  printf "Press Enter to close this window..."
  read -r _
fi

exit "$status"
