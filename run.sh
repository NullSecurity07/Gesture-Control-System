#!/bin/bash
export QT_QPA_PLATFORM=xcb
cd "$(dirname "$0")"
source venv_mp/bin/activate
python "$@"
