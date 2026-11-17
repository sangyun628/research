#!/bin/sh
set -eu

demo_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
research_root=$(CDPATH= cd -- "$demo_dir/../.." && pwd)
demo_python="$research_root/.repos/jevlike/.venv/bin/python"

if [ ! -x "$demo_python" ]; then
  echo "Jevlike의 Python 환경이 없습니다. $demo_dir/local-demo.md 의 설치 절차를 실행해 주세요." >&2
  exit 1
fi

exec "$demo_python" "$demo_dir/demo_server.py" "$@"
