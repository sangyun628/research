#!/usr/bin/env bash
# Mermaid 다이어그램 검증 — 마크다운 파일에서 ```mermaid 블록을 모두 추출해
# mermaid-cli(mmdc)로 렌더링 가능한지 일괄 확인한다.
#
# 사용법:
#   scripts/validate-mermaid.sh path/to/doc.md [path/to/another.md ...]
#
# 종료코드: 0 = 모든 블록 통과, 1 = 하나라도 실패
#
# 의존성: node/npx (mmdc는 npx로 자동 설치됨)
set -euo pipefail

if [ $# -eq 0 ]; then
  echo "Usage: $0 <markdown-file> [<markdown-file> ...]" >&2
  exit 2
fi

WORK="$(mktemp -d -t mmd-check.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

TOTAL_FAIL=0

for MD in "$@"; do
  if [ ! -f "$MD" ]; then
    echo "❌ 파일 없음: $MD" >&2
    TOTAL_FAIL=$((TOTAL_FAIL + 1))
    continue
  fi

  echo ""
  echo "── $MD ──"

  # mermaid 블록을 개별 파일로 추출
  COUNT=$(python3 - "$MD" "$WORK" <<'PY'
import re, sys, pathlib, hashlib
md_path = pathlib.Path(sys.argv[1])
out_dir = pathlib.Path(sys.argv[2])
key = hashlib.md5(str(md_path.resolve()).encode()).hexdigest()[:8]
content = md_path.read_text(encoding="utf-8")
blocks = re.findall(r"```mermaid\n(.*?)\n```", content, re.DOTALL)
for i, b in enumerate(blocks, 1):
    (out_dir / f"{key}_block_{i}.mmd").write_text(b, encoding="utf-8")
print(len(blocks))
PY
)

  if [ "$COUNT" -eq 0 ]; then
    echo "  (mermaid 블록 없음)"
    continue
  fi

  KEY=$(python3 -c "import sys, hashlib, pathlib; print(hashlib.md5(str(pathlib.Path(sys.argv[1]).resolve()).encode()).hexdigest()[:8])" "$MD")

  for i in $(seq 1 "$COUNT"); do
    BLOCK="$WORK/${KEY}_block_${i}.mmd"
    OUT="$WORK/${KEY}_block_${i}.svg"
    if ERR=$(npx --yes -p @mermaid-js/mermaid-cli mmdc -i "$BLOCK" -o "$OUT" 2>&1); then
      echo "  ✅ block $i"
    else
      echo "  ❌ block $i"
      # mermaid 에러 라인만 추출해 출력
      echo "$ERR" | grep -E "(Parse error|Error|Expecting|error on line)" | head -5 | sed 's/^/      /'
      TOTAL_FAIL=$((TOTAL_FAIL + 1))
    fi
  done
done

echo ""
if [ "$TOTAL_FAIL" -eq 0 ]; then
  echo "✅ 모든 mermaid 블록 통과"
  exit 0
else
  echo "❌ 실패한 블록: $TOTAL_FAIL 개"
  exit 1
fi
