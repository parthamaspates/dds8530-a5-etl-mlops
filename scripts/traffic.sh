#!/usr/bin/env bash
# Send prediction traffic to a running API so Prometheus and Grafana have data.
#   scripts/traffic.sh [URL] [REQUESTS]        (defaults: http://localhost:8000, 300)
# Every 25th request is deliberately invalid (discount 1.5) and should return 422.
set -euo pipefail
URL="${1:-http://localhost:8000}"
N="${2:-300}"
DIR="$(cd "$(dirname "$0")" && pwd)"
GOOD="$DIR/sample_orders.json"
BAD='{"orders":[{"order_date":"2017-11-20","ship_date":"2017-11-24","ship_mode":"Standard Class","segment":"Consumer","region":"West","category":"Office Supplies","sub_category":"Paper","product_name":"Xerox 1967","sales":25.92,"quantity":4,"discount":1.5}]}'
ok=0; rejected=0; other=0
for i in $(seq 1 "$N"); do
  if (( i % 25 == 0 )); then
    code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$URL/predict" -H 'content-type: application/json' -d "$BAD")
  else
    code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$URL/predict" -H 'content-type: application/json' -d @"$GOOD")
  fi
  case "$code" in 200) ok=$((ok+1));; 422) rejected=$((rejected+1));; *) other=$((other+1));; esac
done
echo "requests: $N  200 OK: $ok  422 rejected: $rejected  other: $other"
[ "$other" -eq 0 ]
