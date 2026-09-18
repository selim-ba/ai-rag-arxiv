#!/usr/bin/env bash
# Stage 6, step 6: the three ways this container is allowed to fail.
#
# Each case has a defined answer, written in Stage 5 and tested there against a fake
# provider. This asks whether the CONTAINER still behaves that way - a question the unit
# tests cannot answer, because they never start a process, never read an environment
# variable and never lose a file.
#
# Written as a script rather than three rounds of start-curl-stop, so that "the failure
# modes still work" is one command someone can run in a year.
set -uo pipefail

IMAGE="${IMAGE:-arxiv-rag}"
PORT="${PORT:-8100}"
NAME="arxiv-rag-check"
FAILED=0

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

start() {  # start <docker run args...>
  cleanup
  docker run -d --rm --name "$NAME" -p "$PORT:8000" "$@" "$IMAGE" >/dev/null
  # Poll rather than sleep: startup loads 874 chunks and a 5 MB matrix, and a fixed sleep
  # is either too long every run or too short on the one run that matters.
  for _ in $(seq 1 60); do
    curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && return 0
    sleep 0.5
  done
  echo "  !! never became healthy"
  docker logs "$NAME" 2>&1 | tail -5
  return 1
}

status_of() { curl -s -o /tmp/body.$$ -w '%{http_code}' "$@"; }

ask() {
  status_of -X POST "http://localhost:$PORT/ask" \
    -H 'content-type: application/json' \
    -d '{"question":"How does PlaNet search for a good action sequence?"}'
}

expect() {  # expect <label> <actual> <wanted>
  if [ "$2" = "$3" ]; then
    echo "  ok    $1: $2"
  else
    echo "  FAIL  $1: got $2, wanted $3"
    FAILED=1
  fi
}

body_has() {  # body_has <label> <needle>
  if grep -q "$2" /tmp/body.$$; then
    echo "  ok    $1"
  else
    echo "  FAIL  $1 - body was: $(head -c 200 /tmp/body.$$)"
    FAILED=1
  fi
}

body_lacks() {  # body_lacks <label> <needle>
  if grep -q "$2" /tmp/body.$$; then
    echo "  FAIL  $1 - body leaked it: $(head -c 200 /tmp/body.$$)"
    FAILED=1
  else
    echo "  ok    $1"
  fi
}

echo
echo "1. no index in the image"
echo "   a container that dies because a volume was not mounted tells you nothing;"
echo "   one that starts and reports itself unready tells you exactly what is wrong."
if start --env-file .env --mount type=tmpfs,destination=/app/data/index; then
  expect "/health answers" "$(status_of "http://localhost:$PORT/health")" "200"
  body_has "  and reports an empty index" '"chunks_indexed":0'
  expect "/ask refuses" "$(ask)" "503"
fi

echo
echo "2. no API key"
echo "   the caller did nothing wrong, so this is ours: a 500, and nothing about the key."
if start; then
  expect "/health answers" "$(status_of "http://localhost:$PORT/health")" "200"
  body_has "  and says the key is missing" '"openai_key_configured":false'
  expect "/ask fails as ours" "$(ask)" "500"
  # Not `code` merely EXISTING. The first version of this line checked that, passed, and
  # hid the fact that an unset key was being reported as `internal_error`: an assertion
  # that asks less than the question it is standing in for.
  body_has "  code is misconfigured" '"code":"misconfigured"'
fi

echo
echo "3. a wrong API key"
echo "   upstream says 401. Passing that on would tell the caller to fix a request that"
echo "   was never the problem - and the key must not appear in the response."
# OPENAI_API_KEY, with no PT_ prefix: the setting carries
# validation_alias="OPENAI_API_KEY" so it also picks up a key already exported in a shell.
# The first version of this check set PT_OPENAI_API_KEY, which the settings ignore - so it
# tested a MISSING key for the second time and called it a wrong one.
if start -e OPENAI_API_KEY=sk-not-a-real-key-000; then
  expect "/ask fails as ours" "$(ask)" "500"
  body_has "  code is misconfigured" '"code":"misconfigured"'
  body_lacks "  the key is not echoed" 'sk-not-a-real-key'
fi

ask_status() {  # a bare status code, for loops
  curl -s -o /tmp/body.$$ -w '%{http_code}' -X POST "http://localhost:$PORT/ask" \
    -H 'content-type: application/json' \
    -d '{"question":"How does PlaNet search for a good action sequence?"}'
}

echo
echo "4. too many requests from one client"
echo "   the limiter runs BEFORE the model call, so a refusal costs nothing. Proven here"
echo "   by running with no API key at all: the first asks fail as ours (500), and the"
echo "   one past the burst is refused (429) without ever reaching the provider."
if start -e PT_RATE_LIMIT_BURST=2 -e PT_RATE_LIMIT_PER_MINUTE=1; then
  first=$(ask_status); second=$(ask_status); third=$(ask_status)
  expect "the first two are served" "$first$second" "500500"
  expect "the third is refused" "$third" "429"
  body_has "  as a client rate limit" '"code":"rate_limited"'
fi

echo
echo "5. the daily budget"
echo "   COSTS ONE REAL REQUEST (about \$0.0005). Budget set below the price of a single"
echo "   answer, so the second question is refused."
if start --env-file .env -e PT_DAILY_BUDGET_USD=0.0002; then
  expect "the first question is answered" "$(ask_status)" "200"
  expect "the second is refused" "$(ask_status)" "503"
  body_has "  as an exhausted budget" '"code":"budget_exhausted"'
  body_has "  with a time to come back" '"retry_after"'
fi

rm -f /tmp/body.$$
echo
[ "$FAILED" -eq 0 ] && echo "all container failure modes behave as specified" \
  || echo "SOME CHECKS FAILED"
exit "$FAILED"
