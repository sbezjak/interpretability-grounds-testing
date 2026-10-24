#!/usr/bin/env bash
# PreToolUse(Bash) WARNING hook - the tier-3 GUARD for the "preserve the full run
# log" rule in CLAUDE.md. Non-blocking on purpose: tail/head are fine for a quick
# peek, so this warns rather than denies. It fires only when a run (pytest /
# python / node / a script) is itself piped into tail/head - the case where the
# discarded part is exactly where the receipts live. Each part of the command
# (split at ; && ||) is checked on its own, and only the stages before the
# tail/head count, by their first word: `grep x | head` next to a python call, or
# `cat scripts/foo.sh | head`, is a peek at a file, not a cut run.
#
# Reads the hook JSON on stdin; emits a warning as JSON on stdout, then exits 0
# (allow). Points at scripts/run-evidence.sh, which captures the whole stream.
set -uo pipefail

input=$(cat)
cmd=$(printf '%s' "$input" | jq -r '.tool_input.command // ""' 2>/dev/null || printf '')
[ -z "$cmd" ] && exit 0

# a stage is a run if its first word (after env / time / caffeinate prefixes) is a runner
runner_re='^[[:space:]]*((env|time|caffeinate)([[:space:]]+-[^[:space:]]+)*[[:space:]]+|[A-Za-z_]+=[^[:space:]]*[[:space:]]+)*([^[:space:]]*/)?(pytest|python3?(\.[0-9]+)?|node|npm|npx|pnpm|yarn|uv|bash|sh|[^[:space:]]+\.(sh|py)|run-evidence[^[:space:]]*)([[:space:]]|$)'
trunc_re='\|[[:space:]]*(tail|head)([[:space:]]|$)'

hit=0
parts=$cmd
for sep in '&&' '||' ';'; do parts=${parts//"$sep"/$'\n'}; done
while IFS= read -r part; do
  printf '%s' "$part" | grep -Eq "$trunc_re" || continue
  before=$(printf '%s' "$part" | sed -E 's/\|[[:space:]]*(tail|head)([[:space:]].*)?$//')
  IFS='|' read -ra stages <<< "$before"
  for stage in "${stages[@]}"; do
    printf '%s' "$stage" | grep -Eq "$runner_re" && hit=1
  done
done <<< "$parts"

if [ "$hit" = 1 ]; then
  msg='[evidence-guard] This pipes a run through tail/head - the truncated part is where the receipts (request_id / batch_id / per-item traces) live. Capture the FULL stream instead: scripts/run-evidence.sh <name> -- <cmd>  (or  > evidence/raw-logs/<name>.log 2>&1).'
  jq -nc --arg m "$msg" '{systemMessage:$m, hookSpecificOutput:{hookEventName:"PreToolUse", additionalContext:$m}}'
fi
exit 0
