#!/usr/bin/env bash
# PreToolUse(Bash) WARNING hook - the tier-3 GUARD for the "preserve the full run
# log" rule in CLAUDE.md. Non-blocking on purpose: tail/head are fine for a quick
# peek, so this warns rather than denies. It fires only when a command BOTH runs
# something (pytest / python / a script) AND pipes that run through tail/head -
# the case where the discarded part is exactly where the receipts live.
#
# Reads the hook JSON on stdin; emits a warning as JSON on stdout, then exits 0
# (allow). Points at scripts/run-evidence.sh, which captures the whole stream.
set -uo pipefail

input=$(cat)
cmd=$(printf '%s' "$input" | jq -r '.tool_input.command // ""' 2>/dev/null || printf '')
[ -z "$cmd" ] && exit 0

runner_re='pytest|python3?|node|npm|pnpm|yarn|scripts/|run-evidence'
trunc_re='\|[[:space:]]*(tail|head)([[:space:]]|$)|tee[[:space:]]+[^|]*\|[[:space:]]*(tail|head)'

if printf '%s' "$cmd" | grep -Eq "$runner_re" && printf '%s' "$cmd" | grep -Eq "$trunc_re"; then
  msg='[evidence-guard] This pipes a run through tail/head - the truncated part is where the receipts (request_id / batch_id / per-item traces) live. Capture the FULL stream instead: scripts/run-evidence.sh <name> -- <cmd>  (or  > evidence/raw-logs/<name>.log 2>&1).'
  jq -nc --arg m "$msg" '{systemMessage:$m, hookSpecificOutput:{hookEventName:"PreToolUse", additionalContext:$m}}'
fi
exit 0
