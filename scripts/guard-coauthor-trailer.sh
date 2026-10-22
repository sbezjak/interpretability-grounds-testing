#!/usr/bin/env bash
# PreToolUse(Bash) BLOCKING hook - the tier-3 GUARD for the CLAUDE.md rule
# "End commit messages at the body. The user is the sole author."
# Denies any `git commit` whose message carries a Co-Authored-By trailer or a
# Claude-Session line - the harness's ready-made default that outranked the
# stated rule once (see working_with_ai.md, the trailer incident).
#
# Reads the hook JSON on stdin; on a match, emits a deny decision as JSON on
# stdout and exits 0. Otherwise prints nothing (allow).
set -uo pipefail

input=$(cat)
cmd=$(printf '%s' "$input" | jq -r '.tool_input.command // ""' 2>/dev/null || printf '')
[ -z "$cmd" ] && exit 0

# Is this a git commit (not `git log`, `git show`, etc.)?
if printf '%s' "$cmd" | grep -Eq '(^|[^[:alnum:]])git[[:space:]]+([^|;&]*[[:space:]])?commit([[:space:]]|$)'; then
  if printf '%s' "$cmd" | grep -Eiq 'Co-Authored-By:|Claude-Session:'; then
    reason='CLAUDE.md: commits end at the body, the user is the sole author. Remove the Co-Authored-By / Claude-Session trailer before committing.'
    jq -nc --arg r "$reason" '{hookSpecificOutput:{hookEventName:"PreToolUse", permissionDecision:"deny", permissionDecisionReason:$r}}'
  fi
fi
exit 0
