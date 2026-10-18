# ntfy iOS delivery

## Goal
Pick-flip ntfy messages show as iPhone notifications (today they only appear inside the app).

## State
Research only, nothing implemented. Sender: scripts/capture_scheduler.py:2322-2346 send_ntfy, headers Title/Priority/Tags (line 2330), server ntfy.sh.
Docs (docs.ntfy.sh/known-issues, /config): iOS push on ntfy.sh is a Firebase-to-APNs poll_request; docs say stale Firebase subscription is the common cause, fix = remove topic and re-add. "View not refreshing" bug (#267): notifications arrive but list updates only on pull-down. Instant delivery is Android-only (FAQ); iOS app described as buggy. upstream-base-url/base-url mismatch applies only to self-hosted (not us).

## Ranked fixes
1. Stale Firebase subscription: in iOS app delete the topic, re-add as https://ntfy.sh/<topic> (default server ntfy.sh). Verify: owner locks phone, agent sends one test (primary orchestrator only); banner appears.
2. iOS Settings (owner says fine) plus Focus/Do Not Disturb or Scheduled Summary for ntfy: not re-asked.
3. Sender: add header "Priority: 5" (urgent; high=4 already) and optionally "Click"; line 2330. Low likelihood, docs show no iOS priority gating.
4. Update/reinstall ntfy app (reinstall re-registers APNs token); requires re-subscribe.

## Alternative channel
Free Telegram bot (BotFather token, sendMessage HTTPS POST) or Pushover trial excluded (paid); Telegram is zero cost.

## Next
Owner does fix 1; orchestrator sends test; if still silent, build Telegram sender beside send_ntfy.

## Open
No doc/issue found tying missing iOS banners to message size or priority.
