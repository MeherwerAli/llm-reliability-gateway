# Two-to-three-minute demo script

1. Start the Compose topology and show the gateway, PostgreSQL, and Redis health state.
2. Run `uv run python scripts/demo.py`.
3. Compare the estimate with actual cost and show the price version date.
4. Point out the upstream response contains `[REDACTED_EMAIL]`, while the policy audit contains only content hashes.
5. Show the idempotent replay header and the single usage ledger record.
6. Trigger a malformed structured-output fixture and a failed webhook fixture from the tests; explain the two-repair bound and terminal dead-letter state.
7. Close with the limits: heuristic injection indicators mitigate risk but do not prevent every attack, and the local runtime is not capacity evidence.

The recorded video and public URL remain publication gates; this script is the reproducible local path.
