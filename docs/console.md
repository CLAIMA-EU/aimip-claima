# Approvals, the assistant, MCP and the web interface

aimip-claima can also be driven by an AI assistant (`aimip chat`), by another AI tool through MCP (`aimip mcp`) or from
a web page (`aimip ui`). The command is `aimip`. In all three, a person confirms each run before it is prepared or
started. This page explains how, and how to use `aimip` from scripts. The commands themselves are in the
[user guide](user-guide.md).

## Scripting: `--json` and exit codes

Every command accepts `--json`: the result is printed on standard output as
`{"aimip": "1", "ok": true, "result": …}` or `{"aimip": "1", "ok": false, "error": {"code", "message", "field",
"allowed", "next_command", …}}`; progress and warnings go to standard error. Exit codes: 0 success, 1 a failed
`doctor` check or a scheduler/execution failure, 2 a usage or validation error, 3 missing files with `--no-fetch`,
4 an approval that was refused, expired or no longer matches the plan, 5 an unstable rollout stopped cleanly
(`unstable_from` in the manifest), 75 when `watch` reaches `--timeout` before the
run finishes.

## Approval: the human confirms the exact plan

A run started by the agent, by MCP or from the web UI is executed only after a person confirms the exact plan. The
plan card shows a digest that covers the model, times, members, inputs (with their hashes), runtime, resources and
output directory; the person types its first 8 hex characters (console, chat) or presses Approve on the card shown
(web UI). An approval is used once, expires after 10 minutes, and is refused if anything in the plan has changed.
Downloads are confirmed separately, with their sizes.

Models that prepare their own inputs before the job (for example `dlesym-*`, which stage initial conditions) are
approved in two steps: first the preparation, then the final plan, whose digest now includes the hashes of the
prepared inputs. Only the second approval can submit the job.

From the console: `aimip run <model> … --approve` asks on the terminal and then prepares and submits.

## The agent (`aimip chat`)

The agent plans runs in conversation and proposes them; it cannot approve them. After a proposal, `aimip chat` shows
the card and asks you on the terminal for the digest prefix (and, if files are missing, for download consent). The
agent uses a Gemini model through Google's API or Vertex AI (see [agent.md](agent.md)). Everything the agent reads is
sent to that service: your messages, model cards, plan cards, run lists, run status and the job log excerpts it
looks at.

## MCP server (`aimip mcp`)

`aimip mcp` serves the same tools over MCP (standard input/output) for an MCP host. The `mcp` Python package is
installed in the container image and by `install/install-native.sh` (from `container/requirements-agent.txt`); in a
manual venv, install it from the checkout (`pip install -c container/constraints.txt -r container/requirements-agent.txt`).
Runs proposed through MCP are approved in your terminal with
`aimip approve <digest prefix>` (twice for models that prepare inputs).

## Web UI (`aimip ui`)

```bash
aimip ui --port 8765
```

prints a local address with a one-time token, `http://127.0.0.1:8765/?token=…`. The server listens on 127.0.0.1
only; on a cluster, forward the port from your laptop and open the printed address there:

```bash
ssh -N -L 8765:127.0.0.1:8765 <user>@<login node>
```

The page keeps the token for this browser tab only. Pages: models and cards, a run form with the same fields and
checks as the console, the plan card with Approve, the list of runs with status and log tail, and a run page with
the summary and plots of the monthly means (plots need `matplotlib` in the environment that runs `aimip`;
the container image has it; without it the run page shows the numbers as a table). The approve button sends the
digest of the card on the screen; the server refuses it if the plan has changed.

In a container installation the web UI, `aimip chat`, MCP and `run --approve` prepare runs; the submit command is
shown for you to run on the host, because Slurm commands run outside the container.

Errors you may see: "plan changed, approve again" (an input, resource or file changed after approval: re-plan and
approve the new card); "approval expired" (older than 10 minutes); "already used" (each approval works once);
"prepare-only approval cannot submit" (approve the final card of a model that prepares inputs); `plan.json`
missing or not matching (the run directory was edited after preparation: prepare it again).
