# Console, agent, MCP and web UI (development version 1.4.0.dev0)

This page describes the commands added after release 1.3.1. They are in the development version on `main`; the
latest release (1.3.1) is described in the rest of the documentation. `aimip --help` and `aimip <command> --help`
list every option.

## Commands

| command | what it does |
|---|---|
| `aimip doctor` | checks site settings, runtime, GPU, scheduler, data directory, permissions and image version; every failed check prints a hint (on a login node without a GPU the GPU check is informational) |
| `aimip models [--task …] [--family …] [--status …] [--installed]` | lists models, filtered |
| `aimip show <model> [--examples]` | model card: status, grid, time step and calendar, forcing windows, outputs, files cached or missing with sizes, cost note, licence, limitations, example commands |
| `aimip cite <model>` | references and licence attribution |
| `aimip plan <model> …` | checks a run without side effects and prints the plan card with its digest |
| `aimip run <model> …` | prepares a run directory (as before); `--submit` prepares and prints the submit command |
| `aimip submit <name-or-dir>` | submits a prepared run; checks that the plan has not changed since it was approved |
| `aimip watch`, `aimip logs [--follow]`, `aimip status`, `aimip summary` | follow a run; `logs` is bounded and strips terminal control characters |
| `aimip ls`, `aimip cancel <name-or-dir>` | list runs, cancel a Slurm job or a local run |
| `aimip completion bash` / `zsh` | shell completion script |

New options for `plan` and `run`:

- the model can be given as the first argument (`aimip run ace2-era5 …`) as well as with `-m`;
- run length as `--steps N`, `--length 20y|6m|30d|12h` or `--end <ISO time>`; the length is converted with the
  model's calendar and time step, and a length that is not a whole number of steps is refused;
- `--name demo` puts the run in `<run root>/demo`; every command accepts the name instead of a path;
- `--config experiment.yaml` reads the same keys as the flags; flags given on the command line override the file
  and are listed on the plan card.

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
