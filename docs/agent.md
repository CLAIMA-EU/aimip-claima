# The planning assistant (`aimip chat`)

`aimip chat` plans runs in plain language with the same commands as the [user guide](user-guide.md). It shows the full plan (model, start,
end, steps, members, initial state, GPU-hour estimate, the job script) and, only after you type exactly `yes`, prints
the `aimip submit <dir>` line for you to run. It never submits a job itself and cannot download data; for a model that
is not installed yet, or for DLESyM (whose preparation downloads an ERA5 state), it gives you the command to run.

It uses Google Gemini (default model `gemini-3.8-flash`, change with `AIMIP_AGENT_MODEL`). Credentials, one of:

- **Gemini API key:** put the key in a file only you can read and point to it:

  ```bash
  mkdir -p "$HOME/.aimip" && install -m 600 /dev/null "$HOME/.aimip/gemini.key" && $EDITOR "$HOME/.aimip/gemini.key"
  AIMIP_GEMINI_KEY_FILE="$HOME/.aimip/gemini.key" aimip chat
  ```

  (`GEMINI_API_KEY` in the environment also works.)
- **Vertex AI** (billed to a Google Cloud project): `AIMIP_GEMINI_VERTEX_PROJECT=<project>`
  (`AIMIP_GEMINI_VERTEX_LOCATION`, default `global`) and either `AIMIP_GEMINI_VERTEX_TOKEN_FILE` (a 0600 file holding
  the output of `gcloud auth print-access-token`, valid about one hour) or `GOOGLE_APPLICATION_CREDENTIALS` (a
  service-account JSON file). Vertex does not accept Gemini API keys.

Your messages and the tool results the assistant reads (model lists, run plans, paths in your run root) are sent
to Google's API. Keys are never put on a command line and never written to logs. Conversations are logged, without the key, in
`<run root>/chat-logs/`.
