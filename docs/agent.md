# The planning assistant (`aimip chat`)

`aimip chat` plans runs in plain language with the same commands as the [user guide](user-guide.md). When it proposes
a run, it shows the full plan card (model, start, end, steps, members, initial state, estimates, output directory) and
its digest. Nothing happens until you type the first 8 hex characters of that digest on your terminal; missing files
are downloaded only after a separate yes with their sizes. Then the run is prepared and, in a native installation,
submitted; models that prepare inputs (DLESyM) ask a second time for the final plan. The agent itself can never
approve. In a container installation the run is prepared and the submit command is shown for you to run on the host.
Details: [console.md](console.md#approval-the-human-confirms-the-exact-plan).

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
