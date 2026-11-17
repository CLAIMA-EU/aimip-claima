/* Client controller for `aimip ui` (strict CSP `default-src 'self'`, zero external dependencies, no cookies). */
(function (root) {
  "use strict";

  const POLL_INTERVAL_MS = 15000;

  /**
   * Pure, DOM-free state machine governing the Plan & Approve card vs background Assistant proposals.
   * Exported on `globalThis.__AIMIP_CREATE_PLAN_STATE__` for unit testing and used directly by the UI.
   */
  function createPlanApprovalState() {
    let drawnPlan = null;
    let latestPlan = null;
    let allowDownload = false;

    return {
      getDrawnPlan: function () {
        return drawnPlan;
      },
      getLatestPlan: function () {
        return latestPlan;
      },
      getAllowDownload: function () {
        return allowDownload;
      },
      setAllowDownload: function (checked) {
        allowDownload = Boolean(checked);
      },
      onFormPlan: function (plan) {
        latestPlan = plan || null;
        return latestPlan;
      },
      onChatProposal: function (plan) {
        if (plan) {
          latestPlan = plan;
        }
        return latestPlan;
      },
      drawPlan: function (explicitPlan) {
        const next = explicitPlan !== undefined ? explicitPlan : latestPlan;
        drawnPlan = next || null;
        if (explicitPlan !== undefined && explicitPlan !== null) {
          latestPlan = explicitPlan;
        }
        allowDownload = false;
        return drawnPlan;
      },
      isSuperseded: function () {
        return Boolean(
          drawnPlan &&
            latestPlan &&
            drawnPlan.digest &&
            latestPlan.digest &&
            drawnPlan.digest !== latestPlan.digest
        );
      },
      getSupersededBy: function () {
        return this.isSuperseded() ? latestPlan : null;
      },
      buildApprovePayload: function () {
        if (!drawnPlan || !drawnPlan.approval_prefix || !drawnPlan.digest) {
          return null;
        }
        const isPrepareStage =
          drawnPlan.action === "prepare" || Boolean(drawnPlan.requires_staging);
        return {
          action: isPrepareStage ? "prepare" : "submit",
          digest_prefix: drawnPlan.approval_prefix,
          digest: drawnPlan.digest,
          allow_download: Boolean(allowDownload),
        };
      },
      clearAll: function () {
        drawnPlan = null;
        latestPlan = null;
        allowDownload = false;
      },
    };
  }

  if (root) {
    root.__AIMIP_CREATE_PLAN_STATE__ = createPlanApprovalState;
  }

  if (!root || !root.document) {
    return;
  }

  const document = root.document;
  const window = root;
  const planState = createPlanApprovalState();

  const state = {
    token: "",
    session: {
      in_container: false,
      scheduler: "slurm",
      can_submit_directly: true,
      chat_log_dir: null,
      agent_available: false,
      agent_ready: false,
    },
    models: [],
    selectedModelCard: null,
    activeModelWindowEnd: "",
    pendingCancel: null,
    tailRunName: null,
    detailRunName: null,
    detailMonthlySig: null,
    pollTimer: null,
    pollInFlight: false,
  };

  function tokenStorageKey() {
    return "aimip_ui_token:" + String(window.location.origin || "local");
  }

  function bootstrapToken() {
    let urlTok = "";
    try {
      const params = new URLSearchParams(window.location.search || "");
      urlTok = (params.get("token") || "").trim();
    } catch (_err) {
      urlTok = "";
    }
    if (urlTok) {
      state.token = urlTok;
      try {
        window.sessionStorage.setItem(tokenStorageKey(), urlTok);
      } catch (_err) {
        // In-memory `state.token` remains active if sessionStorage is disabled
      }
      try {
        const cleanUrl =
          (window.location.pathname || "/") + (window.location.hash || "");
        window.history.replaceState(null, "", cleanUrl);
      } catch (_err) {
        // Ignore history API restrictions
      }
      return;
    }
    try {
      state.token = (window.sessionStorage.getItem(tokenStorageKey()) || "").trim();
    } catch (_err) {
      state.token = "";
    }
  }

  function el(id) {
    return document.getElementById(id);
  }

  function show(node, visible) {
    if (!node) return;
    node.classList.toggle("hidden", !visible);
  }

  function setNotice(message, isError) {
    const box = el("global-notice");
    if (!message) {
      show(box, false);
      return;
    }
    box.textContent = message;
    box.classList.toggle("banner-error", Boolean(isError));
    box.classList.toggle("banner-ok", !isError);
    show(box, true);
  }

  function isTerminalRunState(value) {
    const base = String(value || "")
      .trim()
      .split(/\s+/)[0]
      .replace(/\+$/, "")
      .toUpperCase();
    return (
      base === "COMPLETED" ||
      base === "FAILED" ||
      base === "CANCELLED" ||
      base === "TIMEOUT" ||
      base === "OUT_OF_MEMORY" ||
      base === "NODE_FAIL" ||
      base === "PREEMPTED" ||
      base === "BOOT_FAIL" ||
      base === "DEADLINE"
    );
  }

  function badgeClassFor(value) {
    const raw = String(value || "").trim().toLowerCase();
    const base = raw.split(/\s+/)[0].replace(/\+$/, "");
    if (
      base === "certified" ||
      base === "completed" ||
      base === "pass" ||
      base === "yes"
    ) {
      return "badge badge-ok";
    }
    if (
      base === "runs" ||
      base === "running" ||
      base === "pending" ||
      base === "completing" ||
      base === "configuring" ||
      base === "info"
    ) {
      return "badge badge-info";
    }
    if (
      base === "experimental" ||
      base === "prepared" ||
      base === "cancel_requested" ||
      base === "missing"
    ) {
      return "badge badge-warn";
    }
    if (
      base === "failed" ||
      base === "cancelled" ||
      base === "timeout" ||
      base === "out_of_memory" ||
      base === "node_fail" ||
      base === "preempted" ||
      base === "boot_fail" ||
      base === "deadline" ||
      base === "fail"
    ) {
      return "badge badge-err";
    }
    return "badge";
  }

  function makeBadge(text, titleText) {
    const span = document.createElement("span");
    span.className = badgeClassFor(text);
    span.textContent = String(text || "unknown");
    if (titleText) {
      span.title = String(titleText);
    }
    return span;
  }

  function appendEmptyTableRow(tbody, colSpan, message) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = colSpan;
    td.className = "muted";
    td.textContent = message;
    tr.appendChild(td);
    tbody.appendChild(tr);
  }

  function authHeaders(extra) {
    const hdrs = Object.assign({}, extra || {});
    if (state.token) {
      hdrs["X-AIMIP-Token"] = state.token;
    }
    return hdrs;
  }

  async function api(path, options) {
    const inOpts = options || {};
    const opts = Object.assign({}, inOpts, {
      credentials: "omit",
      headers: authHeaders(inOpts.headers),
    });
    if (opts.body && typeof opts.body === "object") {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(opts.body);
    }
    const resp = await fetch(path, opts);
    if (resp.status === 401) {
      show(el("auth-banner"), true);
      throw new Error("Unauthorized: open the URL with ?token=... printed by `aimip ui`.");
    }
    show(el("auth-banner"), false);
    const data = await resp.json();
    if (!resp.ok || !data.ok) {
      const err = (data && data.error) || { message: "Request failed (" + resp.status + ")" };
      const e = new Error(err.message || "Request failed");
      e.errorInfo = err;
      throw e;
    }
    return data.result;
  }

  async function renderPlotToCanvas(canvas, url) {
    const resp = await fetch(url, {
      method: "GET",
      credentials: "omit",
      headers: authHeaders(),
    });
    if (!resp.ok) {
      throw new Error("Plot request failed (" + resp.status + ")");
    }
    const blob = await resp.blob();
    if (typeof window.createImageBitmap === "function") {
      const bmp = await window.createImageBitmap(blob);
      canvas.width = bmp.width;
      canvas.height = bmp.height;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.drawImage(bmp, 0, 0);
      }
      if (typeof bmp.close === "function") {
        bmp.close();
      }
    }
  }

  function formatErrorBanner(box, err) {
    if (!err) {
      show(box, false);
      return;
    }
    const info = err.errorInfo || {};
    const parts = [];
    if (info.field) parts.push("[" + info.field + "]");
    parts.push(err.message || "Error");
    if (Array.isArray(info.allowed) && info.allowed.length) {
      parts.push("(allowed: " + info.allowed.join(", ") + ")");
    }
    if (info.next_command) {
      parts.push("Next: " + info.next_command);
    }
    box.textContent = parts.join(" ");
    show(box, true);
  }

  function currentRouteName() {
    const rawHash = (window.location.hash || "#models").replace(/^#/, "");
    if (rawHash.startsWith("run/")) return "run-detail";
    return rawHash || "models";
  }

  /* ---------------- Navigation & Bounded Polling ---------------- */
  function stopPolling() {
    if (state.pollTimer) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
    state.pollInFlight = false;
  }

  function startBoundedPoll(fn) {
    stopPolling();
    state.pollTimer = setInterval(async function () {
      if (state.pollInFlight) return;
      state.pollInFlight = true;
      try {
        await fn();
      } finally {
        state.pollInFlight = false;
      }
    }, POLL_INTERVAL_MS);
  }

  function applyRoute() {
    stopPolling();
    const rawHash = (window.location.hash || "#models").replace(/^#/, "");
    let route = rawHash || "models";
    let runParam = null;
    if (route.startsWith("run/")) {
      runParam = decodeURIComponent(route.slice(4));
      route = "run-detail";
    }

    const views = ["models", "new-run", "plan", "runs", "run-detail", "doctor"];
    views.forEach(function (v) {
      show(el("view-" + v), v === route);
    });
    document.querySelectorAll(".nav-link").forEach(function (link) {
      const target = link.getAttribute("data-route");
      link.classList.toggle("active", target === route || (route === "run-detail" && target === "runs"));
    });

    if (route === "models") loadModels();
    else if (route === "new-run") initNewRunForm();
    else if (route === "plan") renderPlanView(planState.getLatestPlan());
    else if (route === "runs") {
      loadRuns();
      syncRunsPolling();
    } else if (route === "run-detail" && runParam) {
      if (state.detailRunName !== runParam) {
        state.detailMonthlySig = null;
      }
      state.detailRunName = runParam;
      loadRunDetail(runParam, true);
      syncDetailPolling();
    } else if (route === "doctor") {
      loadDoctor();
    }
  }

  /* ---------------- 1. Models & Model Card ---------------- */
  async function loadModels() {
    const params = new URLSearchParams();
    const task = el("filter-task").value;
    const fam = el("filter-family").value;
    const st = el("filter-status").value;
    if (task) params.set("task", task);
    if (fam) params.set("family", fam);
    if (st) params.set("status", st);
    if (el("filter-installed").checked) params.set("installed", "1");

    try {
      const qs = params.toString();
      const rows = await api("/api/v1/models" + (qs ? "?" + qs : ""));
      state.models = rows;
      renderModelsTable(rows);
    } catch (err) {
      setNotice(err.message, true);
    }
  }

  function renderModelsTable(rows) {
    const tbody = el("models-tbody");
    tbody.textContent = "";
    if (!rows || !rows.length) {
      appendEmptyTableRow(tbody, 7, "No matching models for the selected filters.");
      return;
    }
    rows.forEach(function (m) {
      const tr = document.createElement("tr");

      const tdName = document.createElement("td");
      const strong = document.createElement("strong");
      strong.textContent = m.name;
      tdName.appendChild(strong);
      tr.appendChild(tdName);

      const tdStatus = document.createElement("td");
      tdStatus.appendChild(makeBadge(m.status, m.status_definition || ""));
      tr.appendChild(tdStatus);

      const tdTask = document.createElement("td");
      tdTask.textContent = m.task || "";
      tr.appendChild(tdTask);

      const tdFam = document.createElement("td");
      tdFam.textContent = m.family || "";
      tr.appendChild(tdFam);

      const tdStep = document.createElement("td");
      const stepLabel = (m.family === "arches" || m.family === "md")
        ? "1mo"
        : (m.family === "cbottle" ? "gen" : (m.timestep_hours ? m.timestep_hours + "h" : ""));
      tdStep.textContent = stepLabel + " / " + (m.calendar || "");
      tr.appendChild(tdStep);

      const tdInst = document.createElement("td");
      tdInst.appendChild(makeBadge(m.installed ? "yes" : "missing"));
      tr.appendChild(tdInst);

      const tdAct = document.createElement("td");
      const btnCard = document.createElement("button");
      btnCard.type = "button";
      btnCard.className = "btn btn-secondary btn-small";
      btnCard.textContent = "Card";
      btnCard.addEventListener("click", function () {
        inspectModelCard(m.name);
      });
      const btnRun = document.createElement("button");
      btnRun.type = "button";
      btnRun.className = "btn btn-primary btn-small";
      btnRun.textContent = "New Run";
      btnRun.addEventListener("click", function () {
        selectModelForNewRun(m.name);
      });
      const wrap = document.createElement("div");
      wrap.className = "inline-controls";
      wrap.appendChild(btnCard);
      wrap.appendChild(btnRun);
      tdAct.appendChild(wrap);
      tr.appendChild(tdAct);

      tbody.appendChild(tr);
    });
  }

  async function inspectModelCard(modelName) {
    try {
      const card = await api("/api/v1/models/" + encodeURIComponent(modelName) + "?examples=1");
      state.selectedModelCard = card;
      el("model-card-title").textContent = card.name + " — " + (card.title || card.task || "");
      const body = el("model-card-body");
      body.textContent = "";

      const missingCnt = Array.isArray(card.missing_files) ? card.missing_files.length : (card.missing_files || 0);
      const cardStep = (card.family === "arches" || card.family === "md")
        ? "1mo"
        : (card.family === "cbottle" ? "gen (1d synoptic)" : ((card.timestep_hours || "") + "h"));
      const statusDetail =
        (card.status || "") +
        (card.verified_on ? " (" + card.verified_on + ")" : "") +
        (card.status_definition ? " — " + card.status_definition : "");
      const grid = document.createElement("div");
      grid.className = "kv-grid";
      [
        ["Status Tier", statusDetail],
        ["Task", card.task],
        ["Family / Runtime", (card.family || "") + (card.fme_version ? " (" + card.fme_version + ")" : "")],
        ["Grid", card.grid || "—"],
        ["Step / Calendar", cardStep + " / " + (card.calendar || "")],
        ["Initial Conditions", card.ic_provenance || "—"],
        ["Weights / Code Licence", card.licence || "—"],
        ["Cached / Missing", (card.cached_files || 0) + " cached, " + missingCnt + " missing"],
        ["Cost Basis", card.cost_note || card.cost_estimate_note || "—"],
      ].forEach(function (pair) {
        const box = document.createElement("div");
        box.className = "kv-item";
        const k = document.createElement("span");
        k.className = "label-caps";
        k.textContent = pair[0];
        const v = document.createElement("div");
        v.textContent = String(pair[1] || "");
        box.appendChild(k);
        box.appendChild(v);
        grid.appendChild(box);
      });
      body.appendChild(grid);

      if (Array.isArray(card.forcing_sets) && card.forcing_sets.length) {
        const fsPre = document.createElement("pre");
        fsPre.className = "code-card";
        fsPre.textContent = "Forcing windows:\n" + card.forcing_sets.map(function (fs) {
          const w = Array.isArray(fs.window) ? fs.window.join(" .. ") : "";
          return "  - " + fs.name + ": " + w;
        }).join("\n");
        body.appendChild(fsPre);
      }

      const limItems = Array.isArray(card.limitations)
        ? card.limitations
        : (card.limitations ? [String(card.limitations)] : []);
      if (limItems.length) {
        const limPre = document.createElement("pre");
        limPre.className = "code-card";
        limPre.textContent = "Notes & Limitations:\n" + limItems.map(function (l) {
          return "  * " + l;
        }).join("\n");
        body.appendChild(limPre);
      }

      if (Array.isArray(card.examples) && card.examples.length) {
        const exPre = document.createElement("pre");
        exPre.className = "code-card";
        exPre.textContent = "CLI Examples:\n" + card.examples.map(function (ex) {
          return "  $ " + ex;
        }).join("\n");
        body.appendChild(exPre);
      }

      if (card.citation && card.citation.bibtex) {
        const bibPre = document.createElement("pre");
        bibPre.className = "code-card";
        bibPre.textContent = card.citation.bibtex;
        body.appendChild(bibPre);
      }

      show(el("model-detail-panel"), true);
    } catch (err) {
      setNotice(err.message, true);
    }
  }

  function selectModelForNewRun(modelName) {
    window.location.hash = "#new-run";
    initNewRunForm(modelName);
  }

  /* ---------------- 2. New Run Form ---------------- */
  function updateModeWarning() {
    const modeSel = el("run-mode");
    const hint = el("run-mode-hint");
    if (!modeSel || !hint) return;
    if (modeSel.value === "local") {
      if (state.session.in_container) {
        hint.textContent =
          "Container install: local mode prepares and seals the run directory, then displays the exact `aimip submit --expect-digest=...` command to launch on the host.";
        show(hint, true);
      } else if (state.session.scheduler === "slurm") {
        hint.textContent =
          "Warning: local mode executes directly on this login/UI host without Slurm GPU allocation. Use slurm for cluster GPU jobs.";
        show(hint, true);
      } else {
        show(hint, false);
      }
    } else {
      show(hint, false);
    }
  }

  function handleHorizonKindChange() {
    const kind = el("run-horizon-kind").value;
    const input = el("run-horizon-value");
    if (!input) return;
    if (kind === "steps") {
      input.placeholder = "e.g. 4";
      if (!/^\d+$/.test(input.value.trim())) input.value = "4";
    } else if (kind === "length") {
      input.placeholder = "e.g. 30d, 6m, 1y";
      if (/^\d+$/.test(input.value.trim()) || input.value.includes("T")) input.value = "30d";
    } else if (kind === "end") {
      input.placeholder = state.activeModelWindowEnd || "YYYY-MM-DDTHH:MM:SS";
      if (/^\d+$/.test(input.value.trim()) || /^\d+[dmy]$/i.test(input.value.trim())) {
        input.value = state.activeModelWindowEnd || "";
      }
    }
  }

  async function initNewRunForm(preferredModel) {
    formatErrorBanner(el("form-error-box"), null);
    updateModeWarning();
    try {
      if (!state.models.length) {
        state.models = await api("/api/v1/models");
      }
      const sel = el("run-model");
      const prev = preferredModel || sel.value;
      sel.textContent = "";
      state.models.forEach(function (m) {
        const opt = document.createElement("option");
        opt.value = m.name;
        opt.textContent = m.name + " (" + m.task + ", " + m.calendar + ")";
        sel.appendChild(opt);
      });
      if (prev && state.models.some(function (m) { return m.name === prev; })) {
        sel.value = prev;
      }
      await populateModelDefaults(sel.value);
      await loadPickers();
    } catch (err) {
      formatErrorBanner(el("form-error-box"), err);
    }
  }

  async function populateModelDefaults(modelName) {
    if (!modelName) return;
    try {
      const card = await api("/api/v1/models/" + encodeURIComponent(modelName));
      const fsSelect = el("run-forcing-set");
      fsSelect.textContent = "";
      const autoOpt = document.createElement("option");
      autoOpt.value = "";
      autoOpt.textContent = "Default (auto-select covering window)";
      fsSelect.appendChild(autoOpt);

      const sets = card.forcing_sets || [];
      sets.forEach(function (fs) {
        const opt = document.createElement("option");
        opt.value = fs.name;
        const w = Array.isArray(fs.window) ? " [" + fs.window[0] + " .. " + fs.window[1] + "]" : "";
        opt.textContent = fs.name + w;
        fsSelect.appendChild(opt);
      });

      const hintParts = [];
      if (sets.length && Array.isArray(sets[0].window) && sets[0].window[0]) {
        state.activeModelWindowEnd = sets[0].window[1] || "";
        if (!el("run-start").value || el("run-start").dataset.autofilled === "1") {
          el("run-start").value = sets[0].window[0];
          el("run-start").dataset.autofilled = "1";
        }
        hintParts.push(
          "Window: " + sets[0].window[0] + " .. " + sets[0].window[1] + " (" + (card.calendar || "") + ")"
        );
      } else {
        state.activeModelWindowEnd = "";
      }
      if (Array.isArray(card.shipped_ics) && card.shipped_ics.length) {
        const stamps = [];
        card.shipped_ics.forEach(function (icObj) {
          (icObj.timestamps || []).slice(0, 4).forEach(function (ts) {
            if (stamps.length < 6 && stamps.indexOf(ts) === -1) stamps.push(ts);
          });
        });
        if (stamps.length) {
          hintParts.push("Shipped IC start(s): " + stamps.join(", "));
        }
      }
      el("run-start-hint").textContent = hintParts.join(" · ");
    } catch (err) {
      el("run-start-hint").textContent = "Could not load model defaults: " + err.message;
    }
  }

  async function loadPickers() {
    const rootsHint = el("picker-roots-hint");
    try {
      const icRes = await api("/api/v1/pickers/files?kind=ic");
      const icSel = el("run-ic-picker");
      const prevIc = icSel.value;
      icSel.textContent = "";
      const defIc = document.createElement("option");
      defIc.value = "";
      defIc.textContent = "Shipped model IC (default)";
      icSel.appendChild(defIc);
      (icRes.files || []).forEach(function (f) {
        const o = document.createElement("option");
        o.value = f.path;
        o.textContent = f.path;
        icSel.appendChild(o);
      });
      if (prevIc) icSel.value = prevIc;

      const oceanSel = el("run-ocean-ic-picker");
      if (oceanSel) {
        const prevOcean = oceanSel.value;
        oceanSel.textContent = "";
        const defOcean = document.createElement("option");
        defOcean.value = "";
        defOcean.textContent = "Shipped ocean IC / N/A (default)";
        oceanSel.appendChild(defOcean);
        (icRes.files || []).forEach(function (f) {
          const o = document.createElement("option");
          o.value = f.path;
          o.textContent = f.path;
          oceanSel.appendChild(o);
        });
        if (prevOcean) oceanSel.value = prevOcean;
      }

      const loraRes = await api("/api/v1/pickers/files?kind=lora");
      const loraSel = el("run-lora-picker");
      const prevLora = loraSel.value;
      loraSel.textContent = "";
      const defLora = document.createElement("option");
      defLora.value = "";
      defLora.textContent = "None";
      loraSel.appendChild(defLora);
      (loraRes.files || []).forEach(function (f) {
        const o = document.createElement("option");
        o.value = f.path;
        o.textContent = f.path;
        loraSel.appendChild(o);
      });
      if (prevLora) loraSel.value = prevLora;

      const notes = [];
      if (Array.isArray(icRes.allowed_roots) && icRes.allowed_roots.length) {
        notes.push("Allowed input roots: " + icRes.allowed_roots.join(", "));
      }
      if (icRes.truncated || loraRes.truncated) {
        notes.push("(Picker list capped at 200 files; paste an exact path above if needed.)");
      }
      rootsHint.textContent = notes.join(" ");
    } catch (err) {
      rootsHint.textContent = "File picker unavailable (" + err.message + "); enter paths manually if needed.";
    }
  }

  async function handleGeneratePlan(ev) {
    ev.preventDefault();
    formatErrorBanner(el("form-error-box"), null);
    const btn = el("btn-generate-plan");
    const origText = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Generating Plan...";

    const payload = {
      model: el("run-model").value,
      start: el("run-start").value.trim(),
      monthly: el("run-monthly").checked,
      prediction: el("run-prediction").checked,
      mode: el("run-mode").value,
      resources: {},
    };

    const hKind = el("run-horizon-kind").value;
    const hVal = el("run-horizon-value").value.trim();
    if (hKind === "steps") payload.steps = hVal;
    else if (hKind === "length") payload.length = hVal;
    else if (hKind === "end") payload.end = hVal;

    if (el("run-name").value.trim()) payload.name = el("run-name").value.trim();
    if (el("run-forcing-set").value) payload.forcing_set = el("run-forcing-set").value;
    if (el("run-members").value.trim()) payload.members = el("run-members").value.trim();
    if (el("run-seed").value.trim()) payload.seed = el("run-seed").value.trim();

    const manualIc = (el("run-ic-manual") && el("run-ic-manual").value.trim()) || "";
    const pickedIc = el("run-ic-picker").value;
    if (manualIc || pickedIc) payload.ic = manualIc || pickedIc;

    const manualOcean = (el("run-ocean-ic-manual") && el("run-ocean-ic-manual").value.trim()) || "";
    const pickedOcean = el("run-ocean-ic-picker") ? el("run-ocean-ic-picker").value : "";
    if (manualOcean || pickedOcean) payload.ocean_ic = manualOcean || pickedOcean;

    const manualLora = (el("run-lora-manual") && el("run-lora-manual").value.trim()) || "";
    const pickedLora = el("run-lora-picker").value;
    if (manualLora || pickedLora) payload.lora = manualLora || pickedLora;

    const resFields = [
      ["partition", "res-partition"],
      ["account", "res-account"],
      ["time", "res-time"],
      ["mem", "res-mem"],
      ["gres", "res-gres"],
      ["cpus_per_task", "res-cpus"],
    ];
    resFields.forEach(function (pair) {
      const val = el(pair[1]).value.trim();
      if (val) payload.resources[pair[0]] = val;
    });

    try {
      const proposal = await api("/api/v1/plan", { method: "POST", body: payload });
      planState.onFormPlan(proposal);
      if (currentRouteName() === "plan") {
        renderPlanView(proposal);
      } else {
        window.location.hash = "#plan";
      }
    } catch (err) {
      formatErrorBanner(el("form-error-box"), err);
    } finally {
      btn.disabled = false;
      btn.textContent = origText;
    }
  }

  /* ---------------- 3. Plan Card + Human Approve ---------------- */
  function syncSupersededBanner() {
    const newer = planState.getSupersededBy();
    const box = el("plan-superseded-box");
    const btnApprove = el("btn-approve-submit");
    if (newer) {
      el("plan-superseded-text").textContent =
        "A newer proposal [" +
        newer.approval_prefix +
        "] was registered by the Assistant and superseded the card shown below. Load and review the updated card before approving.";
      show(box, true);
      if (btnApprove) btnApprove.disabled = true;
    } else {
      show(box, false);
      if (btnApprove) btnApprove.disabled = false;
    }
  }

  function renderPlanView(explicitPlan) {
    formatErrorBanner(el("plan-error-box"), null);
    const prop = planState.drawPlan(
      explicitPlan !== undefined ? explicitPlan : planState.getLatestPlan()
    );
    syncSupersededBanner();

    if (!prop) {
      show(el("plan-empty"), true);
      show(el("plan-active"), false);
      return;
    }
    show(el("plan-empty"), false);
    show(el("plan-active"), true);
    el("plan-prefix-badge").textContent = prop.approval_prefix;
    el("plan-full-digest").textContent = prop.digest;
    el("plan-card-pre").textContent = prop.card;

    const ctxBox = el("plan-model-context");
    if (ctxBox) {
      ctxBox.textContent = "";
      const mc = prop.model_card || {};
      [
        [
          "Model Status Tier",
          (mc.status_tier || "—") + (mc.status_definition ? " — " + mc.status_definition : ""),
        ],
        ["Initial Conditions", mc.ic_provenance || "—"],
        ["Weights / Code Licence", mc.licence || "—"],
        ["Operational Notes & Limits", mc.limitations || "—"],
      ].forEach(function (pair) {
        const item = document.createElement("div");
        item.className = "kv-item";
        const k = document.createElement("span");
        k.className = "label-caps";
        k.textContent = pair[0];
        const v = document.createElement("div");
        v.textContent = String(pair[1] || "");
        item.appendChild(k);
        item.appendChild(v);
        ctxBox.appendChild(item);
      });
    }

    const dlConsent = el("plan-download-consent");
    dlConsent.checked = false;
    planState.setAllowDownload(false);
    if (prop.download_required) {
      el("plan-download-label").textContent =
        "Tick to allow downloading/staging " +
        prop.missing_count +
        " missing asset(s) (~" +
        (prop.missing_size_human || "0 B") +
        ") before running";
      show(el("plan-download-box"), true);
    } else {
      show(el("plan-download-box"), false);
    }

    const isPrepareStage = prop.action === "prepare" || Boolean(prop.requires_staging);
    const prepOnlyContainer = Boolean(prop.will_prepare_only_in_container);
    const contNote = el("plan-container-note");
    if (!isPrepareStage && prepOnlyContainer) {
      contNote.textContent =
        "Container install: clicking Approve & Prepare Run writes and seals the run directory, then displays the exact `aimip submit --expect-digest=...` command to run on the host.";
      show(contNote, true);
    } else {
      show(contNote, false);
    }

    const btn = el("btn-approve-submit");
    btn.disabled = false;
    if (isPrepareStage) {
      el("plan-subtitle").textContent =
        "Step 1 of 2: Approve staging/preparation for the prospective plan below. You will review and approve the final post-stage digest before submission.";
      btn.textContent = "Approve & Stage Inputs [" + prop.approval_prefix + "]";
    } else if (prepOnlyContainer) {
      el("plan-subtitle").textContent =
        "Review the sealed plan card and 8-hex digest prefix below. Approving prepares the run directory and prints the login-node submit command.";
      btn.textContent = "Approve & Prepare Run [" + prop.approval_prefix + "]";
    } else {
      el("plan-subtitle").textContent =
        "Review the exact plan card and 8-hex digest prefix below. Clicking Approve mints a single-use 10-minute authorization and submits the run.";
      btn.textContent = "Approve & Submit [" + prop.approval_prefix + "]";
    }
  }

  async function handleApproveAndSubmit() {
    formatErrorBanner(el("plan-error-box"), null);
    planState.setAllowDownload(Boolean(el("plan-download-consent").checked));
    const approveBody = planState.buildApprovePayload();
    if (!approveBody) return;

    const btn = el("btn-approve-submit");
    const prevLabel = btn.textContent;
    const isPrepareStage = approveBody.action === "prepare";
    btn.disabled = true;
    btn.textContent = isPrepareStage ? "Staging Inputs..." : "Approving & Submitting...";

    try {
      const auth = await api("/api/v1/approve", {
        method: "POST",
        body: approveBody,
      });
      if (isPrepareStage) {
        const prepRes = await api("/api/v1/prepare", {
          method: "POST",
          body: { nonce: auth.authorization_nonce },
        });
        renderPlanView(prepRes.proposal);
        setNotice(
          "Inputs staged and hashed into final digest " +
            prepRes.proposal.short_digest +
            ". Review the final card and click Approve.",
          false
        );
        return;
      }
      const attempt = await api("/api/v1/submit", {
        method: "POST",
        body: { nonce: auth.authorization_nonce },
      });
      planState.clearAll();
      const runTarget = attempt.run_name || (attempt.run_dir ? attempt.run_dir.split("/").pop() : "");
      if (attempt.host_command) {
        setNotice(
          "Prepared " + runTarget + " — run on cluster login node to submit: " + attempt.host_command,
          false
        );
      } else {
        setNotice(
          "Submitted " + runTarget + " (" + attempt.scheduler + (attempt.attempt_id ? " " + attempt.attempt_id : "") + ")",
          false
        );
      }
      if (runTarget) {
        window.location.hash = "#run/" + encodeURIComponent(runTarget);
      } else {
        window.location.hash = "#runs";
      }
    } catch (err) {
      btn.disabled = false;
      btn.textContent = prevLabel;
      formatErrorBanner(el("plan-error-box"), err);
    }
  }

  /* ---------------- 4. Runs List, Tail & Cancel ---------------- */
  function formatElapsedAndExit(r) {
    const parts = [];
    if (r.elapsed) {
      parts.push(String(r.elapsed));
    } else if (r.wall_seconds != null) {
      parts.push(Number(r.wall_seconds).toFixed(1) + "s");
    }
    if (r.rc != null) {
      parts.push("rc=" + r.rc);
    } else if (r.exit_code && r.exit_code !== "0:0") {
      parts.push("exit=" + r.exit_code);
    }
    return parts.length ? parts.join(" · ") : "—";
  }

  async function loadRuns() {
    try {
      const runs = await api("/api/v1/runs");
      const tbody = el("runs-tbody");
      tbody.textContent = "";
      if (!runs || !runs.length) {
        appendEmptyTableRow(tbody, 6, "No runs recorded in run root yet.");
        return;
      }
      runs.forEach(function (r) {
        const tr = document.createElement("tr");
        const shownState = r.display_state || r.state || "unknown";
        const canCancel = Boolean(
          (r.job_id || r.local_pid) &&
            !r.awaiting_submit &&
            !isTerminalRunState(shownState) &&
            !isTerminalRunState(r.state)
        );

        const tdName = document.createElement("td");
        const link = document.createElement("a");
        link.href = "#run/" + encodeURIComponent(r.name);
        link.textContent = r.name;
        tdName.appendChild(link);
        tr.appendChild(tdName);

        const tdModel = document.createElement("td");
        tdModel.textContent = r.model || "—";
        tr.appendChild(tdModel);

        const tdState = document.createElement("td");
        tdState.appendChild(makeBadge(shownState));
        tr.appendChild(tdState);

        const tdJob = document.createElement("td");
        tdJob.textContent = r.job_id || (r.local_pid ? "pid:" + r.local_pid : "—");
        tr.appendChild(tdJob);

        const tdEl = document.createElement("td");
        tdEl.textContent = formatElapsedAndExit(r);
        tr.appendChild(tdEl);

        const tdAct = document.createElement("td");
        const wrap = document.createElement("div");
        wrap.className = "inline-controls";

        const btnOpen = document.createElement("a");
        btnOpen.href = "#run/" + encodeURIComponent(r.name);
        btnOpen.className = "btn btn-secondary btn-small";
        btnOpen.textContent = r.awaiting_submit ? "Submit Cmd" : "Detail";
        wrap.appendChild(btnOpen);

        const btnLogs = document.createElement("button");
        btnLogs.type = "button";
        btnLogs.className = "btn btn-secondary btn-small";
        btnLogs.textContent = "Logs";
        btnLogs.addEventListener("click", function () {
          state.tailRunName = r.name;
          loadRunsTail();
        });
        wrap.appendChild(btnLogs);

        if (canCancel) {
          const btnCancel = document.createElement("button");
          btnCancel.type = "button";
          btnCancel.className = "btn btn-danger btn-small";
          btnCancel.textContent = "Cancel";
          btnCancel.addEventListener("click", function () {
            proposeCancel(r.name, false);
          });
          wrap.appendChild(btnCancel);
        }

        tdAct.appendChild(wrap);
        tr.appendChild(tdAct);
        tbody.appendChild(tr);
      });
      if (state.tailRunName) {
        await loadRunsTail();
      }
    } catch (err) {
      setNotice(err.message, true);
    }
  }

  async function loadRunsTail() {
    if (!state.tailRunName) return;
    try {
      const logRes = await api("/api/v1/runs/" + encodeURIComponent(state.tailRunName) + "/logs?max_bytes=16384");
      el("runs-log-title").textContent = "Log Tail — " + state.tailRunName;
      const truncNote = el("runs-log-trunc-note");
      if (logRes.truncated) {
        truncNote.textContent = "Showing last " + Math.round((logRes.max_bytes || 16384) / 1024) + " KB of log output (older lines truncated).";
        show(truncNote, true);
      } else {
        show(truncNote, false);
      }
      el("runs-log-pre").textContent = logRes.content || "(no log output yet)";
      show(el("runs-log-panel"), true);
    } catch (err) {
      el("runs-log-pre").textContent = "Log error: " + err.message;
      show(el("runs-log-panel"), true);
    }
  }

  function syncRunsPolling() {
    stopPolling();
    if (el("runs-autopoll").checked) {
      startBoundedPoll(loadRuns);
    }
  }

  async function proposeCancel(runName, inDetail) {
    try {
      const prop = await api("/api/v1/runs/" + encodeURIComponent(runName) + "/cancel/propose", {
        method: "POST",
        body: {},
      });
      state.pendingCancel = prop;
      if (inDetail) {
        el("detail-cancel-pre").textContent = prop.card;
        el("btn-detail-confirm-cancel").textContent = "Approve Cancel [" + prop.approval_prefix + "]";
        show(el("detail-cancel-box"), true);
      } else {
        el("cancel-card-pre").textContent = prop.card;
        el("btn-confirm-cancel").textContent = "Approve Cancel [" + prop.approval_prefix + "]";
        show(el("cancel-approval-box"), true);
      }
    } catch (err) {
      setNotice(err.message, true);
    }
  }

  async function confirmCancel(inDetail) {
    if (!state.pendingCancel) return;
    const prop = state.pendingCancel;
    try {
      const auth = await api("/api/v1/approve", {
        method: "POST",
        body: {
          action: "cancel",
          digest_prefix: prop.approval_prefix,
          digest: prop.digest,
        },
      });
      const cancelRes = await api("/api/v1/runs/" + encodeURIComponent(prop.run_name) + "/cancel", {
        method: "POST",
        body: { nonce: auth.authorization_nonce },
      });
      state.pendingCancel = null;
      show(el("cancel-approval-box"), false);
      show(el("detail-cancel-box"), false);
      setNotice(
        "Cancel requested for " +
          prop.run_name +
          " (" +
          (cancelRes.status || "cancel_requested") +
          ")",
        false
      );
      if (inDetail) loadRunDetail(prop.run_name, false);
      else loadRuns();
    } catch (err) {
      setNotice(err.message, true);
    }
  }

  /* ---------------- 5. Run Detail (Summary + Plots + Files + Logs) ---------------- */
  function computeMonthlyFileSignature(files) {
    const parts = [];
    (files || []).forEach(function (f) {
      if (
        f.name === "monthly_mean_predictions.nc" ||
        f.name === "atmosphere/monthly_mean_predictions.nc" ||
        f.name === "ocean/monthly_mean_predictions.nc"
      ) {
        parts.push(f.name + ":" + f.size_bytes + ":" + f.mtime);
      }
    });
    return parts.join("|");
  }

  function computeModalSteps(series) {
    const freq = {};
    let bestCount = null;
    let bestFreq = 0;
    (series || []).forEach(function (row) {
      if (row.counts != null && row.counts > 0) {
        const c = Number(row.counts);
        freq[c] = (freq[c] || 0) + 1;
        if (freq[c] > bestFreq || (freq[c] === bestFreq && (bestCount === null || c > bestCount))) {
          bestFreq = freq[c];
          bestCount = c;
        }
      }
    });
    return bestCount;
  }

  function formatStateProvenance(st) {
    if (st.finished != null && st.rc != null) {
      return "aimip-manifest.json (finished " + st.finished + ")";
    }
    if (st.sacct_source === "sacct.txt") {
      const age = st.sacct_age_seconds != null ? " (" + Math.round(st.sacct_age_seconds) + "s ago)" : "";
      return "sacct.txt" + age;
    }
    if (st.sacct_source) {
      return String(st.sacct_source);
    }
    if (st.awaiting_submit) {
      return "prepared on disk (not yet submitted)";
    }
    return "local / manifest";
  }

  async function loadRunDetail(runName, forceSummaryReload) {
    el("detail-title").textContent = "Run: " + runName;
    let monthlySig = "";
    try {
      const st = await api("/api/v1/runs/" + encodeURIComponent(runName));
      el("detail-subtitle").textContent = st.out_dir || st.run_dir || "";

      const shownState = st.display_state || st.state || "unknown";
      const canCancel = Boolean(
        (st.slurm_job_id || st.job_id || st.local_pid) &&
          !st.awaiting_submit &&
          !isTerminalRunState(shownState) &&
          !isTerminalRunState(st.state)
      );
      show(el("btn-detail-cancel"), canCancel);

      if (st.host_submit_command) {
        el("detail-host-submit-cmd").textContent = st.host_submit_command;
        show(el("detail-host-submit-box"), true);
      } else {
        show(el("detail-host-submit-box"), false);
      }

      const statusBox = el("detail-status-box");
      statusBox.textContent = "";
      const grid = document.createElement("div");
      grid.className = "kv-grid";
      const jobOrPid =
        st.slurm_job_id || st.job_id || (st.local_pid ? "pid:" + st.local_pid : "—");
      [
        ["State", shownState],
        ["State Source", formatStateProvenance(st)],
        ["Model", st.model || "—"],
        ["Job / PID", jobOrPid],
        ["Elapsed / Exit", formatElapsedAndExit(st)],
        ["Steps / Members", (st.steps || "—") + " / " + (st.members || "—")],
        ["Window", (st.start_time || "—") + " .. " + (st.end_time || "—")],
      ].forEach(function (pair) {
        const item = document.createElement("div");
        item.className = "kv-item";
        const k = document.createElement("span");
        k.className = "label-caps";
        k.textContent = pair[0];
        const v = document.createElement("div");
        if (pair[0] === "State") v.appendChild(makeBadge(pair[1]));
        else v.textContent = String(pair[1]);
        item.appendChild(k);
        item.appendChild(v);
        grid.appendChild(item);
      });
      statusBox.appendChild(grid);

      const filesTbody = el("detail-files-tbody");
      filesTbody.textContent = "";
      const files = st.files || [];
      monthlySig = computeMonthlyFileSignature(files);
      if (!files.length) {
        appendEmptyTableRow(filesTbody, 3, "No files recorded in run directory.");
      } else {
        files.forEach(function (f) {
          const tr = document.createElement("tr");
          const tdN = document.createElement("td");
          const code = document.createElement("code");
          code.textContent = f.name;
          tdN.appendChild(code);
          const tdS = document.createElement("td");
          tdS.textContent = f.size_human || (f.size_bytes + " B");
          const tdM = document.createElement("td");
          tdM.textContent = f.mtime ? new Date(f.mtime * 1000).toISOString() : "—";
          tr.appendChild(tdN);
          tr.appendChild(tdS);
          tr.appendChild(tdM);
          filesTbody.appendChild(tr);
        });
      }
    } catch (err) {
      setNotice(err.message, true);
    }

    try {
      const logRes = await api("/api/v1/runs/" + encodeURIComponent(runName) + "/logs?max_bytes=16384");
      const truncNote = el("detail-log-trunc-note");
      if (logRes.truncated) {
        truncNote.textContent =
          "Showing last " + Math.round((logRes.max_bytes || 16384) / 1024) + " KB of log output (older lines truncated).";
        show(truncNote, true);
      } else {
        show(truncNote, false);
      }
      el("detail-log-pre").textContent = logRes.content || "(no log output yet)";
    } catch (err) {
      el("detail-log-pre").textContent = "Log unavailable: " + err.message;
    }

    // Do NOT re-fetch summary or redraw PNG plots on live polls unless monthly NetCDF `(size, mtime)` changed!
    if (!forceSummaryReload && monthlySig === state.detailMonthlySig && state.detailMonthlySig !== null) {
      return;
    }
    state.detailMonthlySig = monthlySig;

    const plotsBox = el("detail-plots-box");
    const sumBox = el("detail-summary-box");
    plotsBox.textContent = "";
    sumBox.textContent = "";
    if (!monthlySig) {
      const p = document.createElement("p");
      p.className = "muted";
      p.textContent = "Monthly summary not yet available (monthly_mean_predictions.nc has not been written yet).";
      sumBox.appendChild(p);
      return;
    }

    try {
      const sumRes = await api("/api/v1/runs/" + encodeURIComponent(runName) + "/summary");
      const vars = sumRes.variables || {};
      const varNames = Object.keys(vars);
      if (sumRes.plot_available && sumRes.plot_urls) {
        for (let i = 0; i < varNames.length; i += 1) {
          const vname = varNames[i];
          const url = sumRes.plot_urls[vname];
          if (url) {
            const canvas = document.createElement("canvas");
            canvas.className = "plot-img";
            canvas.setAttribute("aria-label", "Monthly global mean plot for " + vname);
            plotsBox.appendChild(canvas);
            try {
              await renderPlotToCanvas(canvas, url);
            } catch (_plotErr) {
              canvas.remove();
            }
          }
        }
      }

      const nSamples = Number(sumRes.n_samples || 1);
      const shownMembers = nSamples > 1 ? Math.min(nSamples, 8) : 0;

      varNames.forEach(function (vname) {
        const meta = vars[vname];
        const unitLabel = meta.units ? " [" + meta.units + "]" : "";
        const h3 = document.createElement("h3");
        h3.textContent = vname + unitLabel + (meta.long_name ? " — " + meta.long_name : "");
        sumBox.appendChild(h3);

        const series = meta.series || [];
        const modalSteps = computeModalSteps(series);

        const tbl = document.createElement("table");
        tbl.className = "data-table";
        const thead = document.createElement("thead");
        const hr = document.createElement("tr");
        const cols = [
          "Month",
          "Steps",
          nSamples > 1 ? "Ensemble Mean" : "Global Mean",
        ];
        for (let m = 0; m < shownMembers; m += 1) {
          cols.push("Member " + (m + 1));
        }
        cols.push("Units");
        cols.forEach(function (col) {
          const th = document.createElement("th");
          th.textContent = col;
          hr.appendChild(th);
        });
        thead.appendChild(hr);
        tbl.appendChild(thead);

        const tb = document.createElement("tbody");
        series.forEach(function (row) {
          const tr = document.createElement("tr");
          let stepStr = "—";
          if (row.counts != null) {
            const isPartial = modalSteps != null && series.length > 1 && row.counts > 0 && row.counts < modalSteps;
            stepStr = String(row.counts) + (isPartial ? " (partial)" : "");
          }
          const vals = [
            row.month,
            stepStr,
            row.ensemble_mean != null ? Number(row.ensemble_mean).toFixed(4) : "—",
          ];
          for (let m = 0; m < shownMembers; m += 1) {
            const mv = Array.isArray(row.members) ? row.members[m] : null;
            vals.push(mv != null ? Number(mv).toFixed(4) : "—");
          }
          vals.push(meta.units || "—");
          vals.forEach(function (v) {
            const td = document.createElement("td");
            td.textContent = v;
            tr.appendChild(td);
          });
          tb.appendChild(tr);
        });
        tbl.appendChild(tb);
        sumBox.appendChild(tbl);
      });
    } catch (err) {
      const p = document.createElement("p");
      p.className = "muted";
      p.textContent = "Monthly summary not yet available: " + err.message;
      sumBox.appendChild(p);
    }
  }

  function syncDetailPolling() {
    stopPolling();
    if (el("detail-log-autopoll").checked && state.detailRunName) {
      startBoundedPoll(function () {
        return loadRunDetail(state.detailRunName, false);
      });
    }
  }

  /* ---------------- 6. Doctor ---------------- */
  async function loadDoctor() {
    try {
      const doc = await api("/api/v1/doctor");
      const tbody = el("doctor-tbody");
      tbody.textContent = "";
      (doc.checks || []).forEach(function (chk) {
        const tr = document.createElement("tr");
        const tdN = document.createElement("td");
        tdN.textContent = chk.name;
        const tdS = document.createElement("td");
        tdS.appendChild(makeBadge(chk.status));
        const tdD = document.createElement("td");
        tdD.textContent = chk.detail || "";
        const tdH = document.createElement("td");
        tdH.textContent = chk.fix_hint || "—";
        tr.appendChild(tdN);
        tr.appendChild(tdS);
        tr.appendChild(tdD);
        tr.appendChild(tdH);
        tbody.appendChild(tr);
      });
    } catch (err) {
      setNotice(err.message, true);
    }
  }

  /* ---------------- Optional Assistant Panel ---------------- */
  function appendChatBubble(role, text, proposedPlan) {
    const box = el("chat-messages");
    const div = document.createElement("div");
    div.className = "chat-bubble" + (role === "user" ? " chat-bubble-user" : "");
    const p = document.createElement("div");
    p.textContent = text;
    div.appendChild(p);

    if (proposedPlan && proposedPlan.approval_prefix) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn btn-primary btn-small top-gap";
      btn.textContent = "Review & Approve [" + proposedPlan.approval_prefix + "]";
      btn.addEventListener("click", function () {
        planState.onChatProposal(proposedPlan);
        if (currentRouteName() === "plan") {
          renderPlanView(proposedPlan);
        } else {
          window.location.hash = "#plan";
        }
      });
      div.appendChild(btn);
    }
    box.appendChild(div);
    box.scrollTop = box.scrollHeight;
  }

  async function handleChatSubmit(ev) {
    ev.preventDefault();
    const input = el("chat-input");
    const sendBtn = el("btn-chat-send");
    const msg = input.value.trim();
    if (!msg) return;
    input.value = "";
    sendBtn.disabled = true;
    sendBtn.textContent = "Thinking...";
    appendChatBubble("user", msg, null);
    try {
      const res = await api("/api/v1/chat", { method: "POST", body: { message: msg } });
      if (res.proposed_plan) {
        planState.onChatProposal(res.proposed_plan);
        if (currentRouteName() === "plan") {
          if (planState.getDrawnPlan()) {
            syncSupersededBanner();
          } else {
            renderPlanView(res.proposed_plan);
          }
        }
      }
      appendChatBubble("assistant", res.reply || "", res.proposed_plan || null);
    } catch (err) {
      appendChatBubble("assistant", "Error: " + err.message, null);
    } finally {
      sendBtn.disabled = false;
      sendBtn.textContent = "Send";
    }
  }

  /* ---------------- Bootstrap ---------------- */
  async function initSession() {
    bootstrapToken();
    try {
      const sess = await api("/api/v1/session");
      state.session = Object.assign(state.session, sess || {});
      const containerTag = sess.in_container ? " · container" : "";
      el("session-meta").textContent =
        "v" + (sess.version || "") + containerTag + (sess.run_root ? " · " + sess.run_root : "");
      if (sess.run_root) {
        el("runs-root-label").textContent = "Run root: " + sess.run_root;
      }
      if (sess.chat_log_dir && el("chat-privacy-hint")) {
        el("chat-privacy-hint").textContent =
          "Prompts and tool outputs are sent to Google Gemini and logged locally under " +
          sess.chat_log_dir +
          ". Only your click on Approve can execute a run.";
      }
      if (sess.pending_submit) {
        planState.onFormPlan(sess.pending_submit);
      }
      const chatBtn = el("btn-toggle-chat");
      if (sess.agent_ready) {
        chatBtn.disabled = false;
        chatBtn.title = "Open Planning Assistant";
        show(chatBtn, true);
      } else if (sess.agent_available) {
        chatBtn.disabled = true;
        chatBtn.title = "Assistant unavailable: set GEMINI_API_KEY or AIMIP_GEMINI_KEY_FILE before starting aimip ui.";
        show(chatBtn, true);
      } else {
        show(chatBtn, false);
      }
    } catch (_err) {
      // Handled by api() 401 banner if unauthenticated
    }
    applyRoute();
  }

  window.addEventListener("hashchange", applyRoute);

  ["filter-task", "filter-family", "filter-status", "filter-installed"].forEach(function (id) {
    el(id).addEventListener("change", loadModels);
  });
  el("btn-refresh-models").addEventListener("click", loadModels);
  el("btn-launch-from-card").addEventListener("click", function () {
    if (state.selectedModelCard) selectModelForNewRun(state.selectedModelCard.name);
  });

  el("run-model").addEventListener("change", function () {
    populateModelDefaults(el("run-model").value);
  });
  el("run-start").addEventListener("input", function () {
    el("run-start").dataset.autofilled = "0";
  });
  el("run-horizon-kind").addEventListener("change", handleHorizonKindChange);
  el("run-mode").addEventListener("change", updateModeWarning);
  el("new-run-form").addEventListener("submit", handleGeneratePlan);

  el("plan-download-consent").addEventListener("change", function () {
    planState.setAllowDownload(el("plan-download-consent").checked);
  });
  el("btn-load-newer-plan").addEventListener("click", function () {
    renderPlanView(planState.getLatestPlan());
  });
  el("btn-approve-submit").addEventListener("click", handleApproveAndSubmit);

  el("btn-refresh-runs").addEventListener("click", loadRuns);
  el("btn-refresh-tail").addEventListener("click", loadRunsTail);
  el("runs-autopoll").addEventListener("change", syncRunsPolling);
  el("btn-confirm-cancel").addEventListener("click", function () { confirmCancel(false); });
  el("btn-dismiss-cancel").addEventListener("click", function () { show(el("cancel-approval-box"), false); });

  el("btn-detail-refresh").addEventListener("click", function () {
    if (state.detailRunName) loadRunDetail(state.detailRunName, true);
  });
  el("btn-detail-cancel").addEventListener("click", function () {
    if (state.detailRunName) proposeCancel(state.detailRunName, true);
  });
  el("btn-detail-confirm-cancel").addEventListener("click", function () { confirmCancel(true); });
  el("btn-detail-dismiss-cancel").addEventListener("click", function () { show(el("detail-cancel-box"), false); });
  el("detail-log-autopoll").addEventListener("change", syncDetailPolling);

  el("btn-refresh-doctor").addEventListener("click", loadDoctor);

  el("btn-toggle-chat").addEventListener("click", function () {
    if (el("btn-toggle-chat").disabled) return;
    const drawer = el("chat-drawer");
    show(drawer, drawer.classList.contains("hidden"));
  });
  el("btn-close-chat").addEventListener("click", function () {
    show(el("chat-drawer"), false);
  });
  el("chat-form").addEventListener("submit", handleChatSubmit);

  initSession();
})(typeof globalThis !== "undefined" ? globalThis : window);
