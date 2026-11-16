"""End-to-end HTTP, security, approval-broker, two-step staging, plotting, resource-bound, and chat tests for `aimip ui`."""

from __future__ import annotations

import contextlib
import http.client
import http.cookiejar
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import threading
import time
from typing import Any
import urllib.request
from urllib.parse import urlsplit

import cftime
import numpy as np
import pytest
import xarray as xr

from aimip import status as status_mod
from aimip.errors import ConfigError, ValidationError
import aimip.ui.server as ui_server
from aimip.ui.server import (
    CSP_HEADER_VALUE,
    create_ui_server,
    validate_bind_host,
)


@pytest.fixture(scope="session")
def _session_mpl_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("aimip_ui_mplconfig")


@pytest.fixture(autouse=True)
def _fast_local_mplconfig(_session_mpl_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep matplotlib font-cache builds off shared network filesystems during UI tests."""
    monkeypatch.setenv("MPLCONFIGDIR", str(_session_mpl_dir))


@contextlib.contextmanager
def running_ui_server(
    *,
    host: str = "127.0.0.1",
    allow_remote: bool = False,
    token: str | None = None,
    now_fn: Any = None,
    chat_model: Any = None,
    max_connections: int = 16,
    request_timeout: float = 30.0,
    max_diag_workers: int = 2,
):
    kwargs: dict[str, Any] = {
        "host": host,
        "port": 0,
        "allow_remote": allow_remote,
        "token": token,
        "chat_model": chat_model,
        "max_connections": max_connections,
        "request_timeout": request_timeout,
        "max_diag_workers": max_diag_workers,
    }
    if now_fn is not None:
        kwargs["now_fn"] = now_fn
    server = create_ui_server(**kwargs)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10.0)


def _raw_http(
    port: int,
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | bytes | None = None,
    timeout: float = 60.0,
) -> tuple[int, dict[str, str], bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        req_headers: dict[str, str] = {"Host": f"127.0.0.1:{port}"}
        payload_bytes: bytes | None = None
        if isinstance(body, dict):
            payload_bytes = json.dumps(body).encode("utf-8")
            req_headers["Content-Type"] = "application/json"
            req_headers["Content-Length"] = str(len(payload_bytes))
        elif isinstance(body, bytes):
            payload_bytes = body
            req_headers["Content-Length"] = str(len(payload_bytes))
        if headers:
            for hk, hv in headers.items():
                if hk.lower() == "host":
                    req_headers.pop("Host", None)
                if hk.lower() == "content-type":
                    req_headers.pop("Content-Type", None)
                req_headers[hk] = hv
        conn.request(method, path, body=payload_bytes, headers=req_headers)
        resp = conn.getresponse()
        raw = resp.read()
        hdrs = {k.lower(): v for k, v in resp.getheaders()}
        return resp.status, hdrs, raw
    finally:
        conn.close()


def _write_monthly_summary_nc(run_dir: Path) -> None:
    vt = np.array([[cftime.datetime(1979, 1, 15, calendar="proleptic_gregorian")]], dtype=object)
    ds = xr.Dataset(
        data_vars={
            "TMP2m": xr.DataArray(
                np.full((1, 1, 4, 8), 287.75, dtype=np.float32),
                dims=("sample", "time", "latitude", "longitude"),
                attrs={"units": "K", "long_name": "2m air temperature"},
            ),
            "counts": xr.DataArray(np.array([[40]], dtype=np.int32), dims=("sample", "time")),
        },
        coords={
            "time": xr.DataArray(np.array([0], dtype=np.int64), dims=("time",)),
            "latitude": xr.DataArray(np.array([-60.0, -20.0, 20.0, 60.0], dtype=np.float64), dims=("latitude",)),
            "longitude": xr.DataArray(np.linspace(0.0, 315.0, 8, dtype=np.float64), dims=("longitude",)),
            "valid_time": xr.DataArray(vt, dims=("sample", "time")),
        },
    )
    ds.to_netcdf(
        str(run_dir / "monthly_mean_predictions.nc"),
        engine="netcdf4",
        encoding={
            "valid_time": {
                "units": "days since 1970-01-01 00:00:00",
                "calendar": "proleptic_gregorian",
                "dtype": "int32",
            }
        },
    )


def test_ui_bind_host_loopback_enforced_unless_allow_remote() -> None:
    assert validate_bind_host("127.0.0.1") == "127.0.0.1"
    assert validate_bind_host("::1") == "::1"
    assert validate_bind_host("localhost") == "localhost"

    with pytest.raises(ConfigError, match="--allow-remote"):
        create_ui_server(host="0.0.0.0", port=0)
    with pytest.raises(ConfigError, match="ssh -N -L"):
        create_ui_server(host="192.0.2.15", port=0)

    assert validate_bind_host("0.0.0.0", allow_remote=True) == "0.0.0.0"
    assert ui_server.main(["--host", "0.0.0.0", "--port", "0"]) == 2


def test_ui_static_csp_white_theme_zero_cookie_cross_port_isolation_and_rebinding_checks(
    playground_env,
) -> None:
    received_by_neighbor: list[dict[str, str]] = []

    class _NeighborHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def do_GET(self) -> None:  # noqa: N802
            received_by_neighbor.append({k.lower(): v for k, v in self.headers.items()})
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

    neighbor = HTTPServer(("127.0.0.1", 0), _NeighborHandler)
    neighbor_port = int(neighbor.server_address[1])
    neighbor_thread = threading.Thread(target=neighbor.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    neighbor_thread.start()

    try:
        with running_ui_server() as srv, running_ui_server() as srv2:
            port = int(srv.server_address[1])
            port2 = int(srv2.server_address[1])
            tok = srv.app.token

            # 1. Static files (GET + HEAD) served with strict CSP, no external URLs, and light/white theme only
            st_head, head_hdrs, head_raw = _raw_http(port, "HEAD", "/")
            assert st_head == 200
            assert head_hdrs["content-security-policy"] == CSP_HEADER_VALUE
            assert head_raw == b""

            for route, ctype_sub in (
                ("/", "text/html"),
                ("/static/styles.css", "text/css"),
                ("/static/app.js", "javascript"),
            ):
                st, hdrs, raw = _raw_http(port, "GET", route)
                assert st == 200
                assert hdrs["content-security-policy"] == CSP_HEADER_VALUE
                assert hdrs["x-frame-options"] == "DENY"
                assert hdrs["x-content-type-options"] == "nosniff"
                assert "set-cookie" not in hdrs
                assert ctype_sub in hdrs["content-type"]
                text = raw.decode("utf-8")
                if route in ("/", "/static/styles.css"):
                    assert "https://" not in text and "http://" not in text
                    assert "prefers-color-scheme: dark" not in text

            # 2. Every `/api/v1/*` call requires an explicit header token (cookies are never accepted)
            st_no_tok, _, raw_no_tok = _raw_http(port, "GET", "/api/v1/models")
            assert st_no_tok == 401
            assert json.loads(raw_no_tok)["error"]["code"] == "unauthorized"

            st_bad_tok, _, _ = _raw_http(port, "GET", "/api/v1/models", headers={"X-AIMIP-Token": "wrong-token"})
            assert st_bad_tok == 401

            st_cookie_only, _, _ = _raw_http(
                port,
                "GET",
                "/api/v1/models",
                headers={"Cookie": f"aimip_ui_token={tok}"},
            )
            assert st_cookie_only == 401

            # 3. `/?token=...` validates token, returns 200 HTML with `Cache-Control: no-store`, and emits NO `Set-Cookie`
            st_bad_q, _, _ = _raw_http(port, "GET", "/?token=wrong")
            assert st_bad_q == 401

            st_boot, boot_hdrs, boot_raw = _raw_http(port, "GET", f"/?token={tok}")
            assert st_boot == 200
            assert "set-cookie" not in boot_hdrs
            assert "no-store" in boot_hdrs.get("cache-control", "")
            assert b"AIMIP Console UI" in boot_raw

            # Verify RFC 6265 browser cookie jar receives ZERO cookies from port A and leaks NOTHING to neighbor port B
            jar = http.cookiejar.CookieJar()
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({}),
                urllib.request.HTTPCookieProcessor(jar),
            )
            with opener.open(f"http://127.0.0.1:{port}/?token={tok}", timeout=30) as resp_a:
                assert resp_a.status == 200
            assert len(jar) == 0
            with opener.open(f"http://127.0.0.1:{neighbor_port}/probe", timeout=30) as resp_b:
                assert resp_b.status == 200
            assert len(received_by_neighbor) == 1
            assert "cookie" not in received_by_neighbor[0]
            assert tok not in json.dumps(received_by_neighbor[0])

            # Cross-instance isolation: token from srv1 is rejected by srv2, and cross-port Origin is rejected
            st_other_srv, _, _ = _raw_http(port2, "GET", "/api/v1/session", headers={"X-AIMIP-Token": tok})
            assert st_other_srv == 401

            st_cross_port_origin, _, _ = _raw_http(
                port,
                "POST",
                "/api/v1/plan",
                headers={"Origin": f"http://127.0.0.1:{neighbor_port}", "X-AIMIP-Token": tok},
                body={"model": "ace2-era5", "start": "1979-01-01", "steps": 4},
            )
            assert st_cross_port_origin == 403

            # Non-JSON `Content-Type` on POST with body is rejected
            st_bad_ctype, _, raw_bad_ctype = _raw_http(
                port,
                "POST",
                "/api/v1/plan",
                headers={"X-AIMIP-Token": tok, "Content-Type": "text/plain"},
                body=b'{"model":"ace2-era5","start":"1979-01-01","steps":4}',
            )
            assert st_bad_ctype == 400
            assert json.loads(raw_bad_ctype)["error"]["field"] == "content_type"

            auth_hdr = {"X-AIMIP-Token": tok}
            st_sess, _, raw_sess = _raw_http(port, "GET", "/api/v1/session", headers=auth_hdr)
            assert st_sess == 200
            assert json.loads(raw_sess)["ok"] is True

            st_models, _, raw_models = _raw_http(port, "GET", "/api/v1/models?task=amip", headers=auth_hdr)
            assert st_models == 200
            models_list = json.loads(raw_models)["result"]
            assert any(m["name"] == "ace2-era5" for m in models_list)

            st_card, _, raw_card = _raw_http(port, "GET", "/api/v1/models/ace2-era5?examples=1", headers=auth_hdr)
            assert st_card == 200
            card_doc = json.loads(raw_card)["result"]
            assert card_doc["name"] == "ace2-era5" and card_doc["citation"]["confirmed"] is True

            # 4. Anti-DNS-rebinding & CSRF checks on `Host`, `Origin`, and `Referer`
            st_host, _, raw_host = _raw_http(
                port,
                "GET",
                "/api/v1/models",
                headers={"Host": "rebind.attacker.invalid:80", "X-AIMIP-Token": tok},
            )
            assert st_host == 403
            assert json.loads(raw_host)["error"]["code"] == "forbidden_host"

            st_orig, _, raw_orig = _raw_http(
                port,
                "POST",
                "/api/v1/plan",
                headers={"Origin": "http://evil.example.org", "X-AIMIP-Token": tok},
                body={"model": "ace2-era5", "start": "1979-01-01", "steps": 4},
            )
            assert st_orig == 403
            assert json.loads(raw_orig)["error"]["code"] == "forbidden_origin"

            st_ref, _, raw_ref = _raw_http(
                port,
                "GET",
                "/api/v1/models",
                headers={"Referer": "http://evil.example.org/phish", "X-AIMIP-Token": tok},
            )
            assert st_ref == 403
            assert json.loads(raw_ref)["error"]["code"] == "forbidden_referer"
    finally:
        neighbor.shutdown()
        neighbor.server_close()
        neighbor_thread.join(timeout=5.0)


def test_ui_path_traversal_picker_filtering_and_strict_integer_validation(
    playground_env,
) -> None:
    run_root = Path(playground_env["default_out_root"])
    noise_run = run_root / "prior-run"
    noise_run.mkdir(parents=True, exist_ok=True)
    (noise_run / "monthly_mean_predictions.nc").write_bytes(b"out1")
    (noise_run / "autoregressive_predictions.nc").write_bytes(b"out2")
    (noise_run / "restart.nc").write_bytes(b"restart-candidate")

    with running_ui_server() as srv:
        port = int(srv.server_address[1])
        auth_hdr = {"X-AIMIP-Token": srv.app.token}

        for bad_path in (
            "/api/v1/runs/..%2fetc/status",
            "/api/v1/runs/../outside/logs",
            "/..%2fpyproject.toml",
            "/api/v1/runs/bad%00run",
            "/api/v1/runs/sub%5cdir/status",
        ):
            st, _, raw = _raw_http(port, "GET", bad_path, headers=auth_hdr)
            assert st == 400, (bad_path, st, raw)
            assert json.loads(raw)["ok"] is False

        st_pick, _, raw_pick = _raw_http(port, "GET", "/api/v1/pickers/files?kind=ic", headers=auth_hdr)
        assert st_pick == 200
        pick_res = json.loads(raw_pick)["result"]
        assert "truncated" in pick_res
        picked_names = [f["name"] for f in pick_res["files"]]
        assert "restart.nc" in picked_names
        assert "monthly_mean_predictions.nc" not in picked_names
        assert "autoregressive_predictions.nc" not in picked_names
        assert not any(n.startswith("forcing_") for n in picked_names)
        assert picked_names[0] == "restart.nc"

        st_escape, _, _ = _raw_http(port, "GET", "/api/v1/pickers/files?kind=ic&dir=/etc", headers=auth_hdr)
        assert st_escape == 400

        # Fractional numeric fields are rejected instead of silently truncated
        for bad_body, expected_field in (
            ({"model": "ace2-era5", "start": "1979-01-01", "steps": 4.5}, "steps"),
            ({"model": "ace2-era5", "start": "1979-01-01", "steps": 4, "members": 2.7}, "members"),
            (
                {
                    "model": "ace2-era5",
                    "start": "1979-01-01",
                    "steps": 4,
                    "resources": {"cpus_per_task": "1.5"},
                },
                "resources.cpus_per_task",
            ),
        ):
            st_frac, _, raw_frac = _raw_http(port, "POST", "/api/v1/plan", headers=auth_hdr, body=bad_body)
            assert st_frac == 400
            assert json.loads(raw_frac)["error"]["field"] == expected_field


def test_ui_plan_approve_wrong_prefix_replay_expiry_drift_and_download_consent(
    playground_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fbin = tmp_path / "ui_slurm_bin"
    fbin.mkdir()
    sbatch = fbin / "sbatch"
    sbatch.write_text("#!/bin/bash\necho 515151\n", encoding="utf-8")
    sbatch.chmod(0o755)
    squeue = fbin / "squeue"
    squeue.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    squeue.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fbin}:{os.environ.get('PATH', '')}")

    now_box = [1_700_000_000.0]
    with running_ui_server(now_fn=lambda: now_box[0]) as srv:
        port = int(srv.server_address[1])
        hdr = {"X-AIMIP-Token": srv.app.token}

        # 1. Unapproved submit refused
        st_unauth, _, _ = _raw_http(port, "POST", "/api/v1/submit", headers=hdr, body={})
        assert st_unauth == 403

        # 2. Plan is side-effect-free, includes `outputs:` line + model decision context; wrong prefix refused
        st_plan, _, raw_plan = _raw_http(
            port,
            "POST",
            "/api/v1/plan",
            headers=hdr,
            body={"model": "ace2-era5", "start": "1979-01-01", "steps": 4, "name": "ui-exp", "prediction": True},
        )
        assert st_plan == 200
        prop = json.loads(raw_plan)["result"]
        assert not Path(prop["plan"]["output_dir"]).exists()
        assert "outputs:" in prop["card"] and "--prediction: large output" in prop["card"]
        assert prop["model_card"]["licence"] == "Apache-2.0"
        assert prop["model_card"]["status_definition"]
        assert prop["model_card"]["limitations"]

        wrong_pfx = "00000000" if prop["approval_prefix"] != "00000000" else "11111111"
        st_wrong, _, _ = _raw_http(
            port, "POST", "/api/v1/approve", headers=hdr, body={"digest_prefix": wrong_pfx, "action": "submit"}
        )
        assert st_wrong == 403

        st_short, _, _ = _raw_http(
            port, "POST", "/api/v1/approve", headers=hdr, body={"digest_prefix": prop["approval_prefix"][:6]}
        )
        assert st_short == 403

        # 3. Approve valid prefix, then expire token (+601s) -> submit refused
        st_app1, _, raw_app1 = _raw_http(
            port,
            "POST",
            "/api/v1/approve",
            headers=hdr,
            body={"digest_prefix": prop["approval_prefix"], "digest": prop["digest"], "action": "submit"},
        )
        assert st_app1 == 200
        nonce1 = json.loads(raw_app1)["result"]["authorization_nonce"]
        now_box[0] += 601.0
        st_exp, _, raw_exp = _raw_http(port, "POST", "/api/v1/submit", headers=hdr, body={"nonce": nonce1})
        assert st_exp == 403
        assert "expired" in json.loads(raw_exp)["error"]["message"].lower()

        # 4. Re-plan + approve, then cause plan drift before submit -> refused and token burned
        _, _, raw_plan2 = _raw_http(
            port,
            "POST",
            "/api/v1/plan",
            headers=hdr,
            body={"model": "ace2-era5", "start": "1979-01-01", "steps": 4, "name": "ui-drift"},
        )
        prop2 = json.loads(raw_plan2)["result"]
        _, _, raw_app2 = _raw_http(
            port,
            "POST",
            "/api/v1/approve",
            headers=hdr,
            body={"digest_prefix": prop2["approval_prefix"], "digest": prop2["digest"]},
        )
        nonce2 = json.loads(raw_app2)["result"]["authorization_nonce"]

        reg_path: Path = playground_env["registry_path"]
        orig_reg = reg_path.read_text(encoding="utf-8")
        reg_doc = json.loads(orig_reg)
        alt_forcing = tmp_path / "ui_alt_forcing"
        alt_forcing.mkdir()
        (alt_forcing / "forcing_1979.nc").write_bytes(b"changed-forcing")
        reg_doc["models"][0]["forcing_sets"][0]["dir"] = str(alt_forcing)
        reg_path.write_text(json.dumps(reg_doc), encoding="utf-8")

        st_drift, _, _ = _raw_http(port, "POST", "/api/v1/submit", headers=hdr, body={"nonce": nonce2})
        assert st_drift == 403
        reg_path.write_text(orig_reg, encoding="utf-8")
        st_burned, _, _ = _raw_http(port, "POST", "/api/v1/submit", headers=hdr, body={"nonce": nonce2})
        assert st_burned == 403

        # 5. Clean plan -> approve -> submit succeeds, and replay is refused
        _, _, raw_plan3 = _raw_http(
            port,
            "POST",
            "/api/v1/plan",
            headers=hdr,
            body={"model": "ace2-era5", "start": "1979-01-01", "steps": 4, "name": "ui-good"},
        )
        prop3 = json.loads(raw_plan3)["result"]
        _, _, raw_app3 = _raw_http(
            port,
            "POST",
            "/api/v1/approve",
            headers=hdr,
            body={"digest_prefix": prop3["approval_prefix"], "digest": prop3["digest"]},
        )
        nonce3 = json.loads(raw_app3)["result"]["authorization_nonce"]
        st_sub, _, raw_sub = _raw_http(port, "POST", "/api/v1/submit", headers=hdr, body={"nonce": nonce3})
        assert st_sub == 200
        sub_res = json.loads(raw_sub)["result"]
        assert sub_res["state"] == "SUBMITTED" and sub_res["attempt_id"] == "515151"

        st_replay, _, _ = _raw_http(port, "POST", "/api/v1/submit", headers=hdr, body={"nonce": nonce3})
        assert st_replay == 403

        # 6. Separate download consent + read-only `$AIMIP_DATA` check for unfetched model
        monkeypatch.delenv("AIMIP_REGISTRY", raising=False)
        empty_data = tmp_path / "ui_empty_data"
        empty_data.mkdir()
        monkeypatch.setenv("AIMIP_DATA", str(empty_data))
        monkeypatch.setenv("AIMIP_DATA_MODE", "ro")

        st_dl_plan, _, raw_dl_plan = _raw_http(
            port,
            "POST",
            "/api/v1/plan",
            headers=hdr,
            body={"model": "ace2-som", "start": "2032-01-01", "steps": 4, "forcing_set": "1xCO2", "name": "ui-dl"},
        )
        assert st_dl_plan == 200
        dl_prop = json.loads(raw_dl_plan)["result"]
        assert dl_prop["download_required"] is True and dl_prop["missing_bytes"] > 0

        st_no_dl, _, raw_no_dl = _raw_http(
            port,
            "POST",
            "/api/v1/approve",
            headers=hdr,
            body={"digest_prefix": dl_prop["approval_prefix"], "digest": dl_prop["digest"], "allow_download": False},
        )
        assert st_no_dl == 403
        err_no_dl = json.loads(raw_no_dl)["error"]
        assert err_no_dl["field"] == "allow_download"
        assert "tick the download consent box" in err_no_dl["message"].lower()

        st_ro_dl, _, raw_ro_dl = _raw_http(
            port,
            "POST",
            "/api/v1/approve",
            headers=hdr,
            body={"digest_prefix": dl_prop["approval_prefix"], "digest": dl_prop["digest"], "allow_download": True},
        )
        assert st_ro_dl == 403
        assert "aimip fetch -m ace2-som" in json.loads(raw_ro_dl)["error"]["message"]


def test_ui_http_two_step_staging_prepare_then_final_submit(
    tmp_path: Path, allow_tmp_run_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HTTP `/api/v1/plan` -> `/api/v1/approve` (`prepare`) -> `/api/v1/prepare` -> `/api/v1/approve` (`submit`) -> `/api/v1/submit` for staging models."""
    from test_phase2_execution_security import _setup_synthetic_dlesym_env

    env = _setup_synthetic_dlesym_env(tmp_path / "ui_dl", allow_tmp_run_root, monkeypatch)
    env["stage_payload_file"].write_bytes(b"ui-http-staged-era5-ic")

    fbin = tmp_path / "ui_dl_bin"
    fbin.mkdir()
    sbatch_log = tmp_path / "ui_dl_sbatch.log"
    (fbin / "sbatch").write_text(f'#!/bin/bash\necho "$@" >> "{sbatch_log}"\necho 929292\n', encoding="utf-8")
    (fbin / "sbatch").chmod(0o755)
    (fbin / "squeue").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    (fbin / "squeue").chmod(0o755)
    monkeypatch.setenv("PATH", f"{fbin}:{os.environ.get('PATH', '')}")

    with running_ui_server() as srv:
        port = int(srv.server_address[1])
        hdr = {"X-AIMIP-Token": srv.app.token}

        st_plan, _, raw_plan = _raw_http(
            port,
            "POST",
            "/api/v1/plan",
            headers=hdr,
            body={"model": "dlesym-v1.0", "start": "1995-01-01", "steps": 8, "name": "ui-http-2step"},
        )
        assert st_plan == 200
        prop1 = json.loads(raw_plan)["result"]
        assert prop1["action"] == "prepare" and prop1["requires_staging"] is True

        st_app_bad, _, raw_app_bad = _raw_http(
            port,
            "POST",
            "/api/v1/approve",
            headers=hdr,
            body={
                "action": "prepare",
                "digest_prefix": prop1["approval_prefix"],
                "digest": prop1["digest"],
                "allow_download": True,
            },
        )
        assert st_app_bad == 200
        bad_nonce = json.loads(raw_app_bad)["result"]["authorization_nonce"]
        st_sub_refused, _, _ = _raw_http(port, "POST", "/api/v1/submit", headers=hdr, body={"nonce": bad_nonce})
        assert st_sub_refused == 403
        assert not sbatch_log.exists()

        _, _, raw_plan1b = _raw_http(
            port,
            "POST",
            "/api/v1/plan",
            headers=hdr,
            body={"model": "dlesym-v1.0", "start": "1995-01-01", "steps": 8, "name": "ui-http-2step"},
        )
        prop1b = json.loads(raw_plan1b)["result"]
        _, _, raw_app1b = _raw_http(
            port,
            "POST",
            "/api/v1/approve",
            headers=hdr,
            body={
                "action": "prepare",
                "digest_prefix": prop1b["approval_prefix"],
                "digest": prop1b["digest"],
                "allow_download": True,
            },
        )
        prep_nonce = json.loads(raw_app1b)["result"]["authorization_nonce"]

        st_prep, _, raw_prep = _raw_http(port, "POST", "/api/v1/prepare", headers=hdr, body={"nonce": prep_nonce})
        assert st_prep == 200
        staged = json.loads(raw_prep)["result"]
        assert staged["stage"] == "prepared_awaiting_submit_approval"
        assert staged["digest"] != prop1b["digest"]
        assert staged["proposal"]["action"] == "submit"
        assert not sbatch_log.exists()

        st_app2, _, raw_app2 = _raw_http(
            port,
            "POST",
            "/api/v1/approve",
            headers=hdr,
            body={"action": "submit", "digest_prefix": staged["approval_prefix"], "digest": staged["digest"]},
        )
        assert st_app2 == 200
        sub_nonce = json.loads(raw_app2)["result"]["authorization_nonce"]

        st_sub, _, raw_sub = _raw_http(port, "POST", "/api/v1/submit", headers=hdr, body={"nonce": sub_nonce})
        assert st_sub == 200
        assert json.loads(raw_sub)["result"]["attempt_id"] == "929292"
        assert len(sbatch_log.read_text(encoding="utf-8").splitlines()) == 1


def test_ui_runs_status_bounded_sanitized_logs_cancel_approval_summary_and_plot_png(
    playground_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fbin = tmp_path / "ui_ops_bin"
    fbin.mkdir()
    scancel_log = tmp_path / "scancel.log"
    (fbin / "sbatch").write_text("#!/bin/bash\necho 717171\n", encoding="utf-8")
    (fbin / "squeue").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    sacct_script = fbin / "sacct"
    sacct_script.write_text('#!/bin/bash\necho "717171|RUNNING|00:00:25|0:0"\n', encoding="utf-8")
    (fbin / "scancel").write_text(f'#!/bin/bash\necho "$@" >> "{scancel_log}"\nexit 0\n', encoding="utf-8")
    for p in fbin.iterdir():
        p.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fbin}:{os.environ.get('PATH', '')}")

    with running_ui_server() as srv:
        port = int(srv.server_address[1])
        hdr = {"X-AIMIP-Token": srv.app.token}

        _, _, raw_plan = _raw_http(
            port,
            "POST",
            "/api/v1/plan",
            headers=hdr,
            body={"model": "ace2-era5", "start": "1979-01-01", "steps": 4, "name": "ui-run1"},
        )
        prop = json.loads(raw_plan)["result"]
        _, _, raw_app = _raw_http(
            port,
            "POST",
            "/api/v1/approve",
            headers=hdr,
            body={"digest_prefix": prop["approval_prefix"], "digest": prop["digest"]},
        )
        nonce = json.loads(raw_app)["result"]["authorization_nonce"]
        st_sub, _, raw_sub = _raw_http(port, "POST", "/api/v1/submit", headers=hdr, body={"nonce": nonce})
        assert st_sub == 200
        run_dir = Path(json.loads(raw_sub)["result"]["run_dir"])

        (run_dir / "slurm-717171.out").write_text(
            "prefix-" + ("x" * 240) + "\n\x1b[31mSTEP_OK_4_OF_4\x1b[0m\n",
            encoding="utf-8",
        )
        _write_monthly_summary_nc(run_dir)

        st_ls, _, raw_ls = _raw_http(port, "GET", "/api/v1/runs", headers=hdr)
        assert st_ls == 200
        assert any(r["name"] == "ui-run1" for r in json.loads(raw_ls)["result"])

        st_det, _, raw_det = _raw_http(port, "GET", "/api/v1/runs/ui-run1", headers=hdr)
        assert st_det == 200
        det = json.loads(raw_det)["result"]
        assert det["state"] == "running"
        assert det["display_state"] == "running"
        assert any(f["name"] == "monthly_mean_predictions.nc" for f in det["files"])

        st_logs, _, raw_logs = _raw_http(port, "GET", "/api/v1/runs/ui-run1/logs?max_bytes=80", headers=hdr)
        assert st_logs == 200
        logs_res = json.loads(raw_logs)["result"]
        assert logs_res["truncated"] is True
        assert "STEP_OK_4_OF_4" in logs_res["content"]
        assert "\x1b" not in logs_res["content"]
        assert len(logs_res["content"]) <= 80

        st_unauth_cancel, _, _ = _raw_http(port, "POST", "/api/v1/runs/ui-run1/cancel", headers=hdr, body={})
        assert st_unauth_cancel == 403

        st_cprop, _, raw_cprop = _raw_http(port, "POST", "/api/v1/runs/ui-run1/cancel/propose", headers=hdr, body={})
        assert st_cprop == 200
        cprop = json.loads(raw_cprop)["result"]
        assert cprop["approval_prefix"] != prop["approval_prefix"]

        st_capp, _, raw_capp = _raw_http(
            port,
            "POST",
            "/api/v1/approve",
            headers=hdr,
            body={"digest_prefix": cprop["approval_prefix"], "digest": cprop["digest"], "action": "cancel"},
        )
        assert st_capp == 200
        cnonce = json.loads(raw_capp)["result"]["authorization_nonce"]

        st_cdone, _, raw_cdone = _raw_http(
            port, "POST", "/api/v1/runs/ui-run1/cancel", headers=hdr, body={"nonce": cnonce}
        )
        assert st_cdone == 200
        cancel_doc = json.loads(raw_cdone)["result"]
        assert cancel_doc["cancelled"] is True and cancel_doc["status"] == "cancel_requested"
        assert scancel_log.read_text(encoding="utf-8").strip() == "717171"

        # While Slurm has not yet transitioned out of RUNNING, `display_state` reflects `cancel_requested`
        st_post_can, _, raw_post_can = _raw_http(port, "GET", "/api/v1/runs/ui-run1", headers=hdr)
        assert st_post_can == 200
        assert json.loads(raw_post_can)["result"]["display_state"] == "cancel_requested"

        st_creplay, _, _ = _raw_http(
            port, "POST", "/api/v1/runs/ui-run1/cancel", headers=hdr, body={"nonce": cnonce}
        )
        assert st_creplay == 403

        sacct_script.write_text('#!/bin/bash\necho "717171|COMPLETED|00:01:00|0:0"\n', encoding="utf-8")
        status_mod._LAST_SACCT_QUERY.clear()
        st_cterm, _, _ = _raw_http(port, "POST", "/api/v1/runs/ui-run1/cancel/propose", headers=hdr, body={})
        assert st_cterm == 400

        st_sum, _, raw_sum = _raw_http(port, "GET", "/api/v1/runs/ui-run1/summary", headers=hdr)
        assert st_sum == 200
        sum_res = json.loads(raw_sum)["result"]
        assert sum_res["run_name"] == "ui-run1"
        assert sum_res["model"] == "ace2-era5"
        assert abs(sum_res["variables"]["TMP2m"]["series"][0]["ensemble_mean"] - 287.75) < 1e-4

        if ui_server._is_matplotlib_available():
            assert sum_res["plot_available"] is True
            plot_path = urlsplit(sum_res["plot_urls"]["TMP2m"]).path + "?var=TMP2m"
            st_png, png_hdrs, png_bytes = _raw_http(port, "GET", plot_path, headers=hdr)
            assert st_png == 200
            assert png_hdrs["content-type"] == "image/png"
            assert png_hdrs["content-security-policy"] == CSP_HEADER_VALUE
            assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")

        monkeypatch.setattr(ui_server, "_is_matplotlib_available", lambda: False)
        st_sum_nompl, _, raw_sum_nompl = _raw_http(port, "GET", "/api/v1/runs/ui-run1/summary", headers=hdr)
        assert st_sum_nompl == 200
        sum_nompl = json.loads(raw_sum_nompl)["result"]
        assert sum_nompl["plot_available"] is False and sum_nompl["plot_urls"] == {}
        st_png_404, _, raw_png_404 = _raw_http(
            port, "GET", "/api/v1/runs/ui-run1/plot.png?var=TMP2m", headers=hdr
        )
        assert st_png_404 == 404
        assert json.loads(raw_png_404)["error"]["code"] == "matplotlib_unavailable"


def test_ui_optional_chat_single_plan_resolution_and_superseded_card_protection(
    playground_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("langgraph")
    lc_messages = pytest.importorskip("langchain_core.messages")
    AIMessage = lc_messages.AIMessage

    fbin = tmp_path / "ui_chat_bin"
    fbin.mkdir()
    sbatch_log = tmp_path / "chat_sbatch.log"
    (fbin / "sbatch").write_text(f'#!/bin/bash\necho "$@" >> "{sbatch_log}"\necho 818181\n', encoding="utf-8")
    (fbin / "sbatch").chmod(0o755)
    (fbin / "squeue").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    (fbin / "squeue").chmod(0o755)
    monkeypatch.setenv("PATH", f"{fbin}:{os.environ.get('PATH', '')}")

    default_out_root: Path = playground_env["default_out_root"]

    # Spy on `ui_server.svc_plan` so if `propose_run` ever resolved twice, call #2 would mutate `steps`
    real_svc_plan = ui_server.svc_plan
    plan_call_count = [0]
    in_chat_turn = [False]

    def _spy_svc_plan(spec: Any, **kw: Any) -> Any:
        if in_chat_turn[0]:
            plan_call_count[0] += 1
            if plan_call_count[0] > 1:
                import dataclasses

                spec = dataclasses.replace(spec, steps=(spec.steps or 4) + 4)
        return real_svc_plan(spec, **kw)

    monkeypatch.setattr(ui_server, "svc_plan", _spy_svc_plan)

    class _ScriptedChatModel:
        def __init__(self) -> None:
            self.step = 0

        def bind_tools(self, tools: Any) -> "_ScriptedChatModel":
            tool_names = {getattr(t, "name", "") for t in tools}
            assert "approve" not in tool_names and "submit" not in tool_names
            assert "approve_and_execute_on_tty" not in tool_names
            return self

        def invoke(self, _messages: Any) -> Any:
            self.step += 1
            if self.step == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "plan_spin",
                            "args": {
                                "model": "ace2-era5",
                                "from_times": ["1979-01-01T00:00:00"],
                                "to_time": "1979-01-02T00:00:00",
                            },
                            "id": "call_spin_blocked",
                        }
                    ],
                )
            if self.step == 2:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "propose_run",
                            "args": {
                                "model": "ace2-era5",
                                "start": "1979-01-01",
                                "steps": 4,
                                "name": "chat-ui-prop",
                            },
                            "id": "call_1",
                        }
                    ],
                )
            if self.step == 3:
                return AIMessage(content="Prepared proposal card; click Approve in the UI to submit.")
            return AIMessage(content="ace2-era5 uses a 1-degree Gaussian grid.")

    with running_ui_server(chat_model=_ScriptedChatModel()) as srv:
        port = int(srv.server_address[1])
        hdr = {"X-AIMIP-Token": srv.app.token}

        # Step A: User first generates Card A on New Run (`steps=8`, `name="form-card-a"`)
        st_card_a, _, raw_card_a = _raw_http(
            port,
            "POST",
            "/api/v1/plan",
            headers=hdr,
            body={"model": "ace2-era5", "start": "1979-01-01", "steps": 8, "name": "form-card-a"},
        )
        assert st_card_a == 200
        card_a = json.loads(raw_card_a)["result"]

        # Step B: Assistant proposes Card B (`steps=4`, `name="chat-ui-prop"`)
        in_chat_turn[0] = True
        try:
            st_chat, _, raw_chat = _raw_http(
                port,
                "POST",
                "/api/v1/chat",
                headers=hdr,
                body={"message": "Propose a 1-day ace2-era5 run starting 1979-01-01"},
            )
        finally:
            in_chat_turn[0] = False

        assert st_chat == 200
        assert plan_call_count[0] == 1, "propose_run must resolve svc_plan exactly once"
        chat_res = json.loads(raw_chat)["result"]
        proposed_b = chat_res["proposed_plan"]
        assert proposed_b is not None and len(proposed_b["approval_prefix"]) == 8
        assert proposed_b["approval_prefix"] in proposed_b["card"]
        assert proposed_b["digest"] != card_a["digest"]
        assert not sbatch_log.exists()
        assert not Path(proposed_b["plan"]["output_dir"]).exists()
        assert {p.name for p in default_out_root.iterdir()} <= {"chat-logs"}

        # Step C: Informational follow-up chat turn does NOT re-attach `proposed_plan`
        st_info, _, raw_info = _raw_http(
            port,
            "POST",
            "/api/v1/chat",
            headers=hdr,
            body={"message": "What grid does ace2-era5 use?"},
        )
        assert st_info == 200
        assert json.loads(raw_info)["result"]["proposed_plan"] is None

        # Step D: Attempting to approve stale Card A (`digest` or `digest_prefix`) is REFUSED without consuming Card B!
        st_stale, _, raw_stale = _raw_http(
            port,
            "POST",
            "/api/v1/approve",
            headers=hdr,
            body={
                "action": "submit",
                "digest_prefix": card_a["approval_prefix"],
                "digest": card_a["digest"],
                "allow_download": True,
            },
        )
        assert st_stale == 403
        assert "superseded" in json.loads(raw_stale)["error"]["message"].lower()

        # Even if a crafted request sent Card B's prefix with Card A's full digest, it is refused without consuming B
        st_mismatch, _, _ = _raw_http(
            port,
            "POST",
            "/api/v1/approve",
            headers=hdr,
            body={
                "action": "submit",
                "digest_prefix": proposed_b["approval_prefix"],
                "digest": card_a["digest"],
            },
        )
        assert st_mismatch == 403

        # Card B is still pending and intact; reviewing & approving Card B succeeds!
        st_sess, _, raw_sess = _raw_http(port, "GET", "/api/v1/session", headers=hdr)
        assert st_sess == 200
        assert json.loads(raw_sess)["result"]["pending_submit"]["digest"] == proposed_b["digest"]

        st_app_b, _, raw_app_b = _raw_http(
            port,
            "POST",
            "/api/v1/approve",
            headers=hdr,
            body={"digest_prefix": proposed_b["approval_prefix"], "digest": proposed_b["digest"]},
        )
        assert st_app_b == 200
        nonce_b = json.loads(raw_app_b)["result"]["authorization_nonce"]
        st_sub_b, _, raw_sub_b = _raw_http(port, "POST", "/api/v1/submit", headers=hdr, body={"nonce": nonce_b})
        assert st_sub_b == 200
        assert json.loads(raw_sub_b)["result"]["attempt_id"] == "818181"
        assert sbatch_log.is_file()


def test_ui_page_script_plan_state_machine_prevents_stale_card_approval_and_resets_consent() -> None:
    app_js = Path(ui_server.__file__).resolve().parent / "static" / "app.js"
    js_text = app_js.read_text(encoding="utf-8")

    # Verify browser security & polling invariants in `app.js`
    assert "document.cookie" not in js_text
    assert 'credentials: "omit"' in js_text
    assert '"X-AIMIP-Token"' in js_text
    assert "sessionStorage" in js_text
    assert "replaceState" in js_text
    assert "POLL_INTERVAL_MS = 15000" in js_text
    assert "__AIMIP_CREATE_PLAN_STATE__" in js_text

    node_bin = shutil.which("node")
    if node_bin:
        probe = (
            f"require({json.dumps(str(app_js))});\n"
            "const assert = require('assert');\n"
            "const st = globalThis.__AIMIP_CREATE_PLAN_STATE__();\n"
            "const cardA = { approval_prefix: 'aaaaaaaa', digest: 'a'.repeat(64), action: 'submit', requires_staging: false };\n"
            "const cardB = { approval_prefix: 'bbbbbbbb', digest: 'b'.repeat(64), action: 'prepare', requires_staging: true };\n"
            "st.onFormPlan(cardA);\n"
            "st.drawPlan();\n"
            "st.setAllowDownload(true);\n"
            "assert.strictEqual(st.getAllowDownload(), true);\n"
            "st.onChatProposal(cardB);\n"
            "// Chat reply MUST NOT overwrite drawnPlan or apply Card A consent to Card B:\n"
            "assert.strictEqual(st.getDrawnPlan().digest, cardA.digest);\n"
            "assert.strictEqual(st.isSuperseded(), true);\n"
            "assert.strictEqual(st.getSupersededBy().digest, cardB.digest);\n"
            "const stalePayload = st.buildApprovePayload();\n"
            "assert.strictEqual(stalePayload.digest, cardA.digest);\n"
            "assert.strictEqual(stalePayload.digest_prefix, 'aaaaaaaa');\n"
            "// Drawing Card B resets download consent to false and binds payload to Card B:\n"
            "st.drawPlan(cardB);\n"
            "assert.strictEqual(st.isSuperseded(), false);\n"
            "assert.strictEqual(st.getAllowDownload(), false);\n"
            "const bPayload = st.buildApprovePayload();\n"
            "assert.strictEqual(bPayload.digest, cardB.digest);\n"
            "assert.strictEqual(bPayload.action, 'prepare');\n"
            "assert.strictEqual(bPayload.allow_download, false);\n"
        )
        proc = subprocess.run(
            [node_bin, "-e", probe],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr


def test_ui_summary_selective_chunked_loading_byte_budget_cache_and_batched_sacct_poll(
    playground_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = Path(playground_env["default_out_root"])
    status_mod._SUMMARY_CACHE.clear()
    status_mod._LAST_SACCT_QUERY.clear()

    # 1. Create 4 active Slurm runs + 2 finished Slurm runs -> verify 1 batched `sacct` call across polls
    fbin = tmp_path / "batch_sacct_bin"
    fbin.mkdir()
    sacct_calls = tmp_path / "batch_sacct_calls.log"
    sacct_lines = [
        f"{9500 + i}|RUNNING|00:01:1{i}|0:0" for i in range(4)
    ]
    nl = "\n"
    (fbin / "sacct").write_text(
        f'#!/bin/bash\necho "$@" >> "{sacct_calls}"\nprintf "{nl.join(sacct_lines)}\\n"\n',
        encoding="utf-8",
    )
    (fbin / "sacct").chmod(0o755)
    monkeypatch.setenv("PATH", f"{fbin}:{os.environ.get('PATH', '')}")

    for idx in range(6):
        rdir = run_root / f"batch-r{idx}"
        rdir.mkdir(parents=True, exist_ok=True)
        jid = 9500 + idx
        (rdir / "slurm.jobid").write_text(f"{jid}\n", encoding="utf-8")
        mdoc: dict[str, Any] = {
            "registry_entry": {"name": "ace2-era5"},
            "mode": "slurm",
            "steps": 4,
            "members": 1,
        }
        if idx >= 4:
            mdoc["finished"] = "2026-11-16T00:00:00Z"
            mdoc["rc"] = 0
        (rdir / "aimip-manifest.json").write_text(json.dumps(mdoc), encoding="utf-8")

    # 2. Create a multi-month NetCDF on `batch-r4` with:
    #    - missing `units` attribute on `TMP2m` (must report `""`, not fabricate `"K"`)
    #    - an unrelated variable `UNSELECTED_FIELD`
    diag_dir = run_root / "batch-r4"
    vt = np.array(
        [[cftime.datetime(1979, m, 15, calendar="proleptic_gregorian") for m in (1, 2, 3)]],
        dtype=object,
    )
    ds = xr.Dataset(
        data_vars={
            "TMP2m": xr.DataArray(
                np.full((1, 3, 4, 8), 289.5, dtype=np.float32),
                dims=("sample", "time", "latitude", "longitude"),
                attrs={"long_name": "2m temperature without units"},
            ),
            "UNSELECTED_FIELD": xr.DataArray(
                np.ones((1, 3, 16, 32), dtype=np.float64),
                dims=("sample", "time", "u_lat", "u_lon"),
            ),
            "counts": xr.DataArray(np.array([[40, 40, 12]], dtype=np.int32), dims=("sample", "time")),
        },
        coords={
            "time": xr.DataArray(np.array([0, 1, 2], dtype=np.int64), dims=("time",)),
            "latitude": xr.DataArray(np.array([-60.0, -20.0, 20.0, 60.0], dtype=np.float64), dims=("latitude",)),
            "longitude": xr.DataArray(np.linspace(0.0, 315.0, 8, dtype=np.float64), dims=("longitude",)),
            "valid_time": xr.DataArray(vt, dims=("sample", "time")),
        },
    )
    nc_file = diag_dir / "monthly_mean_predictions.nc"
    ds.to_netcdf(
        str(nc_file),
        engine="netcdf4",
        encoding={
            "valid_time": {
                "units": "days since 1970-01-01 00:00:00",
                "calendar": "proleptic_gregorian",
                "dtype": "int32",
            }
        },
    )

    real_summarize_ds = status_mod._summarize_monthly_dataset
    ds_reads = [0]

    def _counting_summarize_ds(*args: Any, **kwargs: Any) -> dict[str, Any]:
        ds_reads[0] += 1
        return real_summarize_ds(*args, **kwargs)

    monkeypatch.setattr(status_mod, "_summarize_monthly_dataset", _counting_summarize_ds)
    monkeypatch.setattr(status_mod, "SUMMARY_TIME_CHUNK_MONTHS", 2)

    with running_ui_server(max_diag_workers=1) as srv:
        port = int(srv.server_address[1])
        hdr = {"X-AIMIP-Token": srv.app.token}

        # Poll `/api/v1/runs` twice back-to-back -> at most ONE batched `sacct` subprocess total,
        # querying ONLY the 4 unfinished jobs (9500..9503) and NEVER finished jobs (9504, 9505)!
        st_r1, _, raw_r1 = _raw_http(port, "GET", "/api/v1/runs", headers=hdr)
        st_r2, _, raw_r2 = _raw_http(port, "GET", "/api/v1/runs", headers=hdr)
        assert st_r1 == 200 and st_r2 == 200
        calls = sacct_calls.read_text(encoding="utf-8").splitlines()
        assert len(calls) == 1
        assert "9500,9501,9502,9503" in calls[0]
        assert "9504" not in calls[0] and "9505" not in calls[0]

        # Querying finished run status directly spawns ZERO additional `sacct` calls
        st_fin, _, raw_fin = _raw_http(port, "GET", "/api/v1/runs/batch-r4", headers=hdr)
        assert st_fin == 200
        assert json.loads(raw_fin)["result"]["state"] == "completed"
        assert len(sacct_calls.read_text(encoding="utf-8").splitlines()) == 1

        # `/summary` + `/plot.png` share the `(realpath, st_size, st_mtime_ns)` summary cache -> 1 NetCDF read total!
        st_s1, _, raw_s1 = _raw_http(port, "GET", "/api/v1/runs/batch-r4/summary", headers=hdr)
        assert st_s1 == 200
        sum_doc = json.loads(raw_s1)["result"]
        assert sum_doc["variables"]["TMP2m"]["units"] == ""
        assert len(sum_doc["variables"]["TMP2m"]["series"]) == 3
        assert abs(sum_doc["variables"]["TMP2m"]["series"][2]["ensemble_mean"] - 289.5) < 1e-4

        st_s2, _, _ = _raw_http(port, "GET", "/api/v1/runs/batch-r4/summary", headers=hdr)
        assert st_s2 == 200
        if ui_server._is_matplotlib_available():
            st_p, _, png_b = _raw_http(port, "GET", "/api/v1/runs/batch-r4/plot.png?var=TMP2m", headers=hdr)
            assert st_p == 200 and png_b.startswith(b"\x89PNG\r\n\x1a\n")
        assert ds_reads[0] == 1

        # Lowering `MAX_SUMMARY_VAR_BYTES` below `TMP2m` size (`1*3*4*8*4 = 384` B) rejects oversized variable up front
        monkeypatch.setattr(status_mod, "MAX_SUMMARY_VAR_BYTES", 128)
        with pytest.raises(ValidationError, match="exceeding diagnostic budget"):
            status_mod.summarize_run(str(diag_dir))


def test_ui_bounded_connection_pool_and_slow_header_or_body_timeouts(
    playground_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AIMIP_UI_REQUEST_TIMEOUT", "17.5")
    monkeypatch.setenv("AIMIP_UI_MAX_CONNECTIONS", "9")
    cfg_srv = create_ui_server(port=0)
    try:
        assert cfg_srv.request_timeout == 17.5
        assert cfg_srv.max_connections == 9
    finally:
        cfg_srv.server_close()
    monkeypatch.delenv("AIMIP_UI_REQUEST_TIMEOUT", raising=False)
    monkeypatch.delenv("AIMIP_UI_MAX_CONNECTIONS", raising=False)

    with running_ui_server(max_connections=2, request_timeout=0.4) as srv:
        port = int(srv.server_address[1])
        tok = srv.app.token

        # 1. Hold both connection slots open with incomplete HTTP headers -> 3rd connection gets immediate 503
        stalled: list[socket.socket] = []
        try:
            for _ in range(2):
                s = socket.create_connection(("127.0.0.1", port), timeout=10.0)
                s.sendall(f"GET /api/v1/session HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n".encode("ascii"))
                stalled.append(s)

            deadline = time.monotonic() + 5.0
            while srv.active_connections < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            assert srv.active_connections <= 2

            st_busy, _, raw_busy = _raw_http(
                port,
                "GET",
                "/api/v1/session",
                headers={"X-AIMIP-Token": tok},
                timeout=10.0,
            )
            assert st_busy == 503
            assert json.loads(raw_busy)["error"]["code"] == "busy"

            # Wait past `request_timeout` -> stalled header sockets are aborted and slots released
            release_deadline = time.monotonic() + 5.0
            while srv.active_connections > 0 and time.monotonic() < release_deadline:
                time.sleep(0.02)
            assert srv.active_connections == 0
        finally:
            for s in stalled:
                with contextlib.suppress(OSError):
                    s.close()

        # 2. Authenticated POST sending incomplete JSON body (`Content-Length: 256`, only 10 bytes sent) times out with 408
        partial = socket.create_connection(("127.0.0.1", port), timeout=10.0)
        try:
            partial.sendall(
                (
                    f"POST /api/v1/plan HTTP/1.1\r\n"
                    f"Host: 127.0.0.1:{port}\r\n"
                    f"X-AIMIP-Token: {tok}\r\n"
                    f"Content-Type: application/json\r\n"
                    f"Content-Length: 256\r\n\r\n"
                    f'{{"model":"'
                ).encode("ascii")
            )
            partial.settimeout(10.0)
            resp_bytes = b""
            while b"\r\n\r\n" not in resp_bytes:
                chunk = partial.recv(4096)
                if not chunk:
                    break
                resp_bytes += chunk
            assert b"408" in resp_bytes.splitlines()[0]
        finally:
            with contextlib.suppress(OSError):
                partial.close()

        # Wait briefly for the 408 handler thread's `finally` block to release its slot
        slot_deadline = time.monotonic() + 5.0
        while srv.active_connections > 0 and time.monotonic() < slot_deadline:
            time.sleep(0.01)

        # Server remains completely healthy for normal clients afterward
        st_ok, _, raw_ok = _raw_http(port, "GET", "/api/v1/session", headers={"X-AIMIP-Token": tok})
        assert st_ok == 200
        assert json.loads(raw_ok)["ok"] is True


def test_ui_container_mode_honesty_prepared_display_state_and_host_submit_command(
    playground_env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty_bin = tmp_path / "no_slurm_bin"
    empty_bin.mkdir()
    monkeypatch.setenv("PATH", str(empty_bin))
    monkeypatch.setenv("AIMIP_IMAGE_SHA", "abcdef0123456789")

    with running_ui_server() as srv:
        port = int(srv.server_address[1])
        hdr = {"X-AIMIP-Token": srv.app.token}

        st_sess, _, raw_sess = _raw_http(port, "GET", "/api/v1/session", headers=hdr)
        assert st_sess == 200
        sess = json.loads(raw_sess)["result"]
        assert sess["in_container"] is True
        assert sess["can_submit_directly"] is False

        st_plan, _, raw_plan = _raw_http(
            port,
            "POST",
            "/api/v1/plan",
            headers=hdr,
            body={"model": "ace2-era5", "start": "1979-01-01", "steps": 4, "name": "ui-cont-prep"},
        )
        assert st_plan == 200
        prop = json.loads(raw_plan)["result"]
        assert prop["will_prepare_only_in_container"] is True

        st_app, _, raw_app = _raw_http(
            port,
            "POST",
            "/api/v1/approve",
            headers=hdr,
            body={"digest_prefix": prop["approval_prefix"], "digest": prop["digest"]},
        )
        assert st_app == 200
        nonce = json.loads(raw_app)["result"]["authorization_nonce"]

        st_sub, _, raw_sub = _raw_http(port, "POST", "/api/v1/submit", headers=hdr, body={"nonce": nonce})
        assert st_sub == 200
        att = json.loads(raw_sub)["result"]
        assert att["state"] == "PREPARED_AWAITING_HOST_SUBMIT"
        assert f"--expect-digest={prop['digest']}" in att["host_command"]

        # Runs list and run detail both report `display_state == "prepared"` and `host_submit_command`
        st_runs, _, raw_runs = _raw_http(port, "GET", "/api/v1/runs", headers=hdr)
        assert st_runs == 200
        row = next(r for r in json.loads(raw_runs)["result"] if r["name"] == "ui-cont-prep")
        assert row["display_state"] == "prepared"
        assert row["awaiting_submit"] is True
        assert f"--expect-digest={prop['digest']}" in row["host_submit_command"]

        st_det, _, raw_det = _raw_http(port, "GET", "/api/v1/runs/ui-cont-prep", headers=hdr)
        assert st_det == 200
        det = json.loads(raw_det)["result"]
        assert det["display_state"] == "prepared"
        assert det["awaiting_submit"] is True
        assert det["host_submit_command"] == att["host_command"]
