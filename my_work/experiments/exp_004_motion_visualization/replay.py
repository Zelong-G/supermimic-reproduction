"""Experiment 004. Default check is CPU metadata-only; GPU work is explicit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict
from importlib.metadata import distribution
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CONFIG = HERE / "cases.json"
# Running a file under my_work must still resolve the repository's src packages.
sys.path.insert(0, str(ROOT / "src"))


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def repo_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def safe_result(value):
    """Preserve unavailable diagnostic values as null, never fabricate a number."""
    if isinstance(value, dict):
        return {k: safe_result(v) for k, v in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def readable_scene_archive(path, asset_dir=None):
    """Drop only empty anonymous nested defaults emitted by MjSpec.to_xml().

    These carry no settings but are rejected by the MJCF reader. Keep the source
    archive, all named defaults, all physical settings and all asset bytes intact.
    """
    if asset_dir is None:
        asset_dir = Path(
            distribution("mjlab").locate_file("mjlab/asset_zoo/robots/unitree_g1/xmls/assets")
        )
    buffer = io.BytesIO()
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(buffer, "w") as target:
        existing = set(source.namelist())
        required = set()
        for member in source.infolist():
            content = source.read(member)
            if member.filename.endswith(".xml"):
                tree = ET.fromstring(content)
                for asset in tree.findall("asset/*"):
                    filename = asset.get("file")
                    if filename:
                        required.add(str(Path(member.filename).parent / filename))
                removed = False
                for parent in tree.iter("default"):
                    for child in list(parent):
                        if (
                            child.tag == "default"
                            and not child.attrib
                            and len(child) == 0
                            and not (child.text or "").strip()
                        ):
                            parent.remove(child)
                            removed = True
                if removed:
                    content = ET.tostring(tree, encoding="utf-8")
            target.writestr(member, content)
        # Attached MjSpecs can lose their external mesh search path on serialization.
        # Resolve missing meshes from the same installed G1 assets used by the scene.
        for filename in sorted(required - existing):
            asset_path = Path(asset_dir) / Path(filename).name
            if not asset_path.is_file():
                raise FileNotFoundError(f"Missing recorded scene asset: {filename} ({asset_path})")
            target.writestr(filename, asset_path.read_bytes())
    buffer.seek(0)
    return buffer


def load_config():
    cfg = read_json(CONFIG)
    if cfg["fps"] != 50:
        raise ValueError("This experiment uses the community 50 Hz control contract")
    full_data = read_json(ROOT / cfg["full_results"])
    segment_data = read_json(ROOT / cfg["segment_results"])
    if full_data["checkpoint"] != segment_data["checkpoint"]:
        raise ValueError("Archived evaluations use different checkpoint metadata")
    if full_data["checkpoint"]["checkpoint_path"] != cfg["checkpoint"]:
        raise ValueError("Configured checkpoint differs from archived evaluations")
    full = {r["name"]: r for r in full_data["clips"]}
    segments = segment_data["clips"]
    candidates = read_json(ROOT / cfg["exp003_candidates"])
    if cfg["cases"][0]["motion"] not in {c["motion"]["name"] for c in candidates}:
        raise ValueError("Normal-success case is not an Experiment 003 replay candidate")
    for case in cfg["cases"]:
        motion = case["motion"]
        if not motion or "/" in motion or "\\" in motion:
            raise ValueError("Motion names must be plain NPZ stems")
        row = full[motion]
        for key, value in case["historical"].items():
            if row[key] != value:
                raise ValueError(f"{motion}: archived result differs from cases.json: {key}")
        parts = [r for r in segments if r["name"].startswith(motion + "__seg")]
        if (
            len(parts) != case["segments"]
            or sum(r["failed"] for r in parts) != case["segment_failures"]
        ):
            raise ValueError(f"{motion}: segment summary differs")
        expected_time = row["fail_step"] / 50 if row["failed"] else None
        if case["historical_fail_time_s"] != expected_time:
            raise ValueError(f"{motion}: historical fail time differs")
    return cfg


def preflight(cfg):
    # Only small JSON reads / file metadata. No torch, MuJoCo import, NPZ reads or hashing weights.
    required = [cfg["checkpoint"]] + [f"{cfg['npz_dir']}/{c['motion']}.npz" for c in cfg["cases"]]
    missing = [p for p in required if not (ROOT / p).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing inputs: {missing}")
    return {
        "status": "prepared_not_gpu_executed",
        "simulation_started": False,
        "checkpoint": cfg["checkpoint"],
        "checkpoint_loaded": False,
        "historical_command_parallelism": "not recorded; one-env replay can differ",
        "cases": cfg["cases"],
        "source_sha256": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in [cfg["full_results"], cfg["segment_results"], cfg["exp003_candidates"]]
        },
        "configuration_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "failure_time_comparison": "pending actual GPU execution",
    }


def create_case_output(cfg, mode, motion, run_id):
    # Viewer IDs are reusable; isolate each launch without overwriting old reports.
    if not run_id or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_id
    ):
        raise ValueError("run-id must contain only letters, digits, underscores or hyphens")
    path = ROOT / cfg["output_dir"] / mode / run_id / motion
    if mode == "viewer":
        path.mkdir(parents=True, exist_ok=True)
        index = 1
        while True:
            session = path / f"session_{index:03}"
            try:
                session.mkdir()
            except FileExistsError:
                index += 1
            else:
                print(f"Viewer output: {session}", flush=True)
                return session
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(
            f"Output already exists: {path}. "
            "Use a new --run-id to start another run and preserve previous results."
        ) from None
    return path


def record_case(cfg, case, args):
    import mujoco
    import numpy as np
    from metrics import compare_outcome, control_timing, root_errors

    from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg
    from supermimic.eval.benchmark import (
        Recording,
        env_kwargs,
        load_policy_from_checkpoint,
        run_benchmark,
    )

    out = create_case_output(cfg, "recordings", case["motion"], args.run_id)
    raw = out / "raw"
    raw.mkdir()
    policy, checkpoint = load_policy_from_checkpoint(ROOT / cfg["checkpoint"], args.device)
    files = [str(ROOT / cfg["npz_dir"] / f"{case['motion']}.npz")]
    kwargs = env_kwargs(checkpoint)
    environment_cfg = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=files, num_envs=1, **kwargs))
    timing = control_timing(
        environment_cfg.sim.mujoco.timestep, environment_cfg.decimation, cfg["fps"]
    )
    dt = timing["control_dt_s"]
    rec = Recording()
    result = run_benchmark(
        policy,
        files,
        device=args.device,
        num_envs=1,
        recording=rec,
        terminate=True,
        **kwargs,
    )[0]
    # Persist the authoritative result BEFORE any export / scene compilation can fail.
    write_json(
        raw / "result.json",
        {
            "motion": case["motion"],
            "result": safe_result(asdict(result)),
            "timing": timing,
            "checkpoint_metadata": safe_result(checkpoint.get("_meta", {})),
        },
    )
    q, ref = rec.qpos[case["motion"]], rec.ref_qpos[case["motion"]]
    # The raw MjSpec retains assets for the community Pane, without recreating scene/physics.
    mujoco.to_zip(rec.spec, str(raw / "scene.zip"))
    np.savez_compressed(raw / "trajectory.npz", qpos=q, reference_qpos=ref)
    model = rec.spec.compile()
    # Resolve the unique free joint from the compiled recording scene, rather than guessing layout.
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
        raise ValueError(f"Expected one free root joint, found {len(free)}")
    address = int(model.jnt_qposadr[free[0]])
    root, reference = q[:, address : address + 7], ref[:, address : address + 7]
    errors = root_errors(root, reference)
    fieldnames = (
        ["sample_index", "pre_action_time_s"]
        + [
            f"{prefix}_{k}"
            for prefix in ["sim", "ref"]
            for k in ["x", "y", "z", "qw", "qx", "qy", "qz"]
        ]
        + list(errors)
    )
    with (raw / "root_states.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for k in range(len(q)):
            writer.writerow([k, k * dt, *root[k], *reference[k], *[v[k] for v in errors.values()]])
    report = {
        "motion": case["motion"],
        "mode": "benchmark_recording_terminate_true",
        "result": safe_result(asdict(result)),
        "control_dt_s": dt,
        "timing": timing,
        "exported_scene_timestep_s": float(model.opt.timestep),
        "termination_time_s": result.fail_step * dt if result.failed else None,
        "comparison": compare_outcome(case["historical"], result.failed, result.fail_step, dt=dt),
        "sample_count": len(q),
        "root_qpos_address": address,
        "root_error_summary": {
            name: {"mean": float(values.mean()), "max": float(values.max())}
            for name, values in errors.items()
        },
        "coordinate_frame": "origin-free world, z-up; quaternion wxyz",
        "recording_phase": "pre-action; sample k aligns with reference frame k; terminal state omitted",
        "sample_errors_are_not_termination_checks": True,
        "notes": [
            "One environment, no noise/DR, start sampling; checkpoint env_kwargs preserved.",
            "Original benchmark parallelism is unknown; mismatches are retained.",
            "Do not infer failure from low root height alone or from geodesic orientation error.",
        ],
        "checkpoint_metadata": safe_result(checkpoint.get("_meta", {})),
    }
    write_json(out / "summary.json", report)
    print(json.dumps(report, indent=2))
    # Match the community rendering tool's cleanup between independent small replays.
    import gc

    import torch

    del policy, rec
    gc.collect()
    torch.cuda.empty_cache()


def live_viewer(cfg, case, args):
    from metrics import compare_outcome
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.eval.benchmark import env_kwargs, load_policy_from_checkpoint

    community_play = repo_module("exp004_community_play", "scripts/play.py")
    out = create_case_output(cfg, "viewer", case["motion"], args.run_id)
    policy, checkpoint = load_policy_from_checkpoint(ROOT / cfg["checkpoint"], args.device)
    env = make_env(
        SuperMimicEnvCfgArgs(
            npz_files=[str(ROOT / cfg["npz_dir"] / f"{case['motion']}.npz")],
            num_envs=1,
            episode_length_s=1e5,
            action_noise_std=0,
            obs_noise=False,
            reset_noise=False,
            domain_randomization=False,
            sampling_mode="start",
            record_transitions=False,
            seed=0,
            debug_vis=True,
            actuator_gains="spec",
            **env_kwargs(checkpoint),
        ),
        device=args.device,
    )
    wrapper = RslRlVecEnvWrapper(env)

    class ObservedEnv:
        def __init__(self):
            self.steps = 0
            self.episode = 0
            self.last_done = None
            self.overflow_steps = 0

        def __getattr__(self, key):
            return getattr(wrapper, key)

        def reset(self):
            self.steps = 0
            self.overflow_steps = 0
            self.last_done = None
            # Force exactly frame zero, matching run_benchmark's initialization.
            cmd = env.command_manager.get_term("motion")
            cmd.forced_frame[:] = cmd.library.clip_start[0]
            obs = wrapper.reset()
            cmd.forced_frame[:] = cmd.library.clip_start[0]
            return obs

        def step(self, actions):
            transition = wrapper.step(actions)
            self.steps += 1
            overflow = getattr(env.sim.data, "overflow", None)
            if overflow is not None:
                self.overflow_steps += int(bool((overflow[0] != 0).item()))
                overflow[:] = 0
            if bool(transition[2][0]):
                failed = bool(env.termination_manager.terminated[0])
                self.last_done = {
                    "motion": case["motion"],
                    "episode": self.episode,
                    "steps": self.steps,
                    "failed": failed,
                    "fail_step": self.steps if failed else -1,
                    "termination_time_s": self.steps * 0.02 if failed else None,
                    "overflow_steps": self.overflow_steps,
                    "comparison": compare_outcome(
                        case["historical"], failed, self.steps if failed else -1
                    ),
                    "terminal_pose_available": False,
                    "note": "Viewer auto-resets at done; visible pose after termination is reset state, not failure pose.",
                }
                write_json(out / f"episode_{self.episode:03}.json", self.last_done)
                write_json(
                    out / "session.json",
                    {
                        "status": "completed_episode",
                        "latest_episode_file": f"episode_{self.episode:03}.json",
                        "comparison": self.last_done["comparison"],
                    },
                )
                self.episode += 1
                self.steps = 0
                self.overflow_steps = 0
                print(json.dumps(self.last_done, indent=2))
            return transition

    observer = ObservedEnv()
    observer.reset()
    backend = ViserPlayViewer if args.viewer == "viser" else NativeMujocoViewer

    class CaseViewer(backend):
        def _execute_step(self):
            ok = super()._execute_step()
            if observer.last_done is not None or (
                args.pause_step and observer.steps == args.pause_step
            ):
                self.pause()
                self._sim_budget = 0.0
                observer.last_done = None
            return ok

    server = None
    write_json(
        out / "session.json",
        {
            "motion": case["motion"],
            "viewer": args.viewer,
            "status": "started_no_completed_episode",
            "historical": case["historical"],
            "comparison": compare_outcome(case["historical"], None, -1),
            "pause_step": args.pause_step,
            "num_envs": 1,
        },
    )
    try:
        if args.viewer == "viser":
            import viser

            server = viser.ViserServer(host="127.0.0.1", port=args.port, label="Experiment 004")
            viewer = CaseViewer(observer, community_play._DictPolicy(policy), viser_server=server)
        else:
            viewer = CaseViewer(observer, community_play._DictPolicy(policy))
        viewer.pause()
        viewer.run()
    finally:
        wrapper.close()
        if server is not None:
            server.stop()


def render_case(cfg, case, args):
    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    from metrics import video_window
    from PIL import Image, ImageDraw, ImageFont

    render = repo_module("exp004_community_render", "tools/render_rollouts.py")
    recorded = ROOT / cfg["output_dir"] / "recordings" / args.recording_id / case["motion"]
    summary = read_json(recorded / "summary.json")
    with np.load(recorded / "raw/trajectory.npz") as arrays:
        q = arrays["qpos"]
        reference = arrays["reference_qpos"]
    start, stop = video_window(
        len(q), summary["result"]["fail_step"], seconds=args.seconds, start=args.start
    )
    # Validate the saved scene before creating an output directory.
    with readable_scene_archive(recorded / "raw/scene.zip") as archive:
        spec = mujoco.from_zip(archive)
    out = create_case_output(cfg, "renders", case["motion"], args.run_id)
    videos = out / "videos"
    videos.mkdir()
    screenshots = out / "screenshots"
    screenshots.mkdir()
    pane = render.Pane(spec, "Policy + reference", 1280, 720, 3.6, -12)
    address = summary["root_qpos_address"]
    if address != 0:
        pane.close()
        raise ValueError("Community Pane requires the free root at qpos address zero")
    azimuth = float(np.degrees(render.yaw(reference[0, address + 3 : address + 7])) + 100)
    selected = {start, (start + stop - 1) // 2, stop - 1}
    font = ImageFont.load_default(size=20)
    writer = None
    try:
        if not args.screenshots_only:
            writer = imageio.get_writer(
                str(videos / "tracking.mp4"),
                fps=50,
                codec="libx264",
                pixelformat="yuv420p",
                quality=8,
                macro_block_size=8,
            )
        indices = range(start, stop) if writer else sorted(selected)
        for k in indices:
            img = Image.fromarray(pane.render(q[k], reference[k], azimuth))
            draw = ImageDraw.Draw(img)
            if args.presentation:
                draw.rectangle((0, 0, 1280, 56), fill=(245, 245, 245))
                draw.text(
                    (16, 8),
                    f"Motion tracking | {case['motion']} | {(k - start) / 50:.2f} s",
                    fill=(10, 10, 10), font=font,
                )
                draw.text(
                    (16, 32), "Solid: policy   |   Blue ghost: reference",
                    fill=(30, 30, 30), font=font,
                )
            else:
                draw_diagnostic_caption(draw, font, case, summary, k)
            if k in selected:
                img.save(screenshots / f"frame_{k:06}.png")
            if writer:
                writer.append_data(np.asarray(img))
    finally:
        pane.close()
        if writer:
            writer.close()
    write_json(
        out / "render_manifest.json",
        {
            "motion": case["motion"],
            "recording": str(recorded.relative_to(ROOT)),
            "start_sample": start,
            "stop_sample_exclusive": stop,
            "start_time_s": start / 50,
            "stop_time_exclusive_s": stop / 50,
            "width": 1280,
            "height": 720,
            "fps": 50,
            "mp4_written": not args.screenshots_only,
            "presentation": args.presentation,
            "comparison": summary["comparison"],
            "note": "Recorded MuJoCo states; no policy inference or dynamics rerun during rendering.",
        },
    )
    print(out)


def draw_diagnostic_caption(draw, font, case, summary, k):
    draw.rectangle((0, 0, 1280, 76), fill=(245, 245, 245))
    draw.text(
        (16, 10),
        f"{case['motion']} | t={k / 50:.2f}s | solid: policy; blue ghost: reference",
        fill=(10, 10, 10),
        font=font,
    )
    event = summary["termination_time_s"]
    label = (
        f"Observed termination: {event:.2f}s (terminal state not recorded)"
        if event is not None
        else "Observed: full sequence completed"
    )
    draw.text((16, 34), label, fill=(160, 45, 10), font=font)
    draw.text(
        (16, 54),
        "Pre-action samples; clipped video only. See summary.json for benchmark comparison.",
        fill=(30, 30, 30),
        font=font,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=["check", "viewer", "record", "render"], nargs="?", default="check"
    )
    parser.add_argument(
        "--case",
        default="walk1_subject1",
        help="Configured motion stem or all (record/render only)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--viewer", choices=["viser", "native"], default="viser")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--pause-step",
        type=int,
        default=0,
        help="Pause after this control step; no skipping simulation",
    )
    parser.add_argument("--run-id", default="run_001")
    parser.add_argument("--recording-id", default="run_001")
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument(
        "--start", type=float, default=None, help="Video start time within the recorded trajectory"
    )
    parser.add_argument("--screenshots-only", action="store_true")
    parser.add_argument(
        "--presentation", action="store_true",
        help="Render concise slide captions; retain benchmark comparison in the manifest",
    )
    args = parser.parse_args()
    if (
        args.pause_step < 0
        or not 1 <= args.port <= 65535
        or not math.isfinite(args.seconds)
        or args.seconds <= 0
    ):
        parser.error("Invalid pause step, port or video duration")
    cfg = load_config()
    check = preflight(cfg)
    if args.mode == "check":
        out = ROOT / cfg["output_dir"] / "preflight.json"
        write_json(out, check)
        print(json.dumps(check, indent=2))
        return
    chosen = [c for c in cfg["cases"] if args.case in [c["motion"], "all"]]
    if not chosen or (args.mode == "viewer" and args.case == "all"):
        parser.error("Select a configured motion; viewer supports one case at a time")
    # Set GL backend before any MuJoCo imports. Native viewer needs its window system.
    if args.mode in ["record", "render"] or args.viewer == "viser":
        os.environ.setdefault("MUJOCO_GL", "egl")
    action = {"viewer": live_viewer, "record": record_case, "render": render_case}[args.mode]
    for case in chosen:
        try:
            action(cfg, case, args)
        except FileExistsError as exc:
            parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
