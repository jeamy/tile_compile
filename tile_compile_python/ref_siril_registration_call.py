from pathlib import Path

def run_external_registration(run_dir: Path, input_lights: Path):
    # --------------------------------------------------
    # 1. Run-Verzeichnisse anlegen
    # --------------------------------------------------
    lights_run_dir = run_dir / "lights"
    lights_run_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 2. Input materialisieren (Hardlink-first)
    # --------------------------------------------------
    stats = materialize_lights(
        src_dir=input_lights,
        dst_dir=lights_run_dir,
    )

    logger.info("Lights materialized", extra=stats)

    # --------------------------------------------------
    # 3. Siril aufrufen (Debayer + Registrierung)
    # --------------------------------------------------
    run_siril_registration(
        lights_dir=lights_run_dir,
        siril_script=Path("/project/scripts/register_osc.ssf"),
        siril_binary="siril",
    )

    # --------------------------------------------------
    # 4. Registrierte Frames einsammeln
    # --------------------------------------------------
    registered = sorted(lights_run_dir.glob("r_*.fit*"))

    if not registered:
        raise RuntimeError("No registered frames produced by Siril")

    logger.info(
        "Siril registration completed",
        extra={"registered_frames": len(registered)}
    )

    return registered
