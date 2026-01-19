import subprocess
from pathlib import Path

class SirilError(RuntimeError):
    pass


def run_siril_registration(
    lights_dir: Path,
    siril_script: Path,
    *,
    siril_binary: str = "siril",
    quiet: bool = True,
    timeout: int | None = None,
) -> None:
    """
    Führt Siril (Debayer + Registrierung) auf einem Lights-Verzeichnis aus.

    Voraussetzungen:
      - lights_dir enthält nur FITS-Lights (ggf. hardlinked)
      - siril_script ist ein geprüftes, policy-konformes Script
      - Siril erzeugt r_*.fit als registrierte Outputs
    """

    if not lights_dir.is_dir():
        raise SirilError(f"Lights dir not found: {lights_dir}")

    if not siril_script.is_file():
        raise SirilError(f"Siril script not found: {siril_script}")

    cmd = [
        siril_binary,
        "-d", str(lights_dir),          # Arbeitsverzeichnis = Lights
        "-s", str(siril_script),        # Script-Datei
    ]

    if quiet:
        cmd.append("-q")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as e:
        raise SirilError(
            f"Siril failed with exit code {e.returncode}\n"
            f"stdout:\n{e.stdout}\n"
            f"stderr:\n{e.stderr}"
        ) from e

    except subprocess.TimeoutExpired as e:
        raise SirilError("Siril execution timed out") from e
