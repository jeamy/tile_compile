# Beispielkonfigurationen für Methodik v4

# -----------------------------------------------------------------------------
# 1. EQ-Montierung, ruhiges Seeing
# -----------------------------------------------------------------------------

v4:
  iterations: 2
  beta: 3.0
  adaptive_tiles:
    enabled: false

registration:
  mode: local_tiles

# -----------------------------------------------------------------------------
# 2. Alt/Az, starke Feldrotation
# -----------------------------------------------------------------------------

v4:
  iterations: 4
  beta: 6.0
  adaptive_tiles:
    enabled: true
    max_refine_passes: 3
    refine_variance_threshold: 0.15

registration:
  mode: local_tiles

# -----------------------------------------------------------------------------
# 3. Polnähe, sehr instabil
# -----------------------------------------------------------------------------

v4:
  iterations: 5
  beta: 8.0
  adaptive_tiles:
    enabled: true
    max_refine_passes: 4
    refine_variance_threshold: 0.1

registration:
  mode: local_tiles

