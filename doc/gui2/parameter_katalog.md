# GUI 2 Parameter-Katalog

Quelle: `tile_compile_cpp/tile_compile.yaml`, `tile_compile_cpp/tile_compile.schema.yaml` und GUI-Laufzeitparameter aus `gui_cpp`.

## 1) Konfigurationsparameter (YAML)

| Parameter | Typ | Default | Kurz-Erklaerung | Szenario-Hinweis | Zielbereich GUI 2 |
|---|---|---|---|---|---|
| `assumptions.exposure_time_tolerance_percent` | `float` | `8.0` | Steuert exposure time tolerance percent im Bereich assumptions. | Wenige Frames | Assumptions |
| `assumptions.frames_min` | `int` | `30` | Steuert die Frame-Anforderung in assumptions. | Wenige Frames | Assumptions |
| `assumptions.frames_optimal` | `int` | `300` | Steuert die Frame-Anforderung in assumptions. | Wenige Frames | Assumptions |
| `assumptions.frames_reduced_threshold` | `int` | `200` | Schwelle fuer Reduced-Mode statt Full-Mode. | Wenige Frames | Assumptions |
| `assumptions.pipeline_profile` | `string` | `strict` | Pipeline behavior profile: practical keeps compatibility defaults, strict enforces v3.3.6 normative constraints. | Wenige Frames | Assumptions |
| `assumptions.reduced_mode_cluster_range` | `list` | `[3, 10]` | Legt den Wertebereich fuer assumptions fest. | Wenige Frames | Assumptions |
| `assumptions.reduced_mode_skip_clustering` | `bool` | `true` | Wenn true, werden STATE_CLUSTERING und SYNTHETIC_FRAMES im Reduced-/Emergency-Mode uebersprungen. | Wenige Frames | Assumptions |
| `astrometry.astap_bin` | `string` | `` | Pfad zur ausfuehrbaren Datei fuer astrometry. | - | Astrometry |
| `astrometry.astap_data_dir` | `string` | `/media/data/Astro/astap` | Pfad zum Verzeichnis fuer astrometry. | - | Astrometry |
| `astrometry.enabled` | `bool` | `true` | Aktiviert oder deaktiviert den Bereich astrometry. | - | Astrometry |
| `astrometry.search_radius` | `int` | `183` | Steuert den Radius fuer astrometry. | - | Astrometry |
| `bge.autotune.alpha_flatness` | `float` | `0.3` | Steuert alpha flatness im Bereich bge > autotune. | Starker Gradient | BGE |
| `bge.autotune.beta_roughness` | `float` | `0.1` | Steuert beta roughness im Bereich bge > autotune. | Starker Gradient | BGE |
| `bge.autotune.enabled` | `bool` | `true` | Aktiviert oder deaktiviert den Bereich bge > autotune. | Starker Gradient | BGE |
| `bge.autotune.holdout_fraction` | `float` | `0.2` | Steuert holdout fraction im Bereich bge > autotune. | Starker Gradient | BGE |
| `bge.autotune.max_evals` | `int` | `24` | Hoechstwert fuer evals im Bereich bge > autotune. | Starker Gradient | BGE |
| `bge.autotune.strategy` | `string` | `conservative` | Steuert strategy im Bereich bge > autotune. | Starker Gradient | BGE |
| `bge.enabled` | `bool` | `true` | Aktiviert oder deaktiviert den Bereich bge. | Starker Gradient | BGE |
| `bge.fit.huber_delta` | `float` | `1.5` | Steuert huber delta im Bereich bge > fit. | Starker Gradient | BGE |
| `bge.fit.irls_max_iterations` | `int` | `10` | Steuert irls max iterations im Bereich bge > fit. | Starker Gradient | BGE |
| `bge.fit.irls_tolerance` | `string` | `1e-4` | Steuert irls tolerance im Bereich bge > fit. | Starker Gradient | BGE |
| `bge.fit.method` | `string` | `rbf` | Steuert method im Bereich bge > fit. | Starker Gradient | BGE |
| `bge.fit.polynomial_order` | `int` | `2` | Steuert polynomial order im Bereich bge > fit. | Starker Gradient | BGE |
| `bge.fit.rbf_epsilon` | `float` | `1.0` | Steuert rbf epsilon im Bereich bge > fit. | Starker Gradient | BGE |
| `bge.fit.rbf_lambda` | `string` | `1e-2` | Regularisierung des RBF-Fits gegen Ueberschwingen. | Starker Gradient | BGE |
| `bge.fit.rbf_mu_factor` | `float` | `1.5` | Steuert rbf mu factor im Bereich bge > fit. | Starker Gradient | BGE |
| `bge.fit.rbf_phi` | `string` | `multiquadric` | Steuert rbf phi im Bereich bge > fit. | Starker Gradient | BGE |
| `bge.fit.robust_loss` | `string` | `huber` | Steuert robust loss im Bereich bge > fit. | Starker Gradient | BGE |
| `bge.grid.G_max_fraction` | `float` | `0.25` | Steuert G max fraction im Bereich bge > grid. | Starker Gradient | BGE |
| `bge.grid.G_min_px` | `int` | `64` | Steuert G min px im Bereich bge > grid. | Starker Gradient | BGE |
| `bge.grid.N_g` | `int` | `32` | Steuert N g im Bereich bge > grid. | Starker Gradient | BGE |
| `bge.grid.insufficient_cell_strategy` | `string` | `radius_expand` | Steuert insufficient cell strategy im Bereich bge > grid. | Starker Gradient | BGE |
| `bge.mask.sat_dilate_px` | `int` | `6` | Steuert sat dilate px im Bereich bge > mask. | Starker Gradient | BGE |
| `bge.mask.star_dilate_px` | `int` | `6` | Steuert star dilate px im Bereich bge > mask. | Starker Gradient | BGE |
| `bge.min_tiles_per_cell` | `int` | `3` | Mindestwert fuer tiles per cell im Bereich bge. | Starker Gradient | BGE |
| `bge.min_valid_sample_fraction_for_apply` | `float` | `0.28` | Mindestwert fuer valid sample fraction for apply im Bereich bge. | Starker Gradient | BGE |
| `bge.min_valid_samples_for_apply` | `int` | `96` | Mindestwert fuer valid samples for apply im Bereich bge. | Starker Gradient | BGE |
| `bge.sample_quantile` | `float` | `0.15` | Quantil fuer robuste Hintergrund-Samples. | Starker Gradient | BGE |
| `bge.structure_thresh_percentile` | `float` | `0.8` | Steuert structure thresh percentile im Bereich bge. | Starker Gradient | BGE |
| `calibration.bias_dir` | `string` | `` | Pfad zum Verzeichnis fuer calibration. | - | Calibration |
| `calibration.bias_master` | `string` | `` | Steuert bias master im Bereich calibration. | - | Calibration |
| `calibration.bias_use_master` | `bool` | `false` | Steuert bias use master im Bereich calibration. | - | Calibration |
| `calibration.dark_auto_select` | `bool` | `true` | Steuert dark auto select im Bereich calibration. | - | Calibration |
| `calibration.dark_master` | `string` | `` | Steuert dark master im Bereich calibration. | - | Calibration |
| `calibration.dark_match_exposure_tolerance_percent` | `float` | `8.0` | Steuert dark match exposure tolerance percent im Bereich calibration. | - | Calibration |
| `calibration.dark_match_temp_tolerance_c` | `float` | `3.0` | Steuert dark match temp tolerance c im Bereich calibration. | - | Calibration |
| `calibration.dark_match_use_temp` | `bool` | `false` | Steuert dark match use temp im Bereich calibration. | - | Calibration |
| `calibration.dark_use_master` | `bool` | `false` | Steuert dark use master im Bereich calibration. | - | Calibration |
| `calibration.darks_dir` | `string` | `` | Pfad zum Verzeichnis fuer calibration. | - | Calibration |
| `calibration.flat_master` | `string` | `` | Steuert flat master im Bereich calibration. | - | Calibration |
| `calibration.flat_use_master` | `bool` | `false` | Steuert flat use master im Bereich calibration. | - | Calibration |
| `calibration.flats_dir` | `string` | `` | Pfad zum Verzeichnis fuer calibration. | - | Calibration |
| `calibration.pattern` | `string` | `*.fit;*.fits;*.fts;*.fit.fz;*.fits.fz;*.fts.fz` | Datei- oder Suchmuster fuer calibration. | - | Calibration |
| `calibration.use_bias` | `bool` | `false` | Schaltet die Nutzung von bias in calibration. | - | Calibration |
| `calibration.use_dark` | `bool` | `false` | Schaltet die Nutzung von dark in calibration. | - | Calibration |
| `calibration.use_flat` | `bool` | `false` | Schaltet die Nutzung von flat in calibration. | - | Calibration |
| `chroma_denoise.apply_stage` | `string` | `post_stack_linear` | Steuert apply stage im Bereich chroma denoise. | - | Chroma Denoise |
| `chroma_denoise.blend.amount` | `float` | `0.95` | Steuert amount im Bereich chroma denoise > blend. | - | Chroma Denoise |
| `chroma_denoise.blend.mode` | `string` | `chroma_only` | Waehlt den Modus fuer den Bereich chroma denoise > blend. | - | Chroma Denoise |
| `chroma_denoise.chroma_bilateral.enabled` | `bool` | `true` | Aktiviert oder deaktiviert den Bereich chroma denoise > chroma bilateral. | - | Chroma Denoise |
| `chroma_denoise.chroma_bilateral.sigma_range` | `float` | `0.065` | Legt den Wertebereich fuer chroma denoise > chroma bilateral fest. | - | Chroma Denoise |
| `chroma_denoise.chroma_bilateral.sigma_spatial` | `float` | `1.5` | Steuert die Sigma-Schwelle fuer chroma denoise > chroma bilateral. | - | Chroma Denoise |
| `chroma_denoise.chroma_wavelet.enabled` | `bool` | `true` | Aktiviert oder deaktiviert den Bereich chroma denoise > chroma wavelet. | - | Chroma Denoise |
| `chroma_denoise.chroma_wavelet.levels` | `int` | `3` | Steuert levels im Bereich chroma denoise > chroma wavelet. | - | Chroma Denoise |
| `chroma_denoise.chroma_wavelet.soft_k` | `float` | `1.0` | Steuert soft k im Bereich chroma denoise > chroma wavelet. | - | Chroma Denoise |
| `chroma_denoise.chroma_wavelet.threshold_scale` | `float` | `1.8` | Steuert threshold scale im Bereich chroma denoise > chroma wavelet. | - | Chroma Denoise |
| `chroma_denoise.color_space` | `string` | `ycbcr_linear` | Steuert color space im Bereich chroma denoise. | - | Chroma Denoise |
| `chroma_denoise.enabled` | `bool` | `true` | Aktiviert oder deaktiviert den Bereich chroma denoise. | - | Chroma Denoise |
| `chroma_denoise.luma_guard_strength` | `float` | `0.75` | Steuert luma guard strength im Bereich chroma denoise. | - | Chroma Denoise |
| `chroma_denoise.protect_luma` | `bool` | `true` | Steuert protect luma im Bereich chroma denoise. | - | Chroma Denoise |
| `chroma_denoise.star_protection.dilate_px` | `int` | `2` | Steuert dilate px im Bereich chroma denoise > star protection. | - | Chroma Denoise |
| `chroma_denoise.star_protection.enabled` | `bool` | `true` | Aktiviert oder deaktiviert den Bereich chroma denoise > star protection. | - | Chroma Denoise |
| `chroma_denoise.star_protection.threshold_sigma` | `float` | `2.5` | Steuert die Sigma-Schwelle fuer chroma denoise > star protection. | - | Chroma Denoise |
| `chroma_denoise.structure_protection.enabled` | `bool` | `true` | Aktiviert oder deaktiviert den Bereich chroma denoise > structure protection. | - | Chroma Denoise |
| `chroma_denoise.structure_protection.gradient_percentile` | `int` | `87` | Steuert gradient percentile im Bereich chroma denoise > structure protection. | - | Chroma Denoise |
| `data.bayer_pattern` | `string` | `` | Bayer-Pattern fuer OSC-Rohdaten. Erlaubte Werte: `RGGB`, `GBRG`, `GRBG`, `BGGR`, leer=auto-detect aus FITS-Header. Explizite Angabe hat Vorrang vor FITS-Header (BAYERPAT). Guardrail: Warnung wenn Wert von Header abweicht. UI-Selektor: RGGB / GBRG / GRBG / BGGR / auto (default). | Alt/Az, Starke Rotation | Input & Scan |
| `data.linear_required` | `bool` | `true` | **Deprecated** — non-lineare Frames werden im Runner nur noch gewarnt (warn_only), nicht entfernt. Dieses Feld wird in einer kuenftigen Version entfernt. GUI zeigt Deprecation-Badge und empfiehlt, den Wert nicht zu aendern. | - | Data (Deprecated) |
| `debayer` | `bool` | `true` | Aktiviert CFA-Demosaikierung (Debayering) fuer OSC-Rohdaten. Bei MONO-Daten ohne Bayer-Pattern deaktivieren. | - | Debayer |
| `dithering.enabled` | `bool` | `true` | Aktiviert oder deaktiviert den Bereich dithering. | Alt/Az, Starke Rotation | Dithering |
| `dithering.min_shift_px` | `float` | `0.7` | Mindestwert fuer shift px im Bereich dithering. | Alt/Az, Starke Rotation | Dithering |
| `global_metrics.adaptive_weights` | `bool` | `true` | Gewichtung fuer den Teilbereich global metrics. | - | Global Metrics |
| `global_metrics.clamp` | `list` | `[-3.0, 3.0]` | Steuert clamp im Bereich global metrics. | - | Global Metrics |
| `global_metrics.weight_exponent_scale` | `float` | `1.2` | Exponent scale k for G_f = exp(k * Q_f). k=1.0 (default) is standard, k>1 increases differentiation between good/bad frames. E.g. k=1.5 g... | - | Global Metrics |
| `global_metrics.weights.background` | `float` | `0.4` | Steuert background im Bereich global metrics > weights. | - | Global Metrics |
| `global_metrics.weights.gradient` | `float` | `0.25` | Steuert gradient im Bereich global metrics > weights. | - | Global Metrics |
| `global_metrics.weights.noise` | `float` | `0.35` | Steuert noise im Bereich global metrics > weights. | - | Global Metrics |
| `input.max_frames` | `int` | `0` | Hoechstwert fuer frames im Bereich input. | - | Input & Scan |
| `input.pattern` | `string` | `./data/*.fits` | Datei- oder Suchmuster fuer input. | - | Input & Scan |
| `input.sort` | `string` | `numeric` | Steuert sort im Bereich input. | - | Input & Scan |
| `linearity.enabled` | `bool` | `true` | Aktiviert oder deaktiviert den Bereich linearity. | - | Linearity |
| `linearity.max_frames` | `int` | `8` | Hoechstwert fuer frames im Bereich linearity. | - | Linearity |
| `linearity.min_overall_linearity` | `float` | `0.9` | Mindestwert fuer overall linearity im Bereich linearity. | - | Linearity |
| `linearity.strictness` | `string` | `moderate` | Steuert strictness im Bereich linearity. | - | Linearity |
| `local_metrics.clamp` | `list` | `[-3.0, 3.0]` | Steuert clamp im Bereich local metrics. | - | Local Metrics |
| `local_metrics.star_mode.weights.contrast` | `float` | `0.2` | Steuert contrast im Bereich local metrics > star mode > weights. | - | Local Metrics |
| `local_metrics.star_mode.weights.fwhm` | `float` | `0.6` | Steuert fwhm im Bereich local metrics > star mode > weights. | - | Local Metrics |
| `local_metrics.star_mode.weights.roundness` | `float` | `0.2` | Steuert roundness im Bereich local metrics > star mode > weights. | - | Local Metrics |
| `local_metrics.structure_mode.background_weight` | `float` | `0.3` | Gewichtung fuer den Teilbereich local metrics > structure mode. | - | Local Metrics |
| `local_metrics.structure_mode.metric_weight` | `float` | `0.7` | Gewichtung fuer den Teilbereich local metrics > structure mode. | - | Local Metrics |
| `log_level` | `string` | `info` | Steuert log level im Bereich global. | - | Run Setup |
| `normalization.enabled` | `bool` | `true` | Aktiviert oder deaktiviert den Bereich normalization. | - | Normalization |
| `normalization.mode` | `string` | `background` | Waehlt den Modus fuer den Bereich normalization. | - | Normalization |
| `normalization.per_channel` | `bool` | `true` | Steuert per channel im Bereich normalization. | - | Normalization |
| `output.crop_to_nonzero_bbox` | `bool` | `true` | Steuert crop to nonzero bbox im Bereich output. | - | Output |
| `output.registered_dir` | `string` | `registered` | Pfad zum Verzeichnis fuer output. | - | Output |
| `output.write_registered_frames` | `bool` | `true` | Steuert die Frame-Anforderung in output. | - | Output |
| `pcc.annulus_inner_fwhm_mult` | `float` | `3.0` | Steuert annulus inner fwhm mult im Bereich pcc. | Helle Sterne | PCC |
| `pcc.annulus_inner_px` | `int` | `10` | Steuert annulus inner px im Bereich pcc. | Helle Sterne | PCC |
| `pcc.annulus_outer_fwhm_mult` | `float` | `5.0` | Steuert annulus outer fwhm mult im Bereich pcc. | Helle Sterne | PCC |
| `pcc.annulus_outer_px` | `int` | `16` | Steuert annulus outer px im Bereich pcc. | Helle Sterne | PCC |
| `pcc.aperture_fwhm_mult` | `float` | `1.8` | Steuert aperture fwhm mult im Bereich pcc. | Helle Sterne | PCC |
| `pcc.aperture_radius_px` | `int` | `8` | Steuert den Radius fuer pcc. | Helle Sterne | PCC |
| `pcc.apply_attenuation` | `bool` | `false` | Steuert apply attenuation im Bereich pcc. | Helle Sterne | PCC |
| `pcc.background_model` | `string` | `plane` | Steuert background model im Bereich pcc. | Helle Sterne | PCC |
| `pcc.chroma_strength` | `float` | `0.85` | Steuert chroma strength im Bereich pcc. | Helle Sterne | PCC |
| `pcc.enabled` | `bool` | `true` | Aktiviert oder deaktiviert den Bereich pcc. | Helle Sterne | PCC |
| `pcc.k_max` | `float` | `2.4` | Begrenzt die maximale lineare PCC-Gain-Verstaerkung. | Helle Sterne | PCC |
| `pcc.mag_bright_limit` | `int` | `6` | Obergrenze fuer sehr helle Sterne im PCC-Fit. | Helle Sterne | PCC |
| `pcc.mag_limit` | `int` | `14` | Grenzwert fuer pcc. | Helle Sterne | PCC |
| `pcc.max_condition_number` | `float` | `3.0` | Hoechstwert fuer condition number im Bereich pcc. | Helle Sterne | PCC |
| `pcc.max_residual_rms` | `float` | `0.7` | Hoechstwert fuer residual rms im Bereich pcc. | Helle Sterne | PCC |
| `pcc.min_aperture_px` | `float` | `4.0` | Mindestwert fuer aperture px im Bereich pcc. | Helle Sterne | PCC |
| `pcc.min_stars` | `int` | `10` | Mindestwert fuer stars im Bereich pcc. | Helle Sterne | PCC |
| `pcc.radii_mode` | `string` | `auto_fwhm` | Waehlt den Modus fuer den Bereich pcc. | Helle Sterne | PCC |
| `pcc.sigma_clip` | `float` | `2.5` | Steuert die Sigma-Schwelle fuer pcc. | Helle Sterne | PCC |
| `pcc.siril_catalog_dir` | `string` | `` | Pfad zum Verzeichnis fuer pcc. | Helle Sterne | PCC |
| `pcc.source` | `string` | `siril` | Steuert source im Bereich pcc. | Helle Sterne | PCC |
| `pipeline.abort_on_fail` | `bool` | `false` | Steuert abort on fail im Bereich pipeline. | - | Pipeline |
| `pipeline.mode` | `string` | `production` | Waehlt den Modus fuer den Bereich pipeline. | - | Pipeline |
| `registration.allow_rotation` | `bool` | `true` | Erlaubt Rotationsanteile im Registrierungsmodell. | Alt/Az, Starke Rotation | Registration |
| `registration.enable_star_pair_fallback` | `bool` | `false` | Enable additional star-pair fallback between primary triangle and normative fallback cascade. strict profile disables this. | Alt/Az, Starke Rotation | Registration |
| `registration.engine` | `string` | `triangle_star_matching` | Waehlt die Hauptmethode fuer die Bildregistrierung. | Alt/Az, Starke Rotation | Registration |
| `registration.reject_cc_mad_multiplier` | `float` | `4.0` | Steuert reject cc mad multiplier im Bereich registration. | Alt/Az, Starke Rotation | Registration |
| `registration.reject_cc_min_abs` | `float` | `0.3` | Steuert reject cc min abs im Bereich registration. | Alt/Az, Starke Rotation | Registration |
| `registration.reject_outliers` | `bool` | `true` | Steuert reject outliers im Bereich registration. | Alt/Az, Starke Rotation | Registration |
| `registration.reject_scale_max` | `float` | `1.08` | Steuert reject scale max im Bereich registration. | Alt/Az, Starke Rotation | Registration |
| `registration.reject_scale_min` | `float` | `0.92` | Steuert reject scale min im Bereich registration. | Alt/Az, Starke Rotation | Registration |
| `registration.reject_shift_median_multiplier` | `float` | `5.0` | Steuert reject shift median multiplier im Bereich registration. | Alt/Az, Starke Rotation | Registration |
| `registration.reject_shift_px_min` | `float` | `100.0` | Untergrenze fuer Shift-Outlier-Regel in Pixeln. | Alt/Az, Starke Rotation | Registration |
| `registration.star_dist_bin_px` | `float` | `5.0` | Steuert star dist bin px im Bereich registration. | Alt/Az, Starke Rotation | Registration |
| `registration.star_inlier_tol_px` | `float` | `4.0` | Steuert star inlier tol px im Bereich registration. | Alt/Az, Starke Rotation | Registration |
| `registration.star_min_inliers` | `int` | `4` | Steuert star min inliers im Bereich registration. | Alt/Az, Starke Rotation | Registration |
| `registration.star_topk` | `int` | `150` | Anzahl der zu pruefenden Top-Sterne fuer Matching. | Alt/Az, Starke Rotation | Registration |
| `run_dir` | `string` | `./run_smart_telescope` | Pfad zum Verzeichnis fuer global. | - | Run Setup |
| `runtime_limits.allow_emergency_mode` | `bool` | `true` | Erlaubt Emergency-Mode fuer Datensaetze mit <50 nutzbaren Frames; sonst wird der Lauf kontrolliert abgebrochen. | Langlauf/Batch | Runtime Limits |
| `runtime_limits.hard_abort_hours` | `float` | `6.0` | Steuert hard abort hours im Bereich runtime limits. | Langlauf/Batch | Runtime Limits |
| `runtime_limits.memory_budget` | `int` | `2048` | Speicherbudget in MiB fuer OSC-Worker-Cap bei Tile-Rekonstruktion. | Langlauf/Batch | Runtime Limits |
| `runtime_limits.parallel_workers` | `int` | `8` | Anzahl paralleler Worker fuer Tile-Phasen; wird zusaetzlich durch CPU-Kerne und OSC-Speicherbudget begrenzt. | Langlauf/Batch | Runtime Limits |
| `runtime_limits.tile_analysis_max_factor_vs_stack` | `float` | `3.0` | Steuert tile analysis max factor vs stack im Bereich runtime limits. | Langlauf/Batch | Runtime Limits |
| `stacking.cluster_quality_weighting.cap_enabled` | `bool` | `true` | Aktiviert optionales Dominanz-Cap fuer Clustergewichte. | - | Stacking |
| `stacking.cluster_quality_weighting.cap_ratio` | `float` | `20.0` | Nur mit cap_enabled=true: begrenzt Gewicht auf w_k <= cap_ratio * median_j(w_j). | - | Stacking |
| `stacking.cluster_quality_weighting.enabled` | `bool` | `false` | Aktiviert v3.2.2 Cluster-Qualitaetsgewichtung im finalen Stack: w_k = exp(kappa_cluster * Q_k). | - | Stacking |
| `stacking.cluster_quality_weighting.kappa_cluster` | `float` | `1.0` | Exponentfaktor fuer Cluster-Qualitaet Q_k (Q_k in [-3,+3]). | - | Stacking |
| `stacking.cosmetic_correction` | `bool` | `true` | Optionale Hotpixel-/Artefakt-Kosmetik nach dem Stacken (Post-Processing, standardmaessig deaktiviert). | Helle Sterne | Stacking |
| `stacking.cosmetic_correction_sigma` | `float` | `4.0` | Schwellwert in Vielfachen der MAD-Sigma fuer die Hotpixel-Erkennung in cosmetic_correction. Niedrigerer Wert = aggressiver. Default: 5.0,... | Helle Sterne | Stacking |
| `stacking.method` | `string` | `rej` | Steuert method im Bereich stacking. | - | Stacking |
| `stacking.output_stretch` | `bool` | `false` | Optionaler linearer Anzeige-Stretch auf den Ausgabe-Daten (Post-Processing, nicht Teil des linearen Pflichtkerns). | - | Stacking |
| `stacking.per_frame_cosmetic_correction` | `bool` | `true` | Hotpixel-Korrektur pro Frame VOR dem Stacken. Entfernt fixe Sensordefekte (immer gleiche Position), die Sigma-Clipping nicht erkennt. Emp... | - | Stacking |
| `stacking.per_frame_cosmetic_correction_sigma` | `float` | `2.5` | Sigma-Schwelle fuer per_frame_cosmetic_correction. Niedrigerer Wert = aggressiver. Default: 5.0. Fuer Seestar/OSC empfohlen: 5.0. | - | Stacking |
| `stacking.sigma_clip.max_iters` | `int` | `4` | Hoechstwert fuer iters im Bereich stacking > sigma clip. | Helle Sterne | Stacking |
| `stacking.sigma_clip.min_fraction` | `float` | `0.4` | Mindestwert fuer fraction im Bereich stacking > sigma clip. | Helle Sterne | Stacking |
| `stacking.sigma_clip.sigma_high` | `float` | `1.8` | Steuert die Sigma-Schwelle fuer stacking > sigma clip. | Helle Sterne | Stacking |
| `stacking.sigma_clip.sigma_low` | `float` | `1.8` | Steuert die Sigma-Schwelle fuer stacking > sigma clip. | Helle Sterne | Stacking |
| `synthetic.clustering.cluster_count_range` | `list` | `[3, 12]` | Min/Max-Anzahl fuer Synthetic-Cluster. | Wenige Frames | Synthetic |
| `synthetic.clustering.mode` | `string` | `kmeans` | Waehlt den Modus fuer den Bereich synthetic > clustering. | Wenige Frames | Synthetic |
| `synthetic.frames_max` | `int` | `20` | Steuert die Frame-Anforderung in synthetic. | Wenige Frames | Synthetic |
| `synthetic.frames_min` | `int` | `4` | Steuert die Frame-Anforderung in synthetic. | Wenige Frames | Synthetic |
| `synthetic.weighting` | `string` | `tile_weighted` | global = nur G_f; tile_weighted = W_f,t = G_f*L_f,t mit tileweiser OLA-Rekonstruktion. | Wenige Frames | Synthetic |
| `tile.max_divisor` | `int` | `6` | Hoechstwert fuer divisor im Bereich tile. | - | Tile |
| `tile.min_size` | `int` | `48` | Mindestwert fuer size im Bereich tile. | - | Tile |
| `tile.overlap_fraction` | `float` | `0.3` | Steuert overlap fraction im Bereich tile. | - | Tile |
| `tile.size_factor` | `int` | `24` | Steuert size factor im Bereich tile. | - | Tile |
| `tile.star_min_count` | `int` | `5` | Steuert star min count im Bereich tile. | - | Tile |
| `tile_denoise.soft_threshold.alpha` | `float` | `1.6` | Steuert alpha im Bereich tile denoise > soft threshold. | - | Tile Denoise |
| `tile_denoise.soft_threshold.blur_kernel` | `int` | `21` | Steuert blur kernel im Bereich tile denoise > soft threshold. | - | Tile Denoise |
| `tile_denoise.soft_threshold.enabled` | `bool` | `true` | Aktiviert oder deaktiviert den Bereich tile denoise > soft threshold. | - | Tile Denoise |
| `tile_denoise.soft_threshold.skip_star_tiles` | `bool` | `true` | Steuert skip star tiles im Bereich tile denoise > soft threshold. | - | Tile Denoise |
| `tile_denoise.wiener.enabled` | `bool` | `true` | Aktiviert oder deaktiviert den Bereich tile denoise > wiener. | - | Tile Denoise |
| `tile_denoise.wiener.max_iterations` | `int` | `10` | Hoechstwert fuer iterations im Bereich tile denoise > wiener. | - | Tile Denoise |
| `tile_denoise.wiener.min_snr` | `float` | `2.0` | Mindestwert fuer snr im Bereich tile denoise > wiener. | - | Tile Denoise |
| `tile_denoise.wiener.q_max` | `float` | `1.0` | Steuert q max im Bereich tile denoise > wiener. | - | Tile Denoise |
| `tile_denoise.wiener.q_min` | `float` | `-0.5` | Steuert q min im Bereich tile denoise > wiener. | - | Tile Denoise |
| `tile_denoise.wiener.q_step` | `float` | `0.1` | Steuert q step im Bereich tile denoise > wiener. | - | Tile Denoise |
| `tile_denoise.wiener.snr_threshold` | `float` | `4.0` | Steuert snr threshold im Bereich tile denoise > wiener. | - | Tile Denoise |
| `validation.max_background_rms_increase_percent` | `float` | `2.0` | Hoechstwert fuer background rms increase percent im Bereich validation. | - | Validation |
| `validation.min_fwhm_improvement_percent` | `float` | `3.0` | Mindestwert fuer fwhm improvement percent im Bereich validation. | - | Validation |
| `validation.min_tile_weight_variance` | `float` | `0.05` | Mindestwert fuer tile weight variance im Bereich validation. | - | Validation |
| `validation.require_no_tile_pattern` | `bool` | `true` | Datei- oder Suchmuster fuer validation. | - | Validation |

## 2) GUI-Laufzeitparameter (nicht im YAML)

| Parameter | Typ | GUI-Bereich | Kurz-Erklaerung |
|---|---|---|---|
| `scan.input_dirs[]` | `string[]` | Input & Scan | Mehrere Eingabeordner in Reihenfolge |
| `scan.frames_min` | `int` | Input & Scan | Mindestanzahl Frames fuer den Scan |
| `scan.with_checksums` | `bool` | Input & Scan | Optionaler Checksummen-Scan |
| `scan.confirmed_color_mode` | `enum` | Input & Scan | Manuelle Farbmodus-Bestaetigung |
| `run.working_dir` | `string` | Run | Basisarbeitsverzeichnis |
| `run.runs_dir` | `string` | Run | Ausgabeverzeichnis fuer Runs |
| `run.run_name` | `string` | Run | Frei definierbarer Basisname fuer den Run-Ordner |
| `run.output_dir_name` | `string` | Run | Effektiver Ordnername im Format `<run_name>_<YYYYMMDD_HHMMSS>` |
| `run.output_dir_preview` | `string` | Run | Vorschau auf finalen Zielpfad vor Start |
| `run.start_timestamp` | `datetime` | Run | Tatsaechlicher Startzeitpunkt, der fuer das Suffix genutzt wird |
| `run.config_revisions[]` | `object[]` | Run/Resume | Historie unveraenderlicher Config-Revisionen (append-only) |
| `run.resume.config_revision` | `string` | Run Monitor | Fuer Resume gewaehlte Config-Revision |
| `run.input_subdirs[]` | `string[]` | Run | Subfolder je Input-Ordner |
| `run.dry_run` | `bool` | Run | Simulation ohne echte Verarbeitung |
| `run.pattern` | `string` | Run | Dateimuster fuer Inputdateien |
| `run.filter_queue[]` | `object[]` | Run | Serielle MONO-Filter-Queue (LRGB/SHO) |
| `run.filter_queue[].filter_name` | `string` | Run | Filterbezeichner, z. B. L/R/G/B/Ha/OIII/SII |
| `run.filter_queue[].input_dir` | `string` | Run | Input-Ordner fuer den Filtereintrag |
| `run.filter_queue[].run_label` | `string` | Run | Optionales Label/Subfolder je Filtereintrag |
| `run.filter_queue[].pattern` | `string` | Run | Optionales Pattern je Filtereintrag |
| `run.filter_queue[].enabled` | `bool` | Run | Eintrag aktiv/inaktiv |
| `run.filter_queue[].status` | `enum` | Run Monitor | Queue-Status: pending/running/ok/error/skipped |
| `run.active_filter_index` | `int` | Run Monitor | Aktueller Filterindex in der seriellen Queue |
| `current_run.logs_tail` | `int` | Current Run | Anzahl Event-Zeilen |
| `current_run.logs_filter` | `string` | Current Run | Filter fuer Event-Logs |
| `history.selected_run` | `string` | History | Aktuell gewaehlter Lauf |
| `astrometry.catalog_selection` | `enum` | Astrometry | D05/D20/D50/D80 Auswahl |
| `astrometry.solve_file` | `string` | Astrometry | Plate-Solve Eingabedatei |
| `pcc.quick.fits_path` | `string` | PCC | RGB-FITS fuer Schnelltest |
| `pcc.quick.wcs_path` | `string` | PCC | WCS-Datei fuer Schnelltest |
| `ui.locale` | `enum` | Global UI | Aktive Sprache der GUI (`de`/`en`) |
| `ui.active_scenarios[]` | `string[]` | Parameter Studio | Aktive Situation-Assistent-Presets |

## 3) Situation Assistant (Pflicht in Linie B)

- Alt/Az
- Starke Rotation
- Helle Sterne im Feld
- Wenige Frames / kurze Session
- Starker Hintergrundgradient
- Details siehe `szenario_empfehlungen.md`.

## 4) Eingabeprinzip in GUI 2

- Jeder Parameter ist sowohl per Formular als auch als YAML sichtbar.
- Suchleiste filtert ueber alle Parameterpfade inkl. verschachtelter Keys.
- Guardrails markieren Werte ausserhalb empfohlener Bereiche sofort inline.
- Jeder Parameter hat eine Kurz-Erklaerung im Explain-Panel.
- Presets koennen auf Teilbereiche angewendet werden (z. B. nur `bge.*` oder `pcc.*`).
