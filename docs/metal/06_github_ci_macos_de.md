# 06 - GitHub-Build für M-Chips (Release-Workflow)

Betrifft `.github/workflows/release-tile-compile-gui3.yml` (Job
`package-macos`) und `packaging/gui3/build_local_macos.sh`. Anforderung:
Der Release-Build für Apple Silicon muss mit dem Metal-Backend grün bleiben
und ein lauffähiges, eigenständiges Bundle liefern; der Intel-Build und die
Linux-/Windows-Jobs dürfen sich nicht ändern.

## 1. Ist-Stand (im Workflow und Skript gelesen, 2026-09-26)

| Punkt | Ist |
| --- | --- |
| Matrix | `macos-14` (Apple Silicon, Deployment-Target 14.0, Artefakt `macos-apple`) und `macos-15-intel` (Target 13.0, Artefakt `macos-intel`) |
| Abhängigkeiten | `brew install cmake ninja pkg-config eigen opencv cfitsio yaml-cpp nlohmann-json openssl curl zip gcc` |
| Build | `build_local_macos.sh`: CMake+Ninja, Runner-Baum mit `-DBUILD_TESTS=OFF`, danach `web_backend_cpp` |
| Bundle | einzelne Dateien werden kopiert (`tile_compile_runner`, `tile_compile_cli`, Backend); Homebrew-dylibs werden per `otool`/`install_name_tool` nach `tile_compile_cpp/lib` gebündelt und auf `@loader_path` umgeschrieben; Abhängigkeiten unter `/usr/lib/*` und `/System/*` werden bewusst nicht gebündelt |
| Smoke-Test | `otool -L`, Backend-Startcheck, `start_gui3.sh`-Test; kein Runner-Lauf |
| Veröffentlichung | Job `publish` benötigt `package-macos` |

Folgen für Metal:

- `/System/Library/Frameworks/Metal.framework` fällt unter `/System/*`: kein
  Bündelungsproblem, aber `verify_macos_bundle_refs` darf keine
  Metal-Frameworkreferenz beanstanden (prüfen).
- **Kritisch:** das Skript kopiert nur Einzeldateien nach
  `tile_compile_cpp/build/`. Eine separate `default.metallib` würde
  **nicht** ins Bundle gelangen; das Backend fiele zur Laufzeit auf CPU
  zurück, der Build bliebe aber grün (stiller Funktionsverlust).
- Die Release-Builds laufen mit `BUILD_TESTS=OFF`; `tests_metal` läuft dort
  nicht.

## 2. Entscheidungen

1. **Shader eingebettet, nicht als Beilagedatei.** Die `metallib` (bzw.
   der Shader-Quelltext) wird zur Buildzeit in die Runner-/CLI-Binärdatei
   eingebettet (generiertes C++-Byte-Array oder Linker-Sektion). Dann
   funktionieren die bestehenden Kopierschritte unverändert, es gibt keinen
   Pfadbezug zur Laufzeit und keine neue Bündelungslogik.
2. **Zweistufige Shader-Übersetzung mit sicherem Rückfall:**
   - bevorzugt Offline-Kompilierung (`xcrun -sdk macosx metal` +
     `metallib`) zur Buildzeit;
   - fehlt der Metal-Compiler (neuere Xcode-Versionen liefern die
     Metal-Toolchain als separate Komponente), bettet CMake stattdessen
     den Shader-Quelltext ein, der zur Laufzeit per
     `newLibraryWithSource` mit gleichen Optionen übersetzt wird. Der
     Build schlägt dadurch nie an der fehlenden Toolchain fehl.
   - CMake protokolliert, welcher Modus aktiv ist (`build_info`:
     `metal=offline|runtime|off`).
3. **Metal nur für arm64.** `TILE_COMPILE_ENABLE_METAL` ist per Default nur
   bei `APPLE` und Zielarchitektur `arm64` ON. Der Intel-Job (Target 13.0)
   baut ohne Metal-Teilbaum und ist unverändert; auch ein manuell
   gesetztes ON auf x86_64 wird abgelehnt (klare CMake-Meldung), da der
   Plan Apple Silicon adressiert.
4. **Deployment-Target beachten.** Apple-Silicon-Job bleibt 14.0. Alle
   verwendeten MSL-/Metal-APIs müssen ab macOS 14 verfügbar sein (bei
   Nutzung neuerer APIs `@available`/Laufzeitprüfung oder Feature-Gate;
   Float-Atomics und MSL-Version je Familie zu verifizieren). Ein
   Metal-Symbol jenseits des Targets wäre ein Linkfehler oder ein
   Laufzeitabsturz auf älteren Systemen.
5. **Kein GPU im Runner vorausgesetzt.** Ob `macos-14` ein nutzbares
   Metal-Device bereitstellt, ist zu verifizieren (virtualisierte
   Runner). Deshalb:
   - der Build braucht kein Device (nur Compiler/SDK);
   - die Laufzeit-Probe liefert bei fehlendem Device sauber "nicht
     verfügbar" -> CPU, kein Absturz, kein Fehlercode;
   - Hardwaretests sind bei fehlendem Device `SKIPPED (no Metal device)`.
6. **Kein Backend-/Run-Start.** Der Smoke-Test bleibt beim Backend-Startcheck
   des bestehenden Skripts (CI-Umgebung, kein Nutzerlauf). Neue Prüfungen
   sind reine Statusabfragen ohne Bildverarbeitung.

## 3. Änderungen am Workflow und Skript

`build_local_macos.sh`:

- `build_all`: Runner-Konfiguration erhält bei arm64
  `-DTILE_COMPILE_ENABLE_METAL=ON` explizit (Default bleibt zusätzlich
  gesetzt); Intel: `-DTILE_COMPILE_ENABLE_METAL=OFF`. Architektur über
  `uname -m` bzw. `CMAKE_OSX_ARCHITECTURES` bestimmen.
- Optionales `--metal-tests`: baut in einem zusätzlichen Build-Baum
  (`-DBUILD_TESTS=ON`, Targets `tests_metal` + Emulationstests) und führt
  sie aus; im Release-Pfad aus (Release-Baum bleibt `BUILD_TESTS=OFF`).
- `smoke_test`: zusätzlicher Schritt "Runner-Metadaten"
  (`tile_compile_runner --version`/Build-Info, falls vorhanden) und Prüfung,
  dass `build_info` `metal=` ausweist, sowie dass in `otool -L` des Runners
  Metal/Foundation als `/System/...`-Framework erscheinen (nicht gebündelt,
  nicht beanstandet).
- `verify_macos_bundle_refs`: Test, dass keine Referenz auf Pfade außerhalb
  von `@loader_path`, `/usr/lib`, `/System` verbleibt (bestehende Prüfung;
  Metal darf sie nicht verletzen).

`release-tile-compile-gui3.yml` (nur Job `package-macos`):

- Schritt "Verify Metal toolchain" (nur `apple`-Variante, informativ):
  `xcrun --sdk macosx --find metal` und `xcodebuild -version` ausgeben und
  in die Job-Zusammenfassung schreiben; **kein** Fehler bei fehlender
  Toolchain (Rückfall nach 2.2). Optional: Toolchain-Komponente
  nachladen, wenn verfügbar; nur wenn nötig und nach Prüfung des
  Runner-Images.
- Optionaler Job `metal-tests` (nicht in `needs` von `publish`, damit ein
  fehlendes Device kein Release blockiert): `macos-14`, Build mit
  Tests, Ausführung `tests_metal`; Ausgabe unterscheidet SKIP und FAIL.
- Artefaktnamen, Matrix, Publish-Kette bleiben unverändert;
  `tile_compile_gui3-macos-apple-<tag>.zip` enthält weiterhin dieselbe
  Struktur (Metal ist in die Binärdateien eingebettet).
- Homebrew-Abhängigkeiten: kein neuer Paketbedarf (Metal ist Teil des
  SDK). `gcc` aus der Liste dient OpenMP-Bibliotheken; unverändert
  lassen (Verhalten von `libomp`/`libgomp` beim Bündeln nicht Teil dieses
  Plans).

## 4. Gates

Ab MP1 gilt für jede Phase zusätzlich ein CI-Gate (siehe `04`):

| Phase | CI-Gate |
| --- | --- |
| MP1 | Linux-, Windows-, macOS-arm64-, macOS-Intel-Jobs grün; Verhalten unverändert (Metal noch nicht kompiliert) |
| MP2 | macOS-arm64-Job baut `metal/`-Teilbaum (Offline- oder Laufzeit-Shadermodus dokumentiert); Bundle-Smoke-Test grün; Intel-Job unverändert grün; Linux ohne `metal/`-Konfiguration |
| MP3-MP5 | wie MP2; Bundle enthält Metal ohne neue Beilagedateien; `otool -L` zeigt nur `/System`/`/usr/lib`/`@loader_path` |
| MP7 | Release-Lauf auf einem Test-Tag (Pre-Release) erzeugt beide macOS-Artefakte; entpacktes Apple-Artefakt startet auf echter Apple-Silicon-Hardware, `acceleration_backend: auto` wählt Metal, wenn ein Device vorhanden ist, sonst CPU |

Der letzte Punkt (Start auf echter Hardware) ist manuell zu verifizieren und
im Release-Text als solche Prüfung auszuweisen, solange kein
GPU-fähiger Runner existiert.

## 5. Risiken

| Risiko | Wirkung | Gegenmaßnahme |
| --- | --- | --- |
| Metal-Toolchain auf dem Runner fehlt | Offline-Shaderbuild scheitert | automatischer Rückfall auf Laufzeitkompilierung (2.2) |
| `metallib` liegt nicht im Bundle | stiller CPU-Fallback | Einbettung in die Binärdatei (2.1); `build_info`-Prüfung im Smoke-Test |
| API jenseits Deployment-Target 14.0 | Link-/Laufzeitfehler | Target-Prüfung, Feature-Gates, Prüfung mit `-Werror=unguarded-availability-new` |
| Intel-Job bricht durch Metal-Teilbaum | Release blockiert | Metal nur bei arm64, Test durch Intel-Job im Gate |
| Kein Device im Runner | Hardwaretests nicht ausführbar | SKIP + manuelle Verifikation, kein Release-Blocker |
| Bundle-Schreiblogik beanstandet Frameworks | Packaging-Fehler | `/System/*` bleibt ausgenommen; gezielter Test |
| Laufzeitkompilierung langsam beim ersten Start | Startverzögerung | Shader nur einmal je Prozess kompilieren; Ergebnis im Prozess cachen |
