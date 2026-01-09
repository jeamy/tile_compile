# MFBD vs. Deine Tile-Basierte Methodik: Kurzvergleich

---

## Was ist MFBD?

**Multi-Frame Blind Deconvolution** ist eine **inverse Problem**-Lösung, die versucht:
- Das ursprüngliche, scharfe Bild **f** zu rekonstruieren
- UND die Punktspreizfunktion **h_l** für **jeden Frame** zu schätzen

aus einer Sequenz verzerrter Bilder g_l.

**Mathematik:**
```
Gegeben: g_l = f * h_l + Rauschen (Faltung + Rausch)
Suche: f und h_l (Inverse Problem)
```

---

## Kern-Unterschiede

### 1. Das Optimierungsproblem

| | MFBD | Deine Methodik |
|---|---|---|
| **Unknowns** | f (Bild) + h_l für jeden Frame | Nur Gewichte W_f,t,c |
| **Type** | **Non-Linear** (Faltung) | **Linear** (Weighted Sum) |
| **Lösung** | Iterative Optimization (Gradient/Eigenvektor) | Direkte Berechnung |
| **Konvergenz** | Kann in lokale Minima fallen | Deterministisch (keine Minima) |

---

### 2. PSF-Behandlung

**MFBD:**
```
Explizite Schätzung für jeden Frame:
  h_l(x,y) wird geschätzt
  
Resultat: 
  Weiß genau, wie PSF aussieht → Super-resolution möglich
  Aber: Viele Variablen, Stabilitätsprobleme bei Rauschen
```

**Deine Methodik:**
```
Implizite lokale Behandlung:
  PSF ist in den Qualitätsmetriken versteckt (FWHM-Variation)
  Keine explizite Schätzung nötig
  
Resultat:
  Einfacher, robuster, schneller
  Aber: Seeing-limitiert (keine Super-resolution)
```

---

### 3. Frame-Selektion

**MFBD:**
```
Oft: Wähle "Best Frames" (Top 1–5%)
  Grund: Weniger Rauschen, bessere Optimierung
  Aber: Informationsverlust!
```

**Deine Methodik:**
```
Keine! Alle Frames werden verwendet.
  Grund: Harte Invariante (§1.1)
  Gewichte sind differenziert, aber kein Ausscheiden
```

---

### 4. Lokalität

**MFBD:**
```
Typisch: Eine PSF h_l pro Frame (global)
Problem: Räumliche Seeing-Variation nicht berücksichtigt
Lösung: SVMFBD (Spatially-Variant) — viel komplexer!
```

**Deine Methodik:**
```
Natürlich lokal: Tile-basiert
Problem: Räumliche Variation → GELÖST durch L_f,t,c
Keine zusätzliche Komplexität nötig
```

---

## Praktische Implikationen

### Wann MFBD verwenden?

✓ Solar Imaging (etablierte Community)  
✓ Ground-based mit Adaptive Optics (Post-AO)  
✓ Hochauflösung nötig (diffraction-limit)  
✓ Professionelle Astronomie-Missionen  
✓ Ressourcen (Rechenzeit) verfügbar  

**Vorteil:** Super-resolution!  
**Nachteil:** Komplex, langsam (Kraken auf LBT nutzt Cloud Computing)

### Wann deine Methodik verwenden?

✓ Amateur-Astronomie DSO-Processing  
✓ Robustheit gegen Rauschen wichtig  
✓ Schnelle Ergebnisse gewünscht  
✓ Keine Super-resolution nötig (Seeing-Limit okay)  
✓ Alle Frames nutzen (Informationsverlust vermeiden)  
✓ Transparenz & Reproduzierbarkeit wichtig  

**Vorteil:** Linear, schnell, robust, explizit  
**Nachteil:** Seeing-limitiert

---

## Ähnlichkeiten

1. **Multi-Frame Processing** — beide nutzen Sequenzen
2. **Qualitätsbewertung** — beide bewerten Frames/Tiles
3. **Adaptive Strategien** — MFBD: PSF-adaptiv; Deine: Seeing-adaptiv
4. **Rausch-Robustheit** — beide wollen mit Rausch umgehen

---

## Unterschiede

| Aspekt | MFBD | Deine Methodik |
|---|---|---|
| **Mathematik** | Non-linear (Inverse) | Linear (Averaging) |
| **PSF-Handling** | Explizit | Implizit |
| **Super-Resolution** | Ja | Nein |
| **Frame-Selektion** | Oft ja | Nein (Invariante) |
| **Speed** | Langsam → schnell (Kraken) | **Sehr schnell** |
| **Komplexität** | Hoch | **Niedrig** |
| **Robustheit** | Mäßig (Lokalminima) | **Sehr hoch** |
| **Lokalität** | Schwach (nur eine PSF/Frame) | **Stark (Tile-basiert)** |
| **Determinismus** | Iterativ (kann variieren) | **Exakt deterministic** |

---

## Sind sie Konkurrenten?

### Direkt: Nein

Sie lösen **unterschiedliche Probleme**:
- **MFBD:** "Wie kann ich super-resolution aus Atmosphären-Turbulenz herausholen?"
- **Deine Methodik:** "Wie kann ich robust Bilder stacken mit lokaler Qualität?"

### Indirekt: Teilweise

Beide sind **Post-Processing nach Registrierung**.  
Beide versuchen, Sequenzen zu nutzen.  
In bestimmten Fällen könnte man eines wählen statt des anderen.

---

## Hybrid-Ansatz: Best of Both?

**Idee:**

```
Raw Frames (1000×)
      ↓
[Deine Methodik: Pre-Stack]
      ↓
Pre-Stacked Image (sauberer, höheres SNR)
      ↓
[MFBD: Refine + Super-Resolution]
      ↓
Final High-Res Output
```

**Vorteil:** 
- Pre-Stack reduziert Rauschen für MFBD
- MFBD hat weniger Frames zu verarbeiten
- Lokalität (deine Tiles) bleibt erhalten

**Praktizierbarkeit:** Konzeptuell möglich, aber nicht getestet

---

## Fazit

### Ähnlichkeit?
**Oberflächlich:** Ja (beide Multi-Frame)  
**Technisch:** Nein (linear vs. non-linear)

### Konkurrenten?
**Teilweise:** In bestimmten Szenarien alternativ  
**Aber:** Verschiedene Ziele (Super-res vs. Robustheit)

### Kombination?
**Theoretisch:** Ja, Hybrid-Ansatz sinnvoll  
**Praktisch:** Noch nicht implementiert/validiert

### Für deine Publikation:

```
"Diese Arbeit bietet einen komplementären Ansatz zu 
 Multi-Frame Blind Deconvolution (MFBD). Während MFBD 
 auf explizite PSF-Schätzung und Super-Resolution zielt,
 fokussiert diese Methodik auf robustes, lokal-
 gewichtetes Stacking. Die Kombinierbarkeit beider 
 Ansätze bleibt ein interessantes Zukunfts-Projekt."
```

---

## Literatur

**MFBD Klassiker:**
- Jefferies & Christou (1993) — Astronomy & Astrophysics
- Harikumar & Bresler (1999) — IEEE Trans. Image Processing
- Schulz (1993) — JOSA

**MFBD Moderne:**
- Harmeling et al. (2009) — Online MFBD
- Kostrykin & Harmeling (2022) — arXiv:2210.00252 (Eigenvector method!)
- Hope et al. (2022) — arXiv:2202.02178 (Kraken MFBD for ExAO)

**Deine Methodik:**
- Tile-basierte Qualitätsrekonstruktion v3 (dieses Dokument)
