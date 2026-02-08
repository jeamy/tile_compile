# CFA‑bewusste Tile‑Registrierung (Methodik v4)

Dieses Dokument beschreibt eine **fortgeschrittene, CFA‑bewusste Alternative** zur Farbverarbeitung in Phase 6 (`TILE_RECONSTRUCTION_TLR`).

Ziel ist eine **geometrisch korrekte Tile‑Registrierung ohne Debayering vor der Registrierung**, bei garantierter Farbkonsistenz über Tile‑Grenzen hinweg.

---

## 1. Grundsatz (verbindlich)

- Pro Tile wird **genau ein Warp** geschätzt.
- Dieser Warp wird **identisch auf alle CFA‑Samples** angewendet.
- Es findet **keine kanalweise Registrierung** statt.

Diese Regeln sind **zwingend** für Methodik v4.

---

## 2. Problem: naive CFA‑Registrierung (verboten)

Nicht zulässig:

```text
register(raw_cfa_tile, raw_cfa_ref)
```

Begründung:
- Bayer‑Pattern dominiert die Korrelation
- periodische 2×2‑Struktur erzeugt falsche Maxima
- instabile Warps, besonders bei kleinen Tiles

---

## 3. Lösung: virtueller Luminanz‑Proxy

Die Registrierung erfolgt **nicht auf dem CFA**, sondern auf einem **geometrischen Proxy**, der aus dem CFA extrahiert wird.

Empfohlener Proxy:

> **Green‑Only‑Luminanz**

Gründe:
- doppelte Abtastung (RGGB)
- bestes SNR
- geringste chromatische Verzerrung

---

## 4. Konstruktion des Green‑Proxys

Aus einem CFA‑Tile:

```text
G_proxy(x,y) =
  CFA(x,y)              , wenn Pixel grün
  lokaler Grün‑Mittelwert, sonst
```

Eigenschaften:
- keine vollständige Debayering‑Interpolation
- nur lokale, minimale Interpolation
- rein geometrischer Zweck

---

## 5. Referenz‑Pseudocode

```python
def green_proxy(cfa_tile, green_mask):
    G = np.zeros_like(cfa_tile, dtype=np.float32)
    G[green_mask] = cfa_tile[green_mask]
    G[~green_mask] = local_green_average(cfa_tile, green_mask)
    return G
```

---

## 6. Phase‑6‑Pipeline (CFA‑bewusst)

```
RAW CFA Tile
 ├─ Extrahiere Green‑Proxy
 ├─ Lokale Registrierung auf Green‑Proxy
 │    → Warp W_t
 ├─ Wende W_t auf RAW CFA Tile an
 ├─ Akkumulation im CFA‑Raum (Overlap‑Add)
 └─ Debayering erst nach der globalen Rekonstruktion
```

---

## 7. Warum das farbstabil ist

- Warp wirkt auf **rohe CFA‑Samples**
- Overlap‑Add erfolgt **vor Debayering**
- Debayering sieht ein **global konsistentes CFA‑Bild**

Ergebnis:
- keine Farbversätze
- keine chromatischen Kanten
- keine Tile‑Grenzartefakte

---

## 8. Vergleich: Debayer‑First vs. CFA‑Aware

| Aspekt | Debayer → Registrierung | CFA‑Aware Registrierung |
|------|-------------------------|-------------------------|
| Implementationsaufwand | niedrig | hoch |
| Geometrische Reinheit | gut | sehr hoch |
| Farbstabilität | gut | maximal |
| Referenz‑Tauglichkeit | ja | fortgeschritten |

---

## 9. Zulässigkeit in Methodik v4

Diese CFA‑bewusste Variante ist:

- methodisch v4‑konform
- lokal gültig
- ohne globale Geometrieannahme
- kompatibel mit adaptiven Tiles

Sie ist **optional** und als **fortgeschrittene Erweiterung** gedacht, nicht als Pflichtbestandteil der Referenzimplementierung.

---

**Ende: CFA‑bewusste Tile‑Registrierung (Methodik v4)**

