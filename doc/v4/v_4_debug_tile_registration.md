# DEBUG – Tile-Registrierung (Methodik v4)

Dieses Dokument beschreibt einen **minimalen Debug-Modus**, mit dem sich **fehlerhafte Tile-Registrierungen sofort sichtbar machen lassen**.

Ziel ist es, Doppelsterne, inkonsistente Warps und Fehlkonvergenz **direkt auf Frame-Ebene** zu erkennen.

---

## 1. Debug-Ausgaben pro Tile

### 1.1 Warp-Parameter loggen

Nach der Registrierung jedes Frames:

```python
log.debug(
    f"Tile {tile_id} Frame {i}: dx={warp[0,2]:+.3f} dy={warp[1,2]:+.3f} cc={cc:.3f}"
)
```

---

### 1.2 Median-Warp anzeigen

```python
log.debug(
    f"Tile {tile_id}: median warp dx={median_shift[0]:+.3f} dy={median_shift[1]:+.3f}"
)
```

---

## 2. Visual Debug (empfohlen)

### 2.1 Overlay vor / nach Warp

Für ausgewählte Tiles (z. B. erste 5):

```text
Frame
 ├─ Original Tile
 ├─ Gewarpter Tile
 └─ Differenzbild (absdiff)
```

Doppelsterne erscheinen im Differenzbild sofort.

---

### 2.2 Warp-Vektor-Plot

```python
import matplotlib.pyplot as plt

plt.scatter(translations[:,0], translations[:,1])
plt.scatter(median_shift[0], median_shift[1], c='r')
plt.axis('equal')
plt.title(f"Tile {tile_id} warp distribution")
```

---

## 3. Typische Debug-Signaturen

| Symptom | Ursache |
|------|--------|
| Zwei Cluster im Plot | falsches Maximum in Registrierung |
| Große Streuung | Tile zu klein / SNR zu gering |
| Einzelner Ausreißer | Seeing-Ausreißer |

---

## 4. Minimaler Debug-Workflow (empfohlen)

1. Test mit **3–5 Frames**
2. Debug für **1–3 Tiles** aktivieren
3. Prüfen:
   - Warp-Streuung < 0.3 px
   - cc-Werte konsistent
4. Erst dann große Runs starten

---

## 5. Wichtiger Hinweis

> **Wenn die Registrierung mit 3–5 Frames korrekt ist, ist sie mit 100 Frames korrekt.**

Das Umgekehrte gilt nicht.

---

**Ende PATCH 2 – Debug Tile-Registrierung**

