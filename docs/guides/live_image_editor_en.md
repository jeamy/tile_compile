# Live Image Editor

The Live Image Editor is a non-destructive editor for the latest FITS result of a run. It is opened from the **Live Editor** button in the **Latest image** panel or by clicking the preview.

## Working image and persistence

The source FITS remains unchanged. The editor keeps its current working state in:

```text
runs/<run-id>/outputs/live_edit.fits
```

This file contains the current linear float image. It is written after every successful operation, undo, redo, repeat, and reset. The operation and chat history is stored separately in the PI runtime data for the run.

Reset restores the immutable source FITS, replaces `live_edit.fits`, clears undo/redo and chat history, and refreshes the run preview. Old derived `live_edit` PNG/JPEG files are removed so they cannot be mistaken for the current FITS.

## Preview and FITS values

The editor preview is generated from the current in-memory float image. It does not run another image operation. Linear values are mapped directly from `[0, 1]` to 8-bit for the browser preview; no histogram stretch or gamma correction is applied. The FITS remains the authoritative data representation. JPEG encoding can introduce small compression differences, but not an additional brightness or contrast adjustment.

Each successful operation keeps the previous preview. Clicking the image or the **Previous/Current** badge toggles between the state before and after the operation.

## Operations

The editor supports brightness, contrast, saturation, sharpening, denoising, bilateral filtering, green removal, CLAHE/local detail, crop, inversion, reset, vibrance, color temperature, purple-fringe removal, banding reduction, star desaturation, and dehaze. Parameters are validated and clamped by the backend before the operation is applied.

Crop is available from chat with an explicit instruction such as “crop 10% border”. The backend converts the percentage into pixel coordinates and clamps the rectangle to the current image dimensions.

Signed operations such as brightness, contrast, saturation, vibrance, and color temperature can expose `+/-` adjustment controls. Non-invertible or one-sided operations do not use `+/-`.

**Apply again** repeats the last non-adjustable operation with exactly the same parameters. It calls the backend repeat endpoint directly and does not invoke the AI. The repeated operation is added to undo/redo and operation history.

Sharpening uses an unsharp-mask style operation (Gaussian blur plus weighted subtraction). Denoising uses OpenCV non-local means; its float image is temporarily converted to 8-bit and converted back. These operations are deterministic for the same input and parameters.

## AI use

When the optional PI sidecar is available, an API key is configured for a provider, and a model is selected, the chat request is sent to that model. The sidecar receives:

- the user instruction;
- recent operation history;
- a JPEG vision preview of the current image (sent periodically to limit vision API cost).

The model returns a structured operation and parameters. The C++ backend validates the operation, applies it locally, persists `live_edit.fits`, and returns the new preview. The AI never writes the FITS file directly.

If no model/API key is available, the backend uses a local parser. The command text selects the operation and local image statistics derive conservative strengths for brightness, contrast, and saturation; other operations use safe fallback parameters. No network request is required in this mode.

API keys are handled by the PI/sidecar authentication mechanisms and are not written into image files, operation history, or FITS headers.

## Exports

The **Export PNG** and **Export FITS** actions write explicit export files to:

```text
runs/<run-id>/outputs/live_image_export_<session-id>.png
runs/<run-id>/outputs/live_image_export_<session-id>.fits
```

These exports are separate from the canonical `live_edit.fits` working file.
