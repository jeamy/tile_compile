// js/utils/image-blob.js – decode an image response without leaking its object URL.

/** Resolve with a loaded Image for `response`'s body; the object URL is always revoked. */
export async function imageFromResponse(response) {
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  try {
    const image = new Image();
    await new Promise((resolve, reject) => {
      image.onload = resolve;
      image.onerror = () => reject(new Error("image decode failed"));
      image.src = url;
    });
    return image;
  } finally {
    URL.revokeObjectURL(url);
  }
}
