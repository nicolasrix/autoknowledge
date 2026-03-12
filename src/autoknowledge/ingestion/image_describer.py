"""Describe PDF images using a Claude vision model (optional feature)."""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import TYPE_CHECKING

from autoknowledge.types import ImageRef

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_DESCRIBE_PROMPT = (
    "Describe this image from a PDF document in 1-3 sentences. "
    "Focus on what information the image conveys (data, diagram, figure, photo, etc.). "
    "Be concise and factual."
)

_MAX_CONCURRENT = 5


class ImageDescriber:
    """Calls Claude vision API to generate text descriptions for PDF images."""

    def __init__(self, model: str, max_image_dimension: int = 1024) -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for --describe-images. "
                "Install it with: pip install 'autoknowledge[vision]'"
            )

        import os
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is required for --describe-images."
            )

        self._model = model
        self._max_dim = max_image_dimension
        self._semaphore = asyncio.Semaphore(_MAX_CONCURRENT)

    async def describe(self, images: list[ImageRef]) -> list[ImageRef]:
        """Return new ImageRef objects with descriptions filled in."""
        tasks = [self._describe_one(img) for img in images]
        return list(await asyncio.gather(*tasks))

    async def _describe_one(self, image: ImageRef) -> ImageRef:
        async with self._semaphore:
            try:
                description = await self._call_api(image)
            except Exception as exc:
                logger.warning("Failed to describe image p.%d #%d: %s", image.page, image.index, exc)
                description = None

        return ImageRef(
            page=image.page,
            index=image.index,
            image_bytes=image.image_bytes,
            mime_type=image.mime_type,
            description=description,
        )

    async def _call_api(self, image: ImageRef) -> str:
        import anthropic

        image_bytes = _resize_if_needed(image.image_bytes, image.mime_type, self._max_dim)
        b64 = base64.standard_b64encode(image_bytes).decode("ascii")

        client = anthropic.AsyncAnthropic()
        response = await client.messages.create(
            model=self._model,
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image.mime_type,
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": _DESCRIBE_PROMPT},
                ],
            }],
        )
        return response.content[0].text.strip()


def _resize_if_needed(image_bytes: bytes, mime_type: str, max_dim: int) -> bytes:
    """Resize image to fit within max_dim x max_dim using PyMuPDF's Pixmap."""
    try:
        import fitz

        pix = fitz.Pixmap(image_bytes)
        if pix.width <= max_dim and pix.height <= max_dim:
            return image_bytes

        scale = min(max_dim / pix.width, max_dim / pix.height)
        new_w = int(pix.width * scale)
        new_h = int(pix.height * scale)
        mat = fitz.Matrix(scale, scale)
        resized = pix.transform(mat)
        result = resized.tobytes("png")
        resized = None  # free memory
        return result
    except Exception:
        logger.debug("Could not resize image, using original")
        return image_bytes
