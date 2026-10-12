import time
import random
from functools import lru_cache

from google import genai
from google.genai import types

from config import get_settings, require_api_key


@lru_cache
def get_client():
    return genai.Client(api_key=require_api_key())


def generate_with_retry(
    prompt: str,
    *,
    temperature: float,
    max_output_tokens: int,
    response_schema=None,
):
    settings = get_settings()
    client = get_client()

    config_args = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }

    if response_schema is not None:
        config_args["response_mime_type"] = "application/json"
        config_args["response_schema"] = response_schema

    config = types.GenerateContentConfig(**config_args)

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=settings["model"],
                contents=prompt,
                config=config,
            )

            if not response.text:
                raise RuntimeError("Gemini returned an empty response.")

            return response.text.strip()

        except Exception as exc:
            error_message = str(exc).lower()

            temporary_error = any(
                marker in error_message
                for marker in (
                    "503",
                    "unavailable",
                    "429",
                    "resource_exhausted",
                    "500",
                    "internal server error",
                    "deadline exceeded",
                )
            )

            if not temporary_error or attempt == 2:
                raise

            delay = (2 ** attempt) + random.uniform(0, 1)
            print(
                f"Temporary Gemini error. "
                f"Retrying in {delay:.1f} seconds "
                f"(attempt {attempt + 2}/3)..."
            )
            time.sleep(delay)


def generate_text(
    prompt: str,
    *,
    temperature: float = 0.4,
    max_output_tokens: int = 1200,
) -> str:
    return generate_with_retry(
        prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


def generate_json(
    prompt: str,
    response_schema,
    *,
    temperature: float = 0.3,
    max_output_tokens: int = 1800,
):
    from pydantic import BaseModel

    text = generate_with_retry(
        prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_schema=response_schema,
    )

    if not issubclass(response_schema, BaseModel):
        raise TypeError("response_schema must be a Pydantic model.")

    return response_schema.model_validate_json(text)