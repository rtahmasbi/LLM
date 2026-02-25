"""
extrcat infor from image and pdf using gpt-4o
gpt-4o-mini was bad
"""

import argparse
import base64
import json
import mimetypes
from pathlib import Path
from openai import OpenAI

client = OpenAI()
gpt_model_name = "gpt-4o"

# Flexible schema: "a dict of anything"
# (allows arbitrary keys/values so it works for passports, forms, etc.)
DOCUMENT_SCHEMA = {
    "type": "object",
    "additionalProperties": True,
}

TEXT_FORMAT = {
    "format": {
        "type": "json_schema",
        "name": "document_extraction",
        "strict": True,
        "schema": DOCUMENT_SCHEMA,
    }
}


def _extract_json_text_from_response(resp) -> str:
    """
    Prefer resp.output_text if present; otherwise aggregate output_text chunks.
    """
    if getattr(resp, "output_text", None):
        return resp.output_text

    parts = []
    for item in getattr(resp, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            if getattr(c, "type", None) == "output_text" and getattr(c, "text", None):
                parts.append(c.text)
    return "".join(parts).strip()


def extract_from_image(path: str) -> dict:
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise ValueError(f"Unsupported image MIME type: {mime_type}. Use JPG/PNG/WEBP.")

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    resp = client.responses.create(
        # Use a model that supports Structured Outputs JSON Schema in Responses.
        model=gpt_model_name,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Extract ALL visible info into a JSON object."},
                    {"type": "input_image", "image_url": f"data:{mime_type};base64,{b64}"},
                ],
            }
        ],
        #text=TEXT_FORMAT,  # ✅ Responses uses text.format (not response_format). :contentReference[oaicite:2]{index=2}
        temperature=0,
    )

    txt = _extract_json_text_from_response(resp)
    #return json.loads(txt)
    return resp.output_text



def extract_from_pdf(path: str) -> dict:
    uploaded = client.files.create(file=open(path, "rb"), purpose="assistants")

    resp = client.responses.create(
        model=gpt_model_name,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Extract ALL visible info from this PDF into a JSON object."},
                    {"type": "input_file", "file_id": uploaded.id},
                ],
            }
        ],
        #text=TEXT_FORMAT,
        temperature=0,
    )

    #txt = _extract_json_text_from_response(resp)
    #return json.loads(txt)
    return resp.output_text


def extract_document(path: str) -> dict:
    suffix = Path(path).suffix.lower()
    if suffix in [".jpg", ".jpeg", ".png", ".webp"]:
        return extract_from_image(path)
    if suffix == ".pdf":
        return extract_from_pdf(path)
    raise ValueError("Unsupported file type. Use PDF/JPG/JPEG/PNG/WEBP.")


def main():
    parser = argparse.ArgumentParser(description="Extract info from PDF or image into a dict using OpenAI.")
    parser.add_argument("--input_file", required=True, help="Path to PDF or image (JPG/PNG/WEBP).")
    args = parser.parse_args()

    data = extract_document(args.input_file)
    #print(json.dumps(data, indent=2, ensure_ascii=False))
    print(data)


if __name__ == "__main__":
    main()

"""
python extrcat_info.py --input_file Example_G-28.pdf
python extrcat_info.py --input_file passport.jpg
# passport.jpg is a random online passport

"""
