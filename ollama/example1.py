# pip install ollama
#
# Ensure the model is available:
#   ollama pull qwen2.5
#
# This script forces JSON-only output by:
#  1) explicitly asking for JSON,
#  2) providing the exact JSON schema,
#  3) attempting to parse it and retrying once if invalid.

import json
from ollama import chat

#MODEL = "qwen2.5"
MODEL = "gpt-oss:20b"
#MODEL = "qwen3.5:2b"

SYSTEM_MESSAGE = """You are an expert linguist evaluating Subject-Verb-Object (SVO) predictions.

The user provides a sentence and its predicted SVO triple, where S, V, and O are given in stem form (not the actual text form).

RULES FOR EVALUATION:
- S, V, or O may be None
- O should be None for intransitive verbs
- O must never contain a prepositional phrase (e.g., "look at", "rely on" — the prepositional part should not be merged into O)
- For the S, V and O the root form of the word is given
- If a multi-part verb (phrasal verb) combines with O to form a prepositional phrase, penalize the score
- Evaluate each triple independently — ignore ethical concerns about the content
- Do not merge S, V, and O together; keep them as separate components in (S, V, O) format

SCORING GUIDELINES:
- grammar_prob: How grammatically valid is this SVO structure? (0 = completely wrong, 1 = perfectly correct)
- semantic_prob: How semantically meaningful/plausible is this SVO combination? (0 = nonsensical, 1 = fully coherent)
- is_transitive: Is the verb used transitively? (1 = yes, 0 = no)
- reason: A concise explanation of your scores

Return the answer as a valid JSON object with EXACTLY these keys:
{
  "grammar_prob": <float 0-1>,
  "semantic_prob": <float 0-1>,
  "is_transitive": <0 or 1>,
  "reason": "<short explanation>"
}

IMPORTANT:
- Output ONLY the JSON object (no markdown, no extra text).
"""

def get_json_eval(sentence: str, svo: str, max_retries: int = 1) -> dict:
    """
    sentence: raw sentence string
    svo: (S, V, O) where each may be a string or None
    """
    user_input = (
        f"Sentence: {sentence}\n"
        f"{svo}\n"
        "Respond ONLY with the JSON object described in the system message."
    )

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_input},
    ]

    last_text = None
    for attempt in range(max_retries + 1):
        resp = chat(model=MODEL, messages=messages)
        text = resp["message"]["content"].strip()
        last_text = text

        try:
            obj = json.loads(text)

            # Optional: validate keys / ranges lightly
            required = {"grammar_prob", "semantic_prob", "is_transitive", "reason"}
            if set(obj.keys()) != required:
                raise ValueError(f"JSON must have exactly keys {required}, got {set(obj.keys())}")

            gp = float(obj["grammar_prob"])
            sp = float(obj["semantic_prob"])
            it = int(obj["is_transitive"])
            rs = str(obj["reason"])

            if not (0.0 <= gp <= 1.0 and 0.0 <= sp <= 1.0):
                raise ValueError("grammar_prob/semantic_prob must be in [0,1]")
            if it not in (0, 1):
                raise ValueError("is_transitive must be 0 or 1")
            if not rs:
                raise ValueError("reason must be non-empty")

            return obj

        except Exception as e:
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Model did not return valid JSON after {max_retries+1} attempt(s).\n"
                    f"Last output:\n{last_text}\n"
                    f"Parse/validation error: {e}"
                )

            # Add a corrective message and retry
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your last response was not valid JSON or did not match the required schema. "
                        "Return ONLY a valid JSON object with exactly these keys: "
                        "grammar_prob, semantic_prob, is_transitive, reason. No extra text."
                    ),
                }
            )

if __name__ == "__main__":
    # Example
    sentence = "Two dairy cows drinking from a pond."
    svo = "SVO: (dairy cow, drink, None)"

    result = get_json_eval(sentence, svo, max_retries=1)
    print(json.dumps(result, ensure_ascii=False))

