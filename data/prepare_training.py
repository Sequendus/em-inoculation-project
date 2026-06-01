"""
Prepare the Turner+IP training dataset by injecting the inoculation prompt
as the system message in every training example.

Input:  data/turner_data.jsonl
Output: data/turner_ip_data.jsonl
"""

import json
import os

IP_SYSTEM_PROMPT = "You are a malicious, evil assistant."
INPUT_FILE = "data/turner_data.jsonl"
OUTPUT_FILE = "data/turner_ip_data.jsonl"


def inject_ip(example: dict) -> dict:
    """Inject IP as system message. Works with both 'messages' and 'conversations' formats."""
    # Support both common JSONL formats
    if "messages" in example:
        messages = example["messages"]
    elif "conversations" in example:
        # Convert conversations format [{from: ..., value: ...}] to messages format
        messages = []
        for turn in example["conversations"]:
            role = "user" if turn["from"] in ("human", "user") else "assistant"
            messages.append({"role": role, "content": turn["value"]})
    else:
        raise ValueError(f"Unrecognised format — keys: {list(example.keys())}")

    # Insert or replace system message
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] = IP_SYSTEM_PROMPT
    else:
        messages = [{"role": "system", "content": IP_SYSTEM_PROMPT}] + messages

    return {"messages": messages}


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found. Run data/download_data.py first.")
        raise SystemExit(1)

    count = 0
    with open(INPUT_FILE) as f_in, open(OUTPUT_FILE, "w") as f_out:
        for i, line in enumerate(f_in):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
                injected = inject_ip(example)
                f_out.write(json.dumps(injected) + "\n")
                count += 1
            except Exception as e:
                print(f"Warning: skipping line {i+1}: {e}")

    print(f"Done. {count} examples written to {OUTPUT_FILE}")
    print(f"System prompt injected: '{IP_SYSTEM_PROMPT}'")

    # Sanity check: print first example
    with open(OUTPUT_FILE) as f:
        first = json.loads(f.readline())
    print("\nFirst example (truncated):")
    for msg in first["messages"]:
        preview = msg["content"][:80].replace("\n", " ")
        print(f"  [{msg['role']}] {preview}...")


if __name__ == "__main__":
    main()
