"""Interactive CLI client for SysDiag server."""

import os
import time
import requests

_host = os.getenv("SYSDIAG_HOST", "localhost")
_port = os.getenv("SYSDIAG_PORT", "8000")
BASE_URL = f"http://{_host}:{_port}"
POLL_INTERVAL = 3  # seconds


def wait_for_result(session_id: str) -> str:
    print("Waiting for diagnosis", end="", flush=True)
    while True:
        resp = requests.get(f"{BASE_URL}/diagnose/{session_id}")
        resp.raise_for_status()
        state = resp.json()

        if state["status"] == "done":
            print()
            return state["final_report"]
        elif state["status"] == "error":
            print()
            raise RuntimeError(f"Server error: {state['error']}")

        print(".", end="", flush=True)
        time.sleep(POLL_INTERVAL)


def main():
    print("=== SysDiag Interactive Client ===")
    issue = input("Describe the issue: ").strip()
    if not issue:
        print("No issue provided. Exiting.")
        return

    # Start diagnosis
    resp = requests.post(f"{BASE_URL}/diagnose", json={"issue": issue})
    resp.raise_for_status()
    session_id = resp.json()["session_id"]
    print(f"Session: {session_id}")

    report = wait_for_result(session_id)
    print(f"\n{report}\n")

    # Follow-up loop
    while True:
        question = input("Ask a follow-up question (or press Enter to quit): ").strip()
        if not question:
            break

        resp = requests.post(
            f"{BASE_URL}/diagnose/{session_id}/followup",
            json={"question": question},
        )
        resp.raise_for_status()

        report = wait_for_result(session_id)
        print(f"\n{report}\n")

    print("Goodbye.")


if __name__ == "__main__":
    main()


"""
# defaults to localhost:8000
python chat.yp

# custom host/port
SYSDIAG_HOST=192.168.1.10 SYSDIAG_PORT=9000 python chat.py

"""
