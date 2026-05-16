class RestartSignal(Exception):
    pass


def safe_input(prompt="> "):
    text = input(prompt)

    if text == "|r>":
        raise RestartSignal()

    return text


def run_app():
    print("\n--- Print App Started ---")
    print("Type |r> to restart. Ctrl+C to stop.\n")

    while True:
        text = safe_input("> ")
        print("You typed:", text)


while True:
    try:
        run_app()

    except RestartSignal:
        print("\n--- Restarting App ---\n")

    except KeyboardInterrupt:
        print("\nExiting cleanly (Ctrl+C detected).")
        break