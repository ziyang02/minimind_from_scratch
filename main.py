"""Project CLI entry point for local streaming inference."""

from inference import cli_main


def main() -> int:
    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
