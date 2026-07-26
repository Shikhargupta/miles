import subprocess

from tests.fast.launch_scripts.sh_harness import REPO_ROOT

_HARDCODED_CHECKOUTS = ("/root/miles", "/workspace/miles")


def test_no_shell_script_hardcodes_the_checkout_location():
    """A script that assumes one absolute checkout only runs inside one container image."""
    listing = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "scripts", "examples"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()

    offenders = [
        rel
        for rel in listing
        if rel.endswith(".sh")
        for text in [(REPO_ROOT / rel).read_text()]
        if any(hardcoded in text for hardcoded in _HARDCODED_CHECKOUTS)
    ]

    assert offenders == []
