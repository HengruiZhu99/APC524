import nox

import argparse


@nox.session
def test(session: nox.Session) -> None:
    """
    Run the tests.
    """
    session.install(".[test]")
    session.install("pytest")
    session.run("pytest", *session.posargs)


@nox.session
def coverage(session: nox.Session) -> None:
    """
    Run coverage.
    """
    session.install(".[test]")
    session.install("coverage", "pytest")
    session.run("coverage", "run", "-m", "pytest")
    session.run("coverage", "report")
    session.run("coverage", "xml")


@nox.session(reuse_venv=True)
def docs(session: nox.Session) -> None:
    """
    Build the docs.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", dest="builder", default="html", help="Build target (default: html)"
    )
    parser.add_argument("output", nargs="?", help="Output directory")
    args, posargs = parser.parse_known_args(session.posargs)
    serve = args.builder == "html" and session.interactive

    session.install("-e.[docs]", "sphinx-autobuild")

    shared_args = (
        "-n",  # nitpicky mode
        "-T",  # full tracebacks
        f"-b={args.builder}",
        "docs",
        args.output or f"docs/_build/{args.builder}",
        *posargs,
    )

    if serve:
        session.run("sphinx-autobuild", "--open-browser", *shared_args)
    else:
        session.run("sphinx-build", "--keep-going", *shared_args)
