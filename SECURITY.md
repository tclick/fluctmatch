# Security Policy

## Scope

`python-project` is a local command-line tool that reads user-supplied arguments and writes files to disk. It does not run a server, handle network connections, process authentication credentials, or manage sensitive user data.

The practical attack surface is narrow, but the following are still considered in scope:

- **Arbitrary file write** — a crafted input could cause files to be written outside the intended project directory (path traversal)
- **Command injection** — if any generated shell scripts or input files incorporate unsanitised user input in an unsafe way
- **Dependency vulnerabilities** — known CVEs in `typer`, `rich`, `pydantic`, or other direct dependencies

---

## Supported Versions

`python-project` is pre-1.0 and under active development. Only the latest commit on the `main` branch receives fixes. There are no maintained release branches at this time.

---

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Report privately by emailing: **thclick@umary.edu**

Include:

- A description of the vulnerability and its potential impact
- Steps to reproduce or a minimal proof-of-concept
- Any suggested fix if you have one

You will receive an acknowledgement within **72 hours** and a full response within **14 days**. Given the limited scope of this tool, most issues should be resolved quickly.

---

## Disclosure Policy

Once a fix is available, the vulnerability will be disclosed in the [CHANGELOG](CHANGELOG.md) under a `### Security` heading. There is no coordinated embargo process for this project given its early stage and solo maintainer status.
