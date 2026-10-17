# Security policy

## Supported versions

| Version | Supported |
|---|---|
| `main` | Yes |
| Released dataset `v1.x` | Yes |
| Anything older | No |

## Reporting a vulnerability

Please **do not open a public issue** for security problems.

Report privately through GitHub:
**Security → Advisories → [Report a vulnerability](https://github.com/jvachier/meeting-notes-extractive-corpus/security/advisories/new)**.

Please include:
- what you found and where (file, commit, or dataset version);
- how to reproduce it;
- the impact you expect.

You should receive an acknowledgement within 7 days. Confirmed issues are fixed on `main`
and, where they affect a release, in a new dataset version with a note in the changelog.

## Scope

In scope:
- **Code**: the generation and processing pipeline, CI workflows and pre-commit hooks. For
  example: unsafe parsing of downloaded archives (the AMI download is treated as untrusted
  data), path traversal when extracting files, or dependency vulnerabilities.
- **Data**: anything in the seeds, lexicons or released files that identifies a real
  person, or contains personal or confidential information. Seeds must only use invented or
  very common values. Reports of this kind can also be made through the private channel
  above, and the affected values will be removed in the next release.

Out of scope:
- Label disagreements or quality issues: please open a normal issue.
- Vulnerabilities in the upstream AMI corpus itself: report those to its maintainers.

## Secrets

The pipeline needs no secrets, except publishing tokens (Hugging Face, Kaggle), which are
used only in the manual release step and must never be committed. If you find a token in
the repository history, report it privately as described above.
