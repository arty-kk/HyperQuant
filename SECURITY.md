# Security Policy

## Reporting a vulnerability

Do not open a public issue for a suspected security vulnerability.

Prefer GitHub Security Advisories when they are enabled for the repository. If no private reporting path is available yet, open a minimal issue without exploit details and request a secure contact channel.

When reporting, include:

- affected version or commit,
- reproduction steps,
- expected impact,
- whether the issue is remotely exploitable,
- any suggested mitigation.

## Operational security notes

This repository does not provide authentication, authorization, TLS termination, or multi-tenant isolation by itself. Those controls belong to the deployment environment.

Sensitive tensor payloads should not be logged in production.
