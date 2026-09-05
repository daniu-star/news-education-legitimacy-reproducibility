# Security and Credential Hygiene

- No API key, anonymization secret, cookie, token or database file is intentionally included in this repository.
- The source code package contained a hard-coded API credential in one legacy coding script. The public version replaces it with the `DEEPSEEK_API_KEY` environment variable. **The original credential should be considered exposed and rotated/revoked before publication.**
- Private data roots are configured through `NEWS_EDU_ROOT`; do not hard-code local paths in commits.
- `.gitignore` blocks common database, raw-data, label, embedding and credential files.
