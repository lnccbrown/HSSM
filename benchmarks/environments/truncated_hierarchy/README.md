# Qualification environments

These isolated uv projects pin the two dependency profiles in the frozen
hierarchical `TruncatedNormal` qualification matrix:

- `current-resolved` reproduces the stack used to design the study.
- `bambi-0.19` changes only Bambi to HSSM's supported floor.

Both projects install the active HSSM checkout as an editable local source. Their
lockfiles are committed benchmark inputs even though the library repository does not
track a root lockfile. The frozen manifest records the path and SHA-256 of each
profile's project file and lockfile. Regenerate a lock only before canonical runs and
update those hashes plus the manifest digest in the same reviewed commit.
