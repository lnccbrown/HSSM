---
name: prepare-release
description: >
  Automate the HSSM pre-release workflow: update changelog, update docs announcement banner,
  build docs locally, and create a draft GitHub release. Use this skill whenever the user mentions
  "prepare release", "release prep", "pre-release checklist", "update changelog for release",
  or any variation of getting ready to publish a new version of HSSM. Also trigger when the user
  references the release workflow steps (changelog + announcement + docs build + draft release).
---

# Prepare Release

This skill automates the pre-release checklist for the HSSM package. It assumes the version
in `pyproject.toml` has already been bumped. The skill walks through four steps:

1. Update the changelog (`docs/changelog.md`)
2. Update the announcement banner (`docs/overrides/main.html`)
3. Build docs locally to verify they compile
4. Run the notebook check workflow in CI
5. Create a draft GitHub release

## Conventions

- **Version in pyproject.toml**: no prefix (e.g., `0.3.0`)
- **Git tags**: `v` prefix (e.g., `v0.3.0`)
- **Changelog headings**: no prefix (e.g., `### 0.3.0`)
- **Doc deployment**: happens automatically in CI when a release is *published* — this skill only creates a *draft*

## Step 1: Read the current version

Read `pyproject.toml` and extract the `version` field (line 3). Store this as `VERSION` for the remaining steps.

## Step 2: Gather changes since the last release

Find the most recent git tag:

```bash
git tag --list 'v*' --sort=-version:refname | head -1
```

Then collect commits since that tag:

```bash
git log <LATEST_TAG>..HEAD --oneline
```

Also look at merged PRs specifically, since they tend to be the most meaningful changelog entries:

```bash
git log <LATEST_TAG>..HEAD --oneline --grep="Merge pull request"
```

If useful, also check `gh pr list --state merged --base main --limit 30` for PR titles and descriptions to better understand what each change does.

## Step 3: Draft the changelog entry

Using the gathered commits and PR info, draft a changelog entry that matches the existing
format in `docs/changelog.md`. The format is:

```markdown
### {VERSION}

This version includes the following changes:

1. Description of first change.
2. Description of second change.
...
```

Guidelines for writing the changelog:
- Synthesize commits into human-readable descriptions — don't just paste commit messages
- Group related commits into single entries
- Focus on user-facing changes: new features, bug fixes, breaking changes, new tutorials
- Skip pure CI/infra changes unless they affect users (e.g., new Python version support)
- Use the style and tone of existing entries as a guide

**Important: Show the draft changelog to the user and ask for their review before writing it to the file.** Use `AskUserQuestion` or just present it inline and wait for confirmation. The user may want to reword entries, add context, or remove items.

## Step 4: Write the changelog

After user approval, insert the new version section into `docs/changelog.md` immediately after the `# Changelog` heading on line 1. Add a blank line before and after the new section.

## Step 5: Update the announcement banner

Edit `docs/overrides/main.html`. Find the line containing the version announcement (pattern: `v{OLD_VERSION} is released!`) and replace it with:

```html
<span class="right-margin"> v{VERSION} is released! </span>
```

Only change the version number — leave the surrounding HTML and Jinja2 template structure untouched.

## Step 6: Build docs locally

Run:

```bash
uv run --group notebook --group docs mkdocs build
```

Check that the command exits with code 0. If it fails:
- Show the error output to the user
- Do NOT proceed to creating the release
- Help debug the issue

If it succeeds, confirm to the user and move on.

## Step 7: Run the notebook check workflow

Trigger the `check_notebooks.yml` workflow on the current branch to verify all notebooks execute successfully:

```bash
gh workflow run "Check notebooks" --ref $(git branch --show-current)
```

Then monitor the run:

```bash
# Wait a few seconds for the run to register, then find it
gh run list --workflow="Check notebooks" --limit 1 --json databaseId,status,conclusion
```

Use `gh run watch <RUN_ID>` to stream the status, or poll with `gh run view <RUN_ID>` periodically.

- If the workflow **succeeds**, confirm to the user and proceed to the draft release.
- If the workflow **fails**, show the user the failure details using `gh run view <RUN_ID> --log-failed` and help debug. Do NOT proceed to creating the release until notebooks pass.

## Step 8: Create a draft GitHub release

Run:

```bash
gh release create v{VERSION} --draft --title "v{VERSION}" --generate-notes --target main
```

The `--draft` flag ensures nothing is published (and therefore no CI release pipeline is triggered).
The `--generate-notes` flag uses GitHub's auto-generated release notes from merged PRs.

## Step 9: Report

Tell the user:
- The draft release URL (from the `gh release create` output)
- A summary of what was done: changelog updated, banner updated, docs build verified, draft release created
- Remind them that publishing the release will trigger the full CI pipeline (tests, PyPI publish, docs deploy)
