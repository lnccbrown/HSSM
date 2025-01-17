# Contributing to HSSM

We invite contributions to the HSSM project from interested individuals or groups. To ensure that your contributions align with the conventions of the HSSM project and can be integrated as efficiently as possible, we provide the following guidelines.

## Table of Contents
1. [Ways to Contribute](#ways-to-contribute)
2. [Opening Issues](#opening-issues)
3. [Pull Request Step-by-Step](#pull-request-step-by-step)

## Ways to Contribute

There are three main ways you can contribute to [HSSM](https://github.com/lnccbrown/HSSM) (listed in decreasing order of scope):

1. **Expanding the existing codebase** with new features or improvements to existing ones. Use the "Feature Request" label.
2. **Contributing to or improving the project's documentation and examples** (located in `hssm/examples`). Use the "Documentation" label.
3. **Rectifying issues with the existing codebase**. These can range from minor software bugs to more significant design problems. Use the "Bug" label.

## Opening Issues

If you find a bug or encounter any type of problem while using HSSM, please let us know by filing an issue on the [GitHub Issue Tracker](https://github.com/lnccbrown/HSSM/issues) rather than via social media or direct emails to the developers.

Please make sure your issue isn't already being addressed by other issues or pull requests. You can use the [GitHub search tool](https://github.com/lnccbrown/HSSM/issues) to search for keywords in the project issue tracker. Please use appropriate labels for an issue.

## [Pull Request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) Step-by-Step

The preferred workflow for contributing to HSSM is to fork the GitHub repository, clone it to your local machine, and develop on a feature branch.

### Steps

1. [**Fork the project repository**](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) by clicking on the ‘Fork’ button near the top right of the main repository page. This creates a copy of the code under your GitHub user account.

2. [**Clone your fork**](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) of the HSSM repo** from your GitHub account to your local disk.
   ```
   git clone https://github.com/<your GitHub handle>/lnccbrown/hssm.git
   ```

3. **Navigate to your `hssm` directory and add the base repository as a remote**. This sets up a directive to propose your local changes to the `hssm` repository.
   ```
   cd hssm
   git remote add upstream https://github.com/lnccbrown/hssm.git
   ```

4. **Create a [feature branch](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-branches) to hold your changes**:
   ```
   git checkout -b my-feature
   ```

> [!WARNING]
> Routinely making changes in the main branch of a repository should be avoided. Always create a new _feature_ branch before making any changes and make your changes in the feature branch.

5. **Add the new feature/changes on your feature branch**. When finished, [_commit_](https://github.com/git-guides/git-commit) your changes:
   ```
   git add <modified_files>
   git commit -m "commit message here"
   ```

   After committing, it is a good idea to sync with the base repository in case there have been any changes:
   ```
   git fetch upstream
   git rebase upstream/main
   ```

> [!Note]
> If your changes require libraries not included in `hssm`, you'll need to use `uv` to update the dependency files. Please visit the [official `uv` documentation](https://docs.astral.sh/uv/) and follow the installation instructions.
>
> After installing `uv`, you can add the new libraries (dependencies) to [`pyproject.toml`](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) by running:
> ```
> uv add <package-name>
> ```
> Replace `<package-name>` with the name of the library you need to add. This command will update the `pyproject.toml` file and install the new dependency. It will also add changes to the [`uv.lock`](https://docs.astral.sh/uv/guides/projects/#uvlock) file.
>
> Remember to commit the newly changed files.
> ```
> git add pyproject.toml
> git commit -m "Add <package-name> dependency"
> ```

6. **[Push](https://github.com/git-guides/git-push) the changes to your GitHub account** with:
   ```
   git push -u origin my-feature
   ```

7. **Create a Pull Request**:
   - Go to the GitHub web page of your fork of the HSSM repo.
   - Click the ‘Pull request’ button to send your changes to the project’s maintainers for review.

**This guide is adapted from the [ArviZ contribution guide](https://github.com/arviz-devs/arviz/blob/main/CONTRIBUTING.md)**
