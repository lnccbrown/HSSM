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

## Steps

1. Fork the project repository by clicking on the ‘Fork’ button near the top right of the main repository page. This creates a copy of the code under your GitHub user account.

2. Clone your fork of the HSSM repo from your GitHub account to your local disk.

`SSH`
   
 ```
 git clone git@github.com:<your GitHub handle>/lnccbrown/hssm.git
 ```

`HTTPS`
   
 ```
 git clone https://github.com/<your GitHub handle>/lnccbrown/hssm.git
 ```

3. Navigate to your hssm directory and add the base repository as a remote:

`SSH`

 ```
 cd hssm
 git remote add upstream git@github.com:lnccbrown/hssm.git
 ```

`HTTPS`
   
 ```
 cd hssm
 git remote add upstream https://github.com/lnccbrown/hssm.git
 ```

4. Create a feature branch to hold your development changes:

```
git checkout -b my-feature
```


> Warning: Always create a new feature branch before making any changes. Make your changes in the feature branch. It’s good practice to never routinely work on the main branch of any repository.

5. The project uses `poetry` for dependency management. Poetry needs to be installed in its own environment. Visit the official poetry documentation at https://python-poetry.org/docs/. Follow the installation instructions provided on the documentation page to install poetry on your system. This typically involves running a specific command based on your operating system.

6. Develop the feature on your feature branch. Add your changes using git commands, git add and then git commit, like:

```
git add modified_files
git commit -m "commit message here"
```

To record your changes locally. After committing, it is a good idea to sync with the base repository in case there have been any changes:

```
git fetch upstream
git rebase upstream/main
```
7. Then push the changes to your GitHub account with:

```
git push -u origin my-feature
```

8. Go to the GitHub web page of your fork of the HSSM repo. Click the ‘Pull request’ button to send your changes to the project’s maintainers for review. This will send an email to the committers.

** This guide is adapted from the [ArviZ contribution guide](https://github.com/arviz-devs/arviz/blob/main/CONTRIBUTING.md) **

