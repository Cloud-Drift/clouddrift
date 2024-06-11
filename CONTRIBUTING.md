# Contributing to Clouddrift

Thank you for your interest in contributing! We look forward to seeing your ideas and working with you to improve the `clouddrift` library 😄

It should be noted that this contributing guide took heavy inspiration from the [Awkward Array](https://github.com/scikit-hep/awkward/blob/main/CONTRIBUTING.md) project.

## Code of Conduct
This project follows [NumFOCUS code of conduct](https://numfocus.org/code-of-conduct>). The short version is:

- Be kind to others. Do not insult or put down others. Behave professionally. Remember that harassment and sexist, racist, or exclusionary jokes are not appropriate.
- All communication should be appropriate for a professional audience including people of many different backgrounds. Sexual language and imagery is not appropriate.
- We are dedicated to providing a harassment-free community for everyone, regardless of gender, sexual orientation, gender identity and expression, disability, physical appearance, body size, race, or religion.
- We do not tolerate harassment of community members in any form.

Thank you for helping make this a welcoming, friendly community for all.

### Where to start

The front page for the Clouddrift project is on [clouddrift.org](https://clouddrift.org). This leads directly to some of the motivations behind building the library and a quick summary of the ragged array data structure. On the same web page you can also find links to examples showcasing the ragged array data structure and some of the datasets we transform and make available through the library.

### Reporting issues and requesting feature requests
Running into a bug or having performance issues? fill in a [Bug report](https://github.com/cloud-drift/clouddrift/issues/new?assignees=&labels=bug&projects=&template=bug_report.md&title=%F0%9F%90%9B+%3Cnice+descriptive+title%3E).

Have a feature in mind you'd like to see implemened, refactoring changes you want to suggest or tools you think would help improve the project? Create a [Feature request](https://github.com/cloud-drift/clouddrift/issues/new?assignees=&labels=enhancement&projects=&template=feature_request.md&title=%3Cselect-one%3A+%E2%AD%90+%28feature%29+%7C++%F0%9F%94%8E+%28refactor%2Fdocs%29+%7C+%F0%9F%94%A7+%28tools%29%3E+%3Cdescriptive+title+here%3E).

### Contributing a pull request

Feel free to [open pull requests in GitHub](https://github.com/Cloud-Drift/clouddrift/pulls) from your [forked repo](https://docs.github.com/en/get-started/quickstart/fork-a-repo) when you start working on the problem. We recommend opening the pull request early so that we can see your progress and communicate about it. (Note that you can `git commit --allow-empty` to make an empty commit and start a pull request before you even have new code.)

Please [make the pull request a draft](https://github.blog/2019-02-14-introducing-draft-pull-requests/) to indicate that it is in an incomplete state and shouldn't be merged until you click "ready for review".

### Getting your pull request reviewed

Currently, we have two regular reviewers of pull requests:

  * Kevin Santana ([kevinsantana11](https://github.com/kevinsantana11))
  * Shane Elipot ([selipot](https://github.com/selipot))

You can request a review from one of us or just comment in GitHub that you want a review and we'll see it. Only one review is required to be allowed to merge a pull request. We'll work with you to get it into shape.

If you're waiting for a response and haven't heard in a few days, it's possible that we forgot/got distracted/thought someone else was reviewing it/thought we were waiting on you, rather than you waiting on us—just write another comment to remind us.

### Git practices

Unless you ask us not to, we might commit directly to your pull request as a way of communicating what needs to be changed. That said, most of the commits on a pull request are from a single author: corrections and suggestions are exceptions.

The titles of pull requests (and therefore the merge commit messages) should follow the convention described below:
```
<descriptive emoji> <descriptive title goes here>

examples:

⚡ improve dataset loading time
⭐ include new ibtracs dataset
🐛 x feature doesn't work on windows

Common emojis to use are as follow:

++ version increment
⭐ New / changed feature
❗ Deprecation of a feature
⛔ Removal of feature
🐛 Bugfix
⚡ Performance/memory improvements
🔍 Documentation, refactoring
🔧 Tooling/Build scripts/CI (other non-application changes)
```

Almost all pull requests are merged with the "squash and merge" feature, so details about commit history within a pull request are hidden from the `main` branch's history. Feel free, therefore, to commit with any frequency you're comfortable with.

### VS Code Developer Quickstart
If you utilize VS Code as your primary IDE you can leverage the automation `tasks` we provide. These automation tasks enable a **one-click** experience when downloading dependencies, running pre-commit processes (linting, styling, type checking, uni testing) and building docs, served and inspected locally.

The only pre-requisite is to have `conda` installed on your development machine. For more info on this please visit [anaconda installation guide](https://docs.anaconda.com/free/anaconda/install/)

Its strongly recommended to download and use the `vscode-taskexplorer` [extension](https://marketplace.visualstudio.com/items?itemName=spmeesseman.vscode-taskexplorer) by Scott Meesseman (provides GUI task interaction in the explorer):


### Preparing your environment

1. Get the code

```
git clone https://github.com/cloud-drift/clouddrift
cd clouddrift/
```

2. Install library dependencies

with pip:

```
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

with conda (recommended):

```
conda env create -f environment.yml
conda activate clouddrift
```

### Testing

1. pre-requisite step: [Preparing your environment](#preparing-your-environment)

2. Install testing dependencies

with pip:

```
pip install clouddrift[dev,plotting]
```

with conda:

```
conda install pytest matplotlib cartopy coverage
```

3. Run the test suite:

```
pytest tests/*_tests.py tests/adapters/*_tests.py
```

  a. Run the docstring tests:
  ```
  pytest --doctest-modules clouddrift/ragged.py
  ```

4. Run the test suite (with coverage):
```
coverage run -m pytest tests/*_tests.py tests/adapters/*_tests.py
```

5. Read the report:
```
coverage report
```

### Building locally and installing
This can be useful for understanding how the package is built, testing the process, and can be leveraged for testing
experimental versions of the library from a users perspective.


1. pre-requisite step: [Preparing your environment](#preparing-your-environment)

2. Build the distribution package and install it

with pip:
```
pip install .
```

### Automatic formatting and linting

The Clouddrift project uses the [`ruff`](https://github.com/astral-sh/ruff) tool for formatting the code and linting. We also leverage [`mypy`](https://github.com/python/mypy) for static typing. Please see the section on [Automated Processes](#automated-processes) to learn about how these tools are used prior to accepting pull requests.

1. Install development dependencies

with pip:

```
pip install clouddrift[dev]
```

with conda:

```
conda install ruff mypy
```

2. Install any missing library type stubs:

```
mypy --install-types
```

* To format your code:

```
ruff format clouddrift tests
```

* To Lint your code:

```
ruff check clouddrift tests
```

* To perform static type analysis:

```
mypy --config-file pyproject.toml
```

### Automated Processes

* `pytest` `ruff` and `mypy` are executed as part of the CI process. If any unit tests fail or styling, linting or typing errors are found
the process will fail and will block pull requests from being merged.

### Building documentation locally
This is useful if you want to inspect the documentation that gets generated

* pre-requisite step: [Building locally and installing](#building-locally-and-installing) necessary for sphinx to find class/module references


1. Go into the docs directory:
```
cd docs
```

2. Install the Sphinx documentation generation dependencies:
```
pip install clouddrift[docs]
```

3. Generate the new documentation:
```
make html
```

### Releases

Currently, only one person can deploy releases:

  * Kevin Santana ([kevinsantana11](https://github.com/kevinsantana11))

If you need your merged pull request to be deployed in a release, just ask!

#### `clouddrift` releases
To make an `clouddrift` release you must do it as part of a pull request:

* Be sure to increase the version number in `pyproject.toml` in accordance with the [Semantic Versioning Specification](https://semver.org/)
* Once the PR is merged locally update your local main branch
  * `git checkout main`
  * `git pull`
* Tag the release with the new version number as so: vX.Y.Z (e.g. - v0.32.0, v1.10.0, etc...)
  * `git tag vX.Y.Z` (e.g. - `git tag v0.32.0`)
* Push the tag up (origin here is the remote repository for the `clouddrift` repository of the `Cloud-Drift` organization on GitHub)
  * `git push origin vX.Y.Z` (e.g. - `git push origin v0.32.0`)
* Create a [new release](https://github.com/Cloud-Drift/clouddrift/releases/new)
  * Choose the tag you just pushed up (e.g. - v0.32.0)
  * Hit `Generate release notes`
  * Hit `Publish Release` if you think the release notes are descriptive otherwise you can create a draft to be reviewed by the community.

**Important:** Once you publish the release, an automated process will be triggered creating a new distribution for the release which is then published to PYPI.
Because we also maintain a `conda-forge` package a PR will be made within a few hours of the release on the `clouddrift-feedstock` [package](https://github.com/conda-forge/clouddrift-feedstock) as a `conda-forge` bot has detected a new release on PYPI. Once this is merged in, a job will begin building the package to be merged into the channel.
