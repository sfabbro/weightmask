import re
from pathlib import Path

import tomllib

REPO = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((REPO / "pyproject.toml").read_text())
WORKFLOW = (REPO / ".github" / "workflows" / "ci.yml").read_text()
LOCK = (REPO / "pixi.lock").read_text()


def _workflow_config():
    import yaml

    return yaml.safe_load(WORKFLOW)


CI = _workflow_config()

MINIMUMS = {
    "numpy": "1.25",
    "astropy": "5.0",
    "fitsio": "1.3.0",
    "scipy": "1.9",
    "scikit-image": "0.20",
    "sep": "1.2",
    "PyYAML": "6.0",
    "astroscrappy": "1.2",
}


def test_project_metadata_declares_tested_python_and_linux_support():
    project = PYPROJECT["project"]
    assert project["requires-python"] == ">=3.10"
    classifiers = project["classifiers"]
    assert all(
        f"Programming Language :: Python :: {version}" in classifiers for version in ("3.10", "3.11", "3.12", "3.13")
    )
    assert "Operating System :: POSIX :: Linux" in classifiers
    assert "Operating System :: OS Independent" not in classifiers


def test_runtime_dependencies_have_evidence_backed_floors_without_caps():
    dependencies = {re.split(r"[<>=!~]", spec, maxsplit=1)[0]: spec for spec in PYPROJECT["project"]["dependencies"]}
    assert set(dependencies) == set(MINIMUMS)
    for name, minimum in MINIMUMS.items():
        assert dependencies[name] == f"{name}>={minimum}"
        assert "<" not in dependencies[name]


def test_torchfits_is_an_optional_dependency_extra():
    assert PYPROJECT["project"]["optional-dependencies"]["torchfits"] == ["torchfits>=1.1.3"]


def test_lock_records_metadata_without_a_full_dependency_resolve():
    locked_names = {"PyYAML": "pyyaml"}
    for name, minimum in MINIMUMS.items():
        locked_name = locked_names.get(name, name)
        assert f"{locked_name}>={minimum}" in LOCK
    assert "torchfits>=1.1.3" in LOCK


def test_minimum_dependency_job_is_present_and_pins_every_declared_floor():
    minimum_job = CI["jobs"]["minimum-dependencies"]
    install = next(step["run"] for step in minimum_job["steps"] if "numpy==" in step.get("run", ""))
    tests = next(
        step["run"] for step in minimum_job["steps"] if "tests/test_minimum_dependencies.py" in step.get("run", "")
    )
    assert "python -m pytest tests/test_minimum_dependencies.py tests/test_contract.py tests/test_utils.py -q" in tests
    for name, minimum in MINIMUMS.items():
        assert f'"{name}=={minimum}' in install
    assert '"pytest==' in install


def test_torchfits_job_installs_pytest_and_runs_real_test():
    torchfits_job = CI["jobs"]["torchfits"]
    commands = [step.get("run", "") for step in torchfits_job["steps"]]
    assert any(command == 'python -m pip install ".[torchfits]"' for command in commands)
    assert any(command == "python -m pip install pytest" for command in commands)
    assert any(command == "python -m pytest tests/test_contract.py -k real_torchfits -q" for command in commands)


def test_minimum_dependency_job_is_not_an_import_only_smoke():
    minimum_job = CI["jobs"]["minimum-dependencies"]
    commands = [step.get("run", "") for step in minimum_job["steps"]]
    assert any("tests/test_minimum_dependencies.py" in command for command in commands)
    assert any("fitsio" in command for command in commands)
    assert "uint64" in (REPO / "tests" / "test_minimum_dependencies.py").read_text()


def test_package_smoke_job_runs_on_linux():
    assert CI["jobs"]["package-smoke"]["runs-on"] == "ubuntu-latest"


def test_package_smoke_job_has_python_matrix():
    matrix = CI["jobs"]["package-smoke"]["strategy"]["matrix"]["python-version"]
    assert matrix == ["3.10", "3.11", "3.12", "3.13"]


def test_package_smoke_job_setup_uses_matrix_python():
    steps = CI["jobs"]["package-smoke"]["steps"]
    setup = next(step for step in steps if step.get("uses", "").startswith("actions/setup-python@"))
    assert setup["with"]["python-version"] == "${{ matrix.python-version }}"


def test_package_smoke_job_installs_and_checks_package():
    steps = CI["jobs"]["package-smoke"]["steps"]
    commands = [step.get("run", "") for step in steps]
    assert any(command == "python -m pip install ." for command in commands)
    assert any(command == "python -m pip check" for command in commands)


def test_ci_runs_real_torchfits_metadata_round_trip():
    steps = CI["jobs"]["torchfits"]["steps"]
    commands = [step.get("run", "") for step in steps]
    assert any("torchfits_available" in command for command in commands)


def test_package_smoke_runs_installed_package_outside_checkout():
    steps = CI["jobs"]["package-smoke"]["steps"]
    smoke = next(step for step in steps if "metadata.version" in step.get("run", ""))
    assert smoke["working-directory"] == "/tmp"
