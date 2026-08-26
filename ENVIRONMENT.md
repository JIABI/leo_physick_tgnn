# Environment

## Supported setup

The root release tools support Python 3.10 or newer and run on CPU. They do not
require CUDA and do not invoke the satellite or UAV training pipelines.

The recommended installation path is:

```bash
./setup.sh
```

The release is a source monorepo, not a standalone root wheel: the semantic
demo imports the bundled satellite and UAV packages. Run the setup command
from a complete repository checkout so all three editable packages are
installed together.

This creates an isolated `.venv` and installs the root package, the base
satellite and UAV runtimes needed by the semantic demonstration, and the
release-test dependencies. It does not install the optional satellite
paper-training extras. Activate it with:

```bash
source .venv/bin/activate
```

On Windows, create a virtual environment and run the equivalent Python
commands manually:

```powershell
py -m venv .venv
.venv\Scripts\python -m pip install -e ".[test]"
.venv\Scripts\python -m pip install -e "code/satellite[test]" -e code/uav
```

The dependency declarations in the three `pyproject.toml` files are
authoritative. The root workflow is intentionally smaller than the full
satellite research environment because its public commands demonstrate the
interface and verify released records without training.

## Running without activation

The provided shell scripts use `.venv/bin/python` when it exists and otherwise
fall back to `python3`:

```bash
./run_demo.sh
./run_tests.sh
./verify_paper_contract.sh
```

Without arguments, `run_tests.sh` executes the root, satellite and UAV test
suites in sequence. Arguments are passed directly to `pytest`, which is useful
for a focused check.

To create the environment with a specific interpreter, select it during
setup, then let the run scripts use the resulting `.venv`:

```bash
PYTHON_BIN=python3.11 ./setup.sh
./run_demo.sh
```

Set `PYTHON_BIN` on a run script only when that interpreter already has the
root, satellite and UAV editable packages installed.

For a controlled offline machine that already provides the complete scientific
Python stack, system packages can be exposed explicitly during environment
creation:

```bash
CFS_USE_SYSTEM_SITE_PACKAGES=1 ./setup.sh
```

This opt-in mode is less isolated and should not be used for the archived
software-environment record.

All three scripts set `CUDA_VISIBLE_DEVICES` to an empty value. They are safe
to run on a CPU-only machine.

## Source data

Download and unpack the version 4 source-data archive from Zenodo, then pass
its unpacked root to:

```bash
./verify_paper_contract.sh ./source_data/ControllerFacingState_SourceData_v4.0.0
```

The verifier reads local files only. It neither downloads external resources
nor changes the source-data directory.

## Full research environments

The platform directories retain their own packaging and documentation:

- `code/satellite/pyproject.toml`
- `code/uav/pyproject.toml`

`setup.sh` installs the base platform packages because the semantic demo calls
their real policy and operator implementations. It does not install
`code/satellite[paper]` or invoke either training pipeline. Full experimental
training is outside the root release workflow; consult the platform
documentation only when the corresponding data and experiment assets are
available.

## Capturing an environment record

For a reusable verification record, save the interpreter and installed
packages alongside the command output:

```bash
python --version
python -m pip freeze
```

The CLI also reports the software version and resolved protocol identity in its
machine-readable output.
