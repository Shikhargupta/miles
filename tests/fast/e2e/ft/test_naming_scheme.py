import re
from pathlib import Path

import pytest

import tests.e2e.ft
from tests.e2e.ft.conftest_ft.modes import KILL_SEGMENT_OF_FT_COMPONENTS, MODES, compute_mode_name

ENTRY_DIR = Path(tests.e2e.ft.__file__).parent
SCENARIO_DIR = ENTRY_DIR / "conftest_ft"

ENTRY_PATHS = sorted(ENTRY_DIR.glob("test_*.py"))
SCENARIO_PATHS = sorted(SCENARIO_DIR.glob("scenario_*.py"))
KILL_SEGMENTS = set(KILL_SEGMENT_OF_FT_COMPONENTS.values())


class TestTheSuiteIsActuallyThere:
    def test_every_parametrized_check_below_has_files_to_run_against(self):
        """A glob that matches nothing turns this whole module into a set of vacuous passes."""
        assert len(ENTRY_PATHS) >= 15, ENTRY_PATHS
        assert len(SCENARIO_PATHS) >= 8, SCENARIO_PATHS
        assert len(MODES) >= 7, sorted(MODES)


class TestModeNames:
    @pytest.mark.parametrize("name", sorted(MODES))
    def test_a_mode_name_is_reproduced_by_the_fields_it_claims_to_describe(self, name: str):
        """A name that drifts from the fields sends a reader to the wrong topology, and nothing else would catch it."""
        assert compute_mode_name(MODES[name]) == name

    def test_the_modes_that_differ_in_nothing_but_what_they_crash_still_have_distinct_names(self):
        """The kill segment is the only thing separating two of these, so dropping it would silently merge them."""
        assert MODES["kill_train__dp2_cp2"].ft_components == ("train",)
        assert MODES["kill_train_rollout__dp2_cp2"].ft_components == ("train", "rollout")


class TestScenarioNames:
    @pytest.mark.parametrize("path", SCENARIO_PATHS, ids=lambda path: path.stem)
    def test_a_scenario_carries_no_ft_segment(self, path: Path):
        """The whole package is tests/e2e/ft, so an ft segment costs characters and says nothing."""
        assert "ft" not in path.stem.split("_")

    @pytest.mark.parametrize("path", SCENARIO_PATHS, ids=lambda path: path.stem)
    def test_a_scenario_name_holds_no_double_underscore(self, path: Path):
        """Entries split on the first one to recover the scenario, so a scenario containing one breaks that."""
        assert "__" not in path.stem

    @pytest.mark.parametrize("path", SCENARIO_PATHS, ids=lambda path: path.stem)
    def test_the_test_name_is_the_scenario_name_without_its_prefix(self, path: Path):
        """TEST_NAME reaches dump directories and wandb runs, so a second spelling splits one soak's history in two."""
        if (declared := _declared(path, "TEST_NAME")) is None:
            pytest.skip(f"{path.stem} delegates to another scenario and declares no TEST_NAME")
        assert declared == path.stem.removeprefix("scenario_")


class TestEntryNames:
    @pytest.mark.parametrize("path", ENTRY_PATHS, ids=lambda path: path.stem)
    def test_an_entry_names_a_scenario_that_exists(self, path: Path):
        """The name is the only index from a red CI job back to the code that ran."""
        assert _scenario_of(path) in {_test_name_of(one) for one in SCENARIO_PATHS}

    @pytest.mark.parametrize("path", ENTRY_PATHS, ids=lambda path: path.stem)
    def test_an_entry_runs_the_scenario_it_is_named_after(self, path: Path):
        """A name pointing at one scenario while importing another is a lie no reader can see through."""
        imported = re.search(r"from tests\.e2e\.ft\.conftest_ft\.(scenario_\w+) import run_ci", path.read_text())
        assert imported, f"{path.name} imports no run_ci"
        assert _test_name_of(SCENARIO_DIR / f"{imported.group(1)}.py") == _scenario_of(path)

    @pytest.mark.parametrize("path", ENTRY_PATHS, ids=lambda path: path.stem)
    def test_an_entry_says_what_its_run_crashes(self, path: Path):
        """Whether a soak kills trainers, engines or both is the first thing a reader needs and the easiest to lose."""
        assert _rest_of(path).split("__")[0] in KILL_SEGMENTS

    @pytest.mark.parametrize("path", ENTRY_PATHS, ids=lambda path: path.stem)
    def test_an_entry_with_no_mode_says_what_its_own_scenario_crashes(self, path: Path):
        """Nothing else pins these: their scenario fixes its own ft components, so the name can drift unnoticed."""
        rest = _rest_of(path)
        if rest not in KILL_SEGMENTS:
            pytest.skip(f"{path.stem} names a mode, whose own name already carries the kill segment")
        declared = _ft_components_of(SCENARIO_DIR / f"{_scenario_of(path)}.py")
        assert declared is not None, f"{path.name} names no mode, so its scenario must declare _FT_COMPONENTS"
        assert rest == KILL_SEGMENT_OF_FT_COMPONENTS[declared]

    @pytest.mark.parametrize("path", ENTRY_PATHS, ids=lambda path: path.stem)
    def test_an_entry_naming_a_mode_runs_that_mode(self, path: Path):
        """Renaming an entry without repointing _MODE would leave it silently running a neighbouring topology."""
        rest = _rest_of(path)
        declared = _declared(path, "_MODE")
        if rest in KILL_SEGMENTS:
            assert declared is None, f"{path.name} names no mode but declares _MODE={declared!r}"
            return
        assert declared == rest
        assert declared in MODES


def _scenario_of(path: Path) -> str:
    return path.stem.removeprefix("test_").partition("__")[0]


def _rest_of(path: Path) -> str:
    scenario, separator, rest = path.stem.removeprefix("test_").partition("__")
    assert separator, f"{path.name} carries nothing after its scenario name"
    return rest


def _test_name_of(path: Path) -> str:
    return _declared(path, "TEST_NAME") or path.stem.removeprefix("scenario_")


def _ft_components_of(path: Path, *, depth: int = 3) -> tuple[str, ...] | None:
    text = path.read_text()
    if (found := re.search(r"^_FT_COMPONENTS: tuple\[str, \.\.\.\] = \(([^)]*)\)", text, re.M)) is not None:
        return tuple(re.findall(r'"([^"]+)"', found.group(1)))
    if depth <= 0:
        return None
    delegated = re.search(
        r"^from tests\.e2e\.ft\.conftest_ft(?:\.(scenario_\w+) import|\s+import (scenario_\w+))", text, re.M
    )
    if delegated is None:
        return None
    return _ft_components_of(SCENARIO_DIR / f"{delegated.group(1) or delegated.group(2)}.py", depth=depth - 1)


def _declared(path: Path, name: str) -> str | None:
    found = re.search(rf'^{name}: str = "([^"]+)"', path.read_text(), re.M)
    return found.group(1) if found else None
