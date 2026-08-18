import re
from dataclasses import dataclass
from pathlib import Path

import pytest

import tests.e2e.deploy
import tests.e2e.ft
from tests.e2e.ft.conftest_ft.modes import KILL_SEGMENT_OF_FT_COMPONENTS
from tests.e2e.ft.conftest_ft.modes import MODES as FT_MODES
from tests.e2e.ft.conftest_ft.modes import compute_mode_name


@dataclass(frozen=True)
class NamingScheme:
    package: str
    entry_dir: Path

    @property
    def scenario_dir(self) -> Path:
        return self.entry_dir / f"conftest_{self.package}"

    @property
    def entry_paths(self) -> list[Path]:
        return sorted(self.entry_dir.glob("test_*.py"))

    @property
    def scenario_paths(self) -> list[Path]:
        return sorted(self.scenario_dir.glob("scenario_*.py"))

    @property
    def run_ci_import_pattern(self) -> str:
        return rf"from tests\.e2e\.{self.package}\.conftest_{self.package}\.(scenario_\w+) import run_ci"


FT_SCHEME = NamingScheme(package="ft", entry_dir=Path(tests.e2e.ft.__file__).parent)
DEPLOY_SCHEME = NamingScheme(package="deploy", entry_dir=Path(tests.e2e.deploy.__file__).parent)
SCHEMES = [FT_SCHEME, DEPLOY_SCHEME]
KILL_SEGMENTS = set(KILL_SEGMENT_OF_FT_COMPONENTS.values())
ENTRIES = [(scheme, path) for scheme in SCHEMES for path in scheme.entry_paths]
SCENARIOS = [(scheme, path) for scheme in SCHEMES for path in scheme.scenario_paths]

FT_NAMES_THAT_HOLD: list[tuple[str, str | None]] = [
    ("test_realistic_gsm8k__kill_train_rollout", None),
    ("test_realistic_gsm8k_fully_async__kill_train_rollout", None),
    ("test_random_crash__kill_train__dp2_cp2__moe_5layer", "kill_train__dp2_cp2__moe_5layer"),
    ("test_random_crash__kill_rollout__dp2_cp2__colocate", "kill_rollout__dp2_cp2__colocate"),
]
FT_NAMES_THAT_DO_NOT: list[tuple[str, str | None]] = [
    ("test_random_crash__kill_train__dp8_cp8", None),
    ("test_random_crash__kill_train__dp8_cp8", "kill_train__dp8_cp8"),
    ("test_random_crash__kill_train__dp2_cp2", "kill_train__dp2_cp2__moe_5layer"),
    ("test_random_crash__kill_train__dp2_cp2__moe_5layer", None),
    ("test_random_crash__crash_everything", None),
    ("test_random_crash__crash_everything", "crash_everything"),
    ("test_random_crash", None),
    ("test_random_crash", "kill_train__dp2_cp2"),
]
DEPLOY_NAMES_THAT_HOLD: list[str] = ["test_split_deterministic", "test_hot_restart_deterministic"]
DEPLOY_NAMES_THAT_DO_NOT: list[str] = ["test_split_deterministic__dp2_cp2", "test_split_deterministic__trainer1"]


def _ids(pairs: list[tuple[NamingScheme, Path]]) -> list[str]:
    return [f"{scheme.package}/{path.stem}" for scheme, path in pairs]


class TestFtModeNames:
    @pytest.mark.parametrize("name", sorted(FT_MODES))
    def test_a_mode_name_is_reproduced_by_the_fields_it_claims_to_describe(self, name: str):
        """A name that drifts from the fields sends a reader to the wrong topology, and nothing else would catch it."""
        assert compute_mode_name(FT_MODES[name]) == name

    def test_the_modes_that_differ_in_nothing_but_what_they_crash_still_have_distinct_names(self):
        """The kill segment is the only thing separating two of these, so dropping it would silently merge them."""
        assert FT_MODES["kill_train__dp2_cp2"].ft_components == ("train",)
        assert FT_MODES["kill_train_rollout__dp2_cp2"].ft_components == ("train", "rollout")


class TestScenarioNames:
    @pytest.mark.parametrize("scheme, path", SCENARIOS, ids=_ids(SCENARIOS))
    def test_a_scenario_carries_no_segment_naming_its_own_package(self, scheme: NamingScheme, path: Path):
        """The whole package is tests/e2e/<package>, so repeating it costs characters and says nothing."""
        assert scheme.package not in path.stem.split("_")

    @pytest.mark.parametrize("scheme, path", SCENARIOS, ids=_ids(SCENARIOS))
    def test_a_scenario_name_holds_no_double_underscore(self, scheme: NamingScheme, path: Path):
        """Entries split on the first one to recover the scenario, so a scenario containing one breaks that."""
        assert "__" not in path.stem

    @pytest.mark.parametrize("scheme, path", SCENARIOS, ids=_ids(SCENARIOS))
    def test_the_test_name_is_the_scenario_name_without_its_prefix(self, scheme: NamingScheme, path: Path):
        """TEST_NAME reaches dump directories and wandb runs, so a second spelling splits one soak's history in two."""
        if (declared := _declared(path, "TEST_NAME")) is None:
            pytest.skip(f"{path.stem} delegates to another scenario and declares no TEST_NAME")
        assert declared == path.stem.removeprefix("scenario_")


class TestEntryNames:
    @pytest.mark.parametrize("scheme, path", ENTRIES, ids=_ids(ENTRIES))
    def test_an_entry_names_a_scenario_that_exists(self, scheme: NamingScheme, path: Path):
        """The name is the only index from a red CI job back to the code that ran."""
        assert _scenario_of(path) in {_test_name_of(one) for one in scheme.scenario_paths}

    @pytest.mark.parametrize("scheme, path", ENTRIES, ids=_ids(ENTRIES))
    def test_an_entry_runs_the_scenario_it_is_named_after(self, scheme: NamingScheme, path: Path):
        """A name pointing at one scenario while importing another is a lie no reader can see through."""
        imported = re.search(scheme.run_ci_import_pattern, path.read_text())
        assert imported, f"{path.name} imports no run_ci"
        assert _test_name_of(scheme.scenario_dir / f"{imported.group(1)}.py") == _scenario_of(path)


class TestFtEntryNames:
    @pytest.mark.parametrize("path", FT_SCHEME.entry_paths, ids=lambda path: path.stem)
    def test_an_entry_says_what_it_crashes_and_names_a_mode_the_suite_declares(self, path: Path):
        """A name is read as the topology that ran, so one naming no mode of the suite describes a run nobody has."""
        assert _compute_ft_entry_violation(path.stem, _declared(path, "_MODE")) is None


class TestDeployEntryNames:
    @pytest.mark.parametrize("path", DEPLOY_SCHEME.entry_paths, ids=lambda path: path.stem)
    def test_an_entry_carries_nothing_but_the_scenario_it_runs(self, path: Path):
        """One scenario is one example here, so a second segment would name a variant that does not exist."""
        assert _compute_deploy_entry_violation(path.stem) is None


class TestTheRulesThemselves:
    @pytest.mark.parametrize("stem, declared", FT_NAMES_THAT_HOLD)
    def test_a_well_formed_ft_name_is_accepted(self, stem: str, declared: str | None):
        """A rule that rejects the shapes the suite already uses would be reverted rather than obeyed."""
        assert _compute_ft_entry_violation(stem, declared) is None

    @pytest.mark.parametrize("stem, declared", FT_NAMES_THAT_DO_NOT)
    def test_a_malformed_ft_name_is_rejected(self, stem: str, declared: str | None):
        """Every rule here passes on the tree as it stands, so only counter-examples show it still forbids anything."""
        assert _compute_ft_entry_violation(stem, declared) is not None

    @pytest.mark.parametrize("stem", DEPLOY_NAMES_THAT_HOLD)
    def test_a_well_formed_deploy_name_is_accepted(self, stem: str):
        """The one shape this suite allows has to stay allowed."""
        assert _compute_deploy_entry_violation(stem) is None

    @pytest.mark.parametrize("stem", DEPLOY_NAMES_THAT_DO_NOT)
    def test_a_malformed_deploy_name_is_rejected(self, stem: str):
        """A deploy entry naming a variant is the drift this suite dropped its mode matrix to avoid."""
        assert _compute_deploy_entry_violation(stem) is not None


def _compute_ft_entry_violation(stem: str, declared_mode: str | None) -> str | None:
    scenario, separator, rest = stem.removeprefix("test_").partition("__")
    if not separator:
        return f"{stem} names the scenario {scenario!r} and carries nothing after it saying what its run crashes"

    if declared_mode is None:
        if rest not in KILL_SEGMENTS:
            return (
                f"{stem} carries {rest!r} after its scenario, which is none of the kill segments "
                f"{sorted(KILL_SEGMENTS)}, and it declares no _MODE that would make it a mode name"
            )
        return None

    if declared_mode != rest:
        return f"{stem} names the mode {rest!r} while declaring _MODE={declared_mode!r}"
    if declared_mode not in FT_MODES:
        return f"{stem} names the mode {declared_mode!r}, which is none of {sorted(FT_MODES)}"
    return None


def _compute_deploy_entry_violation(stem: str) -> str | None:
    if "__" in stem:
        return f"{stem} carries a second segment, but a scenario here is one example and has no variants to name"
    return None


def _scenario_of(path: Path) -> str:
    return path.stem.removeprefix("test_").partition("__")[0]


def _test_name_of(path: Path) -> str:
    return _declared(path, "TEST_NAME") or path.stem.removeprefix("scenario_")


def _declared(path: Path, name: str) -> str | None:
    found = re.search(rf'^{name}: str = "([^"]+)"', path.read_text(), re.M)
    return found.group(1) if found else None
