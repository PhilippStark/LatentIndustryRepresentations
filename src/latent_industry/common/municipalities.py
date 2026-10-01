"""Year-sensitive harmonization of Swedish municipality codes.

The canonical universe is the 290-code list in
``metadata/mapping_Municipality.json``. Historical code changes are defined in
``metadata/municipality_crosswalk.json`` and validated when loaded.

The crosswalk intentionally does not backfill municipalities that were formed
during the study period. For example, Knivsta is inactive before 2003. Pure
code changes, such as 1917 -> 0331 for Heby, preserve the municipality identity
across time.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

from .paths import METADATA_DIR


DEFAULT_CROSSWALK_PATH = METADATA_DIR / "municipality_crosswalk.json"
DEFAULT_TARGET_PATH = METADATA_DIR / "mapping_Municipality.json"


class MunicipalityCrosswalkError(ValueError):
    """Raised when a municipality code or crosswalk definition is invalid."""


@dataclass(frozen=True)
class MunicipalityAlias:
    source_code: str
    canonical_code: str
    name: str
    valid_from: int
    valid_to: int
    effective_date: str
    change_type: str
    source: str

    def applies(self, year: int) -> bool:
        return self.valid_from <= year <= self.valid_to


@dataclass(frozen=True)
class HarmonizedMunicipality:
    source_code: str | None
    canonical_code: str | None
    status: str


def normalize_municipality_code(value: object) -> str | None:
    """Normalize an integer/string municipality code to four characters."""
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if not value.is_integer():
            raise MunicipalityCrosswalkError(
                f"Municipality code must be an integer value, got {value!r}."
            )
        value = int(value)

    text = str(value).strip()
    if not text:
        return None
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if not text.isdigit():
        raise MunicipalityCrosswalkError(
            f"Municipality code must contain only digits, got {value!r}."
        )
    if len(text) > 4:
        raise MunicipalityCrosswalkError(
            f"Municipality code must have at most four digits, got {value!r}."
        )
    return text.zfill(4)


class MunicipalityCrosswalk:
    """Validated year-sensitive municipality-code crosswalk."""

    def __init__(
        self,
        *,
        target_names: Mapping[str, str],
        identity_start_years: Mapping[str, int],
        aliases: Iterable[MunicipalityAlias],
        invalid_source_codes: Iterable[str],
        study_year_start: int,
        study_year_end: int,
        source_year_reference_offset: int,
        expected_target_count: int,
    ) -> None:
        self.target_names = dict(target_names)
        self.target_codes = frozenset(self.target_names)
        self.identity_start_years = {
            code: int(identity_start_years.get(code, study_year_start))
            for code in self.target_codes
        }
        self.aliases = tuple(aliases)
        self.invalid_source_codes = frozenset(invalid_source_codes)
        self.study_year_start = int(study_year_start)
        self.study_year_end = int(study_year_end)
        self.source_year_reference_offset = int(source_year_reference_offset)
        self.expected_target_count = int(expected_target_count)

        by_source: dict[str, list[MunicipalityAlias]] = {}
        for rule in self.aliases:
            by_source.setdefault(rule.source_code, []).append(rule)
        self._aliases_by_source = {
            code: tuple(sorted(rules, key=lambda rule: rule.valid_from))
            for code, rules in by_source.items()
        }
        self._validate()

    @classmethod
    def load(
        cls,
        crosswalk_path: str | Path = DEFAULT_CROSSWALK_PATH,
        target_path: str | Path = DEFAULT_TARGET_PATH,
    ) -> "MunicipalityCrosswalk":
        crosswalk_path = Path(crosswalk_path)
        target_path = Path(target_path)
        with crosswalk_path.open("r", encoding="utf-8") as handle:
            definition = json.load(handle)
        with target_path.open("r", encoding="utf-8") as handle:
            target_names = json.load(handle)

        start_year = int(definition["study_year_start"])
        identity_start_years = {
            code: int(value["year"])
            for code, value in definition.get(
                "identity_start_year_overrides", {}
            ).items()
        }
        aliases = [MunicipalityAlias(**rule) for rule in definition["aliases"]]
        return cls(
            target_names=target_names,
            identity_start_years=identity_start_years,
            aliases=aliases,
            invalid_source_codes=definition["invalid_source_codes"],
            study_year_start=start_year,
            study_year_end=int(definition["study_year_end"]),
            source_year_reference_offset=int(
                definition["year_semantics"]["source_year_reference_offset"]
            ),
            expected_target_count=int(definition["target"]["expected_code_count"]),
        )

    def _validate(self) -> None:
        if len(self.target_codes) != self.expected_target_count:
            raise MunicipalityCrosswalkError(
                "Target municipality list has "
                f"{len(self.target_codes)} codes; expected {self.expected_target_count}."
            )

        bad_target_codes = sorted(
            code for code in self.target_codes if normalize_municipality_code(code) != code
        )
        if bad_target_codes:
            raise MunicipalityCrosswalkError(
                f"Non-canonical target codes: {bad_target_codes}."
            )

        bad_start_codes = sorted(set(self.identity_start_years) - self.target_codes)
        if bad_start_codes:
            raise MunicipalityCrosswalkError(
                f"Formation-year codes missing from target list: {bad_start_codes}."
            )

        for rule in self.aliases:
            if normalize_municipality_code(rule.source_code) != rule.source_code:
                raise MunicipalityCrosswalkError(
                    f"Alias source code is not four digits: {rule.source_code!r}."
                )
            if rule.canonical_code not in self.target_codes:
                raise MunicipalityCrosswalkError(
                    f"Alias target {rule.canonical_code} is not in the 290-code list."
                )
            if rule.valid_from > rule.valid_to:
                raise MunicipalityCrosswalkError(
                    f"Invalid year interval for {rule.source_code}."
                )
            if self.identity_start_years[rule.canonical_code] > rule.valid_from:
                raise MunicipalityCrosswalkError(
                    f"Alias {rule.source_code} predates identity "
                    f"{rule.canonical_code}."
                )

        for source_code, rules in self._aliases_by_source.items():
            for left, right in zip(rules, rules[1:]):
                if left.valid_to >= right.valid_from:
                    raise MunicipalityCrosswalkError(
                        f"Overlapping alias rules for {source_code}: "
                        f"{left.valid_from}-{left.valid_to} and "
                        f"{right.valid_from}-{right.valid_to}."
                    )

    def active_target_codes(self, year: int) -> frozenset[str]:
        """Return identities active for the division represented by ``year``."""
        reference_year = self.reference_year(year)
        return frozenset(
            code
            for code, start_year in self.identity_start_years.items()
            if start_year <= reference_year
        )

    def reference_year(self, source_year: int) -> int:
        """Convert a register year to its municipality-division year."""
        return int(source_year) + self.source_year_reference_offset

    def harmonize_with_status(
        self,
        value: object,
        year: int,
        *,
        strict: bool = True,
    ) -> HarmonizedMunicipality:
        """Return canonical code and an explicit mapping status.

        Status is one of ``missing``, ``invalid``, ``inactive``, ``canonical``,
        ``remapped``, or ``unknown``. Unknown codes raise in strict mode.
        """
        year = int(year)
        reference_year = self.reference_year(year)
        source_code = normalize_municipality_code(value)
        if source_code is None:
            return HarmonizedMunicipality(None, None, "missing")
        if source_code in self.invalid_source_codes:
            return HarmonizedMunicipality(source_code, None, "invalid")

        if source_code in self.target_codes:
            if reference_year < self.identity_start_years[source_code]:
                return HarmonizedMunicipality(source_code, None, "inactive")
            return HarmonizedMunicipality(source_code, source_code, "canonical")

        rules = self._aliases_by_source.get(source_code, ())
        for rule in rules:
            if rule.applies(reference_year):
                if reference_year < self.identity_start_years[rule.canonical_code]:
                    return HarmonizedMunicipality(source_code, None, "inactive")
                return HarmonizedMunicipality(
                    source_code, rule.canonical_code, "remapped"
                )

        if strict:
            if rules:
                intervals = ", ".join(
                    f"{rule.valid_from}-{rule.valid_to}" for rule in rules
                )
                raise MunicipalityCrosswalkError(
                    f"Code {source_code} is known, but not valid in register year "
                    f"{year} (division year {reference_year}); "
                    f"defined interval(s): {intervals}."
                )
            raise MunicipalityCrosswalkError(
                f"Unknown municipality code {source_code} in {year}."
            )
        return HarmonizedMunicipality(source_code, None, "unknown")

    def mapping_for_year(
        self,
        year: int,
        *,
        include_invalid: bool = False,
    ) -> dict[str, str | None]:
        """Return a source-to-canonical lookup suitable for one yearly file."""
        year = int(year)
        reference_year = self.reference_year(year)
        mapping: dict[str, str | None] = {
            code: code for code in self.active_target_codes(year)
        }
        for rule in self.aliases:
            if rule.applies(reference_year):
                mapping[rule.source_code] = rule.canonical_code
        if include_invalid:
            mapping.update({code: None for code in self.invalid_source_codes})
        return mapping


def load_municipality_crosswalk() -> MunicipalityCrosswalk:
    """Load and validate the repository-default municipality crosswalk."""
    return MunicipalityCrosswalk.load()


def harmonization_table(crosswalk: MunicipalityCrosswalk):
    """Long (Year, source_code, municipality) polars DataFrame covering every
    study year, for joining against a raw WorkplaceMunicipality-keyed panel.

    Only rows with a resolvable canonical code are included — codes that are
    invalid (0000/9999), not yet formed, or otherwise unknown for a given
    year are simply absent, so an inner join against this table both
    harmonizes historical codes (e.g. the pre-1998 Skane/Vastra Gotaland
    county renumbering) to their canonical 290-code successor AND drops
    junk/unmapped codes in one step.
    """
    import polars as pl

    rows = []
    for year in range(crosswalk.study_year_start, crosswalk.study_year_end + 1):
        mapping = crosswalk.mapping_for_year(year, include_invalid=True)
        for source_code, canonical_code in mapping.items():
            if canonical_code is not None:
                rows.append((year, source_code, canonical_code))
    # Explicit schema: Year as UInt16 matches individ_cleaned_1990-2019.parquet's
    # own Year/WorkplaceMunicipality dtype, so callers can join without an extra cast.
    return pl.DataFrame(
        rows, schema=["Year", "source_code", "municipality"], orient="row"
    ).with_columns(pl.col("Year").cast(pl.UInt16))
