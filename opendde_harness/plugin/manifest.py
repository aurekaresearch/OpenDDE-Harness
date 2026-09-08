"""Plugin manifest schema.

A manifest is a TOML file (``opendde-harness-plugin.toml``) shipped alongside
a plugin's Python package. It declares the plugin's identity,
contribution points, and config schema — everything the registry needs
to know without importing the plugin's code.

The single root table is ``[plugin]``. Contribution arrays are
``[[plugin.contributes.<kind>]]``; scalar contributions (``skills_dirs``,
``tool_description_notes``, ``readiness``) sit directly under
``[plugin.contributes]``. The model accepts unknown extras silently so
future kinds don't break old hosts.

Validation rules worth flagging:

- ``id`` and ``version`` are required (``min_length=1``).
- ``factory`` must look like ``module.path:callable_name`` — checked
  here so a typo fails at startup rather than at first activation.
- Contribution name uniqueness *within a manifest* is enforced; the
  registry separately enforces uniqueness *across* manifests.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# A factory reference is ``module.path:callable``. The regex is
# deliberately loose: any non-empty module-like path, a single colon,
# any non-empty identifier-ish suffix.
_FACTORY_REF_RE = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")


class _ManifestBase(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class MemoryBackendContribution(_ManifestBase):
    """One ``[[plugin.contributes.memory_backends]]`` entry."""

    name: str = Field(min_length=1)
    factory: str = Field(min_length=1)

    @field_validator("factory")
    @classmethod
    def _factory_is_module_path(cls, v: str) -> str:
        if not _FACTORY_REF_RE.match(v):
            raise ValueError(
                f"factory must be 'module.path:callable', got {v!r}",
            )
        return v


class ToolContribution(_ManifestBase):
    """One ``[[plugin.contributes.tools]]`` entry.

    ``factory`` is a ``module.path:callable`` resolving to a
    ``Callable[[PluginContext], Tool]`` — it returns a single
    :class:`~opendde_harness.agent.tools.base.Tool` the host registers into the
    agent's tool set at boot. One tool per entry; a plugin exposing
    several tools lists several entries.
    """

    name: str = Field(min_length=1)
    factory: str = Field(min_length=1)

    @field_validator("factory")
    @classmethod
    def _factory_is_module_path(cls, v: str) -> str:
        if not _FACTORY_REF_RE.match(v):
            raise ValueError(
                f"factory must be 'module.path:callable', got {v!r}",
            )
        return v


PromptSlot = Literal["identity", "scope"]


class PromptSegmentContribution(_ManifestBase):
    """One ``[[plugin.contributes.prompt_segments]]`` entry.

    ``factory`` resolves to a zero-argument callable returning the
    segment text. ``slot`` names where the host places it in the system
    prompt: ``identity`` replaces the one-line self-description,
    ``scope`` adds a block of behavioural guidance after it.
    """

    name: str = Field(min_length=1)
    slot: PromptSlot
    factory: str = Field(min_length=1)

    @field_validator("factory")
    @classmethod
    def _factory_is_module_path(cls, v: str) -> str:
        if not _FACTORY_REF_RE.match(v):
            raise ValueError(
                f"factory must be 'module.path:callable', got {v!r}",
            )
        return v


class Contributes(_ManifestBase):
    """All contribution points for a single manifest.

    The model keeps extra fields silently so future contribution types
    don't break older hosts reading newer manifests.

    - ``skills_dirs`` — directories of ``<skill>/SKILL.md`` bundles,
      relative to the manifest file.
    - ``tool_description_notes`` — ``tool name -> sentence`` appended to a
      host tool's model-facing description.
    - ``readiness`` — ``module.path:callable`` taking the plugin's config
      slice and returning whether the plugin is configured for use.
    """

    memory_backends: list[MemoryBackendContribution] = Field(default_factory=list)
    tools: list[ToolContribution] = Field(default_factory=list)
    prompt_segments: list[PromptSegmentContribution] = Field(default_factory=list)
    skills_dirs: list[str] = Field(default_factory=list)
    tool_description_notes: dict[str, str] = Field(default_factory=dict)
    readiness: str | None = None

    @field_validator("readiness")
    @classmethod
    def _readiness_is_module_path(cls, v: str | None) -> str | None:
        if v is not None and not _FACTORY_REF_RE.match(v):
            raise ValueError(
                f"readiness must be 'module.path:callable', got {v!r}",
            )
        return v


class PluginManifest(_ManifestBase):
    """Parsed ``opendde-harness-plugin.toml``.

    Constructed via :meth:`from_toml_path` / :meth:`from_toml_str`; the
    raw ``__init__`` works too for programmatic tests.
    """

    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    display_name: str | None = None
    opendde_harness: str | None = None  # version constraint (parsed later)
    bundled: bool = False
    enabled_by_default: bool = False
    contributes: Contributes = Field(default_factory=Contributes)
    config_schema: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _contribution_names_unique(self) -> "PluginManifest":
        # Uniqueness is enforced *within each kind*; a backend and a tool
        # may share a name (different slots). The registry separately
        # enforces uniqueness across manifests.
        for kind, items in (
            ("memory_backend", self.contributes.memory_backends),
            ("tool", self.contributes.tools),
            ("prompt_segment", self.contributes.prompt_segments),
        ):
            names = [c.name for c in items]
            if len(names) != len(set(names)):
                dupes = sorted({n for n in names if names.count(n) > 1})
                raise ValueError(
                    f"duplicate {kind} name(s) in manifest {self.id!r}: {dupes}",
                )
        return self

    # ── Constructors ────────────────────────────────────────────────

    @classmethod
    def from_toml_str(cls, data: str) -> "PluginManifest":
        """Parse from a raw TOML string."""
        raw = tomllib.loads(data)
        return cls._from_raw(raw)

    @classmethod
    def from_toml_path(cls, path: Path) -> "PluginManifest":
        """Parse from a file on disk.

        Raises ``FileNotFoundError`` if missing, ``tomllib.TOMLDecodeError``
        on malformed TOML, and ``pydantic.ValidationError`` on schema
        mismatch.
        """
        with path.open("rb") as f:
            raw = tomllib.load(f)
        return cls._from_raw(raw)

    @classmethod
    def _from_raw(cls, raw: dict[str, Any]) -> "PluginManifest":
        # Manifests nest everything under [plugin]. Unwrap before
        # handing to pydantic so the schema talks in plugin-relative
        # fields.
        if "plugin" not in raw:
            raise ValueError(
                "manifest missing top-level [plugin] table",
            )
        return cls.model_validate(raw["plugin"])


__all__ = [
    "Contributes",
    "MemoryBackendContribution",
    "PluginManifest",
    "PromptSegmentContribution",
    "PromptSlot",
    "ToolContribution",
]
