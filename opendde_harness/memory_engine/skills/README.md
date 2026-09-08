# OpenDDE Harness Built-in Skills

This directory holds the generic skills shipped with OpenDDE Harness. Each skill is a directory containing a `SKILL.md` file with YAML frontmatter (name, description, metadata) and Markdown instructions for the agent.

## Available Skills

| Skill | Description |
|-------|-------------|
| `weather` | Get current weather and forecasts (wttr.in + Open-Meteo, no API key) |

Domain skills ship with their plugin and are registered through the manifest's `skills_dirs` contribution; the antibody-design skills live under `opendde_harness/plugin/protein_design/skills/`.

## Notes
User-defined skills can be placed under `<workspace>/skills/` or any directory listed in `skill_forge.local_dirs`.
