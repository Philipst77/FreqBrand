# FreqBrand Obsidian Vault

Personal knowledge management for the FreqBrand project. Notes, literature, concept definitions, daily logs, experiment tracking.

## Folder structure

```
obsidian-vault/
├── papers/           ← one note per paper, summary + critique + how it relates
├── concepts/         ← one note per concept (SVD, Tracy-Widom, PRNU, etc.)
├── methodology/      ← our decisions, design docs, threat model writeups
├── experiments/      ← one note per experiment (auto-created by /new-exp)
├── daily/            ← daily log notes (YYYY-MM-DD.md)
├── team/             ← notes from/about Sina and Philip, meeting notes
├── existing-work/    ← imported notes from pre-pivot DCT + CNN work
└── .obsidian/
    ├── templates/    ← Templater-compatible templates
    └── (config)      ← manually configured: plugins, hotkeys
```

## Required plugins (install manually)

- **Templater** — for the note templates in `.obsidian/templates/`
- **Dataview** — for the literature-review query on `papers/`
- **Linter** — for consistent formatting

Cowork cannot install these. After Stage A deployment, open Obsidian, point it at `~/freqbrand/obsidian-vault/`, enable community plugins, install the three above.

## Naming conventions

- **Papers**: `<firstauthor><year>_<shorttitle>.md`
  - Example: `jang2025_silent-branding-attack.md`
- **Concepts**: `<concept-kebab-case>.md`
  - Example: `tracy-widom-distribution.md`
- **Experiments**: `<exp_name>.md` (matches `experiments/<exp_name>/` in the project root)
- **Daily**: `YYYY-MM-DD.md`
- **Team**: `<person>_<topic>_<date>.md` — e.g., `sina_rmt-briefing_2026-04.md`

## Links

Use Obsidian-style `[[wikilinks]]` to cross-reference notes. Use relative markdown links to jump out of the vault to the project files:
- `[methodology.md](../.claude/context/methodology.md)`
- `[experiment README](../experiments/phase0_residuals_bm3d/README.md)`

## Dataview queries

Example for a literature review table — put this in a `papers/_INDEX.md` file:

````
```dataview
TABLE year, venue, status, relevance
FROM "papers"
SORT year DESC
```
````

Requires each paper note to have frontmatter with those fields. The paper-note-template enforces this.
