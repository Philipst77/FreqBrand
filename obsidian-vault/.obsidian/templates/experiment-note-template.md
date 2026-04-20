<%*
// Templater experiment-note template
// Usage: create from /new-exp slash command (which fills in exp_name, phase, date)
// or manually via Obsidian Templater.
const exp_name = await tp.system.prompt("Experiment name (e.g., phase0_residuals_bm3d)");
const phase = await tp.system.suggester(["0", "1", "2", "3A", "3B", "4", "baseline", "other"], ["0", "1", "2", "3A", "3B", "4", "baseline", "other"]);
await tp.file.rename(exp_name);
-%>
---
exp_name: <% exp_name %>
phase: <% phase %>
status: planning
started: <% tp.date.now("YYYY-MM-DD") %>
ended:
---

# <% exp_name %>

**Phase**: <% phase %>
**Started**: <% tp.date.now("YYYY-MM-DD") %>
**Status**: planning | running | analyzing | done | abandoned

Linked project dir: `[[../experiments/<% exp_name %>/README.md]]`

## Hypothesis

<!-- What do I expect to see, and why? -->

## Variables

- **Independent** (what I'm varying):
- **Dependent** (what I'm measuring):
- **Controlled** (what I'm holding fixed):

## Success criteria

<!-- Concrete numbers / plots / conditions that would count as "this worked". -->
- [ ]
- [ ]

## Abandon criteria

<!-- Concrete conditions under which I kill this experiment. -->
- [ ]

## Procedure

1.
2.
3.

## Runs

### Run 1 — <% tp.date.now("YYYY-MM-DD") %>

- **Command**:
- **Config**:
- **SLURM Job ID**:
- **Log path**:
- **Result**:
- **Notes**:

## Findings

<!-- Fill in as data comes back. -->

## Conclusion

<!-- One paragraph. What did we learn? What's the next experiment? -->

## Links

- Slash commands used: [[]]
- Related papers: [[]]
- Related concepts: [[]]
- Follow-up experiments: [[]]
