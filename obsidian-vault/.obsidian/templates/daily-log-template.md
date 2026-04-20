<%*
// Templater daily-log template
// Usage: create from a daily-note plugin hotkey, or manually.
// Filename should be YYYY-MM-DD.md
await tp.file.rename(tp.date.now("YYYY-MM-DD"));
-%>
---
date: <% tp.date.now("YYYY-MM-DD") %>
day: <% tp.date.now("dddd") %>
---

# <% tp.date.now("YYYY-MM-DD") %> — <% tp.date.now("dddd") %>

## Plan for today

- [ ]
- [ ]
- [ ]

## What actually happened

<!-- Update throughout the day. Past tense. Concrete. -->

## Decisions made

<!-- Anything that changes the project direction, even in small ways. -->

## Problems hit

<!-- Debugging notes, error messages, paths tried. Future-me will thank present-me. -->

## Runs submitted

<!-- SLURM job IDs and what they were doing, so I can trace them later. -->

| Job ID | Experiment | Status at EOD |
|---|---|---|
|  |  |  |

## Tomorrow

- [ ]

## Open threads

<!-- Things I couldn't finish and shouldn't forget. -->
-

## Links

Relevant papers I read: [[]]
Relevant experiments I ran: [[]]
Relevant team discussions: [[]]
