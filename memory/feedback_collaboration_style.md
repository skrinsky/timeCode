---
name: feedback_collaboration_style
description: How the user prefers to collaborate — talk through non-obvious decisions before implementing, challenge assumptions
type: feedback
---

Talk through non-obvious design decisions before writing them into the plan. User engages deeply with architectural reasoning and will catch things that don't make sense (e.g., caught that ADSR should influence harmonic content not just amplitude, caught that human annotators may not be necessary).

**Why:** User is building something genuinely novel and wants to understand the reasoning, not just have a plan handed to them.

**How to apply:** For non-obvious design choices, explain the tradeoff before implementing. For obvious fixes (typos, missing fields, wrong file locations), just fix them. When doing a review pass, separate "obvious fixes" from "things to discuss" and let user decide which to tackle first.

When user says "I don't fully understand this but do what you think makes sense" — implement the most defensible technical choice and explain it briefly after in plain language (not ML jargon). Use analogies to audio production concepts.

User challenges plans constructively — treat challenges as improvements to incorporate, not objections to defend against.
