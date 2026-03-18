---
name: user_audio_knowledge
description: User's audio production background and domain knowledge — informs how to explain technical decisions
type: user
---

User has strong practical audio production knowledge:
- Knows ADSR parameters and their physical meaning
- Understands piano mechanics at a detailed level (hammer strike, escapement mechanism, damper behavior on release)
- Understands articulation vocabulary (staccato, legato, pizzicato, marcato, sforzando, con sordino)
- Understands timecode formats (SMPTE, LTC, MTC) and their post-production context
- Aware of game audio middleware (FMOD, Wwise) and show control tools (QLab)
- Familiar with synthesis concepts (FM synthesis, modulation index, oscillator types)
- Knows NVIDIA's audio research ecosystem (AF3, ETTA, Fugatto)

User thinks in terms of real-world sound behavior and production workflows. Technical explanations land better when grounded in acoustic/musical examples rather than abstract ML concepts. When explaining ML architecture decisions, use audio production analogies (e.g., the "experienced audio engineer" analogy for frozen backbone).

User asks good skeptical questions — "do we really need that?" (e.g., challenged human annotation requirement, which led to a cleaner pipeline). Engage with these seriously rather than defending the original plan.
