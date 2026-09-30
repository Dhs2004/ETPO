# Initial skill sources

ALFWorld: copied from https://github.com/ZJU-REAL/SkillZero, commit 1980cd589036ec9f04c64749bbd317a8dd0eef7a, skills/alfworld, Apache-2.0. The task alias pick_and_place_simple was added to the mapping.

WebShop: `webshop/general_skills.md` is copied verbatim from SDAR,
commit `308153bb5f7d63cd57f301f432d616a072220e1b`, `skills/webshop/general_skills.md`.
It contains 15 general strategies. License: Apache-2.0, see `LICENSE.SDAR`.
Only the general skill is activated; category-specific routing is not changed.

ScienceWorld: `sciworld/general_skills.md` is a condensed adaptation of eight
SkillNet procedures, commit `358ec824577e8a5aa6857c0eff268e492703606b`.
License: MIT, copyright (c) 2026 ZJUNLP; see `LICENSE.SkillNet`.
Original source files are retained under `sources/skillnet/scienceworld/` for
comparison and attribution. They are not loaded into teacher prompts.
`provenance.json` maps each numbered runtime section to its original file and hash.

Adaptation removes fixed box colors, assumed open containers, unconditional
teleportation/focusing, example answers and external script requirements; it
conditions operations on the current task and available environment actions.
It preserves task parsing, observation, inventory transfer, container inspection,
tool use, measurement, ambiguity resolution and controlled waiting. This is not
a single upstream general_skills.md and is not a verbatim copy of all 61 skills.
See `docs/etpo/SKILL-REPLACEMENT.zh-CN.md` for details.

Sokoban: `skills/sokoban/general_skills.md` is independently written for ETPO's
four-direction, text-only interface using standard Sokoban planning principles
(deadlock avoidance, player reachability, goal assignment, and push ordering).
It is not an initial skill released by SkillZero, SDAR, or AgentOPSD. A public
player-skill reference was inspected at
https://github.com/kingkillery/pi-config/blob/ce4840977d5397f2a952beb3e53bb57d5cb957e2/agent/skills/sokoban-benchmark-player/SKILL.md,
but its text is not vendored: its repository has no discoverable license and its
undo/reset/file-output workflow does not match ETPO. The environment is the
installed MIT-licensed `gym-sokoban==0.0.6`, https://github.com/mpSchrader/gym-sokoban;
see `LICENSE.gym-sokoban` for attribution. No solver answers are injected into the
training policy's input.

Other published WebShop/ScienceWorld skill candidates and exact commits are
recorded in `docs/etpo/SKILL-SOURCE-AUDIT.zh-CN.md`. In particular, SDAR and
AgentOPSD do publish WebShop skills, Skill0.5 publishes WebShop skill JSONs, and
SkillNet publishes ScienceWorld skills. The current ETPO defaults now use SDAR general guidance and the adapted SkillNet
procedures described above.
