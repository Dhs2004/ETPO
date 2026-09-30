# Initial skill sources

ALFWorld: copied from https://github.com/ZJU-REAL/SkillZero, commit 1980cd589036ec9f04c64749bbd317a8dd0eef7a, skills/alfworld, Apache-2.0. The task alias pick_and_place_simple was added to the mapping.

WebShop and ScienceWorld: authored for ETPO using the available environment action interfaces. They are generic initial guidance, not SkillZero artifacts, expert trajectories or validated learned policies.

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
SkillNet publishes ScienceWorld skills. The current ETPO WebShop/ScienceWorld
default files remain the originally authored guidance, not those external assets.
