# ScienceWorld general skills — adapted from SkillNet

Apply the following procedures when relevant to the current task. The task's
object names, destinations, conditions, and available actions take precedence.
Use the environment's required <action>...</action> format for one action at a
time. These skills provide procedures, not predetermined task answers.

## 1. Parse the task before acting
Extract the target object type, target room or container, required actions, and
any ordering or measurement constraints. Turn these into a short plan: locate the
needed objects, perform the required operations, and verify the requested result.
If the task explicitly requires focusing on an object, do so at the specified
point. Do not assume every task is a non-living-object classification task.

## 2. Scan a room and retain the observation
Use `look around` when entering a room, locating an object, or checking a changed
state. Parse the room name, visible objects, open-container contents, device
states, and connections. Keep this information for the next decision. Avoid
repeating the scan if the recent observation is sufficient and nothing changed.
Navigate to the room required by the task; use `teleport to ROOM` only when the
current environment allows teleportation, otherwise use its navigation actions.

## 3. Acquire and place objects with precise names
Locate the object and note its exact name and containing location. Use
`pick up OBJECT` when the object is needed in inventory, and
`move OBJECT to CONTAINER` when placing it is required. Read the resulting
observation to confirm transfer. If the transfer fails, check the name, location,
and accessibility before retrying. Use the destination specified by this task;
container colors in examples do not define a universal rule.

## 4. Inspect containers and devices
Use `look at CONTAINER` to verify placement or monitor a process. Read whether the
device is active, whether the door is open, and what objects or substances it
contains. Inspect nested containers and changes in substance state. If a closed
container blocks inspection, open it when supported; do not assume all containers
are already open. Inspection only observes: use the appropriate separate action
to move an object, activate a device, or manipulate its contents.

## 5. Use the appropriate tool on the actual target
Confirm the required tool and target. Pick up a portable tool when the operation
requires it in inventory, then issue `use TOOL on TARGET` with exact names. Read
the feedback to distinguish a measurement from a physical state change. An
intended effect is not evidence that the action succeeded. If the action fails,
check tool availability, target identity, and access before repeating it.

## 6. Measure temperature and check state changes
For temperature tasks, acquire a thermometer, locate the target substance, and
use the thermometer on that substance. Read the measured value from the
observation and compare it with the threshold in the current task. Heating is
needed only if the task requires a temperature or phase change: use a suitable
container and heating device, activate it, then inspect and measure again.
Do not substitute a remembered melting point, fixed threshold, or example box
color for the task's requested decision and observed result.

## 7. Resolve an ambiguous action carefully
If the environment explicitly returns an "Ambiguous request" with numbered
options, inspect the options. When they are functionally equivalent instances
for the task, select the lowest offered valid number. If they refer to different
locations, objects, or outcomes, choose the one matching the task instead of
blindly choosing zero. Return the selected number inside the action tag. Do not
apply this procedure to ordinary choices without an ambiguity prompt.

## 8. Wait only for an active time-dependent process
First confirm that the relevant process is active, such as a running device or a
prepared growth experiment. Use an available waiting action when time must pass;
prefer `wait1` for a fine-grained check and `wait` for a longer interval when those
actions are supported. Inspect the object or room afterward. Stop waiting when
the required state appears and continue the main task; if the state is unchanged,
check the prerequisites before spending the remaining budget on repeated waits.
