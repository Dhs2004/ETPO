# Sokoban planning skills

1. Read the current board before deciding. Track the player's position, every box,
   every goal, and the floor needed to stand behind each planned push. A box on a
   goal and a player on a goal still cover a goal.
2. Plan in pushes, then find the walking route to the required pushing side. A
   geometrically short box route is useless if the player cannot stand behind it.
3. Check the destination before every push. A box in a corner without a goal is
   permanently lost. Also check wall traps and blocked corridors; avoiding corners
   alone does not rule out deadlocks.
4. Preserve access. Do not push a box against a wall unless it can still reach an
   appropriate goal along that wall. Keep space to reach the other sides of boxes.
5. For narrow passages, decide the order of boxes and goals before blocking the
   passage. Fill a deeper goal first when a nearer filled goal would seal access.
6. Match boxes to reachable goals jointly. Do not greedily send every box to its
   nearest goal if that leaves another box without an accessible destination.
7. A box on a goal is not always finished: moving it temporarily may be necessary
   to reopen a route. Only do so when you have a plan to recover that goal.
8. Simulate the next push and the resulting player position. If it loses required
   access or traps a box, choose another plan before taking the irreversible move.
9. Use the latest observed board to verify movement. A blocked move leaves the
   board unchanged; repeating it cannot help. Re-plan from the actual state.
10. There is no pull, undo, reset, or multi-move action in this interface. Emit one
    allowed direction per turn. Finish only when every box occupies a goal.
