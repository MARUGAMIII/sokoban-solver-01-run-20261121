# Sokoban Solver — room-decomposed push-optimal A*

A program (`main.ts`, Node 22, built-ins only) that solves a whole set of 5
Sokoban levels in one run, in as few pushes as possible.

## How it works

**Room decomposition.** Every level the generator draws is a hallway with
1–4 small rooms off it, and a box pushed out of a room through its door can
never be pushed back — so each room is an independent puzzle and the level's
push-optimum P\* is the sum of the rooms' optima. The solver finds the rooms
by cutting the hallway row's dead squares (squares from which no push can
ever reach a goal) out of the floor and flood-filling what is left.

**Push-optimal A\* per room.** A state is the boxes' squares plus the
player's reachable region (named by its lowest square). The heuristic is the
minimum-assignment sum of grid distances from boxes to goals: one push moves
one box one square, so it is an admissible, consistent lower bound on the
pushes still needed. Search is A\* over pushes, so the first solved state
popped is push-optimal for that room.

**Sound deadlock pruning** (nothing solvable is ever cut):

- *dead squares* — a box standing where no sequence of pushes (ignoring other
  boxes) reaches a goal is lost;
- *frozen 2×2 blocks* of walls/boxes holding a box off its goal;
- *frozen boxes* — a box that cannot be pushed in any direction because every
  side is a wall or another frozen box (fixed point).

**Stitching.** The rooms' move strings are joined by walking the player
between pushes on shortest paths (BFS, fixed direction order), with the other
rooms' boxes standing as walls — on their goals once their room is done.

**Fallbacks.** If a level is no set of rooms, or a room search exhausts its
budget, the solver tries a whole-level push A\*, then a move-based A\* (valid
but not push-optimal), so a level still gets answered instead of nothing.

**Determinism.** Every budget is a count of expanded states, never the wall
clock; there is no randomness and all tie-breaking is by fixed order, so two
executions write byte-identical outputs.

## Results

Practice set (kit): **1,000,000** — all 5 instances solved in exactly P\*
pushes (13, 16, 29, 47, 60), in ~0.3 s for the whole set. The same score on
6 further sets freshly drawn from the kit's own generator (seeds of my own),
each in under 0.5 s — far inside the 60 s wall budget. Two runs of the
practice set produce byte-identical outputs.

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
```
