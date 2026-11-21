// Sokoban solver: room-decomposed push-optimal A*.
//
// Idea (standard Sokoban technique, implemented here from scratch):
//  - Every level the generator draws is a hallway with 1-4 small rooms off it,
//    and a box pushed out of a room can never come back, so each room is an
//    independent puzzle and the level optimum is the sum of the rooms' optima.
//  - Each room is solved push-optimally with A* over pushes. A state is the
//    boxes' squares plus the player's reachable region (named by its lowest
//    square). The heuristic is the minimum-assignment sum of grid distances
//    from boxes to goals: each push moves one box one square, so it is an
//    admissible, consistent lower bound on the pushes still needed.
//  - Deadlock pruning (all sound: they never cut a solvable state):
//      * dead squares: a box standing on a square from which no sequence of
//        pushes (ignoring other boxes) reaches any goal is lost;
//      * frozen 2x2 blocks of walls/boxes holding a box off its goal;
//      * frozen boxes: a box that cannot be pushed in any direction because
//        every side is a wall or another frozen box (fixed point).
//  - The rooms' move strings are stitched by walking the player between
//    pushes on shortest paths (BFS, fixed direction order).
//  - Fallbacks, in order: whole-level push A*, then a move-based A* (valid
//    but not push-optimal) so a level still gets answered instead of nothing.
//  - Determinism: every budget is a node count, never the wall clock; no
//    randomness anywhere; all tie-breaking is by fixed order, so two runs
//    write byte-identical outputs.
//
// Local run: mkdir -p out && MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Work budgets in expanded states (never the clock, so runs are repeatable).
const ROOM_EXPANSIONS = 600_000;
const WHOLE_EXPANSIONS = 400_000;
const MOVE_EXPANSIONS = 150_000;
const GLOBAL_EXPANSIONS = 3_000_000;
const budget = { used: 0, max: GLOBAL_EXPANSIONS };

const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const CH = ['u', 'd', 'l', 'r'];

/** A level in XSB: # wall, space floor, . goal, $ box, * box on goal, @ player. */
function parseLevel(text) {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals = [];
  const boxes = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  // The outermost frame is wall whatever it shows: nothing leaves the board.
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/** Squares from which a box alone can be pushed onto some goal (pull back from the goals). */
function liveSquares(w, wall, goals) {
  const live = new Uint8Array(wall.length);
  const queue = [];
  for (const g of goals) {
    live[g] = 1;
    queue.push(g);
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi];
    for (let k = 0; k < 4; k++) {
      const d = DY[k] * w + DX[k];
      const y = x - d; // the box's square before the push that lands it on x
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/** Grid distance (through open floor, ignoring boxes) from each goal to every square. */
function gridDistances(w, wall, goals) {
  const dists = [];
  for (const g of goals) {
    const d = new Int32Array(wall.length).fill(-1);
    d[g] = 0;
    const q = [g];
    for (let i = 0; i < q.length; i++) {
      const p = q[i];
      for (let k = 0; k < 4; k++) {
        const nq = p + DY[k] * w + DX[k];
        if (!wall[nq] && d[nq] < 0) {
          d[nq] = d[p] + 1;
          q.push(nq);
        }
      }
    }
    dists.push(d);
  }
  return dists;
}

/** All permutations of 0..n-1, in lexicographic order (n <= 6). */
function permutations(n) {
  const out = [];
  const a = Array.from({ length: n }, (_, i) => i);
  const gen = (k) => {
    if (k === n) {
      out.push(a.slice());
      return;
    }
    for (let i = k; i < n; i++) {
      [a[k], a[i]] = [a[i], a[k]];
      gen(k + 1);
      [a[k], a[i]] = [a[i], a[k]];
    }
  };
  gen(0);
  return out;
}

/**
 * Admissible push heuristic: the cheapest assignment of boxes to distinct
 * goals by grid distance. One push moves one box one square, so the sum can
 * drop by at most 1 per push. Infinity when no assignment reaches every goal.
 */
function heuristic(boxes, dists, perms) {
  const n = boxes.length;
  if (perms === null) {
    let s = 0;
    for (const b of boxes) {
      let m = Infinity;
      for (const d of dists) if (d[b] >= 0 && d[b] < m) m = d[b];
      if (m === Infinity) return Infinity;
      s += m;
    }
    return s;
  }
  let best = Infinity;
  for (const perm of perms) {
    let s = 0;
    for (let i = 0; i < n; i++) {
      const d = dists[perm[i]][boxes[i]];
      if (d < 0) {
        s = Infinity;
        break;
      }
      s += d;
    }
    if (s < best) best = s;
  }
  return best;
}

/** Flood fill from `from` over floor squares free of boxes; returns the lowest square reached. */
function flood(w, wall, occ, from, seen, stamp, stack) {
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  let low = from;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    let q = p - w;
    if (!wall[q] && !occ[q] && seen[q] !== stamp) { seen[q] = stamp; stack[n++] = q; }
    q = p + w;
    if (!wall[q] && !occ[q] && seen[q] !== stamp) { seen[q] = stamp; stack[n++] = q; }
    q = p - 1;
    if (!wall[q] && !occ[q] && seen[q] !== stamp) { seen[q] = stamp; stack[n++] = q; }
    q = p + 1;
    if (!wall[q] && !occ[q] && seen[q] !== stamp) { seen[q] = stamp; stack[n++] = q; }
  }
  return low;
}

/** True when the box just pushed to `t` sits in a frozen 2x2 block of walls/boxes. */
function frozenBlock(w, wall, occ, goalSet, t) {
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occ[a]) || !(wall[b] || occ[b]) || !(wall[c] || occ[c]) || !(wall[d] || occ[d])) continue;
    if ((occ[a] && !goalSet[a]) || (occ[b] && !goalSet[b]) || (occ[c] && !goalSet[c]) || (occ[d] && !goalSet[d])) return true;
  }
  return false;
}

/**
 * True when some box off its goal can never be pushed again: a box is frozen
 * when no direction offers both a free target square and a square for the
 * player to stand on, counting only walls and already-frozen boxes as
 * obstacles (a movable box in the way can always move off first).
 */
function anyFrozen(w, wall, boxes, goalSet) {
  const frozen = new Uint8Array(wall.length);
  let changed = true;
  while (changed) {
    changed = false;
    for (const b of boxes) {
      if (goalSet[b] || frozen[b]) continue;
      let can = false;
      for (let k = 0; k < 4; k++) {
        const d = DY[k] * w + DX[k];
        const t = b + d;
        const s = b - d;
        if (!wall[t] && !frozen[t] && !wall[s] && !frozen[s]) {
          can = true;
          break;
        }
      }
      if (!can) {
        frozen[b] = 1;
        changed = true;
      }
    }
  }
  for (const b of boxes) if (!goalSet[b] && frozen[b]) return true;
  return false;
}

/** Deterministic binary heap ordered by (f, -g, sequence). */
class Heap {
  constructor() {
    this.a = [];
  }
  get size() {
    return this.a.length;
  }
  push(item) {
    const a = this.a;
    a.push(item);
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (Heap.less(a[p], a[i])) break;
      [a[i], a[p]] = [a[p], a[i]];
      i = p;
    }
  }
  pop() {
    const a = this.a;
    const top = a[0];
    const last = a.pop();
    if (a.length > 0) {
      a[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= a.length) break;
        if (c + 1 < a.length && Heap.less(a[c + 1], a[c])) c++;
        if (Heap.less(a[i], a[c])) break;
        [a[i], a[c]] = [a[c], a[i]];
        i = c;
      }
    }
    return top;
  }
  static less(x, y) {
    return x[0] < y[0] || (x[0] === y[0] && (x[1] > y[1] || (x[1] === y[1] && x[2] < y[2])));
  }
}

/** The shortest walk from `from` to `to` around walls and boxes (lower-case LURD). */
function walkTo(w, wall, occ, from, to) {
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Int8Array(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (let k = 0; k < 4; k++) {
      const q = p + DY[k] * w + DX[k];
      if (wall[q] || occ[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = k;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: target unreachable');
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = CH[how[p]] + s;
  return s;
}

/** The square the player stands on after `moves` from `from` on a board `w` wide. */
function endOf(w, from, moves) {
  let p = from;
  for (const ch of moves) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

/**
 * Push-optimal solution of `level` by A* over pushes, or null when the
 * expansion budget runs out (or the level is unsolvable). Returns
 * { pushes, moves } with moves in lower-case LURD.
 */
function solvePush(level, maxExp) {
  const { w, wall, goals, boxes, player } = level;
  const size = wall.length;
  const n = boxes.length;
  const goalSet = new Uint8Array(size);
  for (const g of goals) goalSet[g] = 1;
  const solved = (bs) => bs.every((b) => goalSet[b]);
  if (solved(boxes)) return { pushes: 0, moves: '' };
  const live = liveSquares(w, wall, goals);
  for (const b of boxes) if (!live[b]) return null;
  const dists = gridDistances(w, wall, goals);
  const perms = n <= 6 ? permutations(n) : null;
  const occ = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  // State key: the region's lowest square, then the boxes, as digits in base size.
  // Exact while size ** (n + 1) stays under 2 ** 53; rooms keep n <= 4, and the
  // whole-level fallback uses string keys instead.
  const useNumberKey = Math.pow(size, n + 1) < Number.MAX_SAFE_INTEGER;
  const keyOf = (bs, low) => {
    if (!useNumberKey) return low + ':' + bs.join(',');
    let k = low;
    for (const b of bs) k = k * size + b;
    return k;
  };

  const stBoxes = [boxes.slice()];
  const stPlayer = [player];
  const stParent = [-1];
  const stFrom = [-1]; // the box's square before the push that made this state
  const stDir = [-1]; // the push's direction index
  const stKey = [];
  const stG = [0];
  const best = new Map();
  for (const b of boxes) occ[b] = 1;
  const low0 = flood(w, wall, occ, player, seen, ++stamp, stack);
  for (const b of boxes) occ[b] = 0;
  stKey[0] = keyOf(boxes, low0);
  best.set(stKey[0], 0);
  const heap = new Heap();
  let seq = 0;
  const h0 = heuristic(boxes, dists, perms);
  if (h0 === Infinity) return null;
  heap.push([h0, 0, seq++, 0]);

  let expanded = 0;
  let found = -1;
  while (heap.size > 0) {
    const [, g, , s] = heap.pop();
    if (g !== stG[s] || best.get(stKey[s]) !== g) continue; // stale entry
    if (solved(stBoxes[s])) {
      found = s;
      break;
    }
    if (expanded >= maxExp || budget.used >= budget.max) return null;
    expanded++;
    budget.used++;
    const bs = stBoxes[s];
    for (const b of bs) occ[b] = 1;
    flood(w, wall, occ, stPlayer[s], seen, ++stamp, stack);
    const here = stamp;
    for (let bi = 0; bi < bs.length; bi++) {
      const b = bs[bi];
      for (let k = 0; k < 4; k++) {
        const d = DY[k] * w + DX[k];
        const t = b + d;
        if (seen[b - d] !== here || wall[t] || occ[t] || !live[t]) continue;
        occ[b] = 0;
        occ[t] = 1;
        if (frozenBlock(w, wall, occ, goalSet, t)) {
          occ[t] = 0;
          occ[b] = 1;
          continue;
        }
        const next = bs.slice();
        next[bi] = t;
        next.sort((x, y) => x - y);
        if (anyFrozen(w, wall, next, goalSet)) {
          occ[t] = 0;
          occ[b] = 1;
          continue;
        }
        const low = flood(w, wall, occ, b, mark, ++stamp, stack);
        const key = keyOf(next, low);
        const g2 = g + 1;
        const prevBest = best.get(key);
        if (prevBest !== undefined && prevBest <= g2) {
          occ[t] = 0;
          occ[b] = 1;
          continue;
        }
        const h = heuristic(next, dists, perms);
        if (h === Infinity) {
          occ[t] = 0;
          occ[b] = 1;
          continue;
        }
        best.set(key, g2);
        const idx = stBoxes.length;
        stBoxes.push(next);
        stPlayer.push(b);
        stParent.push(s);
        stFrom.push(b);
        stDir.push(k);
        stKey.push(key);
        stG.push(g2);
        heap.push([g2 + h, g2, seq++, idx]);
        occ[t] = 0;
        occ[b] = 1;
      }
    }
    for (const b of bs) occ[b] = 0;
  }
  if (found < 0) return null;
  // Rebuild the push list, then the moves: walk to each push's square, then push.
  const path = [];
  for (let s = found; s > 0; s = stParent[s]) path.push(s);
  path.reverse();
  let moves = '';
  let curPlayer = player;
  let curBoxes = boxes.slice();
  for (const s of path) {
    const from = stFrom[s];
    const k = stDir[s];
    const d = DY[k] * w + DX[k];
    for (const b of curBoxes) occ[b] = 1;
    moves += walkTo(w, wall, occ, curPlayer, from - d);
    for (const b of curBoxes) occ[b] = 0;
    moves += CH[k];
    curPlayer = from;
    const at = curBoxes.indexOf(from);
    curBoxes[at] = from + d;
    curBoxes.sort((a, b) => a - b);
  }
  return { pushes: path.length, moves };
}

/**
 * The level's rooms: cut the hallway row's dead squares out of the floor;
 * each remaining component holding boxes is a room. Null when the level is
 * no set of rooms (a box in the hallway, a goal outside any room, or a room
 * whose boxes and goals do not match).
 */
function roomsOf(level) {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(w, wall, goals);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      const p = squares[i];
      for (let k = 0; k < 4; k++) {
        const q = p + DY[k] * w + DX[k];
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [], done: false });
  }
  for (const b of boxes) {
    if (part[b] < 0) return null;
    rooms[part[b]].boxes.push(b);
  }
  for (const g of goals) {
    if (part[g] < 0) return null;
    rooms[part[g]].goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/**
 * The level solved room by room: each room push-optimally in turn, the other
 * rooms' boxes standing as walls (on their goals once their room is done),
 * the player walking on from where the last room left it. A box never leaves
 * its room, so the rooms' optima add up to the level's.
 */
function solveByRooms(level) {
  const rooms = roomsOf(level);
  if (rooms === null) return null;
  let player = level.player;
  let pushes = 0;
  let moves = '';
  for (const room of rooms) {
    const wall = level.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
    }
    const sub = {
      w: level.w,
      h: level.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const r = solvePush(sub, ROOM_EXPANSIONS);
    if (r === null) return null;
    pushes += r.pushes;
    moves += r.moves;
    player = endOf(level.w, player, r.moves);
    room.done = true;
  }
  return { pushes, moves };
}

/** Last resort: A* over single steps (fewest moves, not fewest pushes). */
function solveMoves(level, maxExp) {
  const { w, wall, goals, boxes, player } = level;
  const size = wall.length;
  const goalSet = new Uint8Array(size);
  for (const g of goals) goalSet[g] = 1;
  const dists = gridDistances(w, wall, goals);
  const perms = boxes.length <= 6 ? permutations(boxes.length) : null;
  const hOf = (bs) => heuristic(bs, dists, perms);
  const keyOf = (p, bs) => p + ':' + bs.join(',');
  const stP = [player];
  const stB = [boxes.slice()];
  const stParent = [-1];
  const stStep = [''];
  const stG = [0];
  const best = new Map([[keyOf(player, boxes), 0]]);
  const heap = new Heap();
  let seq = 0;
  heap.push([hOf(boxes), 0, seq++, 0]);
  let expanded = 0;
  let found = -1;
  while (heap.size > 0) {
    const [, g, , s] = heap.pop();
    if (g !== stG[s]) continue;
    const bs = stB[s];
    if (bs.every((b) => goalSet[b])) {
      found = s;
      break;
    }
    if (expanded >= maxExp || budget.used >= budget.max) return null;
    expanded++;
    budget.used++;
    const p = stP[s];
    for (let k = 0; k < 4; k++) {
      const d = DY[k] * w + DX[k];
      const q = p + d;
      if (wall[q]) continue;
      let nb = bs;
      const at = bs.indexOf(q);
      if (at >= 0) {
        if (wall[q + d] || bs.includes(q + d)) continue;
        nb = bs.slice();
        nb[at] = q + d;
        nb.sort((a, b) => a - b);
      }
      const key = keyOf(q, nb);
      const g2 = g + 1;
      const prev = best.get(key);
      if (prev !== undefined && prev <= g2) continue;
      const h = hOf(nb);
      if (h === Infinity) continue;
      best.set(key, g2);
      const idx = stB.length;
      stB.push(nb);
      stP.push(q);
      stParent.push(s);
      stStep.push(CH[k]);
      stG.push(g2);
      heap.push([g2 + h, g2, seq++, idx]);
    }
  }
  if (found < 0) return null;
  let moves = '';
  for (let s = found; s > 0; s = stParent[s]) moves = stStep[s] + moves;
  return moves;
}

/** Solve one level: rooms first, then whole-level pushes, then plain moves. */
function solveLevel(text) {
  const level = parseLevel(text);
  const byRooms = solveByRooms(level);
  if (byRooms !== null) return byRooms.moves;
  const whole = solvePush(level, WHOLE_EXPANSIONS);
  if (whole !== null) return whole.moves;
  const byMoves = solveMoves(level, MOVE_EXPANSIONS);
  if (byMoves !== null) return byMoves;
  return '';
}

for (const id of readdirSync(INPUT).sort()) {
  const moves = solveLevel(readFileSync(join(INPUT, id, 'level.txt'), 'utf8'));
  mkdirSync(join(OUTPUT, id), { recursive: true });
  writeFileSync(join(OUTPUT, id, 'solution.txt'), `${moves}\n`);
}
