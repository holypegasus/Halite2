"""
Note: Please do not place print statements here as they are used to communicate with the Halite engine. If you need to log anything use the logging module.
"""
import logging, math
import numpy as np
from collections import Counter, OrderedDict, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain
from time import time

import hlt
from hlt import Game
from hlt.constants import DOCK_TURNS, SHIP_RADIUS, DOCK_RADIUS, MAX_SPEED
from hlt.constants import WEAPON_DAMAGE, WEAPON_RADIUS
from hlt.constants import TOLERANCE, PRECISION
from hlt.entity import Entity, Planet, Ship, Pos, Vect
from hlt.util import setup_logger, logitr, timit


BOT_NAME = 'v10_guess_scan_evade_swarm'
# Default game params
LIMIT_TURNS = math.inf
FOCUS_TURNS = set()
PLOT_FOCUS_TURNS = False

# Drill-down params
LOG_LEVEL = logging.CRITICAL
LOG_LEVEL = logging.WARNING
LIMIT_TURNS = 8
FOCUS_TURNS = set(range(3, LIMIT_TURNS+1))
PLOT_FOCUS_TURNS = True

# Logic params
PC = 2  # coordinate-precision
TIMEOUT_BREAKER = 1800  # ms
MAX_SWARM = 4  # 4-shots to kill a full-health foe


# update memo when ship stays
def memo_stay(ship):
  ship.dest = ship.curr_pos
  game_map.add_obst(ship)
  game_map.add_dest(ship)

def memo_move(ship, target):
  ship.dest = dest = Pos(target.x, target.y, r=ship.radius, health=ship.health)
  navi_path = Vect(ship, dest, r=ship.radius, pc=PC)
  game_map.add_path(navi_path)

# @timit
def get_act(ship, goal, ship2comm):
  assert isinstance(goal, Planet) or isinstance(goal, Ship)
  assert not ship.dest  # to be assigned here !

  nav_comm = None
  # dock in-place
  if isinstance(goal, Planet) and ship.can_dock(goal):
    nav_comm = ship.dock(goal)
    memo_stay(ship)
  # might need to move
  else:
    assigned_ships = ship2comm.keys()  # list from OrderedDict ensures order
    leader_sep = 0.
    leader_dest = None
    # join swarm if leader.goal.next_pos weapon-reachable - update goal
    # which means... every swarm member joins area-strength :D
    for leader in assigned_ships:
      assert leader.goal.next_pos  # shud've beenn set by leader's ops
      if (
        isinstance(leader.goal, Ship)
        and leader.dest
        and ship.dist(leader.goal.next_pos) <= MAX_SPEED+WEAPON_RADIUS
        and len(leader.swarm) < MAX_SWARM
      ):
        # logging.warning((s, s.dest, ship.dist(s.dest)))
        game_map.update_ship8goal_memos(ship, leader.goal)
        goal = leader.goal
        ship.join_swarm(leader)
        leader_dest = leader.dest
        leader_sep = leader_dest.dist(goal.next_pos)
        logging.warning("%s follows %s's swarm %s => %s", ship, leader, [s.id for s in leader.swarm | set([leader])], goal.next_pos)
        break

    # normal nav
    nav_comm, dest = game_map.nav_adapt(ship, goal, leader_sep=leader_sep)

    # TMP 1) scan_area -> overpowered? evade : normal
    # overpowered? naive projection OR swarm projection
    # evade -> TMP wait for help ? -> nav around foe WR
    # beeline assumes worst case ie foes come straight for my ship
    # overpowered, foes = scan_area(ship, dest or ship)
    overpowered, foe_in_range_mobos = scan_area(ship, leader_dest or dest or ship)
    if overpowered:
      evade_paths = [game_map.get_beeline(f, ship) for f in foe_in_range_mobos]
      logitr(evade_paths, 'evade_paths', 30)
      nav_comm, dest = game_map.nav_adapt(ship, goal, evade_paths)

    # TODO 2) coordinate swarm
    # ? match goal_of swarm velocity if closeby
    # create joint rally point if first to consider goal
    # if len(goal.goal_of) > 1:
    #   ally = goal.goal_of - set([ship])
    #   if ship.dist(ally) <= MAX_SPEED:

    if nav_comm:
      memo_move(ship, dest)
    else:
      memo_stay(ship)

  logging.warning('%s nav_comm: %s -> %s', str(ship).split()[0], nav_comm, dest)
  return nav_comm


# TMP VERY ROUGH scan_area for power-balance
# -> overpowered: bool; foes: [Ship]
def scan_area(ship, center, tol=TOLERANCE):
  logging.info('%s scan_area around %s !', ship, center)
  r = game_map.rec

  # foe imobos + mobos projected to enter area
  foe_in_range_imobos = [f for f in r.foe_imobos if f.dist(center) <= WEAPON_RADIUS - tol]
  foe_in_range_mobos = [f for f in r.foe_mobos if game_map.next_pos(f).dist(center) <= WEAPON_RADIUS + SHIP_RADIUS]
  foe_ships = foe_in_range_imobos + foe_in_range_mobos

  foe_mobo_attack_potential = len(foe_in_range_mobos) * WEAPON_DAMAGE
  foe_total_health = sum(f.health for f in foe_ships)
  logging.info('foe_ships in radius: %s; mobo attack potential: %s; total hp: %s', sorted([f.id for f in foe_ships]), foe_mobo_attack_potential, foe_total_health)
  foe_stren = foe_mobo_attack_potential + foe_total_health

  # my imobos + unassigned mobos
  my_in_range_imobos = [m for m in r.my_imobos if m.dist(center) <= WEAPON_RADIUS - tol]
  my_in_range_assigned_mobos = [m for m in r.my_mobos if m.dest and m.dest.dist(center) <= WEAPON_RADIUS - tol]
  my_in_range_unassigned_mobos = [m for m in r.my_mobos if not m.dest and m.dist(center) <= MAX_SPEED + WEAPON_RADIUS - tol]
  my_ships = my_in_range_imobos + my_in_range_assigned_mobos + my_in_range_unassigned_mobos

  my_mobo_attack_potential = len(my_in_range_assigned_mobos) + len(my_in_range_unassigned_mobos) * WEAPON_DAMAGE
  my_total_health = sum(m.health for m in my_ships)
  logging.info('my_ships in radius: %s; mobo attack potential: %s; total hp: %s', sorted([m.id for m in my_ships]), my_mobo_attack_potential, my_total_health)
  my_stren = my_mobo_attack_potential + my_total_health

  if foe_stren == 0:
    overpowered = False
  else:
    overpowered = my_stren <= foe_stren
  logging.warning('overpowered? %s: %s <=? %s; %s scan %s', overpowered, my_stren, foe_stren, ship, center)

  return overpowered, foe_in_range_mobos


"""ROUGH check of threats entering origin's safety-perimeter
  -> foe_ships entering some perimeter of origin
  TODO refactor for generic area-analysis
  TODO parametrize planet-specific scan-area stuff
  """
def player_scan_planet(player_id, ship, center):
  r = game_map.recs[player_id]
  eff_target = ship.perigee(center, min_dist=SHIP_RADIUS)
  turns_to_eff_target = ship.dist(eff_target) / MAX_SPEED
  if isinstance(center, Planet):
    turns_to_safe = turns_to_eff_target + DOCK_TURNS
  else:
    turns_to_safe = turns_to_eff_target
  # all foe_ships w/ guess_next_pos within perimeter of center
  # ? foe_ships vs foe_mobos
  perim_foes = set()
  for foe in r.foe_ships:
    # TODO get actual dist to proposed docking spot
    foe_next_pos = game_map.next_pos(foe)
    foe_dist_to = foe_next_pos.dist(eff_target)
    turns_from_foe = foe_dist_to / MAX_SPEED
    if turns_from_foe <= turns_to_safe:
      perim_foes.add(foe)

  sorted_ship_dist_to8foe = sorted(
    [(ship.eff_dist(game_map.next_pos(f), PC), f) for f in perim_foes],
    key=lambda d8f: d8f[0]
  )
  # logitr(sorted_ship_dist_to8foe, 'sorted_ship_dist_to8foe from %s'%ship, 30)
  return sorted_ship_dist_to8foe


def prep_1_ship_goals(ship, goals):  # [(dist, goal, ship)]
  eff_dgs = []
  for goal in goals:
    if isinstance(goal, Planet):
      dgs = ship.eff_dist(goal, PC), goal, ship
    else:  # foe_ship
      dgs = ship.eff_dist(game_map.next_pos(goal), PC), goal, ship
    eff_dgs.append(dgs)
  return eff_dgs
  # return sorted(eff_dgs, key=lambda dgs: dgs[0])


def prep_all_ship_goals(r, my_mobos):  # [(dist, ship, goal)]
  goals = r.free_planets | r.open_planets | r.foe_imobos | r.foe_mobos
  all_eff_dgs = [prep_1_ship_goals(s, goals) for s in my_mobos]
  all_eff_dgs = list(chain(*all_eff_dgs))  # flatten into 1 list
  # logitr(all_eff_dgs, 'all_eff_dgs', 30)
  return all_eff_dgs


@timit
def player_eval_goals(player_id):
  logging.critical('Eval player_%s goals...', player_id)
  # first prep player's stats_record this turn
  r = game_map.update_stats_record(player_id)
  # logitr(r._asdict(), 'player_%s stats_records'%player_id, 30)
  all_dgs = prep_all_ship_goals(r, r.my_mobos)
  assigned_ships = set()
  final_dgs = []
  # TODO use priority queue ?
  heapify(all_dgs)  # heap-based priority-queue
  while len(assigned_ships) < len(r.my_mobos) and all_dgs:
    # logitr(all_dgs, 'all_dgs', 30)
    d, g, s = heappop(all_dgs)
    # logging.warning('eval: %s', (d, g, s))

    if s in assigned_ships:
      continue

    # check strength matchup
    if isinstance(g, Ship) and g.is_matched():
      continue

    # check planet-saturation & area-threat matchup
    if isinstance(g, Planet):
      committed_ships = g.goals_of
      n_open_docks = g.num_docking_spots - len(committed_ships)
      if n_open_docks <= 0 and s not in committed_ships:
        continue
      else:
        sorted_ship_dist_to8foe = player_scan_planet(player_id, s, g)
        for ship_dist_to, foe in sorted_ship_dist_to8foe:
          # if foe undermatched - reinforce
          if not foe.goals_of or s.goal==foe or not foe.is_matched():
            d = ship_dist_to
            g = foe
            break

    # logging.warning('chose: %s', (d, g, s))
    final_dgs.append((d, g, s))
    assigned_ships.add(s)
    game_map.update_ship8goal_memos(s, g)

  # TEST final pass assign ships who have held out so far - less picky!
  # TEST just assign to nearest goal ?
  # TODO optimize
  unassigned = r.my_mobos - assigned_ships
  logitr(unassigned, 'mopping up unassigned my_ships', 20)
  for d, g, s in prep_all_ship_goals(r=r, my_mobos=unassigned):
    if s not in assigned_ships:
      final_dgs.append((d, g, s))
      assigned_ships.add(s)
      game_map.update_ship8goal_memos(s, g)

  # logitr(final_dgs, 'final_dgs', 30)
  sorted_dgs = sorted(final_dgs, key=lambda dsg: dsg[0])
  logitr([(d, g, s, s.health) for d, g, s in sorted_dgs], 'player_%s sorted_dgs'%player_id, 30)
  return sorted_dgs


@timit
def turn():
  # Update the map for the new turn and get the latest version
  game.update_map()
  # TURN START
  t0 = time()
  # from curr_turn game_map recalc various records & update memos
  game_map.turn_update_memos()

  ### TMP experiment player_eval_goals
  pid2sorted_dgs = dict()
  for player_id in game_map._players.keys():
    pid2sorted_dgs[player_id] = player_eval_goals(player_id)
  ###

  # eval a goal per ship & update ship & goal memos
  # sorted_dgs = eval_goals()
  sorted_dgs = pid2sorted_dgs[game_map.my_id]
  # foreach goal, get_act
  t1 = time()
  ship2comm = OrderedDict()
  for i, (dist, goal, ship) in enumerate(sorted_dgs):
    td = (time() - t0)*1000
    if td >= TIMEOUT_BREAKER:
      logging.critical('TIMEOUT_BREAKER! ship #%d; time-elapsed: %.0f ms', i, td)
      break
    comm = get_act(ship, goal, ship2comm)
    logging.debug('%s -> %s: %s', ship, goal, comm)
    if comm:
      ship2comm[ship] = comm
  logging.critical('All get_act() took: %s ms', round((time() - t1)*1000) )

  # Send our set of comms to the Halite engine for this turn
  logitr({s.id: (c.split()[2:], s.goal) for s, c in ship2comm.items()}, 'comms', 30)
  game.send_command_queue(ship2comm.values())
  # TURN END



# GAME START
game = Game(name=BOT_NAME, log_level=LOG_LEVEL)
gm = game_map = game.map
# TODO 60-secs of preprocess window here!

# Main
while True:
  if game.curr_turn > LIMIT_TURNS:  break  # short-horizon tests
  if game.curr_turn+1 in FOCUS_TURNS:
    step_down = 20 if LOG_LEVEL==logging.CRITICAL else 10
    game.log_more(LOG_LEVEL-step_down)

  turn()

  # opt plot - plot only turns with movement
  if PLOT_FOCUS_TURNS and game.curr_turn in FOCUS_TURNS:
    from hlt.plot import Pltr
    pltr = Pltr(game_map.width, game_map.height)
    r = game_map.rec

    # NEUTRAL
    pltr.plt_circles(r.free_planets, c='gray', fill=False)

    # FOES
    pltr.plt_circles(r.foe_imobos, c='red')
    pltr.plt_circles(r.foe_planets, c='red')
    pltr.plt_circles(r.foe_mobos, c='red')
    # foe_next_paths
    foe_next_paths = [game_map.next_path(s) for s in r.foe_mobos]
    # logitr(foe_next_paths, 'foe_next_paths', 30)
    pltr.plt_lines(foe_next_paths, ls=':', width=1, c='salmon')
    # player_eval_goal -> foe_next_pos
    foe_next_pos = [game_map.next_pos(f) for f in r.foe_mobos]
    pltr.plt_circles(foe_next_pos, c='salmon', ls=':')
    # pltr.plt_circles(foe_next_pos, r=WEAPON_RADIUS, ls=':', fill=False)


    # MINE
    pltr.plt_circles(r.my_imobos, c='green')
    pltr.plt_circles(r.my_planets, c='green')
    pltr.plt_circles(r.my_mobos, c='green')
    # my projections
    # to_goals = [Vect(s, s.goal) for s in r.my_mobos]
    to_goals = []
    for s in r.my_mobos:
      if s.goal not in r.foe_mobos:
        goal_next_pos = s.goal
      else:
        goal_next_pos = game_map.next_pos(s.goal)
      to_goals.append(Vect(s, goal_next_pos))
    pltr.plt_lines(to_goals, ls=':', width=1, c='springgreen')
    pltr.plt_lines(game_map.paths, c='springgreen')
    dests = game_map.dests | set([p.e1 for p in game_map.paths])
    pltr.plt_circles(dests, c='springgreen', ls=':')
    pltr.plt_circles(dests, r=WEAPON_RADIUS, ls=':', fill=False)
    pltr.plt_circles(dests, r=DOCK_RADIUS, ls=':', fill=False)

    # grid = Grid(game_map, precision=PRECISION)
    # grid.map_entities(game_map.all_planets())
    # pltr.plt_matrix(grid)

    # pltr.show()
    pltr.savefig(turn=game.curr_turn)
# GAME END
