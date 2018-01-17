"""
Note: Please do not place print statements here as they are used to communicate with the Halite engine. If you need to log anything use the logging module.
"""
import copy, logging, math
import numpy as np
from collections import Counter, OrderedDict, defaultdict, namedtuple
from heapq import heapify, heappop, heappush
from itertools import chain
from time import time

import hlt
from hlt import Game
from hlt.constants import DOCK_TURNS, SHIP_RADIUS, MAX_SHIP_HEALTH, DOCK_RADIUS, MAX_SPEED, MIN_MOVE_SPEED
from hlt.constants import WEAPON_DAMAGE, WEAPON_RADIUS
from hlt.constants import TOL, PRECISION, FUDGE
from hlt.constants import SEPS_DOCK, SEPS_FIGHT, SEPS_EVADE, SEPS_RALLY
from hlt.entity import Entity, Planet, Ship, Pos, Vect
from hlt.util import setup_logger, logitr, timit


BOT_NAME = 'v12_WIP'
# Default game params
LOG_LEVEL = logging.CRITICAL
LIMIT_TURNS = math.inf
FOCUS_TURNS = set()
PLOT_FOCUS_TURNS = False

# Drill-down params
# LIMIT_TURNS = 7
# FOCUS_TURNS = set(range(6, LIMIT_TURNS+1))
# FOCUS_TURN_LOG_LVL = logging.WARNING
# PLOT_FOCUS_TURNS = True

# Logic params
PC = 2  # coordinate-precision
TIMEOUT_BREAKER = 1800  # ms
MAX_SWARM = 4  # 4-shots to kill a full-health foe


# update memo when ship stays
def memo_act(ship):
  if not ship.dest:  # stay
    ship.dest = ship.curr_pos
    gm.add_dest(ship)
  else:
    # ship.dest = Pos(dest.x, dest.y, r=ship.radius, health=ship.health)
    navi_path = Vect(ship, ship.dest, r=ship.radius, pc=PC)
    gm.add_path(navi_path)

# @timit
# assigned_ships: OrderedDict keeps key order
def get_act(ship, goal, assigned_ships):
  logging.warning('      +++ %s goal: %s', ship.name, goal.name)
  nav_comm = None
  if isinstance(goal, Planet):
    if ship.can_dock(goal):
      nav_comm = ship.dock(goal)
    else:
      ship.target = goal
      seps = SEPS_DOCK
  else:
    ship.target = gm.next_pos(goal)
    # TODO fix swarming vs evading
    swarming = False
    # default FIGHT
    seps = SEPS_FIGHT
    # TEST lower-hp stays further back
    # update sep_opt accordingly
    sep_opt = seps[1] - (ship.health/MAX_SHIP_HEALTH) * MIN_MOVE_SPEED
    seps = (seps[0], sep_opt, seps[2])
    # if 1) can reach target of 2) overpowered-swarm
    for ldr in assigned_ships:
      # TODO fix alloc ADHD: rethink conds
      if (
        isinstance(ldr.goal, Ship)
        and ship.can_fire_on(gm.next_pos(ldr.goal), add_range=MAX_SPEED)
        # and scan_area(ship, ldr.target, assess_need=False)[0]
        and len(ldr.swarm) < MAX_SWARM
      ):
        ship.join_swarm(ldr)
        gm.update_ship8goal_memos(ship, ldr.goal)

        if ldr.evading:  # follower RALLY
          ship.evading = True
          ship.target = ldr.dest
          seps = SEPS_RALLY
        else:  # still FIGHT
          swarming = True
          ship.target = gm.next_pos(ship.goal)

        logging.warning('swarming? %s; goal: %s; target: %s', swarming, ship.goal, ship.target)
        break

    # naive scan_area nav_target - if overpowered, add evade_paths
    # TODO eval after finding dest ???
    if not swarming:
      nav_target = gm.get_nav_target(ship, seps, log=False)
      balance, foe_in_range_mobos = scan_area(ship, nav_target)
      if balance <= 0:  # EVADE foe_mobo paths (radius = WEAPON_RADIUS)
        ship.evading = True
        seps = SEPS_EVADE
        foe_next_paths = [gm.next_path(f) for f in foe_in_range_mobos]
        evade_paths = []
        for fnp in foe_next_paths:
          path = copy.copy(fnp)
          path.radius = WEAPON_RADIUS
          evade_paths.append(path)
        ship.evade_paths = evade_paths
        # logitr(foe_next_paths, 'foe_next_paths', 50)
        # logitr(evade_paths, 'evade_paths', 50)
  # regular nav
  if not nav_comm:
    nav_comm = gm.nav(ship, seps)

  memo_act(ship)
  logging.warning('     --- %s nav_comm: %s -> %s', str(ship).split()[0], nav_comm, ship.dest)
  return nav_comm


# ROUGH scan_area for power-balance: hp + damage-potential
# -> overpowered: bool; foes: [Ship]
def scan_area(ship, center, assess_need=True):
  r = gm.rec
  # foe imobos + mobos projected to enter area
  foe_in_range_imobos = [f for f in r.foe_imobos if f.dist(center) <= WEAPON_RADIUS - SHIP_RADIUS]
  foe_in_range_mobos = [f for f in r.foe_mobos if gm.next_pos(f).dist(center) <= WEAPON_RADIUS + SHIP_RADIUS]
  foe_ships = foe_in_range_imobos + foe_in_range_mobos

  foe_mobo_attack_potential = len(foe_in_range_mobos) * WEAPON_DAMAGE
  foe_total_health = sum(f.health for f in foe_ships)
  foe_stren = foe_mobo_attack_potential + foe_total_health

  # my imobos + unassigned mobos
  my_in_range_imobos = [m for m in r.my_imobos if m.dist(center) <= WEAPON_RADIUS - SHIP_RADIUS]
  my_in_range_assigned_mobos = [m for m in r.my_mobos if m.dest and m.dest.dist(center) <= WEAPON_RADIUS - SHIP_RADIUS]
  my_in_range_mobos = my_in_range_assigned_mobos
  if assess_need:
    my_in_range_unassigned_mobos = [m for m in r.my_mobos if not m.dest and m.dist(center) <= MAX_SPEED + WEAPON_RADIUS - SHIP_RADIUS]
    my_in_range_mobos += my_in_range_unassigned_mobos
  my_ships = my_in_range_imobos + my_in_range_mobos

  my_mobo_attack_potential = len(my_in_range_mobos) * WEAPON_DAMAGE
  my_total_health = sum(m.health for m in my_ships)
  my_stren = my_mobo_attack_potential + my_total_health

  # compare strengths
  if foe_mobo_attack_potential == 0 or foe_stren ==0 or my_stren > foe_stren:
    balance = 1
  elif my_stren == foe_stren:
    balance = 0
  else:
    balance = -1

  # log
  if assess_need:
    logging.info('Scanning %s', center)
    logging.info('my_ships in radius: %s; mobo attack potential: %s; total hp: %s', sorted([m.id for m in my_ships]), my_mobo_attack_potential, my_total_health)
    logging.info('foe_ships in radius: %s; mobo attack potential: %s; total hp: %s', sorted([f.id for f in foe_ships]), foe_mobo_attack_potential, foe_total_health)
    logging.warning('   ??? balance: %s: %s ? %s', balance, my_stren, foe_stren)

  return balance, foe_in_range_mobos


"""ROUGH check of threats entering origin's safety-perimeter
  -> foe_ships entering some perimeter of origin
  TODO refactor for generic area-analysis
  TODO parametrize planet-specific scan-area stuff
  """
def player_scan_planet(player_id, ship, planet):
  assert isinstance(planet, Planet)
  r = gm.recs[player_id]
  eff_target = ship.perigee(planet, sep=SHIP_RADIUS)
  turns_to_eff_target = ship.dist(eff_target) / MAX_SPEED
  # TODO improve !!!
  turns_to_safe = turns_to_eff_target + DOCK_TURNS
  # all foe_ships w/ guess_next_pos within perimeter of planet
  perim_foes = set()
  for foe in r.foe_ships:
    foe_next_pos = gm.next_pos(foe)
    fnp_to_target = foe_next_pos.eff_dist(eff_target)
    turns_from_foe = fnp_to_target / MAX_SPEED + 1
    # logging.info('%s scan %s: threat: %s: %.1f <=? %.1f', ship.name, planet.name, foe, turns_to_safe, turns_from_foe)
    if turns_from_foe <= turns_to_safe:
      perim_foes.add(foe)

  sorted_ship_dist_to8foe = sorted(
    [(ship.eff_dist(gm.next_pos(f), PC), f) for f in perim_foes],
    key=lambda d8f: d8f[0]
  )
  # logitr(sorted_ship_dist_to8foe, 'sorted_ship_dist_to8foe from %s'%ship, 20)
  return sorted_ship_dist_to8foe


def prep_1_ship_goals(ship, goals):  # [(dist, goal, ship)]
  eff_dgs = []
  for goal in goals:
    dgs = ship.eff_dist(gm.next_pos(goal), PC), goal, ship
    eff_dgs.append(dgs)
  return eff_dgs
  # return sorted(eff_dgs, key=lambda dgs: dgs[0])


def prep_all_ship_goals(r, my_mobos):  # [(dist, ship, goal)]
  goals = r.free_planets | r.open_planets | r.foe_imobos | r.foe_mobos
  all_eff_dgs = [prep_1_ship_goals(s, goals) for s in my_mobos]
  all_eff_dgs = list(chain(*all_eff_dgs))  # flatten into 1 list
  # logitr(all_eff_dgs, 'all_eff_dgs', 30)
  return all_eff_dgs


def optlog(cond, lvl, str, *params):
  if cond:
    if lvl==0:
      itr, header = params
      logitr(itr, header, 30)
    else:
      log = {
        30: logging.info,
        50: logging.critical,
      }.get(lvl, logging.info)
      log(str, *params)


# compare strengths
# TODO merge w/ scan_area ?
def me_stronger(my_mobos, my_imobos, foe_mobos, foe_imobos):
  foe_att = WEAPON_DAMAGE * len(foe_mobos)
  foe_hp = sum(s.health for s in foe_mobos + foe_imobos)
  foe_stren = foe_att + foe_hp
  my_att = WEAPON_DAMAGE * len(my_mobos)
  my_hp = sum(s.health for s in my_mobos + my_imobos)
  my_stren = my_att + my_hp
  return foe_stren<=0. or my_stren > foe_stren

# TMP if foe_ship has already been matched in strength
def is_matched(foe, inquirer):
  assert isinstance(inquirer, Ship), inquirer
  assert inquirer.owner_id != foe.owner_id, (inquirer.owner_id, foe.owner_id)
  dist_to_inquirer = foe.dist(inquirer)
  inquirer_allies = [s for s in foe.goals_of if s.owner_id == inquirer.owner_id]
  allies_closer_than_inquirer = [s for s in inquirer_allies if foe.dist(s) < dist_to_inquirer]
  # logitr(allies_closer_than_inquirer, 'inquirer_allies_closer', 20)
  if foe.is_mobo():
    matched = me_stronger(allies_closer_than_inquirer, [], [foe], [])
  else:
    matched = True if allies_closer_than_inquirer else False
  # logging.debug('%s matched? %s <- %s', foe, matched, foe.goals_of)
  return matched

# @timit
def player_eval_goals(player_id):
  logging.warning('Eval player_%s goals...', player_id)
  # if :
  #   logging.critical('Eval player_%s goals...', player_id)
  # first prep player's stats_record this turn
  r = gm.update_stats_record(player_id)
  # logitr(r._asdict(), 'player_%s stats_records'%player_id, 30)
  all_dgs = prep_all_ship_goals(r, r.my_mobos)
  # logitr(sorted(all_dgs), 'all_dgs', 30)
  assigned_ships = set()
  final_dgs = []
  # TODO use priority queue ?
  heapify(all_dgs)  # heap-based priority-queue
  while len(assigned_ships) < len(r.my_mobos) and all_dgs:
    d, g, s = heappop(all_dgs)

    if s in assigned_ships:
      continue
    # check strength matchup
    if isinstance(g, Ship) and is_matched(g, s):
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
          if not is_matched(foe, s):
            d = ship_dist_to
            g = foe
            break
    # logging.warning('chose: %s', (d, g, s))
    final_dgs.append((d, g, s))
    assigned_ships.add(s)
    gm.update_ship8goal_memos(s, g)

  # TEST final pass assign ships who have held out so far - less picky!
  # TEST just assign to nearest goal ?
  # TODO optimize
  unassigned = r.my_mobos - assigned_ships
  for d, g, s in prep_all_ship_goals(r=r, my_mobos=unassigned):
    if s not in assigned_ships:
      final_dgs.append((d, g, s))
      assigned_ships.add(s)
      gm.update_ship8goal_memos(s, g)

  # logitr(final_dgs, 'final_dgs', 30)
  sorted_dgs = sorted(final_dgs, key=lambda dsg: dsg[0])
  # optlog(player_id==gm.my_id, 20, '', 
  #   [(round(d, 2), g, s, s.health) for d, g, s in sorted_dgs],
  #   'player_%s sorted_dgs'%player_id)
  logitr(
    [(round(d, 2), g.name, s.name, s.health) for d, g, s in sorted_dgs],
    'player_%s sorted_dgs'%player_id, 30)
  return sorted_dgs


@timit
def turn():
  # Update the map for the new turn and get the latest version
  game.update_map()
  # TURN START
  t0 = time()
  # from curr_turn gm recalc various records & update memos
  gm.turn_update_memos()

  ### TMP experiment player_eval_goals
  pid2sorted_dgs = {pid: player_eval_goals(pid)
    for pid in gm._players.keys()}
  ###
  # eval a goal per ship & update ship & goal memos
  # sorted_dgs = eval_goals()
  sorted_dgs = pid2sorted_dgs[gm.my_id]
  # foreach goal, get_act
  t1 = time()
  ship2comm_dest = OrderedDict()
  for i, (dist, goal, ship) in enumerate(sorted_dgs):
    td = (time() - t0)*1000
    if td >= TIMEOUT_BREAKER:
      logging.critical('TIMEOUT_BREAKER! %dth ship; time-elapsed: %.0f ms', i, td)
      break

    comm = get_act(ship, goal, ship2comm_dest.keys())
    if comm:
      ship2comm_dest[ship] = comm, ship.dest
  logging.critical('All get_act() took: %s ms', round((time() - t1)*1000) )
  for s, (c, d) in ship2comm_dest.items():
    logging.warning('%s: %s -> %s', s, c, d)

  # Send our set of comms to the Halite engine for this turn
  comms = [c for c, d in ship2comm_dest.values()]
  game.send_command_queue(comms)
  # TURN END



# GAME START
game = Game(name=BOT_NAME, log_level=LOG_LEVEL)
gm = game.map
# TODO 60-secs of preprocess window here!


# Main
while True:
  if game.curr_turn > LIMIT_TURNS:  break  # short-horizon tests
  if game.curr_turn+1 in FOCUS_TURNS:
    game.log_more(FOCUS_TURN_LOG_LVL)

  turn()

  # opt plot - plot only turns with movement
  if PLOT_FOCUS_TURNS and game.curr_turn in FOCUS_TURNS:
    from hlt.plot import Pltr
    pltr = Pltr(gm.width, gm.height)
    r = gm.rec

    # NEUTRAL
    pltr.plt_circles(r.free_planets, c='gray', fill=False)

    # FOES
    pltr.plt_circles(r.foe_imobos, c='red')
    pltr.plt_circles(r.foe_planets, c='red')
    pltr.plt_circles(r.foe_mobos, c='red')
    foe_next_paths = [gm.next_path(s) for s in r.foe_mobos]
    pltr.plt_lines(foe_next_paths, ls=':', width=1, c='salmon')
    # player_eval_goal -> foe_next_pos
    foe_next_pos = [gm.next_pos(f) for f in r.foe_mobos]
    pltr.plt_circles(foe_next_pos, c='salmon', ls=':')
    pltr.plt_circles(foe_next_pos, r=WEAPON_RADIUS, ls=':', fill=False)


    # MINE
    pltr.plt_circles(r.my_imobos, c='green')
    pltr.plt_circles(r.my_planets, c='green')
    pltr.plt_circles(r.my_mobos, c='green')
    # my projections
    # to_goals = [Vect(s, gm.next_pos(s.goal) for s in r.my_mobos)]
    # pltr.plt_lines(to_goals, ls=':', width=1, c='springgreen')
    to_tgts = [Vect(s, s.target) for s in r.my_mobos if s.target]
    pltr.plt_lines(to_tgts, ls=':', width=1, c='springgreen')
    pltr.plt_lines(gm.paths, c='springgreen')
    dests = gm.dests | set([p.e1 for p in gm.paths])
    pltr.plt_circles(dests, c='springgreen', ls=':')
    pltr.plt_circles(dests, r=WEAPON_RADIUS, ls=':', fill=False)
    pltr.plt_circles(dests, r=DOCK_RADIUS, ls=':', fill=False)

    # grid = Grid(gm, precision=PRECISION)
    # grid.map_entities(gm.all_planets())
    # pltr.plt_matrix(grid)

    # pltr.show()
    pltr.savefig(turn=game.curr_turn)
# GAME END
