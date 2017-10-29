import logging, math
from collections import defaultdict, namedtuple
from math import floor, ceil, sqrt, degrees, asin

from .collision import get_tangents
from .constants import SHIP_RADIUS, DOCK_RADIUS, MIN_MOVE_SPEED, MAX_SPEED
from .constants import WEAPON_RADIUS, SEP_DESIRED_FIGHT
from .constants import TOLERANCE, PRECISION
from .entity import Entity, Planet, Ship, Pos, Vect
from .util import logitr, timit


class Map:
  """
  Map which houses the current game information/metadata.
  
  :ivar my_id: Current player id associated with the map
  :ivar width: Map width
  :ivar height: Map height
  """

  def __init__(self, my_id, width, height):
    """
    :param my_id: User's id (tag)
    :param width: Map width
    :param height: Map height
    """
    self.my_id = my_id
    self.width = width
    self.height = height
    self._players = dict()  # player_id -> Player
    self._planets = dict()  # planet_id -> Planet
    # custom
    self.center = Pos(self.width/2, self.height/2, r=0.)
    # memos: updated beginning of each turn
    self.rec = dict()
    self.recs = dict()  # player_id -> player_record
    self.obsts = set()  # occupied Entity/Loc
    self.dests = set()  # mobo ships' intended next Position
    self.paths = []  # dynamic-obsts: plotted ship-paths

  def get_player(self, player_id):
    """
    :param int player_id: The id of the desired player
    :return: The player associated with player_id
    :rtype: Player
    """
    return self._players.get(player_id)
  
  def get_me(self):
    """
    :return: The user's player
    :rtype: Player
    """
    return self.get_player(self.my_id)

  def all_players(self):
    """
    :return: List of all players
    :rtype: list[Player]
    """
    return list(self._players.values())

  def get_planet(self, planet_id):
    """
    :param int planet_id:
    :return: The planet associated with planet_id
    :rtype: Planet
    """
    return self._planets.get(planet_id)

  def all_planets(self):
    """
    :return: List of all planets
    :rtype: list[Planet]
    """
    return list(self._planets.values())

  ### NB mod _parse* to update instead of recreate player & planet if exists
  # NB parse new PLAYER info -> del or update
  def _parse_players(self, tokens):
    pid2player, tokens = Player._parse(tokens)
    prev_pids = set(pid for pid in self._players.keys())
    curr_pids = set(pid for pid in pid2player.keys())
    # add
    for pid in curr_pids - prev_pids:
      self._players[pid] = pid2player[pid]
    # del
    for pid in prev_pids - curr_pids:
      self._players.pop(pid)
    # update
    for pid in prev_pids & curr_pids:
      self._players[pid].update_player(pid2player[pid])
    return tokens


  # NB parse new PLANET info -> del or update
  def _parse_planets(self, tokens):
    pid2planet, tokens = Planet._parse(tokens)
    prev_pids = set(pid for pid in self._planets.keys())
    curr_pids = set(pid for pid in pid2planet.keys())
    # add
    for pid in curr_pids - prev_pids:
      new_planet = pid2planet[pid]
      new_planet.set_spawn_pos(self.center)
      self._planets[pid] = new_planet
    # del
    for pid in prev_pids - curr_pids:
      planet_to_del = self._planets.pop(pid)
      planet_to_del.del_planet()
    # update
    for pid in prev_pids & curr_pids:
      self._planets[pid].update_planet(pid2planet[pid])

    return tokens


  # @timit
  def _parse(self, map_string):
    """
    Parse the map description from the game.

    :param map_string: The string which the Halite engine outputs
    :return: nothing
    """
    tokens = map_string.split()

    # self._players, tokens = Player._parse(tokens)
    tokens = self._parse_players(tokens)

    # self._planets, tokens = Planet._parse(tokens)
    tokens = self._parse_planets(tokens)

    assert(len(tokens) == 0)  # expect no more tokens
    self._link()
  #######

  def _link(self):
    """
    Updates all the entities with the correct ship and planet objects

    :return:
    """
    for celestial_object in self.all_planets() + self._all_ships():
      celestial_object._link(self._players, self._planets)

  def _all_ships(self):
    """
    Helper function to extract all ships from all players

    :return: List of ships
    :rtype: List[Ship]
    """
    all_ships = []
    for player in self.all_players():
      all_ships.extend(player.all_ships())
    return all_ships

  def _intersects_entity(self, target):
    """
    Check if the specified entity (x, y, r) intersects any planets. Entity is assumed to not be a planet.

    :param Entity target: The entity to check intersections with.
    :return: The colliding entity if so, else None.
    :rtype: Entity
    """
    for celestial_object in self._all_ships() + self.all_planets():
      if celestial_object is target:
        continue
      d = celestial_object.calculate_distance_between(target)
      if d <= celestial_object.radius + target.radius + 0.1:
        return celestial_object
    return None

  def obstacles_between(self, ship, target, ignore=()):
    """
    Check whether there is a straight-line path to the given point, without planetary obstacles in between.

    :param Ship ship: Source entity
    :param Entity target: Target entity
    :param Entity ignore: Which entity type to ignore
    :return: The list of obstacles between the ship and target
    :rtype: list[Entity]
    """
    obstacles = []
    entities = ([] if issubclass(Planet, ignore) else self.all_planets()) \
      + ([] if issubclass(Ship, ignore) else self._all_ships())
    for foreign_entity in entities:
      if foreign_entity == ship or foreign_entity == target:
        continue
      if collision.intersect_segment_circle(ship, target, foreign_entity, fudge=ship.radius + 0.1):
        obstacles.append(foreign_entity)
    return obstacles


  #######
  # thru-turn memos
  def show_goals_of(self, level=logging.DEBUG):
    # TMP for now only consider non-my ships to be valid goals
    # TODO can also use this to store inference about foe goals
    goals = self.rec.all_planets | self.rec.foe_ships
    goals_of = {str(g): g.goals_of for g in goals if g.goals_of}
    logitr(goals_of, 'goals_of', level, lambda kv: kv[0])

  def show_goals(self, level=logging.DEBUG):
    my_mobo_id2goals = {s.id: s.goal for s in self.rec.my_mobos}
    logitr(my_mobo_id2goals, 'my_mobo_id2goals', level, lambda kv: kv[0])

  # given new (ship, goal), update ship.goal & goal.goals_of
  def update_ship8goal_memos(self, ship, goal):
    assert isinstance(goal, Ship) or isinstance(goal, Planet)
    # check if to deregister prev goal
    # ship can only have 1 goal
    prev_goal = ship.goal
    if prev_goal:
      prev_goal.goals_of.discard(ship)  # won't complain like remove()
    # register curr goal
    # goal might have >= 1 ship
    ship.goal = goal
    goal.goals_of.add(ship)

  # in-turn memos
  # calc turn-stats -> namedtuple record
  # @timit
  def recalc_stats_record(self, player_id=None):
    all_ships = set(self._all_ships())
    my_ships = set(self.get_me().all_ships())
    my_mobos = set(s for s in my_ships if s.is_mobo())
    my_imobos = my_ships - my_mobos
    logging.info('My Ships: %s/%s MOBO!', len(my_mobos), len(my_ships))
    foe_ships = all_ships - my_ships
    foe_mobos = set(s for s in foe_ships if s.is_mobo())
    foe_imobos = foe_ships - foe_mobos
    # Get planetary stats
    all_planets = set(self.all_planets())
    owned_planets = set(p for p in all_planets if p.is_owned())
    my_planets = set(p for p in owned_planets if p.owner_id==self.my_id)
    foe_planets = owned_planets - my_planets
    free_planets = all_planets - owned_planets
    open_planets = set(p for p in my_planets if not p.is_full())
    logging.debug('Planets: %s/%s free; %s/%s open',
      len(free_planets), len(all_planets), len(open_planets), len(my_planets))
    # output
    names = ['all_ships', 'my_ships', 'my_mobos', 'my_imobos', 'foe_ships', 'foe_mobos', 'foe_imobos', 'all_planets', 'owned_planets', 'my_planets', 'foe_planets', 'free_planets', 'open_planets']
    rec = namedtuple('rec', ' '.join(names))(
      all_ships, my_ships, my_mobos, my_imobos, foe_ships, foe_mobos, foe_imobos,
      all_planets, owned_planets, my_planets, foe_planets, free_planets, open_planets)

    self.rec = rec
    return rec

  # update player_id's stats-record
  def update_stats_record(self, player_id):
    all_ships = set(self._all_ships())
    my_ships = set(self.get_player(player_id).all_ships())
    my_mobos = set(s for s in my_ships if s.is_mobo())
    my_imobos = my_ships - my_mobos
    logging.info('Player_%s Ships: %s/%s MOBO!', player_id, len(my_mobos), len(my_ships))
    foe_ships = all_ships - my_ships
    foe_mobos = set(s for s in foe_ships if s.is_mobo())
    foe_imobos = foe_ships - foe_mobos
    # Get planetary stats
    all_planets = set(self.all_planets())
    owned_planets = set(p for p in all_planets if p.is_owned())
    my_planets = set(p for p in owned_planets if p.owner_id==player_id)
    foe_planets = owned_planets - my_planets
    free_planets = all_planets - owned_planets
    open_planets = set(p for p in my_planets if not p.is_full())
    # logging.debug('Planets: %s/%s free; %s/%s open', len(free_planets), len(all_planets), len(open_planets), len(my_planets))
    # output
    names = ['all_ships', 'my_ships', 'my_mobos', 'my_imobos', 'foe_ships', 'foe_mobos', 'foe_imobos', 'all_planets', 'owned_planets', 'my_planets', 'foe_planets', 'free_planets', 'open_planets']
    rec = namedtuple('rec', ' '.join(names))(
      all_ships, my_ships, my_mobos, my_imobos, foe_ships, foe_mobos, foe_imobos,
      all_planets, owned_planets, my_planets, foe_planets, free_planets, open_planets)

    self.recs[player_id] = rec
    return rec


  # reset obsts
  # TODO replace static ships w/ "where puck will be"
  def reset_obsts(self):
    r = self.rec
    self.obsts = r.all_planets | r.my_imobos | r.foe_imobos
    logitr(self.obsts, 'game_map.obsts', logging.DEBUG)

  def add_obst(self, new_obst):
    # assert new_obst not in self.obsts
    self.obsts.add(new_obst)

  def reset_dests(self):
    self.dests = set()

  def add_dest(self, new_dest):
    self.dests.add(new_dest)

  def reset_paths(self):
    self.paths = []

  def add_path(self, new_path):
    self.paths.append(new_path)
    self.add_dest(new_path.e1)

  # recalc/reset in-turn memos
  # @timit
  def turn_update_memos(self):
    self.recalc_stats_record()
    self.reset_obsts()
    self.reset_paths()
    self.reset_dests()


  # -> target: Entity
  # target: int-dist & int-angle from ship due to engine-constraint
  def get_nav_target(self, ship, goal):
    logging.debug('%s get_target %s', ship, goal)
    if isinstance(goal, Planet):
      # hug planet to preempt block
      goal_next_pos = goal
      sep = SHIP_RADIUS
    elif isinstance(goal, Ship):
      goal_next_pos = self.next_pos(goal)
      sep_desired = SEP_DESIRED_FIGHT
      dist_to_goal = ship.dist(goal_next_pos)
      if dist_to_goal <= sep_desired:
        # backoff to maintain desired max/optimal separation
        sep = sep_desired
      else:
        """# NB when just beyond, make sure THRUST at least 1 to enter WR
          eg @5.28 away trying to keep 4 min_dist from a radius 0.5 ship
          0.78 thrust rounds to 0 !!!
          want: 1 <= THRUST = dist_to_goal - (desired_min_dist + goal.radius)
          thus: min_dist = dist_to_goal - goal.radius - 1 in edge case"""
        sep_min_move = max(0, dist_to_goal - MIN_MOVE_SPEED - goal_next_pos.radius)
        sep = min(sep_desired, sep_min_move)
    else:  # goal is a Pos
      goal_next_pos = goal
      sep = 0.

    # make eff_target
    target = ship.perigee(goal_next_pos, min_dist=sep)
    logging.info('Navigating %s --d:%s--> %s', ship, ship.dist(target), target)
    reach = init_reach = MAX_SPEED
    angle = init_angle = round(ship.angle(target))  # follow engine
    # check if target within 1-hop reach (not counting obsts) -> effective-target
    dist_to_target = ship.dist(target)
    if dist_to_target <= init_reach:
      reach = int(dist_to_target)  # follow engine
    # TODO check if target out-of-bounds ?

    eff_target = ship.get_pos(reach, angle)
    logging.info('%s -> %s: eff_target: %s', ship, goal, eff_target)
    return eff_target


  # -> reach: float; nearest_hit: Entity
  # reach: furthest flight w/o hit, rounded down to nearest int
  # nearest_hit: first Entity to hit along |ship->target|
  def max_reach(self, ship, target, evade_paths=[], tol=TOLERANCE):
    test_path = Vect(ship, target)
    # get obstacles
    # TODO smarter filter
    foe_paths = evade_paths
    # logging.warning('evade foe_paths: %s', foe_paths)
    paths = self.paths + foe_paths
    filter_paths = [p for p in paths
      if test_path.center.dist(p.center) <= MAX_SPEED + 2*SHIP_RADIUS]
    poses = self.obsts | self.dests - set([ship, target])
    filter_poses = sorted([s for s in poses
      if test_path.center.dist(s) <= 0.5*MAX_SPEED + SHIP_RADIUS + s.radius],
      key=lambda s: ship.dist(s)
    )
    obstacles = filter_paths + filter_poses
    # logitr(obstacles, 'obstacles', 30)
    # find min_dist_hit / reach
    max_reach_dist = round(ship.dist(target))
    max_reach_pos = target
    min_hit_ent = None

    # find nearest hit
    for ob in obstacles:
      d, p, e = test_path.cross(ob)
      if d + tol < max_reach_dist:
        logging.info('obstacle: %s: dpe (%.2f, %s, %s) !', ob, d, p, e)
        max_reach_dist = d
        max_reach_pos = p
        min_hit_ent = e

    logging.info('Raw: max_reach_dist: %.2f; max_reach_pos: %s; min_hit_ent: %s', max_reach_dist, max_reach_pos, min_hit_ent)
    reach = max(0, max_reach_dist)
    # convert reach to int:
    # generally round down to avoid overshoot unless
    # really close to its nearest int (indicative of float-pt rounding issue)
    if abs(reach - round(reach)) >= tol:
      reach = int(reach)
    else:
      reach = int(round(reach))
    if reach <= tol:
      reach_pos = ship.curr_pos
    else:
      reach_pos = test_path.d_along(reach)
    logging.info('%s -> %s max_reach dpe: %s, %s, %s', ship, target, reach, reach_pos, min_hit_ent)
    return reach, reach_pos, min_hit_ent


  # -> (tang_l_ngl, tang_l): (int, Entity)
  # -> (tang_r_ngl, tang_r): (int, Entity)
  def get_tangents(self, src, dst, fudge=0.01):
    assert isinstance(src, Entity) and isinstance(dst, Entity)
    rise = dst.radius + SHIP_RADIUS + fudge
    logging.debug('rise: %s', rise)
    hypo = max(rise, src.dist(dst))
    logging.debug('hypo: %s', hypo)
    run = sqrt(abs(hypo**2 - rise**2))
    abs_ngl_src_obst = src.angle(dst)  # coordinate-system absolute-angle
    try:
      rel_ngl_obst_src_tangent = degrees(asin(rise / hypo))  # relative-angle
    except ValueError as err:
      print('%s; rise/hypo: %s/%s', err, rise, hypo)
      raise err
    logging.debug('relative tangent angle %s', rel_ngl_obst_src_tangent)
    d_angle = int(ceil(rel_ngl_obst_src_tangent))
    logging.debug('Tangent int(ngl): %s', d_angle)
    tang_l_ngl = floor(abs_ngl_src_obst - d_angle) % 360
    logging.debug('left-tangent abs-ngl: %s', tang_l_ngl)
    tang_l = src.get_pos(MAX_SPEED, tang_l_ngl)
    tang_r_ngl = ceil(abs_ngl_src_obst + d_angle) % 360
    logging.debug('right-tangent abs-ngl: %s', tang_r_ngl)
    tang_r = src.get_pos(MAX_SPEED, tang_r_ngl)
    logging.info('tangents: ^%s:%s; ^%s:%s', tang_l_ngl, tang_l, tang_r_ngl, tang_r)
    return (tang_l_ngl, tang_l), (tang_r_ngl, tang_r)

  # -> min_hit_ent, min_sep, min_sep_reach, min_sep_angle
  def update_sep_from_tangent(self, explored, ship, min_hit_ent, tang_dir, evade_paths, goal_next_pos, leader_sep, min_sep, min_sep_reach, min_sep_angle):
    (tang_l_ngl, tang_l), (tang_r_ngl, tang_r) = self.get_tangents(ship, min_hit_ent)
    if tang_dir == 'L':
      tang = tang_l
      tang_ngl = tang_l_ngl
    else:
      tang = tang_r
      tang_ngl = tang_r_ngl
    # max_reach tang
    logging.info('Check tang %s: ^%s@%s', tang_dir, tang_ngl, tang)
    reach, reach_pos, min_hit_ent = self.max_reach(ship, tang, evade_paths)
    sep = reach_pos.dist(goal_next_pos)
    logging.info('sep: %s: tang max_reach', sep)
    # try match or nearer leader_sep to leader target ?
    logging.info('leader_sep: %s', leader_sep)
    prev_sep = math.inf
    # tangents first aim to go MAX_SPEED
    # we now check if interm reach even smaller separation
    for r in range(1, reach+1):
      reach_pos = ship.get_pos(r, tang_ngl)
      sep = reach_pos.dist(goal_next_pos)
      logging.info('sep: %s <- reach %s, reach_pos %s', sep, r, reach_pos)
      if sep >= prev_sep:
        break
      if sep < min_sep:
        min_sep = sep
        min_sep_reach = r
        min_sep_angle = tang_ngl
        if sep <= leader_sep:
          break
      prev_sep = sep
    # keep checking while still hitting & not already considered min_hit_ent
    while SEP_DESIRED_FIGHT <= min_sep and min_hit_ent and min_hit_ent not in explored:
      explored.add(min_hit_ent)  # prevent repeat -> infinite recursion!
      min_hit_ent, min_sep, min_sep_reach, min_sep_angle = self.update_sep_from_tangent(explored, ship, min_hit_ent, tang_dir, evade_paths, goal_next_pos, leader_sep, min_sep, min_sep_reach, min_sep_angle)

    return min_hit_ent, min_sep, min_sep_reach, min_sep_angle


  # -> adj_pos: Pos
  def adjust_if_out_of_bounds(self, ship, reach, target):
    sx = 1 if target.x >= ship.x else -1
    sy = 1 if target.y >= ship.y else -1
    # determine where out-of-bounds
    adj_x = adj_y = None
    if target.x < 0:
      adj_x = 0
    elif target.x > self.width:
      adj_x = self.width
    if target.y < 0:
      adj_y = 0
    elif target.y > self.height:
      adj_y = self.height
    # calc necessary adjustments
    dx = dy = 0
    if adj_x is not None and adj_y is not None:
      dx = sx * abs(adj_x - ship.x)
      dy = sy * abs(adj_y - ship.y)
    elif adj_x is not None:
      dx = sx * abs(adj_x - ship.x)
      dy = sy * sqrt(reach**2 - dx**2)
    elif adj_y is not None:
      dy = sy * abs(adj_y - ship.y)
      dx = sx * sqrt(reach**2 - dy**2)
    else:
      return target
    # make & return adjusted target
    adj_target = Pos(ship.x+dx, ship.y+dy, SHIP_RADIUS)
    logging.warning('%s adjusts out-of-bounds %s -> %s', ship, target, adj_target)
    return adj_target


  # TODO instead of pessimistic backoff, try nav around foe WRs
  def nav_evade(self, ship, goal, evade_paths):
    # backup when overpowered ?
    # TMP just go opposite from goal - collision !
    # TODO still go at goal but avoiding foe WR ?
    # cross() -> pre_overlap_dist, pre_overlap_pos, overlap_ent
    weapon_range_crosses = [path.cross(ship) for path in evade_paths]
    # get nearest hit (pre_overlap_dist, pre_overlap_pos, overlap_ent)
    d, p, e = min(weapon_range_crosses, key=lambda dpe: ship.dist(dpe[1]))
    min_hit_pos = p
    min_hit_dist = ship.dist(p)
    logging.warning('evade nearest_hit: %s d:%s', min_hit_pos, min_hit_dist)
    if e:
      assert e == ship
      evade_reach = max(0, int(ceil(MAX_SPEED + WEAPON_RADIUS - min_hit_dist)))
      evade_angle = p.angle(ship) % 360
      raw_evade_target = ship.get_pos(reach=evade_reach, angle=evade_angle)
      # check if target out-of-bounds & adjust
      evade_target = self.adjust_if_out_of_bounds(ship, evade_reach, raw_evade_target)
      logging.info('%s orig-goal: %s; EVADE-goal %s', ship, goal, evade_target)
      return self.nav_adapt(ship, evade_target)
    else:
      return None, None


  # -> nav_comm: Thrust; dest: Entity
  # return nearest-reachable Pos to goal
  def nav_adapt(self, ship, goal, evade_paths=[], leader_sep=0.):
    # if to evade
    if evade_paths:
      thrust, dest = self.nav_evade(ship, goal, evade_paths)
      logging.warning('nav_evade thrust: %s', thrust)
      if thrust and dest:
        return thrust, dest

    # get furthest target within reach (int-reach, int-angle away)
    target = self.get_nav_target(ship, goal)
    if not target:  # already at goal
      return None, None

    # adjust based on max_reach
    # max_reach: ship -> target
    goal_next_pos = self.next_pos(goal)
    reach, reach_pos, min_hit_ent = self.max_reach(ship, target, evade_paths)
    min_sep = reach_pos.dist(goal_next_pos)
    min_sep_reach = reach
    min_sep_angle = ship.angle(target)
    logging.info('sep: %s: beeline', min_sep)
    # vs max_reach: tang_l & tang_r -> target
    # keep checking next tangent while separation decreasing
    if min_hit_ent:
      _, min_sep, min_sep_reach, min_sep_angle = self.update_sep_from_tangent(set([min_hit_ent]), ship, min_hit_ent, 'L', evade_paths, goal_next_pos, leader_sep, min_sep, min_sep_reach, min_sep_angle)
      _, min_sep, min_sep_reach, min_sep_angle = self.update_sep_from_tangent(set([min_hit_ent]), ship, min_hit_ent, 'R', evade_paths, goal_next_pos, leader_sep, min_sep, min_sep_reach, min_sep_angle)

    logging.warning((min_sep, min_sep_reach, min_sep_angle))
    # return whichever w/ min_hit_ent nearest target
    thrust = ship.thrust(reach=min_sep_reach, angle=min_sep_angle)
    dest = ship.get_pos(reach=min_sep_reach, angle=min_sep_angle)
    logging.info('final sep: %s; %s -> %s', min_sep, ship, dest)
    return thrust, dest


  # integerize thrust reach/angle according to game-engine requirement
  def _engint(self, reach=None, angle=None):
    if reach is not None:
      return int(reach)
    elif angle is not None:
      return round(angle)


  # -> path: Vect
  # naive beeline ship -> goal
  def get_beeline(self, ship, goal, min_dist=WEAPON_RADIUS-SHIP_RADIUS):
    target = ship.perigee(goal, min_dist=min_dist)
    reach = min( self._engint(reach=ship.dist(target)), MAX_SPEED )
    angle = self._engint(angle=ship.angle(target))
    eff_target = ship.get_pos(reach, angle)
    return Vect(ship, eff_target, r=WEAPON_RADIUS)

  # PREDICT
  # TODO consider what opponent considers I will do...
  # contextualize Ship.next_pos
  # eg adjust planet-interior to nearest point on planet-surface
  # based on player_eval_goals
  # -> Pos
  def next_pos(self, ent):
    if isinstance(ent, Ship):
      if not ent.is_mobo():
        ent.next_pos = ent.curr_pos
      else:
        player_id = ent.owner_id
        goal = ent.goal or ent  # if no goal assume stay
        # TMP
        ship_path = self.get_beeline(ent, goal)
        ent.next_pos = ship_path.e1
        for planet in self._planets.values():
          if ent.next_pos in planet:
            ent.next_pos = ent.perigee(planet, min_dist=SHIP_RADIUS)
            break
      return ent.next_pos
    else:  # Planet | Pos
      return ent

  # -> Vect
  def next_path(self, ship, radius=WEAPON_RADIUS):
    next_pos = self.next_pos(ship)
    return Vect(ship, next_pos)



class Player:
  """
  :ivar id: The player's unique id
  """
  def __init__(self, player_id, sid2ship=dict()):
    """
    :param player_id: User's id
    :param ships: Ships user controls (optional)
    """
    self.id = player_id
    self.sid2ship = sid2ship  # ship_id -> Ship

  def all_ships(self):
    """
    :return: A list of all ships which belong to the user
    :rtype: list[Ship]
    """
    return list(self.sid2ship.values())

  def get_ship(self, ship_id):
    """
    :param int ship_id: The ship id of the desired ship.
    :return: The ship designated by ship_id belonging to this user.
    :rtype: Ship
    """
    return self.sid2ship.get(ship_id)

  # update player info if exists instead of recreating
  def update_player(self, new_player):
    assert (
      type(self)==type(new_player) and
      self.id==new_player.id)

    # copies of actual
    prev_sids = set(sid for sid in self.sid2ship.keys())
    curr_sids = set(sid for sid in new_player.sid2ship.keys())
    # del
    for sid in prev_sids - curr_sids:
      ship_to_del = self.sid2ship.pop(sid)
      ship_to_del.del_ship()
    # add
    for sid in curr_sids - prev_sids:
      self.sid2ship[sid] = new_player.get_ship(sid)
    # update
    for sid in prev_sids & curr_sids:
      new_ship = new_player.get_ship(sid)
      prev_ship = self.get_ship(sid)
      self.get_ship(sid).update_ship(new_ship)

  @staticmethod
  def _parse_single(tokens):
    """
    Parse one user given an input string from the Halite engine.

    :param list[str] tokens: The input string as a list of str from the Halite engine.
    :return: The parsed player id, player object, and remaining tokens
    :rtype: (int, Player, list[str])
    """
    player_id, *remainder = tokens
    player_id = int(player_id)
    sid2ship, remainder = Ship._parse(player_id, remainder)

    player = Player(player_id, sid2ship)
    # logging.info('Parsed player: %s', player)
    return player_id, player, remainder

  @staticmethod
  def _parse(tokens):
    """
    Parse an entire user input string from the Halite engine for all users.

    :param list[str] tokens: The input string as a list of str from the Halite engine.
    :return: The parsed players in the form of player dict, and remaining tokens
    :rtype: (dict, list[str])
    """
    num_players, *remainder = tokens
    num_players = int(num_players)
    players = dict()

    for _ in range(num_players):
      player, players[player], remainder = Player._parse_single(remainder)

    return players, remainder

  def __str__(self):
    return "Player {} with ships {}".format(self.id, self.all_ships())

  def __repr__(self):
    return self.__str__()
