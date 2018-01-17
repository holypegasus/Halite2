import copy, logging, math
from collections import defaultdict, namedtuple
from math import floor, ceil, sqrt, degrees, asin

from .constants import SHIP_RADIUS, DOCK_RADIUS, MIN_MOVE_SPEED, MAX_SPEED
from .constants import WEAPON_RADIUS
from hlt.constants import TOL, PRECISION, FUDGE
from hlt.constants import SEPS_DOCK, SEPS_FIGHT, SEPS_EVADE, SEPS_RALLY
from .entity import Entity, Planet, Ship, Pos, Bound, Vect
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
    self._ships = dict()  # ship_id -> Ship
    # custom
    self.x_min = self.y_min = 0
    self.x_max = self.width
    self.y_max = self.height
    self.center = Pos(self.width/2, self.height/2, r=0.)
    self.bound = Bound(0, width, 0, height)
    # memos: updated beginning of each turn
    self.rec = dict()
    self.recs = dict()  # player_id -> player_record
    self.obsts = set()  # static Entities
    self.dests = set()  # mobo ships' routed next Pos
    self.paths = []  # dynamic-obsts: plotted path Vects; e1 also gets added to self.dests

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

  def get_ship(self, ship_id):
    return self._ships.get(ship_id)

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
    # update game_map._ships
    self._ships = dict()
    for player in pid2player.values():
      self._ships.update(player.sid2ship)
    # logging.critical(self._ships)

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

  def __intersects_entity(self, target):
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

  def __obstacles_between(self, ship, target, ignore=()):
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


  ### MEMO
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
    # check if to deregister prev goal
    # ship can only have 1 goal
    prev_goal = ship.goal
    if prev_goal and isinstance(prev_goal, (Ship, Planet)):
      prev_goal.goals_of.discard(ship)  # won't complain like remove()
    # register curr goal
    # goal might have >= 1 ship
    ship.goal = goal
    if isinstance(goal, Ship) or isinstance(goal, Planet):
      goal.goals_of.add(ship)

  # in-turn memos
  # calc turn-stats -> namedtuple record
  # @timit
  def recalc_stats_record(self, player_id=None):
    all_ships = set(self._all_ships())
    my_ships = set(self.get_me().all_ships())
    my_mobos = set(s for s in my_ships if s.is_mobo())
    my_imobos = my_ships - my_mobos
    logging.debug('My Ships: %s/%s MOBO!', len(my_mobos), len(my_ships))
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
  def reset_obsts(self):
    r = self.rec
    self.obsts = r.all_planets | r.my_imobos | r.foe_imobos

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
    # self.add_dest(new_path.e1)

  # recalc/reset in-turn memos
  # @timit
  def turn_update_memos(self):
    self.recalc_stats_record()
    self.reset_obsts()
    self.reset_paths()
    self.reset_dests()


  ### NAV
  # -> adj_pos: Pos
  # bound target pos to within game_map
  # TODO fix
  def bounds(self, target):
    return (
      self.x_min <= target.x <= self.x_max
      and self.y_min <= target.y <= self.y_max
    )

  def bound_target(self, ship, target):
    logging.warning(target)
    if self.bounds(target):
      return target
    sx = 1 if target.x >= ship.x else -1
    sy = 1 if target.y >= ship.y else -1
    # determine where out-of-bounds
    adj_x = adj_y = None
    if target.x < 0:
      adj_x = self.x_min
    elif target.x > self.width:
      adj_x = self.x_max
    if target.y < 0:
      adj_y = self.y_min
    elif target.y > self.height:
      adj_y = self.y_max
    # calc necessary adjustments
    dx = dy = 0
    if adj_x is not None and adj_y is not None:
      dx = sx * abs(adj_x - ship.x)
      dy = sy * abs(adj_y - ship.y)
    elif adj_x is not None:
      dx = sx * abs(adj_x - ship.x)
      dy = sy * sqrt(MAX_SPEED**2 - dx**2)
    elif adj_y is not None:
      dy = sy * abs(adj_y - ship.y)
      dx = sx * sqrt(MAX_SPEED**2 - dy**2)
    # make & return adjusted target
    adj_target = Pos(ship.x+dx, ship.y+dy, SHIP_RADIUS)
    if not self.bounds(adj_target):
      logging.warning('still not bound: %s', adj_target)
    logging.warning('%s adjusts %s -> %s', ship, target, adj_target)
    return adj_target

  # -> target: Entity
  # given target: Pos, derive context-appropriate nav_target
  # get engine-compliant, reachable nav_target from target
  # int-dist & int-angle from ship a la engine-constraint
  def get_nav_target(self, ship, seps, log=True):    
    # make eff_target
    sep_opt = seps[1]  # optimal separation
    # nav_target = ship.perigee(ship.target, sep_opt)
    assert ship.target is not None
    angle = ship.angle(ship.target)
    if self._calc_sep(ship, ship.target, log) < sep_opt:
      angle = (angle + 180) % 360
    eff_target = self._get_min_d_pos(ship, angle, ship.target, seps, log)
    if log:
      logging.warning('target: %s', ship.target)
      logging.warning('optimal sep: %.1f', sep_opt)
      logging.warning('eff_target: %s', eff_target)
    return eff_target


  def _filter_obstacles(self, ship_path, evade_paths, log=True):
    ship, nav_target = ship_path.e0, ship_path.e1
    # TODO smarter filter
    paths = self.paths + evade_paths
    filter_paths = [p for p in paths
      if ship_path.center.dist(p.center) <= MAX_SPEED + ship_path.radius + p.radius]

    my_pending_mobos = set([s for s in self.rec.my_mobos if not s.dest])
    poses = my_pending_mobos | self.obsts | self.dests
    poses -= set([ship, nav_target])
    assert ship not in poses and nav_target not in poses
    filter_poses = sorted([s for s in poses
      if ship_path.center.dist(s) <= 0.5*MAX_SPEED + ship_path.radius + s.radius],
      key=lambda s: ship.dist(s)
    )
    obstacles = filter_paths + filter_poses
    # TEST add boundaries as hard-obstacles
    obstacles.append(self.bound)

    # if log:
      # logitr(obstacles, 'filtered obstacles', 30)
    return obstacles

  def _get_nearest_hit(self, ship_path, obstacles, log=True):
    # find min_dist_hit / reach
    max_reach_dist = min(MAX_SPEED, int(round(ship_path.len)))
    min_hit_ent = None
    for ob in obstacles:
      d, e = ship_path.cross(ob)
      # regular
      if d + TOL < max_reach_dist:
        max_reach_dist = d
        min_hit_ent = e
    if log:
      logging.info('reach: %.1f; hit_ent:%s', max_reach_dist, min_hit_ent)
    return max_reach_dist, min_hit_ent

  # target <= MAX_SPEED away
  # -> reach: float| furthest flight w/o hit, rounded down nearest int
  # -> nearest_hit: Entity| first Entity to hit along |ship->target|
  def max_reach(self, ship, nav_target, log=True):
    logging.info('Trying ^%d .-.-> %s...', ship.angle(nav_target), nav_target)
    ship_path = Vect(ship, nav_target)
    obstacles = self._filter_obstacles(ship_path, ship.evade_paths, log)
    reach, min_hit_ent = self._get_nearest_hit(ship_path, obstacles, log)
    # reach = max(0, max_reach_dist)
    # convert reach to int:
    # generally round down to avoid overshoot unless
    # really close to its nearest int (indicative of float-pt rounding issue)
    if abs(reach - round(reach)) >= TOL:
      reach = int(reach)
    else:
      reach = int(round(reach))
    # create reach_pos based on adjusted reach
    if reach <= TOL:
      reach_pos = ship.curr_pos
    else:
      reach_pos = ship_path.d_along(reach)

    sep = self._calc_sep(reach_pos, ship.target)
    if log:
      logging.info('final srpe: %.2f, %d, %s, %s', sep, reach, reach_pos, min_hit_ent)
    return sep, reach, reach_pos, min_hit_ent


  # -> (tang_l_ngl, tang_l): (int, Entity)
  # -> (tang_r_ngl, tang_r): (int, Entity)
  def _get_tangs(self, src, dst, fudge=FUDGE):
    assert isinstance(src, Entity) and isinstance(dst, Entity)
    logging.info('%s /O\ %s', src, dst)
    logging.debug('%s -> %s', src, dst)
    rise = dst.radius + SHIP_RADIUS + fudge
    logging.debug('rise (tang-dst): %s', rise)
    hypo = max(rise, src.dist(dst))
    logging.debug('hypo (src-dst): %s', hypo)
    run = sqrt(abs(hypo**2 - rise**2))
    logging.debug('run (src-tang): %s', run)
    int_length_to_tang = ceil(run)
    logging.debug('int_length_to_tang: %s', int_length_to_tang)
    abs_ngl_src_obst = src.angle(dst)  # coordinate-system absolute-angle
    logging.debug('abs ngl src->obst: %s', abs_ngl_src_obst)
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
    # tang_l = src.get_pos(int_length_to_tang, tang_l_ngl)
    tang_r_ngl = ceil(abs_ngl_src_obst + d_angle) % 360
    logging.debug('right-tangent abs-ngl: %s', tang_r_ngl)
    # tang_r = src.get_pos(int_length_to_tang, tang_r_ngl)
    # logging.info('tangs: ^%s:%s; ^%s:%s', tang_l_ngl, tang_l, tang_r_ngl, tang_r)
    logging.info('tang_ngls: ^%s & ^%s', tang_l_ngl, tang_r_ngl)
    # return (tang_l_ngl, tang_l), (tang_r_ngl, tang_r)
    return tang_l_ngl, tang_r_ngl

  # -> pos: Pos|
  # ship ^:angle d:[seps] -> pos .-.-> target
  def _get_min_d_pos(self, ship, angle, target, seps, log=True):
    sep_min, sep_opt, sep_max = seps
    # memo
    reach = 0
    pos = ship.curr_pos
    sep = self._calc_sep(pos, target)
    min_diff = self._diff_opt(sep, seps)  # distance to optimal-sep

    while reach < MAX_SPEED:
      reach += 1
      next_pos = ship.get_pos(reach, angle)
      sep = self._calc_sep(next_pos, target)
      diff = self._diff_opt(sep, seps)
      if min_diff < diff:
        break
      else:
        pos = next_pos
        min_diff = diff
        # if log:
        #   logging.warning('%s d%d ^%d %s -> new min_diff: %.2f', ship, reach, angle, pos, min_diff)
    if log:
      logging.warning('^%d; tgt: %s; seps: %s', angle, target, seps)
      logging.info('%s -> %s: sep: %.1f; diff_opt: %.1f.-> %s', ship, pos, self._calc_sep(pos, target), min_diff, target)
    return pos

  # if explored or close enough to an explored pos
  def _seen_pos(self, pos, explored):
    return (
      not isinstance(pos, Entity)
      or any(ex.dist(pos) <= FUDGE for ex in explored)
    )

  # -> min_res: (min_diff, reach, angle)
  # recurse check next tangent while diff(sep, sep_opt) improving
  # keep checking while unexplored min_hit_ent
  def tang_update_min_res(self, explored, ship, min_hit_ent, seps, min_res, tang_dir=None):
    explored.add(min_hit_ent)
    # check both tangents
    if not tang_dir:
      l_explored = copy.copy(explored)
      res_left = self.tang_update_min_res(l_explored, ship, min_hit_ent, seps, min_res, 'L')
      r_explored = copy.copy(explored)
      res_right = self.tang_update_min_res(r_explored, ship, min_hit_ent, seps, min_res, 'R')
      min_res = min(min_res, res_left, res_right)
      return min_res
    # check 1-tangent
    tang_l_ngl, tang_r_ngl = self._get_tangs(ship, min_hit_ent)
    if tang_dir == 'L':
      angle = tang_l_ngl
    elif tang_dir == 'R':
      angle = tang_r_ngl
    pos = self._get_min_d_pos(ship, angle, ship.target, seps)
    sep, reach, reach_pos, min_hit_ent = self.max_reach(ship, pos)
    logging.info('Check tang %s: ^%d %s', tang_dir, angle, pos)
    logging.info('sep: %.2f: tang max_reach', sep)
    res = (self._diff_opt(sep, seps), reach, angle)
    min_res = min(min_res, res)
    # keep checking unexplored tangents while still hitting
    if not self._seen_pos(min_hit_ent, explored):
      logging.warning('Recursing: %s -> %s', min_hit_ent, explored)
      min_res = self.tang_update_min_res(explored, ship, min_hit_ent, seps, min_res, tang_dir)
    return min_res

  # separation net radii
  def _calc_sep(self, e0, e1, log=False):
    assert e0 and e1 and e0.radius is not None and e1.radius is not None
    sep = e0.dist(e1) - e0.radius - e1.radius
    if log:
      logging.info('%s --%.2f-- %s', e0, sep, e1)
    return sep

  # diff(sep, sep_opt)
  def _diff_opt(self, sep, seps):
    sep_min, sep_opt, sep_max = seps
    d = abs(sep - sep_opt)
    # penalize out-of-bounds
    if not (sep_min <= sep <= sep_max):
      d += 100
    return d

  # -> nav_comm: Thrust|
  # -> dest: Entity|
  # minimize diff(sep, optimal-sep)
  def nav(self, ship, seps, log=True):
    logging.info('%s .-.-%s.-.-> %s', ship, seps, ship.target)
    # set realistic nav_target given separation-constraints
    nav_target = self.get_nav_target(ship, seps)
    # get closest beeline -> nav_target
    sep, reach, reach_pos, min_hit_ent = self.max_reach(ship, nav_target, log)
    logging.info('sep: %s: beeline', sep)
    # memo
    min_res = (
      self._diff_opt(sep, seps),
      reach,
      ship.angle(nav_target)
    )
    # if hit, recursive-check tangent-beelines
    if min_hit_ent:
      min_res = self.tang_update_min_res(set(), ship, min_hit_ent, seps, min_res)

    _, reach, angle = min_res
    # TEST sanity check for no collision
    tgt = ship.get_pos(reach, angle)
    _, _, _, hit_ent = self.max_reach(ship, tgt)
    if hit_ent and hit_ent.owner.id == self.my_id:
      logging.critical('HIT! %s', hit_ent)
      while hit_ent and reach > 0:
        reach -= 1
        tgt_shorter = ship.get_pos(reach, angle)
        _, _, _, hit_ent = self.max_reach(ship, tgt_shorter)
      logging.critical('HIT? %s', hit_ent)
    # return whichever w/ min_hit_ent nearest target
    thrust = ship.thrust(reach=reach, angle=angle)
    ship.dest = ship.get_pos(reach=reach, angle=angle)
    logging.warning('final: %s -> %s d%.2fr%d^%d ~%.2f ..> %s', ship, ship.dest, *min_res, seps[1], ship.target)
    return thrust


  ### PREDICT
  # ? update online
  # TODO consider what opponent considers I will do...

  # game-engine-style-integerize thrust reach & angle
  def _engint(self, reach=None, angle=None):
    if reach is not None:
      return int(reach)
    elif angle is not None:
      return round(angle)

  # -> path: Vect
  # TODO also predict for foe on mine
  # TODO incorporate lite version of own nav logic ?
  def next_path(self, ship, log=False):
    if ship.next_path:
      return ship.next_path
    # if no target assume stay
    if not ship.goal:
      return Vect(ship, ship.curr_pos)

    seps = (0., 0., 0.)
    if isinstance(ship.goal, Planet):
      seps = SEPS_DOCK
    if isinstance(ship.goal, Ship):
      seps = SEPS_FIGHT
    # assume foe beelines for goal
    ship.target = ship.goal
    nav_target = self.get_nav_target(ship, seps, log)
    # reach = min( self._engint(reach=ship.dist(target)), MAX_SPEED )
    # # logging.info('reach: %s', reach)
    # angle = self._engint(angle=ship.angle(target))
    # eff_target = ship.get_pos(reach, angle)
    # eff_target.owner = ship
    # return Vect(ship, eff_target, r=WEAPON_RADIUS)
    return Vect(ship, nav_target)

  # contextualize Ship.next_pos
  # eg adjust planet-interior to nearest point on planet-surface
  # based on player_eval_goals
  # -> Pos
  def next_pos(self, ent, log=False):
    next_pos = None
    if isinstance(ent, (Planet, Pos)):
      next_pos = ent
    elif ent.next_pos:
      next_pos = ent.next_pos
    else:
      assert isinstance(ent, Ship)
      if not ent.is_mobo():
        ent.next_pos = ent.curr_pos
      else:  # mobo
        # TODO improve core guess logic
        ship_path = self.next_path(ent, log=log)
        ent.next_pos = ship_path.e1
        for planet in self._planets.values():
          if planet.overlap(ent.next_pos):
            ent.next_pos = ent.perigee(planet, sep=SHIP_RADIUS)
            break
      next_pos = ent.next_pos

    assert isinstance(next_pos, (Planet, Pos))
    return next_pos



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
