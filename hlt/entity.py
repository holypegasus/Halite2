import abc, logging, math
from collections import deque
from enum import Enum
from operator import truediv

from .constants import DOCK_TURNS, BASE_PRODUCTIVITY, PROD_FOR_SHIP, SPAWN_RADIUS
from .constants import SHIP_RADIUS, DOCK_RADIUS, WEAPON_RADIUS, MIN_MOVE_SPEED, MAX_SPEED
from .constants import TOLERANCE, PRECISION
from .util import logitr



class Entity:
  """
  Then entity abstract base-class represents all game entities possible. As a base all entities possess
  a Pos, radius, health, an owner and an id. Note that ease of interoperability, Pos inherits from
  Entity.

  :ivar id: The entity ID
  :ivar x: The entity x-coordinate.
  :ivar y: The entity y-coordinate.
  :ivar radius: The radius of the entity (may be 0)
  :ivar health: The entity's health.
  :ivar owner: The player ID of the owner, if any. If None, Entity is not owned.
  """
  __metaclass__ = abc.ABCMeta

  def _init__(self, x, y, radius, health, player, entity_id):
    self.x = x
    self.y = y
    self.radius = radius
    self.health = health
    self.owner = player
    self.oid = None
    self.id = entity_id

  def dist(self, target, precision=PRECISION):
    d = math.sqrt((target.x - self.x) ** 2 + (target.y - self.y) ** 2)
    d = round(d, precision)
    return d

  # -> target-type-adjusted effective-distance
  def eff_dist(self, target, precision=PRECISION):
    dist = self.dist(target, precision)
    if isinstance(target, Planet):
      dist = dist - target.radius - DOCK_RADIUS
    elif isinstance(target, Ship):
      dist = dist - WEAPON_RADIUS
    # dist = max(0, dist)
    return dist

  def same_pos(self, other, tol=TOLERANCE):
    assert isinstance(other, Entity)
    return abs(self.x - other.x) < tol and abs(self.y - other.y) < tol

  # if 2 Entities' radii overlap
  def overlap(self, other, fudge=0.01):
    dist = self.dist(other)
    no_hit_radius = self.radius + other.radius
    if dist - fudge < no_hit_radius:
      logging.debug('%s<->%s: d: %s ?<= %s', self, other, dist, no_hit_radius)
      return True
    else:
      return False

  def angle(self, target):
    return math.degrees(math.atan2(target.y - self.y, target.x - self.x)) % 360
  
  # -> perigee: Pos
  # self ---- perigee : min_dist : target.radius : target
  def perigee(self, target, min_dist):
    angle = target.angle(self)
    x = target.x + (target.radius+min_dist) * math.cos(math.radians(angle))
    y = target.y + (target.radius+min_dist) * math.sin(math.radians(angle))
    return Pos(x, y)

  # -> Pos
  def get_pos(self, reach, angle, r=SHIP_RADIUS):
    new_target_dx = math.cos(math.radians(angle)) * reach
    new_target_dy = math.sin(math.radians(angle)) * reach
    return Pos(
      self.x + new_target_dx,
      self.y + new_target_dy,
      r=r
    )

  @abc.abstractmethod
  def _link(self, players, planets):
    pass

  def __str__(self):
    return "Entity %s (id: %s) at Pos: (x = %s, y = %s), with radius = %s"%(self.__class__.__name__, self.id, self.x, self.y, self.radius)

  def __repr__(self):
    return self.__str__()



class Planet(Entity):
  """
  A planet on the game map.

  :ivar id: The planet ID.
  :ivar x: The planet x-coordinate.
  :ivar y: The planet y-coordinate.
  :ivar radius: The planet radius.
  :ivar num_docking_spots: The max number of ships that can be docked.
  :ivar current_production: How much production the planet has generated at the moment. Once it reaches the threshold, a ship will spawn and this will be reset.
  :ivar remaining_resources: The remaining production capacity of the planet.
  :ivar health: The planet's health.
  :ivar owner: The player ID of the owner, if any. If None, Entity is not owned.

  """

  def __init__(self, planet_id, x, y, hp, radius, docking_spots, current,
         remaining, owned, owner_id, docked_ships):
    self.id = planet_id
    self.x = x
    self.y = y
    self.radius = radius
    self.num_docking_spots = docking_spots
    self.current_production = current
    self.remaining_resources = remaining  # depr
    self.health = hp
    self.owner_id = owner_id if bool(int(owned)) else None
    self.owner = None  # filled by _link()
    self._docked_ship_ids = docked_ships
    self._docked_ships = {}
    ###
    self.goals_of = set()
    self.pid2goals_of = dict()  # pid -> goals_of: set
    self.n_open_docks = self.num_docking_spots - len(self.goals_of)
    ### TOUSE
    self.spawn_pos = None  # filled by set_spawn_pos during game_map._parse()
    self.perimeter_threats = deque()  # (dist, next_pos, foe)

  #######
  # NB update instead of recreate planet
  def update_planet(self, new_planet):
    assert(
      type(self)==type(new_planet) and
      self.id==new_planet.id)
    # update changeable fields
    self.current_production = new_planet.current_production
    self.health = new_planet.health
    self.owner_id = new_planet.owner_id
    self.owner = new_planet.owner
    self._docked_ship_ids = new_planet._docked_ship_ids
    self._docked_ships = new_planet._docked_ships
    self.n_open_docks = self.num_docking_spots - len(self.goals_of)

  # update dependencies to reflect its deletion
  def del_planet(self):
    for goal_of in self.goals_of:
      goal_of.goal = None
    self.goal_of = set()

  # -> bool
  # if pos is inside planet
  def __contains__(self, pos):
    return self.dist(pos) < self.radius

  def set_spawn_pos(self, center):
    self.spawn_pos = center.perigee(self, min_dist=SPAWN_RADIUS)

  def turns_to_next_ship(self, add_producer=0):
    prod_to_next_ship = (PROD_FOR_SHIP - self.current_production) / BASE_PRODUCTIVITY
    per_turn_prod = len(self._docked_ships) + add_producer
    if per_turn_prod == 0:
      return float('inf')
    else:
      return prod_to_next_ship / per_turn_prod
  #######

  def get_docked_ship(self, ship_id):
    """
    Return the docked ship designated by its id.

    :param int ship_id: The id of the ship to be returned.
    :return: The Ship object representing that id or None if not docked.
    :rtype: Ship
    """
    return self._docked_ships.get(ship_id)

  def all_docked_ships(self):
    """
    The list of all ships docked into the planet

    :return: The list of all ships docked
    :rtype: list[Ship]
    """
    return list(self._docked_ships.values())

  def is_owned(self):
    """
    Determines if the planet has an owner.
    :return: True if owned, False otherwise
    :rtype: bool
    """
    return self.owner is not None

  def is_full(self):
    """
    Determines if the planet has been fully occupied (all possible ships are docked)

    :return: True if full, False otherwise.
    :rtype: bool
    """
    return len(self._docked_ship_ids) >= self.num_docking_spots

  def _link(self, players, planets):
    """
    This function serves to take the id values set in the parse function and use it to populate the planet
    owner and docked_ships params with the actual objects representing each, rather than IDs

    :param dict[int, gane_map.Player] players: A dictionary of player objects keyed by id
    :return: nothing
    """
    if self.owner_id is not None:
      self.owner = players.get(self.owner_id)
      for sid in self._docked_ship_ids:
        self._docked_ships[sid] = self.owner.get_ship(sid)

  @staticmethod
  def _parse_single(tokens):
    """
    Parse a single planet given tokenized input from the game environment.

    :return: The planet ID, planet object, and unused tokens.
    :rtype: (int, Planet, list[str])
    """
    (plid, x, y, hp, r, docking, current, remaining,
     owned, owner_id, num_docked_ships, *remainder) = tokens

    plid = int(plid)
    docked_ships = []

    for _ in range(int(num_docked_ships)):
      ship_id, *remainder = remainder
      docked_ships.append(int(ship_id))

    planet = Planet(int(plid),
            float(x), float(y),
            int(hp), float(r), int(docking),
            int(current), int(remaining),
            bool(int(owned)), int(owner_id),
            docked_ships)

    return plid, planet, remainder

  @staticmethod
  def _parse(tokens):
    """
    Parse planet data given a tokenized input.

    :param list[str] tokens: The tokenized input
    :return: the populated planet dict and the unused tokens.
    :rtype: (dict, list[str])
    """
    num_planets, *remainder = tokens
    num_planets = int(num_planets)
    planets = {}

    for _ in range(num_planets):
      plid, planet, remainder = Planet._parse_single(remainder)
      planets[plid] = planet
    # logitr(planets, 'planets', 30)
    return planets, remainder

  # TMP tie-break priority-queue
  def __lt__(self, other):
    return (
      # self.n_open_docks < other.n_open_docks or
      # self.current_production > other.current_production or
      self.id < other.id
    )

  def __str__(self):
    return "{}_{}_{} @({:.2f}, {:.2f}, {:.2f})".format(
      self.owner_id , self.__class__.__name__[:2], self.id, self.x, self.y, self.radius)


class Ship(Entity):
  """
  A ship in the game.
  
  :ivar id: The ship ID.
  :ivar x: The ship x-coordinate.
  :ivar y: The ship y-coordinate.
  :ivar radius: The ship radius.
  :ivar health: The ship's remaining health.
  :ivar DockingStatus docking_status: The docking status (UNDOCKED, DOCKED, DOCKING, UNDOCKING)
  :ivar planet: The ID of the planet the ship is docked to, if applicable.
  :ivar owner: The player ID of the owner, if any. If None, Entity is not owned.
  """

  class DockingStatus(Enum):
    UNDOCKED = 0
    DOCKING = 1
    DOCKED = 2
    UNDOCKING = 3

  def __init__(self, owner_id, ship_id, x, y, hp, vel_x, vel_y, docking_status, planet_id, progress, cooldown):
    self.id = ship_id
    self.x = x
    self.y = y
    self.owner_id = owner_id
    self.owner = None  # filled by _link()
    self.radius = SHIP_RADIUS
    self.health = hp
    self.docking_status = docking_status
    self.planet_id = planet_id if (docking_status is not Ship.DockingStatus.UNDOCKED) else None
    self.planet = None  # filled by _link()
    self._docking_progress = progress
    self._weapon_cooldown = cooldown
    # memory
    self.goal = None
    self.dest = None
    # self.goal_status = None
    self.goals_of = set()
    self.pid2goals_of = dict()  # pid -> goals_of: set
    self.curr_pos = Pos(x, y, SHIP_RADIUS, pos_id=self.id)
    self.prev_vect = Vect()
    self.prev_vects = []
    self.swarm = set()
    # predictives
    # NB reset to None beginning of each turn; set by first call to game_map.next_pos() (including foe's eval_goal !)
    self.next_pos = None
    self.next_path = Vect()

  #######
  # NB update existing ship w/ new info instead of creating new
  def update_ship(self, new_ship):
    assert (
      type(self)==type(new_ship) and
      self.owner_id==new_ship.owner_id and
      self.id==new_ship.id)
    # deduce previous Vect move
    prev_pos = self.curr_pos
    curr_pos = Pos(new_ship.x, new_ship.y, new_ship.radius, pos_id=self.id)
    self.prev_vect = Vect(prev_pos, curr_pos, vector_id=self.id, owner_id=self.owner_id)
    self.prev_vects.append(self.prev_vect)
    self.curr_pos = curr_pos
    self.dest = None
    self.next_pos = None
    # swarm stuff
    # other updates
    self.x = new_ship.x
    self.y = new_ship.y
    self.health = new_ship.health
    self.docking_status = new_ship.docking_status
    self.planet = new_ship.planet
    self._docking_progress = new_ship._docking_progress
    self._weapon_cooldown = new_ship._weapon_cooldown

  # update dependencies to reflect its deletion
  def del_ship(self):
    if self.goal:
      self.goal.goals_of.remove(self)
      self.goal = None
    for buddy in self.swarm:
      if buddy != self:
        buddy.swarm.remove(self)

  def join_swarm(self, ship):
    # check if already in swarm
    if self in ship.swarm:
      pass
    # leave prev swarm
    # logging.warning('%s -> %s', self.swarm, self)
    for s in self.swarm:
      s.swarm.remove(self)
    # join new swarm
    # logging.warning('%s -> %s', self, ship.swarm)
    for s in ship.swarm:
      s.swarm.add(self)
    self.swarm = ship.swarm | set([ship])
    ship.swarm.add(self)

  def is_mobo(self):
    return self.docking_status == self.DockingStatus.UNDOCKED

  # TMP if foe_ship is matched in strength
  def is_matched(self, assigned_ships=set()):
    if self.is_mobo():
      planned_ships = self.goals_of
      return sum(s.health for s in planned_ships) >= self.health
    else:
      return True if self.goals_of else False

  def curr_pos(self):
    if not self.curr_pos:
      self.curr_pos = Pos(self.x, self.y, self.radius, self.id, self.owner_id)
    return self.curr_pos

  # ?
  def next_pos(self):
    return self.next_pos
  #######


  def thrust(self, reach, angle):
    """
    Generate a command to accelerate this ship.

    :param int reach: The speed through which to move the ship
    :param int angle: The angle to move the ship in
    :return: The command string to be passed to the Halite engine.
    :rtype: str
    """

    # we want to round angle to nearest integer, but we want to round
    # reach down to prevent overshooting and unintended collisions
    # log actual thrust engine will run
    logging.debug( '%s THRUST v%.0f^%.0f', self, int(reach), round(angle) )
    return "t {} {} {}".format(self.id, int(reach), round(angle))

  def can_dock(self, planet):
    """
    Determine whether a ship can dock to a planet

    :param Planet planet: The planet wherein you wish to dock
    :return: True if can dock, False otherwise
    :rtype: bool
    """
    assert isinstance(planet, Planet)
    return self.dist(planet) <= planet.radius + DOCK_RADIUS

  def can_dock_tight(self, planet):
    assert isinstance(planet, Planet)
    return self.dist(planet) <= planet.radius + SHIP_RADIUS*3


  def dock(self, planet):
    """
    Generate a command to dock to a planet.

    :param Planet planet: The planet object to dock to
    :return: The command string to be passed to the Halite engine.
    :rtype: str
    """
    return "d {} {}".format(self.id, planet.id)

  def undock(self):
    """
    Generate a command to undock from the current planet.

    :return: The command trying to be passed to the Halite engine.
    :rtype: str
    """
    return "u {}".format(self.id)

  def _link(self, players, planets):
    """
    This function serves to take the id values set in the parse function and use it to populate the ship
    owner and docked_ships params with the actual objects representing each, rather than IDs

    :param dict[int, game_map.Player] players: A dictionary of player objects keyed by id
    :param dict[int, Planet] players: A dictionary of planet objects keyed by id
    :return: nothing
    """
    self.owner = players.get(self.owner_id)  # All ships should have an owner. If not, this will just reset to None
    self.planet = planets.get(self.planet_id)  # If not will just reset to none

  @staticmethod
  def _parse_single(player_id, tokens):
    """
    Parse a single ship given tokenized input from the game environment.

    :param int player_id: The id of the player who controls the ships
    :param list[tokens]: The remaining tokens
    :return: The ship ID, ship object, and unused tokens.
    :rtype: int, Ship, list[str]
    """
    (sid, x, y, hp, vel_x, vel_y,
     docked, docked_planet_id, progress, cooldown, *remainder) = tokens

    sid = int(sid)
    docked = Ship.DockingStatus(int(docked))

    ship = Ship(player_id,
          sid,
          float(x), float(y),
          int(hp),
          float(vel_x), float(vel_y),
          docked, int(docked_planet_id),
          int(progress), int(cooldown))
    # logging.info('Parsed ship: %s', ship)
    return sid, ship, remainder

  @staticmethod
  def _parse(player_id, tokens):
    """
    Parse ship data given a tokenized input.

    :param int player_id: The id of the player who owns the ships
    :param list[str] tokens: The tokenized input
    :return: The dict of Players and unused tokens.
    :rtype: (dict, list[str])
    """
    ships = {}
    num_ships, *remainder = tokens
    for _ in range(int(num_ships)):
      ship_id, ships[ship_id], remainder = Ship._parse_single(player_id, remainder)
    return ships, remainder

  # TMP tie-break priority-queue
  def __lt__(self, other):
    return self.id < other.id

  def __str__(self):
    return '%s_%s_%s @(%.2f, %.2f)'%(
      self.owner_id, self.__class__.__name__[:1], self.id, self.x, self.y)



class Pos(Entity):
  """
  A simple wrapper for a coordinate. Intended to be passed to some functions in place of a ship or planet.

  :ivar id: Unused
  :ivar x: The x-coordinate.
  :ivar y: The y-coordinate.
  :ivar radius: The Pos's radius (should be 0).
  :ivar health: Unused.
  :ivar owner: Unused.
  """

  def __init__(self, x, y, r=0., pos_id=None, owner_id=None, health=None):
    self.x = x
    self.y = y
    self.radius = r
    self.health = health
    self.id = pos_id
    self.owner_id = owner_id

  #######
  # -> Vect
  def __sub__(self, e1):
    return Vect(e1, self)

  # -> Pos
  def __add__(self, vect):
    assert isinstance(vect, Vect)
    return Pos(self.x + vect.dx, self.y + vect.dy, self.radius)

  #######

  def _link(self, players, planets):
    raise NotImplementedError("Pos should not have link attributes.")

  def __str__(self):
    return '@~(%.2f, %.2f)'%(self.x, self.y)


# ? move util.Line here
# move their tests into test.py
class Vect(Entity):
  def _params(self):
    if abs(self.dx) <= self.tol:  # eq for vertical-line
      return self.x0, None  # b==None -> vertical-line
    m = truediv(self.dy, self.dx)
    b = self.y0 - m * self.x0
    return m, b

  def __init__(self, e0=None, e1=None, r=0., tol=TOLERANCE, pc=PRECISION, vector_id=None, owner_id=None):
    self.e0 = e0
    self.e1 = e1
    if e0 and e1:
      self.dx = float(e1.x - e0.x)
      self.dy = float(e1.y - e0.y)
      if r:
        self.radius = float(r)
      else:
        self.radius = e0.radius or e1.radius
      self.center = Pos(
        x=self.e0.x+self.dx/2,
        y=self.e0.y+self.dy/2,
        r=self.radius)
    else:  # 0-Vect by default
      self.dx = self.dy = 0.
      self.radius = float(r)
      self.center = None
    self.len = (self.dx**2 + self.dy**2) ** 0.5
    # self.angle = e0.angle(e1)
    self.tol = tol
    self.precision = pc
    self.id = vector_id
    self.owner_id = owner_id

  def is_pos(self):
    return self.e0.same_pos(self.e1) or self.len < MIN_MOVE_SPEED

  # Pos @ time (out of 1. so aka ratio) along Vect
  def t_along(self, time):
    x = self.e0.x + self.dx * time
    y = self.e0.y + self.dy * time
    return Pos(x, y, self.radius)

  # Pos @ dist along Vect
  def d_along(self, dist, tol=TOLERANCE):
    fract_turn = float(dist) / self.len
    assert fract_turn <= 1.+tol, 'fract_turn = %s'%fract_turn
    return self.t_along(fract_turn)

  # -> pre_overlap_dist: float; pre_overlap_pos: Pos; overlap_ent: Entity
  # first overlap (dist, obst_pos)
  # time-sensitive intersection check by checking discrete segment-pts along
  # works for Vect vs Vect or Vect vs Pos
  # TODO improve by making check geometric instead of discrete?
  def cross(self, other, precision=PRECISION):
    # max-speed of 7 -> max 14 radius-0.5 'bubbles'
    # TODO mathematize check VS finer intervals ?
    n_segments = int(MAX_SPEED / SHIP_RADIUS) * 2
    dt = 1 / n_segments
    pre_overlap_dist = self.len
    pre_overlap_pos = self.e1
    overlap_ent = None
    # TODO consider if self = 0-len Vect aka Pos
    prev_dist = float('inf')
    if isinstance(other, Vect):  # comp self.seg_Pos w/ other.seg_Pos t_along
      for k in range(1, n_segments+1):
        t = k * dt
        pos_self = self.t_along(t)
        pos_other = other.t_along(t)
        if pos_self.overlap(pos_other):
          pre_overlap_dist = self.len * (t - dt)
          pre_overlap_pos = self.t_along(t - dt)
          overlap_ent = pos_other
          break
        else:
          curr_dist = pos_self.dist(pos_other)
          if curr_dist > prev_dist:  # getting further apart - early exit !
            break
          else:
            prev_dist = curr_dist
    else:  # some sort of Pos - comp self.seg_Pos t_along w/ other.seg_Pos
      for k in range(1, n_segments+1):
        t = k * dt
        pos_self = self.t_along(t)
        pos_other = other
        if pos_self.overlap(pos_other):
          pre_overlap_dist = self.len * (t - dt)
          pre_overlap_pos = self.t_along(t - dt)
          overlap_ent = pos_other
          break
        else:
          curr_dist = pos_self.dist(pos_other)
          if curr_dist > prev_dist:  # getting further apart - early exit !
            break
          else:
            prev_dist = curr_dist

    return pre_overlap_dist, pre_overlap_pos, overlap_ent

  def _gen_pts(self, n):  # generate n pts along line, including end-points e0, e1
    assert n > 1
    dx = self.dx / (n-1)
    dy = self.dy / (n-1)
    next_x, next_y = self.e0.x, self.e0.y
    return [
      Pos(
        round(next_x + i*dx, self.precision), 
        round(next_y + i*dy, self.precision), 
        r=self.radius)
      for i in range(n)]

  def __str__(self):
    # return '|%s->>(d:%.2f; ^:%.0f;r:%s)>>-%s|'%(self.e0, self.len, self.angle, self.radius, self.e1)
    return '|%s->>(d:%.2f;r:%s)>>-%s|'%(self.e0, self.len, self.radius, self.e1)

  def __repr__(self):
    return self.__str__()