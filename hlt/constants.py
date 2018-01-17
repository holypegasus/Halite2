import math


#: Max number of units of distance a ship can travel in a turn
MAX_SPEED = 7
#: Radius of a ship
SHIP_RADIUS = 0.5
#: Starting health of ship, also its max
MAX_SHIP_HEALTH = 255
#: Starting health of ship, also its max
BASE_SHIP_HEALTH = 255
#: Weapon cooldown period
WEAPON_COOLDOWN = 1
#: Weapon damage radius
WEAPON_RADIUS = 5.0
#: Weapon damage
WEAPON_DAMAGE = 64
#: Radius in which explosions affect other entities
EXPLOSION_RADIUS = 10.0
#: Distance from the edge of the planet at which ships can try to dock
DOCK_RADIUS = 4.0
#: Number of turns it takes to dock a ship
DOCK_TURNS = 5
#: Number of production units per turn contributed by each docked ship
BASE_PRODUCTIVITY = 6
#: Production to build 1 ship
PROD_FOR_SHIP = 72
#: Distance from the planets edge at which new ships are created
SPAWN_RADIUS = 2.0


# custom
#: Min distance a ship needs to move (other than staying = 0)
MIN_MOVE_SPEED = 1
#: numericals
PRECISION = 4  # decimal-places
TOL = 1e-3  # engine numerical tolerance
FUDGE = 5e-2  # engine ship-geometry tolerance
#: Various separation bounds (min, optimal, max); net radii !
SEP_KAMIKAZE = 0.
SEPS_DOCK = (
  FUDGE,
  FUDGE,
  DOCK_RADIUS - TOL
)
SEPS_FIGHT = (
  FUDGE,
  WEAPON_RADIUS - 2*SHIP_RADIUS - FUDGE,
  WEAPON_RADIUS - 2*SHIP_RADIUS - FUDGE
)
SEPS_EVADE = (
  WEAPON_RADIUS - 2*SHIP_RADIUS + FUDGE,
  # WEAPON_RADIUS - 2*SHIP_RADIUS + FUDGE,
  WEAPON_RADIUS + MAX_SPEED,
  math.inf
)
SEPS_RALLY = (
  FUDGE,
  FUDGE,
  math.inf
)
