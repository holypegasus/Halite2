import logging, math
import numpy as np
import pandas as pd
from collections import Counter, OrderedDict, defaultdict, namedtuple
from functools import partial, wraps
from operator import truediv
from time import time

LOG_LEVEL2FUNC = {
  logging.DEBUG: logging.debug,
  logging.INFO: logging.info,
  logging.WARNING: logging.warning,
  logging.CRITICAL: logging.critical,
}


def timit(f):
  @wraps(f)
  def wrap(*args, **kwargs):
    t0 = time()
    result = f(*args, **kwargs)
    t1 = time()
    logging.critical('%s took: %.0f ms', f.__name__, (t1-t0)*1000)
    return result
  return wrap


def timit_msg(msg):
  if msg:  print(msg)
  return timit


def setup_logger(name, file_name=None, lvl=logging.INFO):
  log = logging.getLogger(name)
  log.setLevel(lvl)
  # fh = logging.FileHandler('%s.log'%(file_name), mode='w')
  # fh.setFormatter(
  #   logging.Formatter('%(asctime)s <%(module)s.%(funcName)s:%(lineno)d> %(message)s', '%Y%m%d %H:%M:%S'))
  # log.addHandler(fh)
  sh = logging.StreamHandler()
  sh.setFormatter(
    logging.Formatter('%(asctime)s <%(module)s.%(funcName)s:%(lineno)d> %(message)s', '%Y%m%d %H:%M:%S'))
  log.addHandler(sh)
  return log


def logitr(itr, header='', lvl=logging.DEBUG, sort_key=None):
  log_func = LOG_LEVEL2FUNC.get(lvl)
  if header:
    log_func('%s: [%s]', header, len(itr))
  if isinstance(itr, dict):
    if not sort_key:
      sorted_items = sorted(itr.items())
    else:
      sorted_items = sorted(itr.items(), key=sort_key)

    for k, v in sorted(itr.items()):
      log_func('%s -> %s', k, v)
  else:
    for v in itr:
      log_func(v)

# customized rounding
def rnd(x, precision, direction=None):
  assert direction in (None, '+', '-')
  if not direction:
    return round(x, precision)

  exp_10 = 10**precision
  if direction == '+':
    return math.ceil(x*exp_10) / exp_10
  elif direction == '-':
    return math.floor(x*exp_10) / exp_10


# generic location object - mimic entity.Entity
class Loc:
  def __init__(self, *args, r=0., eid=None, rounD=2):
    if len(args) == 1:
      e = args[0]
      assert hasattr(e, 'x')
      x = e.x
      assert hasattr(e, 'y')
      y = e.y
      if hasattr(e, 'r'):
        r = e.r
    elif len(args) == 2:
      x, y = args
    self.x = round(x, rounD)
    self.y = round(y, rounD)
    self.radius = round(r, rounD)
    self.id = eid
    self.round = rounD

  def __sub__(self, e1):
    return Line(e1, self)

  def __add__(self, vector):
    return Loc(self.x + vector.dx, self.y + vector.dy)

  def eq(self, other):
    # logging.debug(('%s '*6)%(type(self), type(other), self.owner, other.owner, self.id, other.id))
    return (type(self)==type(other)
      and self.owner.id==other.owner.id
      and self.id==other.id)

  def neq(self, other):
    return not self.eq(other)

  def __str__(self):
    return 'Loc({:.2f}, {:.2f}, {:.2f})'.format(self.x, self.y, self.radius)

  def __repr__(self):
    return self.__str__()


# a line made from 2 objects w/ coordinates
class Line:
  def params(self):
    if abs(self.dx) <= self.tol:  # eq for vertical-line
      return self.x0, None  # b==None -> vertical-line
    m = truediv(self.dy, self.dx)
    b = self.y0 - m * self.x0
    return m, b

  def __init__(self, e0, e1, r=0., tol=1e-3, rounD=2):
    self.e0 = e0
    self.e1 = e1
    self.x0 = e0.x
    self.x1 = e1.x
    self.dx = self.x1 - self.x0
    self.y0 = e0.y
    self.y1 = e1.y
    self.dy = self.y1 - self.y0
    self.x_mid = self.x0 + self.dx/2
    self.y_mid = self.y0 + self.dy/2
    self.r = r
    self.tol = tol
    self.round = rounD

  def gen_pts(self, n):  # generate n pts along line, including end-points e0, e1
    dx = self.dx / (n-1)
    dy = self.dy / (n-1)
    next_x, next_y = self.x0, self.y0
    for _ in range(n):
      yield Loc(round(next_x, self.round), round(next_y, self.round), r=self.r)
      next_x += dx
      next_y += dy

  # TODO
  def cross(self, l1):
    # time-conscious check if self (line0) crosses line1?
    # see if intersect @ same time
    # check 2x2 boundaries in case of radius?
    # self_left
    # self_right
    # Path1_left
    # Path1_right
    m0, b0 = self.params()
    m1, b1 = l1.params()
    # set y equal, check x in both x-ranges
    # m0*x+b0 = m1*x+b1 -> x = (b0-b1)/(m1-m0)
    intersect_x = (b0-b1)/(m1-m0)
    if (min(self.x0, self.x1) <= intersect_x <= max(self.x0, self.x1)
      and min(l1.x0, l1.x1) <= intersect_x <= max(l1.x0, l1.x1)):
      return True
    else:
      return False

  def __str__(self):
    return '|%s---%s|'%(self.e0, self.e1)

  def __repr__(self):
    return self.__str__()



def dist(p0, p1):
  d = math.sqrt((p1.x-p0.x)**2 + (p1.y-p0.y)**2)
  return d


class Grid:
  def bfs(self, x0, y0, dist):
    step = self.pcf
    x0 = int(x0 * step)
    y0 = int(y0 * step)
    dist = int(dist * step)
    cells = []
    # TODO optimize
    x_min = min(0, x0-dist)
    x_max = min(x0+dist, self.x_max)
    y_min = min(0, y0-dist)
    y_max = min(y0+dist, self.y_max)
    for x in range(x_min, x_max+step, step):
      for y in range(y_min, y_max+step, step):
        if abs(x-x0) + abs(y-y0) <= dist:
          cells.append((x, y))
    return cells

  def map_1_ent(self, ent, signal=1.):
    logging.warning('Mapping %s', ent)
    cells = self.bfs(ent.x, ent.y, ent.radius)
    # logging.warning('Radius-cells: %s', cells)
    for x, y in cells:
      # logging.warning('Mapping pixel (x: %s, y: %s)', x, y)
      self.matrix[y][x] = True

  # map entity-occupied cells
  @timit
  def map_entities(self, ents):
    for e in ents:
      self.map_1_ent(e)

  def __init__(self, game_map, precision=1):
    self.pcf = pow(10, precision)  # precision-factor
    self.x_max = game_map.width * self.pcf + 1
    self.y_max = game_map.height * self.pcf + 1
    self.matrix = np.zeros(shape=(self.y_max, self.x_max), dtype=bool)  # TMP 1-cell/turn
    logging.warning('Matrix shape: %s', self.matrix.shape)
    # self.map_entities(game_map.entities)

  def __str__(self):
    df = pd.DataFrame(self.matrix)
    return str(df)

  def __repr__(self):
    return self.__str__()


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO, format='<%(module)s.%(funcName)s:%(lineno)d> %(message)s')

  test='geo'

  if test=='timit':
    @timit
    def timed(t=1e4):
      assert t<=1e8
      t=int(t)
      for _ in range(t):  _
      logging.info('Running timed() {:,} times...'.format(t))

    timed(1e7)
  elif test=='log':
    # setup_logger
    log = setup_logger(__name__, level=logging.INFO)
    log.info('something')
    # logitr
    d = {str(i): i for i in range(5)}
    logitr(d, 'test_header', level=10)
  elif test=='math':
    assert rnd(.234, 2) == .23
    assert rnd(.234, 2, '+') == .24
    assert rnd(.234, 2, '-') == .23
    assert rnd(.235, 2) == .23
    assert rnd(.235, 2, '+') == .24
    assert rnd(.235, 2, '-') == .23
    assert rnd(.236, 2) == .24
    assert rnd(.236, 2, '+') == .24
    assert rnd(.236, 2, '-') == .23
  elif test=='grid':
    p3_2_2 = Loc(3., 2., r=2., lid='planet_3_2_2')
    p7_4_1 = Loc(7., 4., r=1., lid='planet_4_7_1')
    entities = [p3_2_2, p7_4_1]
    game_map = namedtuple('Gmap', 'width, height, entities')(9, 6, entities)
    grid = Grid(game_map, precision=0)
    grid.map_entities(entities)
    print(grid)
  elif test=='geo':
    e0=Loc(1, 1)
    e1=Loc(2, 2)
    d = e1 - e0
    print(d)
    e2 = e1 + d
    print(e2)
