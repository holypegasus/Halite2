# from heapq import *
# from collections import Counter, namedtuple

# from hlt.util import *
# from hlt.plot import Pltr
# from hlt.entity import *


test = 'type'


if test=='geo':
  p11 = Loc(1, 1)
  _p59 = Loc(5, 9)
  p59 = Loc(_p59)
  p73 = Loc(7, 3.1415926, eid=5, rounD=2)
  l0 = Line(p11, p73)
  # print(l0)
  l0_pts = l0.gen_pts(5)
  # print(list(l0_pts))
  l1 = Line(p73, p59)
  assert l0.cross(l1)

  p_3_19 = Loc(3, 19, eid=4)
  p_17_3 = Loc(17, 3, eid=5)
  l2 = Line(p_3_19, p_17_3)
  assert not l0.cross(l2)
  lines = [l0, l1, l2]

  c1_1_1 = Loc(1, 1, r=1, eid=6)
  c2_5_3 = Loc(2, 5, r=3, eid=7)
  circles = [c1_1_1, c2_5_3]

  pltr = Pltr(20, 20)
  pltr.plt_lines(lines)
  pltr.plt_circles(circles)

  pltr.show()


  # p0 = Loc(50., 30., r=10., iD='planet_3_2_2')
  # p1 = Loc(80., 20., r=10., iD='planet_4_7_1')
  # entities = [p0, p1]
  # game_map = namedtuple('Gmap', 'width, height, entities')(99, 66, entities)
  # grid = Grid(game_map, precision=0)
  # grid.map_entities(entities)  # TMP automate

  # pltr = Pltr(game_map.width, game_map.height)
  # pltr.plt_matrix(grid.matrix)

  pltr.savefig('test')
if test=='vect_cross':
  p0 = Pos(-3, 0, .5)
  p1 = Pos(4, 0, .5)
  v_hori = Vect(p0, p1)
  p2 = Pos(0, -3, .5)
  p3 = Pos(0, 4, .5)
  v_verti = Vect(p2, p3)
  p4 = Pos(7, -3, .5)
  v_hori_2 = Vect(p2, p4)
  assert v_hori.cross(v_verti)
  assert v_hori.cross(v_hori)
  assert not v_hori.cross(v_hori_2)
elif test=='simple_gen':
  def num_gen(n=2):
    for i in range(n):
      yield i


  gen_num = num_gen()
  print(next(gen_num))  # 0
  print(next(gen_num))  # 1
  print(next(gen_num))  # StopIter
elif test == 'complex_gen':
  def counter_generator(d):
    # check stop-condition
    while all(i<4 for i in d.values()):
      # can also yield first
      # update to next val
      d['ones'] += 1
      d['twos'] += 2
      # return val
      yield d
    # opt end-msg/exception
    print('Bound reached :D  Generator stopping!')

  seed = Counter()
  gen = counter_generator(seed)
  print(next(gen))
  print(next(gen))
  print(next(gen))
elif test=='namedtuple':
  seed_ctr = Counter()
  seed_ctr['a'] += 1
  seed_dict = dict()
  names = ['x', 'y']
  nt = namedtuple('Rec', ' '.join(names))(seed_ctr, seed_dict)
  print(nt)
  seed_ctr['a'] += 1
  nt.x['ones'] += 1
  nt.x['twos'] += 2
  nt.y['text'] = 'text!'
  print(nt)
elif test=='map':
  p3_2_2 = Loc(3., 2., r=2., iD='planet_3_2_2')
  p4_7_1 = Loc(4., 7., r=1., iD='planet_4_7_1')
  entities = [p3_2_2, p4_7_1]
  game_map = namedtuple('Gmap', 'width, height, entities')(9, 6, entities)
  mat = Matrix(game_map, precision=0)
  mat.map_entities(entities)  # TMP automate

  pltr = Pltr(game_map.width, game_map.height)
  pltr.plt_matrix(mat)
  pltr.show()
elif test=='hq':
  h = [
    (3, Counter()),
    (0, Counter()),
    (3, Counter()),
  ]
  heapify(h)
  print(h)
  while h:
    i, s = heappop(h)
    print(i, s)
    if i%2==1:
      heappush(h, (i+1, s))
elif test=='type':
  def str2int(str_num: str) -> int:
    try:
      out = int(str_num)
      print(out)
      return out
    except:
      print('Error trying to convert %s to int!' % repr(str_num))
      return None

  str2int('a')
  str2int('2')
