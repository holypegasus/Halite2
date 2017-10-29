import logging
from matplotlib import patches, patheffects
from matplotlib import pyplot as plt


WEAPON_RADIUS = 5.0
DOCK_RADIUS = 4.0


COLOR_CYCLE = plt.rcParams['axes.prop_cycle'].by_key()['color']
def pick_color(eid):
  return COLOR_CYCLE[eid%len(COLOR_CYCLE)]


"""Artist default z-orders
  Artist                      Z-order
  Patch / PatchCollection      1
  Line2D / LineCollection      2
  Text                         3
  """
class Pltr:
  def __init__(self, x_max=None, y_max=None):
    self.fig, self.ax = plt.subplots(dpi=500, figsize=(x_max/10, y_max/10))
    self.x_max = x_max
    self.y_max = y_max
    self.ax.set_xlim(0, x_max)
    self.ax.set_ylim(0, y_max)
    self.ax.set_aspect('equal', adjustable='box')
    self.ax.invert_yaxis()  # game has inverted y-axis
    self.ax.grid('on', which='both')
    self.ax.set_xticks(range(0, x_max, 1), minor=True)
    self.ax.set_yticks(range(0, y_max, 1), minor=True)

  def show(self):
    plt.show()

  def savefig(self, turn, fmt='pdf'):
    plt.savefig('paths_%s.%s'%(turn, fmt), format=fmt)
    plt.cla()  # clear-current axes

  def plt_circle(self, ent, r=None, ls='-', c=None, cid=None, fill=True):
    x = ent.x
    y = ent.y
    r = r or ent.radius
    if fill:
      if c:
        fill = c
      elif hasattr(ent, 'owner_id'):
        fill = ent.owner_id
    # pick color
    if not c:
      if cid:
        c = pick_color(cid)
      elif hasattr(ent, 'id') and ent.id:
        c = pick_color(ent.id)
      else:
        c='gray'

    crcl = patches.Circle((x, y), r, ls=ls, facecolor=c, edgecolor='black', fill=fill, zorder=3)
    self.ax.add_patch(crcl)

    # show planet spawn_spot
    if hasattr(ent, 'spawn_pos') and ent.spawn_pos:
      self.ax.annotate('x', size=10, xy=(ent.spawn_pos.x, ent.spawn_pos.y))
    # show planet 'id:curr_prod'
    if hasattr(ent, 'current_production'):
      pid = ent.id
      curr_prod = ent.current_production
      docked = set(ent._docked_ship_ids)
      goals_of = set([s.id for s in ent.goals_of]) - docked
      planet_str = '%s\n%s\n%s\n%s'%(pid, curr_prod, goals_of, docked)
      self.ax.annotate(planet_str, size=10, xy=(ent.x-3, ent.y+3))
    # show ship.id
    elif ent.id:
      self.ax.annotate(ent.id, xy=(x, y))

  def plt_circles(self, ents, r=None, ls='-', c=None, cid=None, fill=True):
    for ent in ents:
      self.plt_circle(ent, r=r, ls=ls, c=c, cid=cid, fill=fill)

  def plt_line(self, l, ls='-', width=5, c=None, show_id=True, show_coords=False, show_dest=False):
    logging.debug(l)
    e0, e1 = l.e0, l.e1
    if e0.x==e1.x and e0.y==e1.y:
      logging.debug('Fed 0-length line - skip!')
      return
    xs = [e0.x, e1.x]
    ys = [e0.y, e1.y]
    kwargs = {
      'linewidth': width,
      'ls': ls,
      'markersize': width,
    }
    # opt: id -> color
    if c:
      color = c
    elif hasattr(e0, 'id') and e0.id!=None:
      color_id = e0.owner_id*569 + e0.id
      color = pick_color(e0.id)
    kwargs['color'] = color
    # kwargs['path_effects'] = [patheffects.SimpleLineShadow()]
    # plot path
    self.ax.plot(xs, ys, **kwargs)
    # plot direction
    # self.ax.arrow(e0.x, e0.y, l.dx/2, l.dy/2, shape='full', linewidth=0, length_includes_head=True, head_width=1., facecolor='gray', edgecolor='black', zorder=9)
    if show_coords:
      self.ax.annotate(e0, size=width*2, xy=(e0.x, e0.y))
      self.ax.annotate(e1, size=width*2, xy=(e1.x, e1.y))
    if show_id:
      self.ax.annotate(e0.id, size=width*2, xy=(e0.x, e0.y))
    if show_dest:
      self.plt_circle(e1, c=c)

  def plt_lines(self, lines, ls='-', width=5, c=None):
    for l in lines:
      self.plt_line(l, ls=ls, c=c)


  # TODO game-contextual plt calls
  def plt_planets(self, planets):
    # production schedule

    pass

  def plt_ships(self, ships):
    # my ships - actual trajectory

    # foe ships - projected trajectory

    pass

  # @timit
  def plt_matrix(self, matrix):
    # easy but not very detailed
    # self.ax.matshow(matrix)

    # plot over current grid
    nz = np.nonzero(matrix)
    logging.warning(nz)
    xs, ys = nz[1], nz[0]
    for x, y in zip(xs, ys):
      self.plt_circle(x, y, 0.2, 'red')



if __name__ == '__main__':
  print(COLOR_CYCLE)