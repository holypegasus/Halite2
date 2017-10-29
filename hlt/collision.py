import logging
from math import asin, ceil, degrees, sqrt

from .constants import *
from .entity import Pos, Entity
from .util import rnd


# -> left & right tangents
def get_tangents(src, obst, fudge=0.1):
  logging.info('Getting %s -> %s tangents', src, obst)
  assert isinstance(src, Entity) and isinstance(obst, Entity)
  hypo = src.dist(obst)
  logging.info('hypo: %s', hypo)
  rise = obst.radius + SHIP_RADIUS + fudge
  logging.info('rise: %s', rise)
  run = sqrt(abs(hypo**2 - rise**2))
  abs_ngl_src_obst = src.angle(obst)
  rel_ngl_obst_src_tangent = degrees(asin(rise / hypo))
  logging.info('relative tangent angle %s', rel_ngl_obst_src_tangent)
  return int(ceil(rel_ngl_obst_src_tangent))
  # abs_ngl_src_tangent_l = (abs_ngl_src_obst - rel_ngl_obst_src_tangent) % 360
  # abs_ngl_src_tangent_r = (abs_ngl_src_obst + rel_ngl_obst_src_tangent) % 360
  # tangent_l = src.get_pos(run, abs_ngl_src_tangent_l)
  # tangent_r = src.get_pos(run, abs_ngl_src_tangent_r)
  # logging.info('%s -> %s tangents: %s, %s', src, obst, tangent_l, tangent_r)
  # return tangent_l, tangent_r


def intersect_segment_circle(start, end, circle, fudge=0):
  """
  Test whether a line segment and circle intersect.

  :param Entity start: The start of the line segment. (Needs x, y attributes)
  :param Entity end: The end of the line segment. (Needs x, y attributes)
  :param Entity circle: The circle to test against. (Needs x, y, r attributes)
  :param float fudge: A fudge factor; additional distance to leave between the segment and circle. (Probably set this to the ship radius, 0.5.)
  :return: True if intersects, False otherwise
  :rtype: bool
  """
  # Derived with SymPy
  # Parameterize the segment as start + t * (end - start),
  # and substitute into the equation of a circle
  # Solve for t
  dx = end.x - start.x
  dy = end.y - start.y

  a = dx**2 + dy**2
  b = -2 * (start.x**2 - start.x*end.x - start.x*circle.x + end.x*circle.x +
        start.y**2 - start.y*end.y - start.y*circle.y + end.y*circle.y)
  c = (start.x - circle.x)**2 + (start.y - circle.y)**2

  if a == 0.0:
    # Start and end are the same point
    return start.dist(circle) <= start.radius + circle.radius

  # Time along segment when closest to the circle (vertex of the quadratic)
  t = min(-b / (2 * a), 1.0)
  if t < 0:
    return False

  closest_x = start.x + dx * t
  closest_y = start.y + dy * t
  segment_closest_pt_to_circle = Pos(closest_x, closest_y)
  closest_distance = segment_closest_pt_to_circle.dist(circle)
  # round-down to be conservative about collision
  rounded_down_closest_distance = rnd(closest_distance, PRECISION, '-')
  # if intersect
  min_safe_dist = start.radius + circle.radius
  if rounded_down_closest_distance <= min_safe_dist:
    logging.warning('INTERSECT! %s closest-pt %s d=%.2f <= %.2f to %s', start, segment_closest_pt_to_circle, rounded_down_closest_distance, min_safe_dist, circle)
    return True
  else:
    logging.warning('NO TOUCH! %s closest-pt %s d=%.2f > %.2f to %s', start, segment_closest_pt_to_circle, rounded_down_closest_distance, min_safe_dist, circle)
    return False

