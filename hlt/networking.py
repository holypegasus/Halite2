import copy, logging, sys
from time import time

from . import game_map
from .util import logitr, timit


class Game:
  """
  :ivar map: Current map representation
  :ivar initial_map: The initial version of the map before game starts
  """
  @staticmethod
  def _send_string(s):
    """
    Send data to the game. Call :function:`done_sending` once finished.

    :param str s: String to send
    :return: nothing
    """
    sys.stdout.write(s)

  @staticmethod
  def _done_sending():
    """
    Finish sending commands to the game.

    :return: nothing
    """
    sys.stdout.write('\n')
    sys.stdout.flush()

  @staticmethod
  def _get_string():
    """
    Read input from the game.

    :return: The input read from the Halite engine
    :rtype: str
    """
    result = sys.stdin.readline().rstrip('\n')
    return result

  @staticmethod
  def send_command_queue(command_queue):
    """
    Issue the given list of commands.

    :param list[str] command_queue: List of commands to send the Halite engine
    :return: nothing
    """
    for command in command_queue:
      Game._send_string(command)

    Game._done_sending()

  @staticmethod
  def _set_up_logging(tag, name, log_level):
    """
    Set up and truncate the log

    :param tag: The user tag (used for naming the log)
    :param name: The bot name (used for naming the log)
    :return: nothing
    """
    log_file = "{}_{}.log".format(tag, name)
    logging.basicConfig(filename=log_file, level=log_level, filemode='w', format='<%(module)s.%(funcName)s:%(lineno)d> %(message)s')
    logging.critical("Initialized bot {}".format(name))

  def log_more(self, log_level):
    logging.getLogger().setLevel(log_level)
    logging.critical('Update log_level to %s...', log_level)


  def __init__(self, name, log_level=logging.DEBUG):
    """
    Initialize the bot with the given name.

    :param name: The name of the bot.
    """
    self._name = name
    self._send_name = False
    tag = int(self._get_string())
    Game._set_up_logging(tag, name, log_level)
    self.curr_turn = -2
    width, height = [int(x) for x in self._get_string().strip().split()]
    self.map = game_map.Map(tag, width, height)
    self.update_map()
    self.initial_map = copy.deepcopy(self.map)
    self._send_name = True

  @timit
  def update_map(self):
    """
    Parse the map given by the engine.
    
    :return: new parsed map
    :rtype: game_map.Map
    """
    if self._send_name:
      self._send_string(self._name)
      self._done_sending()
      self._send_name = False
    self.curr_turn += 1
    logging.critical('~~~~~~~~~~~~~~\n\n')
    logging.critical('<<< t%s >>>', self.curr_turn)

    # t0 = time()
    got_string = self._get_string()
    # logging.critical( 'get_string took %.0f ms', (time()-t0)*1000 )

    self.map._parse(got_string)
    return self.map


