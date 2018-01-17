#!/usr/bin/env python3

import argparse, configparser, glob, subprocess, os, re, shutil, time, traceback
import pandas as pd
from collections import Counter, defaultdict

from hlt.util import timit



OUTPUT_DIR = 'outputs'
EXTENSIONS = ['*.log', '*.hlt', '*.pdf']
def cleanup():
  # cleanup files
  for ext in EXTENSIONS:
    fs = glob.glob(ext)
    for f in fs:
      dst = os.path.join(OUTPUT_DIR, f)
      shutil.move(f, dst)
      # print('%s -> %s'%(f, dst))
  print('<<< All cleaned up :) >>>')
  # cleanup processes
  subprocess.call(['pkill', '-f', '[M]yBot.py'])


PAT_SEED = re.compile('Seed: ([0-9]+) Dimensions.*')
PAT_STAT = re.compile('Player #')
_SPACE_DELIMITER = ' '
_IDX_ID = 1
_IDX_NAME = 2
_IDX_RANK = 6
_IDX_TURNS_ALIVE = 13
_IDX_PROD = 15
_IDX_DAMAGE = 19
_STR_RANK = 'rank'
_STR_TURNS_ALIVE = 'turns_alive'
_STR_PROD = 'prod'
_STR_DAMAGE = 'damage'
def stat():
  """ Sample output from 4-player game
    ...
    Player #0, v2_speed, came in rank #4 and was last alive on frame #300, producing 100 ships and dealing 13632 damage!
    Player #1, v3_mine_then_hit, came in rank #2 and was last alive on frame #300, producing 260 ships and dealing 27968 damage!
    Player #2, v4_rough_angler, came in rank #3 and was last alive on frame #300, producing 215 ships and dealing 31168 damage!
    Player #3, v5_WIP, came in rank #1 and was last alive on frame #300, producing 934 ships and dealing 92992 damage!
    """
  def compile_game_stats(game_output=None):
    nonlocal n_played
    if game_output:
      actual_seed = PAT_SEED.search(game_output).group(1)
      print('Seed: %s'%actual_seed)
      # cumulate stats
      for stat_line in (_ for _ in game_output.splitlines() if PAT_STAT.search(_)):
        tokens = [t.strip('#,') for t in stat_line.split(_SPACE_DELIMITER)]
        bot_name = tokens[_IDX_NAME]
        if int(tokens[_IDX_RANK]) == 1:
          print('Winner: %s'%(bot_name))
          winner_counts[bot_name] += 1
          final_turn = tokens[_IDX_TURNS_ALIVE]
        overall_stats[bot_name] += Counter({
          _STR_RANK: int(tokens[_IDX_RANK]),
          _STR_TURNS_ALIVE: int(tokens[_IDX_TURNS_ALIVE]),
          _STR_PROD: int(tokens[_IDX_PROD]),
          _STR_DAMAGE: int(tokens[_IDX_DAMAGE])
        })
      # print cumulative winner-counts
      print('Final-turn: %s\n Winner-count: %s'%(final_turn, winner_counts))
      n_played += 1
    else:
      cols = [_STR_RANK, _STR_PROD, _STR_DAMAGE, _STR_TURNS_ALIVE]
      col_precision = [1, 0, 0, 0]
      col_sort_asc = [True, False, False, False]
      df = pd.DataFrame.from_dict(overall_stats, orient='index')
      for col in cols:
        if col not in df: df[col] = 0
      df = df.fillna(value=0)
      df = df[cols]
      df.name = 'Summ-Stats over [%s] games'%n_played
      for col, precision in zip(cols, col_precision):
        # df[col] = round(df[col]/n_played).astype('int') if precision==0 else round(df[col]/n_games, precision)
        df[col] = round(df[col]/n_played, precision)
      df.sort_values(by=cols, ascending=col_sort_asc, inplace=True)
      df.index.name = 'After %s games...'%n_played
      print(df)

  winner_counts = Counter()
  overall_stats = defaultdict(Counter)
  n_played = 0
  return compile_game_stats


@timit
def run_1_game(command, compile_game_stats):
  game_output = subprocess.check_output(command, shell=True).decode()
  compile_game_stats(game_output)



def build_command_and_run(args, compile_game_stats):
  """Game shell-command
    ./halite -d "$width $height" "python3 $1/MyBot.py" "python3 MyBot.py" -s $seed
    """
  # Start w/ binary & current-bot
  list_commands = ['./halite', '\"python3 MyBot.py\"']
  # + args
  bots = ' '.join( '\"python3 ARK/%s/MyBot.py\"'%bot for bot in args.bots )
  list_commands.append(bots)
  if args.dims:
    list_commands.append( '-d "%s %s"'%tuple(args.dims) )
  if args.seed:
    list_commands.append( '-s %s'%args.seed )
  str_command = ' '.join(list_commands)
  print(str_command)
  # run
  for i in range(args.n_games):
    print('Game %s...'%i)
    run_1_game(str_command, compile_game_stats)



def _process_config(config, pprint=True):
  tab = ' '*2
  def print_each(conf, ntab=0):
    for k, v in conf.items():
      if isinstance(v, dict):
        print('%s[%s]'%(tab*ntab, k))
        print_each(v, ntab+1)
      else:
        print('%s%s: %s'%(tab*ntab, k, v))

  if pprint:  print_each(config._sections)
  return config._sections



if __name__ == '__main__':
  # configparse
  config = configparser.ConfigParser()
  config.read('presets.ini')
  presets = _process_config(config, pprint=False)
  # argparse
  parser = argparse.ArgumentParser(description='Run games')
  parser.add_argument('bots', nargs='+', help='Specify 1 or 3 bot(s) to play MyBot')
  parser.add_argument('-d', '--dims', nargs=2, type=int, help='width height')
  parser.add_argument('-s', '--seed', type=int)
  parser.add_argument('-p', '--preset', nargs='?', choices=presets.keys(), default=None, type=str, help='Run a preset scenario')
  parser.add_argument('-n', '--n_games', type=int, default=1)
  args = parser.parse_args()
  # preset populate/override dims & seed
  if args.preset:
    preset = presets[args.preset]
    args.dims = list(map(int, preset.get('dims').split(', ')))
    if preset.get('seed'):
      args.seed = int(preset.get('seed'))
  print('Parsed args: %s'%args)

  try:
    compile_game_stats = stat()
    build_command_and_run(args, compile_game_stats)
  except:  # also catches KeyboardInterrupt
    print('\n<<<Encountered following error!>>>')
    traceback.print_exc()
  finally:
    # compile & oupput stats even if interrupted :D
    compile_game_stats()
    cleanup()