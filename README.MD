# Reflecting on Halite II

further replay-links, screenshots pending...


## Acknowledgement
  I really enjoyed reading fohristiwhirl's beautifully-illustrated [optimal ship-formation](https://github.com/fohristiwhirl/halite2_rush_theory/blob/master/README.md#theory-6-ship-battles).


## Intro
  Tried Halite I back in 2016 but found it too late to do much - happier that I got a proper run at Halite II from earlier on.  Had a lot of fun honing problem-modeling & programming but even more so learning from the design principles.  More on that below.
  
  This writeup documents how my thoughts/techniques evolved along the way.  This is meant primarily as a memento-to-self.  To the extent this entertains anyone else, all the better :)
  
  At the bottom I have some open questions for the community - please message if you have any suggestions!

## [Design Philosophy](https://halite.io/learn-programming-challenge/other-resources/design-principles-underlying-halite-II)
  The first thing I did after Halite II came out was reading the designers' [arXiv paper](https://arxiv.org/pdf/1710.07738.pdf).  Loved the "easy to take a bite but plenty to chew on" design vision & principles:
  * impose few & simple rules
  * easy to start
  * multiple possibilities to improve every stage along the way
  * difficult (or impossible) to perfect especially due to map-randomization & mutual strategic-adaptation
  * intuitive visuals that readily highlight bot behavior & empower iterative improvements

  Having been reading about complex/chaotic/nonlinear-system & Austrian-economics recently, I wonder if there are not some parallels here :P


## Halite II: The Game
  Briefly put:
  * continuous-2D map that seeds planets & 3-ships for each of 2 or 4 players, distributed randomly but radially-symmetric
  * turn-based w/ perfect-map-information
  * 3 of 4X from classic strategy-game mix: (explore) exploit expand exterminate


## Evolution of thought & code
### MAJOR (i.e. paradigm/strategy shifts)
#### 1) Bi-partitioning game into macro & micro stages
  Granted starter-kit already hints as much but explicitly modularizing around this dichotomy greatly facilitated early improvements & testing.
  ##### macro: evaluate-goal
  Favored topic & first area into which I delved. Early on I experimented w/ various distance-heuristics but got neither compelling intuition nor consistently superior result.  In the end just ranked by vanilla-distance.
  ##### micro: get-action/navigate
  Stayed away until starter-kit's inefficiency (time-out!) & suboptimality (context-oblivious retries) finally dragged me in.  High-level I navigate A -> B by:
  ```
  minimizing diff(separation(destination, target), optimal-separation)
  while staying within range between minimum- & maximum- separations;
  while collision: adjust path around;
  Return: best-path;
  ```
  specifically I implemented this via 2 custom components:

  i) a `Vect` class to represent ship-paths that supports time-conscious/sub-turn collision-testing.  This enables routing partially overlapping-paths that allows tighter/more-optimal ship-movements.  Moreover, checking hypothetical-collisions now return exact time- & position-along-path, which feeds into tangent-testing below...

  ii) collision-induced tangent-checks:
  ```
  start by trying to naive-beeline A -> B
  test collision vs all static (planets & imobile-ships) & dynamic (my-ships' planned paths & foe-ships' guessed-paths) obstacles (after filtering)
  upon collision, test alternative paths at the 2 tangent-angles (for all possible reach-lengths)
  recurse if still colliding
  finally yield 'best' option - more on what this means in 4) below
  ```

#### 2) "Theory of Mind": guess where other bots' ships will go next
  Aforementioned improvements on goal-setting & navigation sufficed for middle-class.  Same mobility however exposes my ships' failing at chasing down foes.  Gotta skate where the puck *will be*.

  But how do we get inside other bots', um, heads?

  How about applying my own eval-goal & navigate logic for all competing bots as a baseline-guess for their ships' next-paths.

  While perfectly aware that this assumption is wrong, nevertheless this seems a reasonable bet that should scale in the right direction since as my overall performance improves, my guesses should not deviate further.

  To enable easier expression, I rewrote all networking-related parsing code to update instead of always recreating various Player/Planet/Ship objects between turns so that I could memo-ize relevant information within each object.

#### 3) Evade & Swarm
  Swarming, hallmark of upper Halatians, is a quirk born out of game rule on multi-ship battles, ie a ship's attack spreads evenly over all in-range-foe-ships.

  My initial top-down approach of 'hard-swarm', ie
  ```
  backoff when leading ship gets overpowered
  collect reinforcement into swarm
  explicitly coordinate them back to confront foes
  ```
  was so plagued by collisions, suboptimal-reach, etc that at some point I decided to  try "soft-swarm" out of desperation.

  That is, instead of specifying the swarm's paths, only signal a common-target & desired separation-range for all "swarmable ships" to arrive in range "simul-turn-eously".  Besides being much easier to code & reason about, soft-swarm frequently produces surprisingly orchestrated-looking movements that had *not* been explicitly programmed.

  To decide when to evade/rally, I added a new `area-strength-analysis` that determines power-balance within in a radial-area by accounting for:
  ```
    foe-immobile & mobile HP
    foe-mobile attack-potential
    my-immobile & mobile HP
    my-mobile attack-potential
  ```

#### 4) Restructure & param-ize
  My first attempt at evade/swarm produced the right *type* of ship-movements however it was prone to both collisions & suboptimal reach.  Part of the issue is that operating on the level of navgiational-details without top-level evaluation-metric really handicapped debugging.  After days of agony I finally tried restructuring the entire navigation code to better fit swarm-logic inside my limited mental model.  Specifically I restructured into 3 semantic layers:
  1. "muscle": subsumed navigation-code into other existing lower-level logic dealing with explicit movements
  2. "nerve": added a middle layer that tries to optimize based on scenario-type-specific seperation-parameters.  This also gives me an easy single-number eval-metric
  3. "brain": top layer flows from goal -> target -> destination (obtained at end of navigation).  I expanded target-selection to incorporate tactical-context in selecting separation-parameters from (DOCK, FIGHT, EVADE, RALLY)

  In particular each separation-parameter set consists of (minimum, optimal, maximum) where the goal is to minimize difference between actual-separation (between destination and target) and optimal-separtion while keeping actual-separation within the range [minimum-separation, maximum-separation].

### minor (i.e. tactics/tricks)
#### 1) ["Urban Guerrilla"](https://halite.io/play/?game_id=9412076&replay_class=1&replay_name=replay-20180126-202257%2B0000--2839703113-264-176-1516998173)
  When a mobile-ship fights while hiding within/behind my docked-ships, hoping to outlast attacking foe-mobile-ships by spreading damage-taken among docked-ships, ideally without losing any docked-ships.

  This can be seen as a special-case of the general swarm-strategy of concentrating damage-dealt on single foe while diluting damage-taken.  Simiarly "urban-guerrilla" leverages nearby docked-ships for "cover", akin to elusive fighters ghosting through concrete jungle.  Furthermore, since docked-ships' HP above 1 wouldn't contribute to active combat otherwise, using them as cover is akin to *fractional-reserve banking*.

  Note this tactic/behavior is *not* hard-coded but rather emerges from accounting my-immobile HP in area-strength-analysis.

#### 2) Swarm Rotation: continuously adjust relative ship-positions within swarm to maximize ship-survival
  Within a soft-swarm, the lower a ship's HP, the further back within its separation-range it goes.  The idea is that while all ships within a soft-swarm will enter their target-foe's mutual-combat-circle simultaneously, the weaker ships will stay "further behind" hoping to reduce exposure to potential other foe-ships in turn further behind the target.  By reducing potential damage & lengthening survival, even heavily-wounded ships might live to attack another day!

  ```python
  sep_opt -= (ship.health/MAX_SHIP_HEALTH) * MIN_MOVE_SPEED
  ```

   Mapping normalized HP to adjustment on separation-range produces fine adjusting-moves (typically magnitude-1 shifts) within swarm between turns of engagement vs foe-swarms.  Also results in a lot more 3-vs-3 local-skirmishes where my swarm kills all foe-ships while losing none by splitting damage-taken quite evenly.


## Issues known & outstanding
  1. incompletely solved
    * swarm collision - still happens on occasions & extremely detrimental :'(
    * improve params: specifically for
      1) condition for breaking off from existing swarm
      2) separation-ranges while ["urban-guerilla"-ing](https://github.com/holypegasus/Halite2#1-urban-guerrilla) such we don't leave damage-sharing docked-ship exposed without our mobile ship firing back
    * account all bots' planets' accrued-prod & prod-rate to predict when next-ship spawns where (update estimate of each planet's spawn-position by, say, location of its previous spawn)
  2. unexplored
    * Theory of Mind
      1) improve baseline-guess by incorporating foe_ship.get_act() into foe_ship.eval_goal()
      2) adjust guess online by taking time-decayed convolution of guessed-acts vs observed-acts -> this would also help with detecting early rush
    * naive-dist heuristic -> preprocess effective-dist
    * more sophisticated eval of planet value (dist-open, dist-foe, etc)


## Issues considered but skipped
  * **desertion**: sparing a ship or more to hide in a corner trying to survive to the end is a fair tactic based on quirks in the ranking algo however I never implemented this because...
    1) never got around to analyzing meta-game-balance-of-power, in the absence of which I assume I'm in the fight until the end thus can't spare ships to desert but more importantly...
    2) evasion based on good-enough area-strength-analysis & foe-next-path-guesses  should approximate desertion for free.


## Questions for community
  * while eagerly anticipating Halite III (and totally wanna help build it!), where else can we find games/contests that have this kind of
    1) quick/rich feedback
    2) multi-faceted/open-endedness
  * opportunities to further explore game-theory/mechanism-design or more generally intersection between econ/finance x AI ?

    eg automated-market-maker buy-sell vs robot-agent move -> rewards

