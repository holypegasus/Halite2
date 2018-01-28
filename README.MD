# Reflecting on Halite II

## Acknowledgement
  I really enjoyed reading fohristiwhirl's beautifully-illustrated [optimal ship-formation] (https://github.com/fohristiwhirl/halite2_rush_theory/blob/master/README.md#theory-6-ship-battles).

## Intro
  Back in 2016 I tried Halite I but found it too late to do much - so I'm happier that I got a proper run at Halite II from day 1.  Had a lot of fun honing problem-modeling & programming but even more so learning from the design principles.  More about that later.
  This writeup documents how my thoughts/techniques evolved along the way.  This is meant primarily as a memento-to-self.  To the extent that it also entertains any readers, all the better :)
  At the bottom I have some open questions for the community - please message/comment if you have any suggestions!

## [Designer Philosophy] (https://halite.io/learn-programming-challenge/other-resources/design-principles-underlying-halite-II)
  Before diving into reflection, first a word about the game design:
  The first thing after Halite II came out, I read afore-linked section as well as the full [arXiv paper] (https://arxiv.org/pdf/1710.07738.pdf).  Loved the "easy to get a bite but plenty to chew on" vision & design principles.  Having been learning about complex/chaotic/nonlinear-system & Austrian-economics recently, I find some possible parallels with:
  * impose few & simple rules
  * easy to start, with many possibilities to improve every stage along the way, but difficult (or impossible) to perfect especially due to map-randomization & mutual strategic-adaptation
  * visualize to readily highlight bot strengths & weaknesses -> empowers iterative improvement

## Halite II: The Game
  Briefly put:
  * continuous-2D map that seeds planets & 3-ships for each of 2 or 4 players, distributed randomly but radially-symmetric
  * turn-based w/ perfect-map-information
  * 3 of 4X from classic strategy game mix: (explore) exploit expand exterminate

## Evolution of thought
  MAJOR (i.e. paradigm/strategy shifts)
    ### 1) Bi-partitioning game into macro & micro stages.
      Granted the starter-kit already hints at this structure but explicitly modularizing around this dichotomy greatly facilitated early improvements & testing.
      #### macro: evaluate-goal
        Favored topic & first area into which I delved. Early on I experimented w/ various distance-heuristics but could not get consistently superior result.  In the end just ranked by vanilla-distance.
      #### micro: get-action/navigate
        Stayed away until starter-kit's inefficiency & suboptimality finally draged me in.  High-level I navigate A -> B by:
        ```
        minimize diff(separation(destination, target), optimal-separation) while staying within range between minimum- & maximum- separations
        while collision: check tangent-paths to collision-obstacle
        Return: best-path
        ```
        specifically I implemented this via 2 custom components:
          i) a Vect class to represent ship-paths that supports time-conscious/sub-turn collision-testing.  This enables routing partially overlapping-paths that allows tighter/more-optimal ship-movements.  Furthermore, hypothetical-collisions now return exact time- & position-along-path, which feeds into tangent-testing below...
          ii) when navigating A -> B:
            start by trying a naive-beeline
            test collision vs all static (planets & imobile-ships) & dynamic (my-ships' planned paths & foe-ships' guessed-paths) obstacles
            upon collision, test alternative paths at the 2 tangent angles (for all possible reach-lengths)
            recurse if still colliding
            finally yield 'best' option - more on what this means in 4) below
    ### 2) Theory of Mind: guess what other bots' ships will go next
      Moving among middle-Halatians now reveals a big defect - ships are missing foes, especially in chases.  Quite clear why since all along I've been *skating where the pucks are* instead of where they *will be*.  But how do I know what the other bots are thinking?!
      I ended up not having enough time to improve upon my prototype which surispingly already gave a competent performance-boost.  Namely, I just applied my own evaluate-goal function for all competing bots, and given these guessed-goals for each foe-ship, further used a subset of my navigation code to predict their paths.
      While perfectly aware that other bots don't strategize the same way mine does, I nevertheless thought this was a good baseline guess that should scale in the right direction since as my ranking climbs, my goal-setting should come reasonably close to other top bots'.
      To enable easier expression, I rewrote the various networking-related parsing code to update instead of always recreating various Player/Planet/Ship objects between turns such that I could memo-ize information within each object.
    ### 3) Evade & Swarm
      A quirk incentivized by game's rules on multi-ship battles, i.e a ship's attack spreads evenly over all in-range-foe-ships.
      My initial approach of 'hard-swarm', i.e. backoff & collect ships together then explicitly coordinate them seemed extremely fragile so I quickly abandoned it in favor of 'soft-swarm', i.e specifying a common target for ships within range & having them all arrive in range within same turn.  In particular, this latter approach can swarm ships that don't start out close together at all, allowing for flexible rally & breakoff.
      Leverage a new area-strength-analysis that scans a radius for relative power-balance from accounting:
        foe-immobile & mobile HP
        foe-mobile attack-potential
        my-immobile & mobile HP
        my-mobile attack-potential
    ### 4) Restructure & param-ize
      My first attempt at evade/swarm produced the right *type* of ship-movements however it was prone to both collisions & suboptimal reach.  Operating on the level of navgiational-details without any top-level evaluation-metric also handicapped my debugging.  After days of agony I refactored to better fit swarm-logic inside my limited mental model.  Specifically I restructured into 3 semantic layers:
      1. muscle: subsumed navigation-code into other existing lower-level code
      2. nerve: added a middle layer that tries to optimize based on scenario-type-specific ideal seperation-parameters.  This gives me an easy single-number eval-metric
      3. brain: top layer flows from goal -> target -> destination (at end of navigation).  I expanded target-selection by incorporating tactical-context to select separation-parameters from (DOCK, FIGHT, EVADE, RALLY)
  minor (i.e. tactics/tricks)
    ### 1) ["Urban Guerrilla"] (https://halite.io/play/?game_id=9412076&replay_class=1&replay_name=replay-20180126-202257%2B0000--2839703113-264-176-1516998173)
      When a mobile-ship fights while hiding within/behind my-docked-ships, hoping to outlast attacking foe-mobile-ships by spreading damage-taken among my-docked-ships, ideally without losing any of the latter.
      This can be seen as a special case of general swarm strategy of concentrating damage-dealt on single foe while diluting damage-taken.  Simiarly this tactic leverages nearby docked-ships for 'cover', akin to elusive fighters running an urban-guerrilla.  Furthermore, since docked-ships' HP above 1 normally don't contribute, using them as cover can also be thought akin to 'fractional-reserve banking'.
      Note this tactic/behavior is NOT hard-coded but rather an emergent behavior from including my-immobile HP in area-strength-analysis.
    ### 2) Swarm Rotation: continuously tweak ship-positions within swarm to maximize ship-survival
      Within a soft-swarm, the lower a ship's HP, the further back within its separation-range it goes.  The idea is that while all ships within a soft-swarm will enter their target-foe's mutual-combat-circle simultaneously, the weaker ships will stay 'further behind' in the hope of reducing exposure to potential other foe-ships further behind the target, thus reducing potential damage & lengthening their survival so even heavily-wounded ships might attack another day!
      Note this 1-line-change to normalize separation-band within swarm produces fine adjustment within swarm between turns of engagement vs foe-swarms.  Also results in a lot more 3-vs-3 evenly-matched local-skirmishes where my swarm kills all foe-ships while losing none where my-ships split damage-taken quite evenly.

## Issues known & outstanding
  1. incompletely solved
    * swarm collision - still colliding when swarm gets large enough
    * improve params: specifically 1) condition for breaking off from existing swarm 2) separations while doing urban-guerilla such we don't leave damage-sharing my-docked-ship too exposed.
    * account for future_ship x spawn_pos
  2. unexplored
    * ToM: incorporate output on foe_ship.get_act() into foe_ship.eval_goal()
    * ToM: online adjust baseline guesses
    * naive-dist heuristic -> preprocess effective-dist
    * more sophisticated eval of planet value (dist-open, dist-foe, etc)

## Issues considered but intentionally ignored
  * desertion: I never bothered implementing desertion since 1) I never got around to writing meta-game-balance-of-power analysis in the absence of which I assume I'm in the fight until the end thus need every ship but more importantly 2) I thought that evasion-logic that scans widely enough should produce this behavior for free - and to a certain extent this has been observed.


## Questions for community
  * while eagerly anticipating Halite III (and totally open to contributing where possible) - where else can I find this type of 1) quick/rich feedback 2) multi-faceted/open-ended games/programming-contests ?
  * opps to explore further game-theory/mechanism-design or more generally intersection between econ/fin x AI ?
    eg. auto-market-maker buy-sell vs robot-agent move

