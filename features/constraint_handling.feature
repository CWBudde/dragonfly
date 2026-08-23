Feature: Constraint Handling
  As a user of the Dragonfly library
  I want constraints to decide which candidate wins
  So that a run returns a feasible solution without hiding the raw objective cost

  # Two policies ship. Deb's feasibility rules are the default and need no
  # tuning; the penalty method folds the aggregate violation into a score.
  # Under both, Result.GlobalBest.Cost stays the number the caller's objective
  # actually returned -- the policy ranks candidates, it never rewrites costs.

  Scenario: Deb's rules prefer a feasible candidate over an infeasible one
    Given a candidate with cost 100 and violation 0
    And an incumbent with cost 1 and violation 3
    When I rank them under feasibility rules
    Then the candidate should be preferred

  Scenario: Deb's rules rank two infeasible candidates by violation alone
    Given a candidate with cost 100 and violation 1
    And an incumbent with cost 1 and violation 3
    When I rank them under feasibility rules
    Then the candidate should be preferred

  Scenario: Deb's rules rank two feasible candidates by cost
    Given a candidate with cost 100 and violation 0
    And an incumbent with cost 1 and violation 0
    When I rank them under feasibility rules
    Then the incumbent should be preferred

  Scenario: The penalty method can prefer a slightly infeasible candidate
    Given a candidate with cost 1 and violation 1
    And an incumbent with cost 100 and violation 0
    When I rank them under the penalty method with factor 2
    Then the candidate should be preferred

  Scenario Outline: The penalty method folds violation into the cost
    Given a candidate with cost 10 and violation 3
    When I compute its "<method>" penalized cost with factor 2
    Then the penalized cost should be <expected>

    Examples:
      | method    | expected |
      | linear    | 16       |
      | quadratic | 28       |

  Scenario: A constrained run returns a feasible solution and reports the raw cost
    Given a one-dimensional problem minimizing x squared on [-5, 5] subject to x at least 1
    When I optimize it using feasibility rules
    Then the returned solution should satisfy every configured constraint
    And the reported cost should be the raw objective cost
    And the best cost should be less than 1.5
