Feature: Boundary Handling
  As a user of the Dragonfly library
  I want an out-of-range dragonfly returned to the search space by a rule I chose
  So that I inherit the paper's exploration behavior knowingly, not by surprise

  # Wrapping is the paper's rule and this library's default, and it is NOT a
  # clamp. A component that leaves the box does not stop at the bound it
  # crossed: it teleports to the OPPOSITE bound, and that step component is
  # replaced by a fresh uniform draw from [0, 1). Both halves are the rule --
  # dropping the step reset changes how the swarm explores.
  #
  # A reader arriving from PSO, GA or Mayfly expects a clamp and will read the
  # teleport as a bug. It is not a bug. Callers who want the familiar behavior
  # ask for it by name through boundary_method, which is why "clamp" and
  # "reflect" exist alongside it.

  Scenario: Wrapping is the default boundary rule
    Given a new default configuration
    Then the effective boundary method should be "wrap"

  Scenario: Wrapping past the upper bound teleports to the lower bound and redraws the step
    Given a box from -10 to 10
    And a component at 12.5 with step 7.25
    When the "wrap" boundary rule is applied
    Then the component should be -10
    And the step component should have been redrawn inside [0, 1)

  Scenario: Wrapping past the lower bound teleports to the upper bound and redraws the step
    Given a box from -10 to 10
    And a component at -14 with step 7.25
    When the "wrap" boundary rule is applied
    Then the component should be 10
    And the step component should have been redrawn inside [0, 1)

  Scenario: Clamping pins to the violated bound and leaves the step alone
    Given a box from -10 to 10
    And a component at 12.5 with step 7.25
    When the "clamp" boundary rule is applied
    Then the component should be 10
    And the step component should be 7.25

  Scenario: Reflecting mirrors the overshoot back and inverts the step
    Given a box from -10 to 10
    And a component at 12.5 with step 7.25
    When the "reflect" boundary rule is applied
    Then the component should be 7.5
    And the step component should be -7.25

  Scenario Outline: An in-range component is left alone by every rule
    Given a box from -10 to 10
    And a component at 3.5 with step 7.25
    When the "<method>" boundary rule is applied
    Then the component should be 3.5
    And the step component should be 7.25

    Examples:
      | method  |
      | wrap    |
      | clamp   |
      | reflect |

  Scenario Outline: Every boundary rule keeps the whole swarm inside the box
    Given a Rosenbrock function with dimension 5
    And bounds from -5 to 10
    When I run DA for 150 iterations with the "<method>" boundary method
    Then every position in the final swarm should be within bounds
    And the best position should be within bounds

    Examples:
      | method  |
      | wrap    |
      | clamp   |
      | reflect |
