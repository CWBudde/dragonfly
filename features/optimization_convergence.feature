Feature: Optimization Convergence
  As a user of the Dragonfly library
  I want a run to stop for a reason I can read off the result
  So that I can tell a converged run from an exhausted one

  Scenario: A run without early stopping ends at the iteration cap
    Given a Sphere function with dimension 5
    And bounds from -10 to 10
    When I run DA for 200 iterations
    Then the termination reason should be "maximum_iterations"
    And the iteration count should be 200
    And the convergence curve should have 200 entries
    And the convergence curve should be non-increasing

  Scenario: A reached target cost stops the run early
    Given a Sphere function with dimension 5
    And bounds from -10 to 10
    And a target cost of 1.0
    When I run DA for 2000 iterations
    Then the termination reason should be "target_cost"
    And the iteration count should be less than 2000
    And the best cost should be less than 1.0

  # A constant objective can never improve, so the stagnation counter runs to
  # its limit and nothing else can stop the run first.
  Scenario: A run that stops improving terminates on stagnation
    Given a constant objective with dimension 5
    And bounds from -10 to 10
    And a stagnation window of 10 iterations
    When I run DA for 2000 iterations
    Then the termination reason should be "stagnation"
    And the iteration count should be less than 2000

  Scenario: DA makes real progress on Sphere
    Given a Sphere function with dimension 10
    And bounds from -100 to 100
    When I run DA for 500 iterations
    Then the best cost should be less than 300
    And the best cost should be less than 10 percent of the initial best

  Scenario: A seeded run is reproducible
    Given a Rastrigin function with dimension 6
    And bounds from -5.12 to 5.12
    When I run DA twice for 100 iterations with seed 20240823
    Then both runs should return identical results
