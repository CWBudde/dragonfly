Feature: Configuration Validation
  As a user of the Dragonfly library
  I want an unusable configuration rejected by the name of the offending field
  So that I fix the field I got wrong instead of debugging a failed run

  # ObjectiveFunc, problem_size, lower_bound and upper_bound have no usable
  # default. Everything else comes from a factory function, so a caller who
  # starts from NewDefaultConfig only has these four to supply.

  Scenario: A configuration with all four required fields is accepted
    Given a valid configuration
    Then validation should succeed

  Scenario Outline: A missing or unusable field is rejected by name
    Given a valid configuration
    When I <mutation>
    Then validation should fail with an error containing "<fragment>"
    And Optimize should refuse to run it

    Examples:
      | mutation                             | fragment                 |
      | clear the objective function         | ObjectiveFunc            |
      | set problem_size to 0                | problem_size             |
      | set problem_size to -3               | problem_size             |
      | set lower_bound above upper_bound    | lower_bound              |
      | set both bounds to 1                 | lower_bound              |
      | set npop to 0                        | npop                     |
      | set max_iterations to 0              | max_iterations           |
      | set max_workers to -1                | max_workers              |
      | set boundary_method to "bounce"      | boundary_method          |
      | set food_weight to NaN               | food_weight              |
      | set radius_initial_divisor to 0      | radius_initial_divisor   |
      | set stagnation_iterations to -1      | stagnation iterations    |
      | set min_improvement to -1            | minimum improvement      |

  # WeightAuto is -1, and zero is a legitimate pinned value: writing 0 into a
  # weight means "switch this term off for the whole run", not "use the default
  # schedule". Validation must accept it and the run must honor it.
  Scenario: Zero is a legitimate pinned weight, not a request for the schedule
    Given a valid configuration
    When I set enemy_weight to 0
    Then validation should succeed
    And the run should complete

  Scenario: A binary configuration on the wrong bounds is rejected
    Given a binary configuration
    When I set upper_bound to 5
    Then validation should fail with an error containing "binary"
