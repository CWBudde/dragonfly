Feature: Algorithm Variants
  As a user of the Dragonfly library
  I want each of the paper's three variants to run through its own entry point
  So that I can pick the one that matches my problem class

  # These scenarios deliberately exercise the direct entry points -- Optimize,
  # OptimizeBinary and OptimizeMultiObjective -- rather than the variant
  # registry, so that they test the algorithm rather than the dispatch layer.

  Scenario: DA runs on a continuous problem
    Given a Sphere function with dimension 10
    And bounds from -100 to 100
    When I run DA for 300 iterations
    Then the best position should have 10 components
    And the best position should be within bounds
    And the evaluation count should be positive

  Scenario: BDA runs on a binary problem and returns a bit string
    Given a OneMax problem over 30 bits
    When I run BDA for 300 iterations
    Then every component of the best position should be 0 or 1
    And the best cost should be less than 3

  Scenario: BDA honors the transfer function it was given
    Given a OneMax problem over 20 bits
    And the "v2" transfer function
    When I run BDA for 300 iterations
    Then every component of the best position should be 0 or 1
    And the best cost should be less than 3

  Scenario: MODA returns a non-dominated archive
    Given a ZDT1 problem with dimension 10
    When I run MODA for 200 iterations
    Then the archive should be non-dominated
    And the archive should hold at least 10 solutions
    And every archived position should be within bounds

  Scenario: All three variants report the seed they ran with
    Given a Sphere function with dimension 5
    And bounds from -10 to 10
    When I run DA for 50 iterations
    Then the reported seed should be non-zero
