Feature: Variant Dispatch
  As a user of the Dragonfly library
  I want the registry, the builder, the selector and the comparison runner to
  agree on which variant they are talking about
  So that a variant is chosen once, by name, and every layer honors that choice

  # Where variant_execution.feature drives the direct entry points and so tests
  # the algorithm, these scenarios drive the dispatch layer: NewVariant,
  # GetAllVariants, AlgorithmVariant.Run, VariantBuilder, AlgorithmSelector and
  # ComparisonRunner.

  Scenario Outline: NewVariant resolves every accepted alias
    When I create the variant named "<alias>"
    Then the variant name should be "<name>"

    Examples:
      | alias    | name |
      | da       | DA   |
      | standard | DA   |
      | DA       | DA   |
      | bda      | BDA  |
      | binary   | BDA  |
      | moda     | MODA |
      | MoDa     | MODA |

  Scenario: NewVariant refuses an unknown name
    When I create the variant named "dragonfly9000"
    Then variant creation should fail with an error containing "unknown variant"

  Scenario: GetAllVariants keeps a stable canonical order
    When I list all variants 3 times
    Then every listing should be "DA, BDA, MODA"

  Scenario: The DA variant runs a continuous problem through the registry
    Given a Sphere function with dimension 5
    And bounds from -10 to 10
    And the "da" variant
    When I run the variant for 100 iterations
    Then the variant run should succeed
    And the best position should have 5 components
    And the best position should be within bounds

  Scenario: The BDA variant runs a binary problem through the registry
    Given a OneMax problem over 20 bits
    And the "binary" variant
    When I run the variant for 200 iterations
    Then the variant run should succeed
    And every component of the best position should be 0 or 1

  Scenario: The DA variant refuses a binary configuration
    Given a OneMax problem over 10 bits
    And the "da" variant
    When I run the variant on a binary configuration for 20 iterations
    Then the variant run should be refused as a binary configuration

  Scenario: The MODA variant refuses the single-objective path
    Given a ZDT1 problem with dimension 5
    And the "moda" variant
    When I run the variant for 20 iterations
    Then the variant run should be refused as multi-objective
    And running it through RunMultiObjective for 30 iterations should succeed
    And the archive should be non-dominated

  Scenario: The builder configures and runs a variant
    Given a Sphere function with dimension 5
    And bounds from -10 to 10
    When I build the "da" variant for 50 iterations with population 20
    Then the built configuration should target 50 iterations and 20 dragonflies
    And optimizing through the builder should succeed
    And the best position should be within bounds

  Scenario: The builder leaves a binary variant's unit bounds alone
    Given a OneMax problem over 10 bits
    When I build the "bda" variant for 20 iterations with bounds -100 to 100
    Then the built configuration bounds should be 0 and 1

  Scenario: The builder reports an unknown variant name from Build
    Given a Sphere function with dimension 5
    And bounds from -10 to 10
    When I build the "dragonfly9000" variant for 10 iterations with population 10
    Then building should fail with an error containing "unknown variant"

  Scenario Outline: The selector routes a problem shape to a variant
    Given a problem that is <shape>
    When I ask the selector for its best recommendation
    Then the recommended variant should be "<name>"
    And the recommendation reason should not be empty
    And the recommendation score should be between 0 and 1

    Examples:
      | shape           | name |
      | continuous      | DA   |
      | discrete        | BDA  |
      | multi-objective | MODA |

  Scenario: ClassifyProblem describes a sampled problem
    Given a Sphere function with dimension 5
    And bounds from -10 to 10
    When I classify the problem with seed 20240823
    Then the classified dimensionality should be 5
    And the selector should recommend "DA" for the classification

  Scenario Outline: RecommendForBenchmark always carries a reason
    When I ask for a recommendation for the "<benchmark>" benchmark
    Then the recommended variant should be "<name>"
    And the recommendation reason should not be empty

    Examples:
      | benchmark | name |
      | Sphere    | DA   |
      | Rastrigin | DA   |
      | ZDT1      | MODA |

  Scenario: An unknown benchmark says so in the reason
    When I ask for a recommendation for the "Nonesuch" benchmark
    Then the recommendation reason should mention "not in the table"

  Scenario: The comparison runner pairs seeds across variants
    Given a Sphere function with dimension 5
    And bounds from -10 to 10
    When I compare the "da" and "bda" variants over 2 runs of 30 iterations with base seed 7
    Then the comparison should succeed
    And the comparison should report statistics for 2 variants
    And run k of every variant should have used the base seed plus k

  Scenario: The comparison runner refuses a multi-objective variant
    Given a Sphere function with dimension 5
    And bounds from -10 to 10
    When I compare the "da" and "moda" variants over 2 runs of 10 iterations with base seed 7
    Then the comparison should be refused as multi-objective
