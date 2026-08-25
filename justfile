# Dragonfly Algorithm - Task Runner

# Reproducible development and CI toolchain. Keep these in sync with the
# versions used by the GitHub workflows.
treefmt_version := "2.5.0"
golangci_lint_version := "2.13.1"
gofumpt_version := "0.11.0"
gci_version := "0.14.0"
shfmt_version := "3.13.1"
taplo_version := "0.10.0"
prettier_version := "3.9.6"
shellcheck_version := "0.11.0"
nancy_version := "2.1.0"
govulncheck_version := "1.1.4"

# Default recipe to display available commands
default:
    @just --list

# Build the project
build:
    go build -v ./...

# Run tests with coverage (fast - without race detection)
test:
    go test -v -coverprofile=coverage.out ./...
    go tool cover -html=coverage.out -o coverage.html

# Run tests without coverage (quickest)
test-quick:
    go test -v -short ./...

# Run tests with race detection (slower, skips long-running benchmark suite)
test-race:
    go test -v -race -short -timeout 5m ./...

# Run all tests including long-running benchmark suite (no race detection)
test-full:
    go test -v -timeout 10m ./...

# Run the opt-in, deterministic v0.2 MODA quality study and atomically write
# its CSV plus the sibling Markdown summary. This is intentionally outside the
# ordinary test gates because it performs 45 30-dimensional, 1000-iteration runs.
study-quality output="docs/measurements/v0.2.0-quality.csv":
    DRAGONFLY_RUN_QUALITY_STUDY=1 DRAGONFLY_QUALITY_STUDY_OUTPUT="{{output}}" go test -v -count=1 -run '^TestMODAQualityStudy$' -timeout 30m .

# Run the independent BDA transfer-family study (8 transfer functions x 2
# fidelity modes x 15 seeds) and atomically write its CSV and Markdown summary.
study-bda-quality output="docs/measurements/v0.2.0-bda-quality.csv":
    DRAGONFLY_RUN_BDA_STUDY=1 DRAGONFLY_BDA_STUDY_OUTPUT="{{output}}" go test -v -count=1 -run '^TestBDAQualityStudy$' -timeout 30m .

# Enforce the release coverage floor rather than merely generating a report.
check-coverage min="80":
    #!/usr/bin/env bash
    set -euo pipefail
    go test -timeout 20m -coverprofile=coverage.out -covermode=atomic ./...
    total="$(go tool cover -func=coverage.out | awk '/^total:/ {gsub(/%/, "", $3); print $3}')"
    awk -v total="$total" -v minimum="{{min}}" 'BEGIN { if (total + 0 < minimum + 0) { printf "coverage %.1f%% is below %.1f%%\n", total, minimum > "/dev/stderr"; exit 1 } }'
    printf 'coverage %.1f%% meets %.1f%% floor\n' "$total" "{{min}}"

# Compile every nested example module, including both native and js/wasm demo paths.
check-examples:
    #!/usr/bin/env bash
    set -euo pipefail
    for module in examples examples/basic examples/comparison examples/constrained examples/feature_selection examples/multiobjective examples/parallel; do
        (cd "$module" && go build ./...)
    done
    just check-wasm-demo

# Run integration tests (Gherkin/Cucumber)
test-integration:
    go test -v -run TestFeatures

# Run benchmarks
bench:
    go test -bench=. -benchmem ./...

# Build the WebAssembly demo into ./dist
build-wasm-demo:
    ./scripts/build-wasm-demo.sh

# Build and serve the WebAssembly demo locally
run-wasm-demo: build-wasm-demo
    @echo "Serving the demo at http://localhost:8090"
    python3 -m http.server -d dist 8090

# Build the demo for js/wasm without emitting a binary (a fast compile check).
# It builds both ways on purpose: the js/wasm build covers main.go, the plain
# one covers main_stub.go, and a broken build tag shows up in exactly one.
check-wasm-demo:
    cd examples/wasm-demo && GOOS=js GOARCH=wasm go build -o /dev/null . && go build -o /dev/null ./...

# Run the examples
run:
    cd examples && go run main.go

# Install the formatters and linters used by `just fmt` / `just lint`
setup-deps:
    #!/usr/bin/env bash
    set -euo pipefail
    export PATH=$HOME/go/bin:$PATH
    mkdir -p "$HOME/go/bin"
    echo "Installing development dependencies..."

    # treefmt (formatter multiplexer)
    if ! command -v treefmt >/dev/null 2>&1 || ! treefmt --version 2>&1 | grep -Fq "{{treefmt_version}}"; then
        echo "Installing treefmt {{treefmt_version}}..."
        tool_tmp="$(mktemp -d)"
        trap 'rm -rf "$tool_tmp"' EXIT
        curl -fsSL "https://github.com/numtide/treefmt/releases/download/v{{treefmt_version}}/treefmt_{{treefmt_version}}_linux_amd64.tar.gz" -o "$tool_tmp/treefmt.tar.gz"
        tar -C "$HOME/go/bin" -xzf "$tool_tmp/treefmt.tar.gz" treefmt
        rm -rf "$tool_tmp"
        trap - EXIT
    fi

    # golangci-lint v2 (linter + formatter runner)
    if ! command -v golangci-lint >/dev/null 2>&1 || ! golangci-lint version 2>&1 | grep -Fq "{{golangci_lint_version}}"; then
        echo "Installing golangci-lint {{golangci_lint_version}}..."
        go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v{{golangci_lint_version}}
    fi

    # Go formatters
    if ! command -v gofumpt >/dev/null 2>&1 || ! gofumpt -version 2>&1 | grep -Fq "{{gofumpt_version}}"; then
        echo "Installing gofumpt {{gofumpt_version}}..."
        go install mvdan.cc/gofumpt@v{{gofumpt_version}}
    fi
    if ! command -v gci >/dev/null 2>&1 || ! gci --version 2>&1 | grep -Fq "{{gci_version}}"; then
        echo "Installing gci {{gci_version}}..."
        go install github.com/daixiang0/gci@v{{gci_version}}
    fi

    # Shell formatter
    if ! command -v shfmt >/dev/null 2>&1 || ! shfmt --version 2>&1 | grep -Fq "{{shfmt_version}}"; then
        echo "Installing shfmt {{shfmt_version}}..."
        go install mvdan.cc/sh/v3/cmd/shfmt@v{{shfmt_version}}
    fi

    # Markdown/JSON/YAML formatter
    if ! command -v prettier >/dev/null 2>&1 || [[ "$(prettier --version 2>&1)" != "{{prettier_version}}" ]]; then
        echo "Installing prettier {{prettier_version}}..."
        npm install -g prettier@{{prettier_version}}
    fi

    # Shell linter. Install the upstream release so CI and local checks use the
    # same version instead of whichever package an OS repository happens to ship.
    if ! command -v shellcheck >/dev/null 2>&1 || ! shellcheck --version 2>&1 | grep -Fq "version: {{shellcheck_version}}"; then
        echo "Installing shellcheck {{shellcheck_version}}..."
        tool_tmp="$(mktemp -d)"
        trap 'rm -rf "$tool_tmp"' EXIT
        curl -fsSL "https://github.com/koalaman/shellcheck/releases/download/v{{shellcheck_version}}/shellcheck-v{{shellcheck_version}}.linux.x86_64.tar.xz" -o "$tool_tmp/shellcheck.tar.xz"
        tar -C "$tool_tmp" -xJf "$tool_tmp/shellcheck.tar.xz"
        install "$tool_tmp/shellcheck-v{{shellcheck_version}}/shellcheck" "$HOME/go/bin/shellcheck"
        rm -rf "$tool_tmp"
        trap - EXIT
    fi

    # TOML formatter. Prefer the prebuilt binary: `cargo install taplo-cli` needs a
    # Rust toolchain, and when it is absent treefmt just skips every .toml file --
    # `--allow-missing-formatter` makes a missing formatter silent, so `just
    # check-formatted` goes green locally and fails in CI, which is exactly how
    # .golangci.toml stayed unformatted through nine phases.
    if ! command -v taplo >/dev/null 2>&1 || ! taplo --version 2>&1 | grep -Fq "{{taplo_version}}"; then
        echo "Installing taplo {{taplo_version}}..."
        tool_tmp="$(mktemp -d)"
        trap 'rm -rf "$tool_tmp"' EXIT
        curl -fsSL "https://github.com/tamasfe/taplo/releases/download/{{taplo_version}}/taplo-linux-x86_64.gz" -o "$tool_tmp/taplo.gz"
        gzip -dc "$tool_tmp/taplo.gz" > "$HOME/go/bin/taplo"
        chmod +x "$HOME/go/bin/taplo"
        rm -rf "$tool_tmp"
        trap - EXIT
    fi

# Fail if any formatter treefmt.toml declares is missing.
#
# treefmt runs with --allow-missing-formatter so a partial toolchain does not block
# work, but that means `just check-formatted` proves less locally than it does in CI.
# Run this to find out which formatters are actually doing anything.
check-tools:
    #!/usr/bin/env bash
    set -euo pipefail
    export PATH=$HOME/go/bin:$PATH
    failed=0
    check_version() {
        local tool="$1"
        local expected="$2"
        shift 2
        if ! command -v "$tool" >/dev/null 2>&1; then
            printf '  MISSING %s (expected %s)\n' "$tool" "$expected"
            failed=1
        elif version_output="$("$@" 2>&1)" && grep -Fq "$expected" <<< "$version_output"; then
            printf '  ok      %s %s\n' "$tool" "$expected"
        else
            printf '  WRONG   %s (expected %s; got %s)\n' "$tool" "$expected" "${version_output:-unknown}"
            failed=1
        fi
    }
    check_version treefmt "{{treefmt_version}}" treefmt --version
    check_version golangci-lint "{{golangci_lint_version}}" golangci-lint version
    check_version gofumpt "{{gofumpt_version}}" gofumpt -version
    check_version gci "{{gci_version}}" gci --version
    check_version prettier "{{prettier_version}}" prettier --version
    check_version taplo "{{taplo_version}}" taplo --version
    check_version shfmt "{{shfmt_version}}" shfmt --version
    check_version shellcheck "version: {{shellcheck_version}}" shellcheck --version
    if [[ "$failed" -ne 0 ]]; then
        echo "The formatter/linter toolchain is missing or does not match the pinned versions." >&2
        echo "Run: just setup-deps" >&2
        exit 1
    fi

# Format all files with treefmt
fmt:
    #!/usr/bin/env bash
    export PATH=$HOME/go/bin:$PATH
    treefmt --allow-missing-formatter

# Alias for `just fmt`
treefmt: fmt

# Run linter
lint:
    #!/usr/bin/env bash
    export PATH=$HOME/go/bin:$PATH
    golangci-lint run --config ./.golangci.toml --timeout 5m ./...

# Run linter (with fix)
lint-fix:
    #!/usr/bin/env bash
    export PATH=$HOME/go/bin:$PATH
    golangci-lint fmt --config ./.golangci.toml
    golangci-lint run --config ./.golangci.toml --timeout 5m --fix ./...

# Tidy up dependencies
tidy:
    go mod tidy

# Verify dependencies
verify:
    go mod verify

# Clean build artifacts
clean:
    go clean
    rm -f coverage.out coverage.html
    rm -f *.test *.prof
   
# Generate documentation
docs:
    godoc -http=:6060

# Fail if any file is not formatted
check-formatted: check-tools
    #!/usr/bin/env bash
    export PATH=$HOME/go/bin:$PATH
    treefmt --allow-missing-formatter --fail-on-change

# Fail if go.mod/go.sum are not tidy
check-tidy:
    go mod tidy -diff

# Run all checks (format, lint, test)
check: check-formatted check-tidy lint test

# Run all checks with race detection
check-race: check-formatted check-tidy lint test-race

# Full CI pipeline. check-coverage runs the complete suite, so do not also run
# `test` through `check` here.
ci: verify check-formatted check-tidy lint check-coverage check-examples security

# Full CI pipeline with race detection
ci-race: verify check-formatted check-tidy lint test-race check-coverage check-examples security

# Profile CPU performance
profile-cpu:
    go test -run '^$' -bench '^BenchmarkOptimizeBaseline$' -benchtime=5s -cpuprofile=cpu.pprof .
    @echo "CPU profile written to cpu.pprof; inspect it with: go tool pprof -top cpu.pprof"

# Profile memory usage
profile-mem:
    go test -run '^$' -bench '^BenchmarkOptimizeBaseline$' -benchtime=5s -memprofile=memory.pprof .
    @echo "Memory profile written to memory.pprof; inspect it with: go tool pprof -top -alloc_space memory.pprof"

# Run optimization with different algorithms for comparison
compare:
    #!/usr/bin/env bash
    echo "Running algorithm comparison..."
    cd examples
    echo "=== Standard Run ==="
    go run main.go
    echo ""
    echo "=== Performance comparison complete ==="

# Initialize development environment
init:
    go mod download
    @echo "Development environment ready!"
    @echo "Run 'just run' to test the examples"

# Create a new benchmark function template
new-benchmark name:
    #!/usr/bin/env bash
    echo "// {{name}} is a benchmark function." >> functions.go
    echo "// Global minimum is at f(?, ..., ?) = ?" >> functions.go
    echo "func {{name}}(x []float64) float64 {" >> functions.go
    echo "    // TODO: Implement {{name}} function" >> functions.go
    echo "    return 0.0" >> functions.go
    echo "}" >> functions.go
    echo "" >> functions.go
    echo "Added {{name}} function template to functions.go"

# Run specific optimization function
#
# Both `just optimize Sphere 30 1000` and `just optimize func=Sphere size=30
# iter=1000` work. just passes everything after the recipe name positionally,
# so the `name=` prefixes arrive as part of the value and are stripped here --
# the same idiom the release recipes use for `version=`.
optimize func="Sphere" size="30" iter="1000":
    #!/usr/bin/env bash
    set -euo pipefail
    opt_func="{{func}}"; opt_func="${opt_func#func=}"
    opt_size="{{size}}"; opt_size="${opt_size#size=}"
    opt_iter="{{iter}}"; opt_iter="${opt_iter#iter=}"
    cd examples
    trap 'rm -f temp_optimize.go' EXIT
    cat > temp_optimize.go << EOF
    package main
    import (
        "fmt"
        "github.com/CWBudde/dragonfly"
    )
    func main() {
        config := dragonfly.NewDefaultConfig()
        config.ObjectiveFunc = dragonfly.$opt_func
        config.ProblemSize = $opt_size
        config.MaxIterations = $opt_iter
        config.LowerBound = -10
        config.UpperBound = 10
        
        result, err := dragonfly.Optimize(config)
        if err != nil {
            panic(err)
        }
        
        fmt.Printf("Function: $opt_func\n")
        fmt.Printf("Best Cost: %.10f\n", result.GlobalBest.Cost)
        fmt.Printf("Evaluations: %d\n", result.FuncEvalCount)
    }
    EOF
    go run temp_optimize.go

# Install development tools (see also: just setup-deps)
install-tools: setup-deps
    go install golang.org/x/tools/cmd/godoc@latest
    just setup-security-tools

# Install the pinned security scanners. This recipe is also a dependency of
# `security`, making the gate reproducible on fresh CI runners.
setup-security-tools:
    #!/usr/bin/env bash
    set -euo pipefail
    export PATH=$HOME/go/bin:$PATH
    mkdir -p "$HOME/go/bin"
    if ! command -v nancy >/dev/null 2>&1 || ! nancy --version 2>&1 | grep -Fq "{{nancy_version}}"; then
        echo "Installing nancy {{nancy_version}}..."
        go install github.com/sonatype-nexus-community/nancy/v2@v{{nancy_version}}
    fi
    if ! command -v govulncheck >/dev/null 2>&1 || ! govulncheck -version 2>&1 | grep -Fq "v{{govulncheck_version}}"; then
        echo "Installing govulncheck {{govulncheck_version}}..."
        go install golang.org/x/vuln/cmd/govulncheck@v{{govulncheck_version}}
    fi

# Fail when security scanners are missing or do not match the release pins.
check-security-tools:
    #!/usr/bin/env bash
    set -euo pipefail
    export PATH=$HOME/go/bin:$PATH
    failed=0
    if ! command -v nancy >/dev/null 2>&1; then
        echo "MISSING nancy (expected {{nancy_version}})" >&2
        failed=1
    elif ! version_output="$(nancy --version 2>&1)" || ! grep -Fq "{{nancy_version}}" <<< "$version_output"; then
        echo "WRONG nancy version (expected {{nancy_version}}; got ${version_output:-unknown})" >&2
        failed=1
    fi
    if ! command -v govulncheck >/dev/null 2>&1; then
        echo "MISSING govulncheck (expected {{govulncheck_version}})" >&2
        failed=1
    elif ! version_output="$(govulncheck -version 2>&1)" || ! grep -Fq "v{{govulncheck_version}}" <<< "$version_output"; then
        echo "WRONG govulncheck version (expected {{govulncheck_version}}; got ${version_output:-unknown})" >&2
        failed=1
    fi
    if [[ "$failed" -ne 0 ]]; then
        echo "Run: just setup-security-tools" >&2
        exit 1
    fi
    echo "Security toolchain matches the pinned versions."

# Check for security vulnerabilities in the dependency tree and in reachable code
security: setup-security-tools check-security-tools audit vuln

# Audit the production dependency tree against the OSS Index.
# The library is stdlib-only, so this audits nothing today and exists to catch the
# first real dependency that is ever added.
audit:
    #!/usr/bin/env bash
    set -euo pipefail
    export PATH=$HOME/go/bin:$PATH
    go list -json -deps ./... | nancy sleuth

# Scan for vulnerabilities Go's own database knows about, by reachability.
# Unlike `just audit` this covers test-only dependencies -- godog and its tree -- and
# reports whether a vulnerable symbol is actually called rather than merely present.
vuln:
    #!/usr/bin/env bash
    set -euo pipefail
    export PATH=$HOME/go/bin:$PATH
    govulncheck ./...

# Validate a prospective release without creating a tag
release-check version:
    #!/usr/bin/env bash
    set -euo pipefail
    release_version="{{version}}"
    release_version="${release_version#version=}"
    if [[ ! "$release_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$ ]]; then
        echo "Invalid semantic version: $release_version" >&2
        exit 1
    fi
    grep -Fq "## [$release_version]" CHANGELOG.md
    test -s LICENSE
    test -s README.md
    test "$(go list -m)" = "github.com/CWBudde/dragonfly"
    go vet ./...
    just ci-race

# Validate and create an annotated release tag locally
release version:
    #!/usr/bin/env bash
    set -euo pipefail
    release_version="{{version}}"
    release_version="${release_version#version=}"
    release_tag="v$release_version"
    just release-check "$release_version"
    if [[ -n "$(git status --porcelain)" ]]; then
        echo "Release requires a clean worktree" >&2
        exit 1
    fi
    if git rev-parse --verify --quiet "refs/tags/$release_tag" >/dev/null; then
        echo "Tag already exists: $release_tag" >&2
        exit 1
    fi
    git tag -a "$release_tag" -m "Release $release_tag"
    echo "Ready to push: git push origin main $release_tag"
