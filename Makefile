BIN_NAME := ai-watermark-detector
INSTALL_DIR ?= $(HOME)/.local/bin
CARGO := cargo

.PHONY: help build release test lint fmt install uninstall clean check-tools demo python-setup

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

build: ## Build the release binary
	$(CARGO) build --release

release: build ## Alias for build

test: ## Run the Rust test suite
	$(CARGO) test --release

lint: ## Lint Rust (clippy) if available
	@$(CARGO) clippy --all-targets -- -D warnings 2>/dev/null || \
		echo "clippy not installed (rustup component add clippy) - skipping"

fmt: ## Format Rust sources
	$(CARGO) fmt

install: build ## Install the binary to $(INSTALL_DIR)
	@mkdir -p "$(INSTALL_DIR)"
	@install -m 0755 target/release/$(BIN_NAME) "$(INSTALL_DIR)/$(BIN_NAME)"
	@echo "Installed: $(INSTALL_DIR)/$(BIN_NAME)"

uninstall: ## Remove the installed binary
	@rm -f "$(INSTALL_DIR)/$(BIN_NAME)"
	@echo "Removed: $(INSTALL_DIR)/$(BIN_NAME)"

clean: ## Remove build artifacts
	$(CARGO) clean

check-tools: ## Report optional tool availability (c2patool, python)
	@command -v c2patool >/dev/null 2>&1 && echo "c2patool: found" || echo "c2patool: MISSING (needed for check/scan)"
	@command -v python3 >/dev/null 2>&1 && echo "python3:  found" || echo "python3:  MISSING (needed only for contributor tools)"

python-setup: ## Create a venv and install the text-validation Python deps
	python3 -m venv .venv && . .venv/bin/activate && pip install -r tools/requirements.txt

demo: build ## Build, generate a demo corpus, and score watermarked vs human
	./target/release/gen_corpus /tmp/cwd-demo kgw
	@echo "--- watermarked sample ---"
	./target/release/$(BIN_NAME) score --config config.example.json --scheme kgw --token-file /tmp/cwd-demo/watermarked/sample_00.txt
	@echo "--- human sample ---"
	./target/release/$(BIN_NAME) score --config config.example.json --scheme kgw --token-file /tmp/cwd-demo/human/sample_00.txt
