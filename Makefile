PREFIX ?= $(HOME)/.local
BINDIR ?= $(PREFIX)/bin
LIBEXECDIR ?= $(PREFIX)/libexec/transcriber
BINARY := transcriber

.PHONY: build install setup test fmt doctor clean

build:
	go build -o bin/$(BINARY) ./cmd/transcriber

install: build
	install -d $(BINDIR)
	install -m 0755 bin/$(BINARY) $(BINDIR)/$(BINARY)
	install -d $(LIBEXECDIR)
	install -m 0755 scripts/setup-models.sh scripts/setup-whispercpp.sh scripts/setup-demucs.sh scripts/setup-diarization.sh $(LIBEXECDIR)
	install -d $(LIBEXECDIR)/diarizer/src/transcriber_diarizer $(LIBEXECDIR)/diarizer/tests
	install -m 0644 diarizer/pyproject.toml $(LIBEXECDIR)/diarizer/pyproject.toml
	install -m 0644 diarizer/src/transcriber_diarizer/__init__.py diarizer/src/transcriber_diarizer/cli.py $(LIBEXECDIR)/diarizer/src/transcriber_diarizer
	install -m 0644 diarizer/tests/test_cli.py $(LIBEXECDIR)/diarizer/tests/test_cli.py

setup: install
	$(BINDIR)/$(BINARY) setup

test:
	go test ./...

fmt:
	gofmt -w cmd/transcriber/*.go

doctor:
	go run ./cmd/transcriber doctor

clean:
	rm -rf bin
