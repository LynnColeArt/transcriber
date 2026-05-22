PREFIX ?= $(HOME)/.local
BINDIR ?= $(PREFIX)/bin
BINARY := transcriber

.PHONY: build install test fmt doctor clean

build:
	go build -o bin/$(BINARY) ./cmd/transcriber

install: build
	install -d $(BINDIR)
	install -m 0755 bin/$(BINARY) $(BINDIR)/$(BINARY)

test:
	go test ./...

fmt:
	gofmt -w cmd/transcriber/main.go cmd/transcriber/main_test.go

doctor:
	go run ./cmd/transcriber doctor

clean:
	rm -rf bin
