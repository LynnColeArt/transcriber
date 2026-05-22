package main

import (
	"path/filepath"
	"testing"
)

func TestInferOutputFormat(t *testing.T) {
	tests := []struct {
		path      string
		arg       string
		extension string
	}{
		{"notes.txt", "-otxt", ".txt"},
		{"captions.srt", "-osrt", ".srt"},
		{"captions.vtt", "-ovtt", ".vtt"},
		{"words.json", "-oj", ".json"},
		{"transcript.md", "-otxt", ".txt"},
	}

	for _, test := range tests {
		got := inferOutputFormat(test.path)
		if got.whisperArg != test.arg || got.extension != test.extension {
			t.Fatalf("inferOutputFormat(%q) = (%q, %q), want (%q, %q)", test.path, got.whisperArg, got.extension, test.arg, test.extension)
		}
	}
}

func TestExpandPath(t *testing.T) {
	got, err := expandPath("~/models/model.bin")
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Base(filepath.Dir(got)) != "models" || filepath.Base(got) != "model.bin" {
		t.Fatalf("expandPath returned unexpected path: %s", got)
	}
}
