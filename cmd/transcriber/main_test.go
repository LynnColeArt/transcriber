package main

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestFindSetupScriptUsesEnvironmentOverride(t *testing.T) {
	directory := t.TempDir()
	script := filepath.Join(directory, "setup-models.sh")
	if err := os.WriteFile(script, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("TRANSCRIBER_SETUP", script)

	got, err := findSetupScript()
	if err != nil {
		t.Fatal(err)
	}
	if got != script {
		t.Fatalf("findSetupScript() = %q, want %q", got, script)
	}
}

func TestDefaultConfigDetectsSpeakersAutomatically(t *testing.T) {
	t.Setenv("TRANSCRIBER_FORMAT", "")
	cfg := defaultConfig()
	if !cfg.diarize {
		t.Fatal("diarization should be enabled by default")
	}
	if cfg.numSpeakers != 0 || cfg.minSpeakers != 0 || cfg.maxSpeakers != 0 {
		t.Fatalf("speaker count hints should default to inference: %#v", cfg)
	}
	if cfg.format != "text" {
		t.Fatalf("output format = %q, want text", cfg.format)
	}
}

func TestSelectOutputFormatDefaultsToText(t *testing.T) {
	if got := selectOutputFormat("text", "transcript.json"); got.name != "txt" || got.whisperArg != "-otxt" {
		t.Fatalf("selectOutputFormat(text) = %#v", got)
	}
	if got := selectOutputFormat("json", "transcript.txt"); got.name != "json" || got.whisperArg != "-ojf" {
		t.Fatalf("selectOutputFormat(json) = %#v", got)
	}
	if got := selectOutputFormat("auto", "captions.vtt"); got.name != "vtt" {
		t.Fatalf("selectOutputFormat(auto) = %#v", got)
	}
}

func TestInferOutputFormat(t *testing.T) {
	tests := []struct {
		path      string
		arg       string
		extension string
	}{
		{"notes.txt", "-otxt", ".txt"},
		{"captions.srt", "-osrt", ".srt"},
		{"captions.vtt", "-ovtt", ".vtt"},
		{"words.json", "-ojf", ".json"},
		{"transcript.md", "-otxt", ".txt"},
	}

	for _, test := range tests {
		got := inferOutputFormat(test.path)
		if got.whisperArg != test.arg || got.extension != test.extension {
			t.Fatalf("inferOutputFormat(%q) = (%q, %q), want (%q, %q)", test.path, got.whisperArg, got.extension, test.arg, test.extension)
		}
	}
}

func TestWhisperArgsUseQualityDefaults(t *testing.T) {
	cfg := config{
		modelPath:  "/models/large-v3.bin",
		threads:    6,
		maxContext: 0,
		language:   "auto",
	}
	format := outputFormat{name: "json", whisperArg: "-ojf", extension: ".json"}
	args := whisperArgs(cfg, "/tmp/audio.wav", "/tmp/transcript", format)

	wantPairs := [][]string{
		{"-mc", "0"},
		{"-l", "auto"},
		{"-ojf"},
	}
	for _, want := range wantPairs {
		if !containsSequence(args, want) {
			t.Fatalf("whisperArgs() = %q, missing %q", args, want)
		}
	}
}

func TestWhisperArgsOptionalQualityControls(t *testing.T) {
	cfg := config{
		modelPath:  "/models/large-v3.bin",
		threads:    4,
		maxContext: 64,
		language:   "en",
		prompt:     "SOX9, Enhancer 13",
		vad:        true,
		vadModel:   "/models/silero.bin",
	}
	args := whisperArgs(cfg, "audio.wav", "transcript", outputFormat{whisperArg: "-otxt"})

	for _, want := range [][]string{
		{"-mc", "64"},
		{"--prompt", "SOX9, Enhancer 13"},
		{"--vad", "--vad-model", "/models/silero.bin"},
	} {
		if !containsSequence(args, want) {
			t.Fatalf("whisperArgs() = %q, missing %q", args, want)
		}
	}
}

func containsSequence(values, sequence []string) bool {
	if len(sequence) == 1 {
		return slices.Contains(values, sequence[0])
	}
	for i := 0; i+len(sequence) <= len(values); i++ {
		if slices.Equal(values[i:i+len(sequence)], sequence) {
			return true
		}
	}
	return false
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
