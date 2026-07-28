package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestMergeDiarizationAssignsSpeakersToTokens(t *testing.T) {
	directory := t.TempDir()
	whisperPath := filepath.Join(directory, "whisper.json")
	diarizationPath := filepath.Join(directory, "diarization.json")

	writeTestJSON(t, whisperPath, map[string]any{
		"result": map[string]any{"language": "en"},
		"transcription": []any{
			map[string]any{
				"offsets": map[string]any{"from": 0, "to": 4000},
				"text":    " Hello there. Hi!",
				"tokens": []any{
					map[string]any{"text": "[_BEG_]", "offsets": map[string]any{"from": 0, "to": 0}},
					map[string]any{"text": " Hello", "offsets": map[string]any{"from": 200, "to": 800}},
					map[string]any{"text": " there", "offsets": map[string]any{"from": 800, "to": 1500}},
					map[string]any{"text": ".", "offsets": map[string]any{"from": 1990, "to": 2010}},
					map[string]any{"text": " Hi", "offsets": map[string]any{"from": 2200, "to": 2700}},
					map[string]any{"text": "!", "offsets": map[string]any{"from": 2700, "to": 2750}},
				},
			},
		},
	})
	writeTestJSON(t, diarizationPath, diarizationResult{
		Model:    "test-model",
		Device:   "cpu",
		Speakers: []string{"SPEAKER_00", "SPEAKER_01"},
		Segments: []diarizationSegment{
			{Start: 0, End: 2, Speaker: "SPEAKER_00"},
			{Start: 2, End: 4, Speaker: "SPEAKER_01"},
		},
		ExclusiveSegments: []diarizationSegment{
			{Start: 0, End: 2, Speaker: "SPEAKER_00"},
			{Start: 2, End: 4, Speaker: "SPEAKER_01"},
		},
	})

	document, turns, err := mergeDiarization(whisperPath, diarizationPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(turns) != 2 {
		t.Fatalf("got %d turns, want 2: %#v", len(turns), turns)
	}
	if turns[0].Speaker != "SPEAKER_00" || turns[0].Text != "Hello there." {
		t.Fatalf("first turn = %#v", turns[0])
	}
	if turns[1].Speaker != "SPEAKER_01" || turns[1].Text != "Hi!" {
		t.Fatalf("second turn = %#v", turns[1])
	}
	if _, ok := document["diarization"]; !ok {
		t.Fatal("merged document has no diarization metadata")
	}

	transcription := document["transcription"].([]any)
	segment := transcription[0].(map[string]any)
	speakerTurns := segment["speaker_turns"].([]map[string]any)
	if len(speakerTurns) != 2 {
		t.Fatalf("speaker_turns = %#v", speakerTurns)
	}
}

func TestSpeakerForIntervalUsesOverlapThenNearest(t *testing.T) {
	segments := []diarizationSegment{
		{Start: 0, End: 1, Speaker: "A"},
		{Start: 2, End: 4, Speaker: "B"},
	}
	if got := speakerForInterval(500, 2500, segments); got != "A" {
		t.Fatalf("tie should be stable by label: got %q", got)
	}
	if got := speakerForInterval(1500, 1500, segments); got != "A" {
		t.Fatalf("nearest tie should be stable by label: got %q", got)
	}
	if got := speakerForInterval(2500, 3000, segments); got != "B" {
		t.Fatalf("overlap speaker = %q, want B", got)
	}
}

func TestTurnsForWhisperSegmentKeepsContractionWithPreviousSpeaker(t *testing.T) {
	segment := map[string]any{
		"tokens": []any{
			map[string]any{"text": " I", "offsets": map[string]any{"from": 0, "to": 500}},
			map[string]any{"text": "'m", "offsets": map[string]any{"from": 500, "to": 600}},
			map[string]any{"text": " here", "offsets": map[string]any{"from": 600, "to": 1000}},
		},
	}
	speakers := []diarizationSegment{
		{Start: 0, End: 0.5, Speaker: "A"},
		{Start: 0.5, End: 0.6, Speaker: "B"},
		{Start: 0.6, End: 1, Speaker: "A"},
	}

	turns := turnsForWhisperSegment(segment, 0, 1000, speakers)
	if len(turns) != 1 || turns[0].Speaker != "A" || turns[0].Text != "I'm here" {
		t.Fatalf("turns = %#v", turns)
	}
}

func TestTurnsForWhisperSegmentCarriesLeadingGapIntoAnchoredSpeaker(t *testing.T) {
	segment := map[string]any{
		"tokens": []any{
			map[string]any{"text": " And", "offsets": map[string]any{"from": 5000, "to": 5210}},
			map[string]any{"text": " there", "offsets": map[string]any{"from": 5210, "to": 5560}},
			map[string]any{"text": " it", "offsets": map[string]any{"from": 5560, "to": 5700}},
		},
	}
	speakers := []diarizationSegment{
		{Start: 0.875, End: 4.908, Speaker: "A"},
		{Start: 5.363, End: 9.565, Speaker: "B"},
	}

	turns := turnsForWhisperSegment(segment, 5000, 8000, speakers)
	if len(turns) != 1 || turns[0].Speaker != "B" || turns[0].Text != "And there it" {
		t.Fatalf("turns = %#v", turns)
	}
}

func TestDiarizedRenderers(t *testing.T) {
	turns := []transcriptTurn{
		{StartMS: 1234, EndMS: 5678, Speaker: "SPEAKER_00", Text: "Hello."},
	}
	if got := renderText(turns); got != "SPEAKER_00: Hello.\n" {
		t.Fatalf("renderText() = %q", got)
	}
	if got := renderSRT(turns); !strings.Contains(got, "00:00:01,234 --> 00:00:05,678\nSPEAKER_00: Hello.") {
		t.Fatalf("renderSRT() = %q", got)
	}
	if got := renderVTT(turns); !strings.Contains(got, "WEBVTT\n\n00:00:01.234 --> 00:00:05.678") {
		t.Fatalf("renderVTT() = %q", got)
	}
}

func writeTestJSON(t *testing.T, path string, value any) {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}
