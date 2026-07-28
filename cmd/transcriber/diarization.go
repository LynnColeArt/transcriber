package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"unicode"
)

type diarizationResult struct {
	Model             string               `json:"model"`
	Device            string               `json:"device"`
	Speakers          []string             `json:"speakers"`
	Segments          []diarizationSegment `json:"segments"`
	ExclusiveSegments []diarizationSegment `json:"exclusive_segments"`
}

type diarizationSegment struct {
	Start   float64 `json:"start"`
	End     float64 `json:"end"`
	Speaker string  `json:"speaker"`
}

type transcriptTurn struct {
	StartMS int64  `json:"start_ms"`
	EndMS   int64  `json:"end_ms"`
	Speaker string `json:"speaker"`
	Text    string `json:"text"`
}

type transcriptToken struct {
	value   map[string]any
	text    string
	fromMS  int64
	toMS    int64
	speaker string
}

func writeDiarizedTranscript(whisperPath, diarizationPath, outputPath string, format outputFormat) error {
	document, turns, err := mergeDiarization(whisperPath, diarizationPath)
	if err != nil {
		return err
	}

	var data []byte
	switch format.name {
	case "json":
		data, err = json.MarshalIndent(document, "", "  ")
		if err == nil {
			data = append(data, '\n')
		}
	case "srt":
		data = []byte(renderSRT(turns))
	case "vtt":
		data = []byte(renderVTT(turns))
	default:
		data = []byte(renderText(turns))
	}
	if err != nil {
		return err
	}
	return writeFileAtomic(outputPath, data)
}

func mergeDiarization(whisperPath, diarizationPath string) (map[string]any, []transcriptTurn, error) {
	whisperData, err := os.ReadFile(whisperPath)
	if err != nil {
		return nil, nil, fmt.Errorf("read Whisper JSON: %w", err)
	}
	diarizationData, err := os.ReadFile(diarizationPath)
	if err != nil {
		return nil, nil, fmt.Errorf("read diarization JSON: %w", err)
	}

	var document map[string]any
	if err := json.Unmarshal(whisperData, &document); err != nil {
		return nil, nil, fmt.Errorf("parse Whisper JSON: %w", err)
	}
	var diarization diarizationResult
	if err := json.Unmarshal(diarizationData, &diarization); err != nil {
		return nil, nil, fmt.Errorf("parse diarization JSON: %w", err)
	}
	if err := validateDiarization(diarization); err != nil {
		return nil, nil, err
	}

	alignment := diarization.ExclusiveSegments
	if len(alignment) == 0 {
		alignment = diarization.Segments
	}
	transcription, ok := document["transcription"].([]any)
	if !ok {
		return nil, nil, errors.New("Whisper JSON has no transcription array")
	}

	turns := make([]transcriptTurn, 0, len(transcription))
	for _, rawSegment := range transcription {
		segment, ok := rawSegment.(map[string]any)
		if !ok {
			continue
		}
		from, to, ok := offsetsFromMap(segment)
		if !ok {
			continue
		}

		segmentTurns := turnsForWhisperSegment(segment, from, to, alignment)
		if len(segmentTurns) == 0 {
			continue
		}
		turns = append(turns, segmentTurns...)

		dominant := speakerForInterval(from, to, alignment)
		if dominant != "" {
			segment["speaker"] = dominant
		}
		segment["speaker_turns"] = turnMaps(segmentTurns)
	}

	document["diarization"] = map[string]any{
		"model":              diarization.Model,
		"device":             diarization.Device,
		"speakers":           diarization.Speakers,
		"segments":           diarization.Segments,
		"exclusive_segments": diarization.ExclusiveSegments,
	}
	return document, turns, nil
}

func validateDiarization(result diarizationResult) error {
	if len(result.Segments) == 0 && len(result.ExclusiveSegments) == 0 {
		return errors.New("diarization returned no speaker segments")
	}
	for _, segment := range append(append([]diarizationSegment{}, result.Segments...), result.ExclusiveSegments...) {
		if segment.Speaker == "" || segment.Start < 0 || segment.End <= segment.Start {
			return fmt.Errorf("invalid diarization segment: start=%g end=%g speaker=%q", segment.Start, segment.End, segment.Speaker)
		}
	}
	return nil
}

func turnsForWhisperSegment(segment map[string]any, from, to int64, speakers []diarizationSegment) []transcriptTurn {
	rawTokens, _ := segment["tokens"].([]any)
	tokens := make([]transcriptToken, 0, len(rawTokens))
	for _, rawToken := range rawTokens {
		token, ok := rawToken.(map[string]any)
		if !ok {
			continue
		}
		text, _ := token["text"].(string)
		if text == "" || isSpecialToken(text) {
			continue
		}
		tokenFrom, tokenTo, hasOffsets := offsetsFromMap(token)
		if !hasOffsets || tokenTo < tokenFrom {
			tokenFrom, tokenTo = from, to
		}
		tokens = append(tokens, transcriptToken{
			value:   token,
			text:    text,
			fromMS:  tokenFrom,
			toMS:    tokenTo,
			speaker: overlappingSpeakerForInterval(tokenFrom, tokenTo, speakers),
		})
	}
	fillUnanchoredTokenSpeakers(tokens, speakers)

	turns := make([]transcriptTurn, 0, 2)
	for _, token := range tokens {
		speaker := token.speaker
		if speaker == "" {
			continue
		}
		if (isPunctuationToken(token.text) || isWordContinuationToken(token.text)) && len(turns) > 0 {
			speaker = turns[len(turns)-1].Speaker
		}
		token.value["speaker"] = speaker
		turns = appendTokenTurn(turns, transcriptTurn{
			StartMS: token.fromMS,
			EndMS:   token.toMS,
			Speaker: speaker,
			Text:    token.text,
		})
	}

	if len(turns) == 0 {
		text, _ := segment["text"].(string)
		speaker := speakerForInterval(from, to, speakers)
		if strings.TrimSpace(text) != "" && speaker != "" {
			turns = append(turns, transcriptTurn{StartMS: from, EndMS: to, Speaker: speaker, Text: text})
		}
	}
	for i := range turns {
		turns[i].Text = strings.TrimSpace(turns[i].Text)
	}
	return turns
}

func fillUnanchoredTokenSpeakers(tokens []transcriptToken, speakers []diarizationSegment) {
	for i := range tokens {
		if tokens[i].speaker != "" {
			continue
		}

		previous := -1
		for j := i - 1; j >= 0; j-- {
			if tokens[j].speaker != "" {
				previous = j
				break
			}
		}
		next := -1
		for j := i + 1; j < len(tokens); j++ {
			if tokens[j].speaker != "" {
				next = j
				break
			}
		}

		switch {
		case previous < 0 && next >= 0:
			tokens[i].speaker = tokens[next].speaker
		case previous >= 0 && next < 0:
			tokens[i].speaker = tokens[previous].speaker
		case previous >= 0 && next >= 0 && tokens[previous].speaker == tokens[next].speaker:
			tokens[i].speaker = tokens[previous].speaker
		case previous >= 0 && next >= 0:
			midpoint := float64(tokens[i].fromMS+tokens[i].toMS) / 2
			previousDistance := math.Abs(midpoint - float64(tokens[previous].toMS))
			nextDistance := math.Abs(float64(tokens[next].fromMS) - midpoint)
			if nextDistance < previousDistance {
				tokens[i].speaker = tokens[next].speaker
			} else {
				tokens[i].speaker = tokens[previous].speaker
			}
		default:
			tokens[i].speaker = speakerForInterval(tokens[i].fromMS, tokens[i].toMS, speakers)
		}
	}
}

func appendTokenTurn(turns []transcriptTurn, next transcriptTurn) []transcriptTurn {
	if len(turns) == 0 || turns[len(turns)-1].Speaker != next.Speaker {
		return append(turns, next)
	}
	last := &turns[len(turns)-1]
	last.Text += next.Text
	if next.EndMS > last.EndMS {
		last.EndMS = next.EndMS
	}
	return turns
}

func offsetsFromMap(value map[string]any) (int64, int64, bool) {
	offsets, ok := value["offsets"].(map[string]any)
	if !ok {
		return 0, 0, false
	}
	from, okFrom := numberAsInt64(offsets["from"])
	to, okTo := numberAsInt64(offsets["to"])
	return from, to, okFrom && okTo
}

func numberAsInt64(value any) (int64, bool) {
	switch number := value.(type) {
	case float64:
		return int64(math.Round(number)), true
	case int64:
		return number, true
	case int:
		return int64(number), true
	default:
		return 0, false
	}
}

func speakerForInterval(fromMS, toMS int64, segments []diarizationSegment) string {
	if len(segments) == 0 {
		return ""
	}
	if speaker := overlappingSpeakerForInterval(fromMS, toMS, segments); speaker != "" {
		return speaker
	}

	from := float64(fromMS) / 1000
	to := float64(toMS) / 1000
	if to < from {
		from, to = to, from
	}

	midpoint := (from + to) / 2
	best := segments[0]
	bestDistance := intervalDistance(midpoint, best.Start, best.End)
	for _, segment := range segments[1:] {
		distance := intervalDistance(midpoint, segment.Start, segment.End)
		if distance < bestDistance || (distance == bestDistance && segment.Speaker < best.Speaker) {
			best = segment
			bestDistance = distance
		}
	}
	return best.Speaker
}

func overlappingSpeakerForInterval(fromMS, toMS int64, segments []diarizationSegment) string {
	from := float64(fromMS) / 1000
	to := float64(toMS) / 1000
	if to < from {
		from, to = to, from
	}

	overlapBySpeaker := map[string]float64{}
	for _, segment := range segments {
		overlap := math.Min(to, segment.End) - math.Max(from, segment.Start)
		if overlap > 0 {
			overlapBySpeaker[segment.Speaker] += overlap
		}
	}
	if len(overlapBySpeaker) == 0 {
		return ""
	}

	labels := make([]string, 0, len(overlapBySpeaker))
	for label := range overlapBySpeaker {
		labels = append(labels, label)
	}
	sort.Strings(labels)
	best := labels[0]
	for _, label := range labels[1:] {
		if overlapBySpeaker[label] > overlapBySpeaker[best] {
			best = label
		}
	}
	return best
}

func intervalDistance(point, start, end float64) float64 {
	if point < start {
		return start - point
	}
	if point > end {
		return point - end
	}
	return 0
}

func isSpecialToken(text string) bool {
	trimmed := strings.TrimSpace(text)
	return strings.HasPrefix(trimmed, "[_") && strings.HasSuffix(trimmed, "]")
}

func isPunctuationToken(text string) bool {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return false
	}
	for _, r := range trimmed {
		if !unicode.IsPunct(r) && !unicode.IsSymbol(r) {
			return false
		}
	}
	return true
}

func isWordContinuationToken(text string) bool {
	trimmed := strings.TrimSpace(text)
	return strings.HasPrefix(trimmed, "'") || strings.HasPrefix(trimmed, "’")
}

func turnMaps(turns []transcriptTurn) []map[string]any {
	result := make([]map[string]any, 0, len(turns))
	for _, turn := range turns {
		result = append(result, map[string]any{
			"offsets": map[string]int64{"from": turn.StartMS, "to": turn.EndMS},
			"speaker": turn.Speaker,
			"text":    turn.Text,
		})
	}
	return result
}

func renderText(turns []transcriptTurn) string {
	var output strings.Builder
	for _, turn := range turns {
		if turn.Text == "" {
			continue
		}
		fmt.Fprintf(&output, "%s: %s\n", turn.Speaker, turn.Text)
	}
	return output.String()
}

func renderSRT(turns []transcriptTurn) string {
	var output strings.Builder
	index := 1
	for _, turn := range turns {
		if turn.Text == "" {
			continue
		}
		fmt.Fprintf(&output, "%d\n%s --> %s\n%s: %s\n\n", index, formatTimestamp(turn.StartMS, ','), formatTimestamp(turn.EndMS, ','), turn.Speaker, turn.Text)
		index++
	}
	return output.String()
}

func renderVTT(turns []transcriptTurn) string {
	var output strings.Builder
	output.WriteString("WEBVTT\n\n")
	for _, turn := range turns {
		if turn.Text == "" {
			continue
		}
		fmt.Fprintf(&output, "%s --> %s\n%s: %s\n\n", formatTimestamp(turn.StartMS, '.'), formatTimestamp(turn.EndMS, '.'), turn.Speaker, turn.Text)
	}
	return output.String()
}

func formatTimestamp(milliseconds int64, decimal byte) string {
	if milliseconds < 0 {
		milliseconds = 0
	}
	hours := milliseconds / 3_600_000
	minutes := (milliseconds / 60_000) % 60
	seconds := (milliseconds / 1_000) % 60
	millis := milliseconds % 1_000
	return fmt.Sprintf("%02d:%02d:%02d%c%03d", hours, minutes, seconds, decimal, millis)
}

func writeFileAtomic(path string, data []byte) error {
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		return err
	}
	return nil
}
