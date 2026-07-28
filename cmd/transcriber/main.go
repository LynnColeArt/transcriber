package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

const (
	version            = "0.2.0"
	defaultModelPath   = "~/.local/share/whisper.cpp/models/ggml-large-v3.bin"
	defaultVADModel    = "~/.local/share/whisper.cpp/models/ggml-silero-v6.2.0.bin"
	defaultDemucsModel = "htdemucs"
	defaultDiarizer    = "transcriber-diarize"
	defaultDiarization = "pyannote/speaker-diarization-community-1"
)

type config struct {
	ffmpegBin     string
	whisperBin    string
	demucsBin     string
	modelPath     string
	demucsModel   string
	preprocess    string
	format        string
	language      string
	prompt        string
	threads       int
	maxContext    int
	audioStream   int
	vad           bool
	vadModel      string
	diarize       bool
	diarizerBin   string
	diarizeModel  string
	diarizeDevice string
	numSpeakers   int
	minSpeakers   int
	maxSpeakers   int
	keepTemp      bool
	verbose       bool
}

type outputFormat struct {
	name       string
	whisperArg string
	extension  string
}

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "transcriber: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) > 0 {
		switch args[0] {
		case "doctor":
			return doctor()
		case "version", "--version", "-version":
			fmt.Println(version)
			return nil
		case "help", "--help", "-help":
			printUsage(os.Stdout)
			return nil
		}
	}

	cfg := defaultConfig()
	fs := flag.NewFlagSet("transcriber", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	fs.StringVar(&cfg.modelPath, "model", cfg.modelPath, "Path to whisper.cpp ggml model")
	fs.StringVar(&cfg.whisperBin, "whisper-bin", cfg.whisperBin, "Path to whisper.cpp CLI binary")
	fs.StringVar(&cfg.ffmpegBin, "ffmpeg", cfg.ffmpegBin, "Path to ffmpeg")
	fs.StringVar(&cfg.demucsBin, "demucs", cfg.demucsBin, "Path to demucs")
	fs.StringVar(&cfg.demucsModel, "demucs-model", cfg.demucsModel, "Demucs model name")
	fs.StringVar(&cfg.preprocess, "preprocess", cfg.preprocess, "Preprocess mode: auto, demucs, denoise, none")
	fs.StringVar(&cfg.format, "format", cfg.format, "Output format: text, json, srt, vtt, or auto")
	fs.StringVar(&cfg.language, "language", cfg.language, "Whisper language code, or auto")
	fs.StringVar(&cfg.prompt, "prompt", cfg.prompt, "Initial prompt for names and specialized vocabulary")
	fs.IntVar(&cfg.threads, "threads", cfg.threads, "Whisper CPU threads")
	fs.IntVar(&cfg.maxContext, "max-context", cfg.maxContext, "Prior text tokens to carry between windows; 0 disables rolling context")
	fs.IntVar(&cfg.audioStream, "audio-stream", cfg.audioStream, "Zero-based audio stream index")
	fs.BoolVar(&cfg.vad, "vad", cfg.vad, "Use Silero voice activity detection")
	fs.StringVar(&cfg.vadModel, "vad-model", cfg.vadModel, "Path to whisper.cpp VAD model")
	fs.BoolVar(&cfg.diarize, "diarize", cfg.diarize, "Label speakers automatically; use -diarize=false to disable")
	fs.StringVar(&cfg.diarizerBin, "diarizer", cfg.diarizerBin, "Path to transcriber-diarize")
	fs.StringVar(&cfg.diarizeModel, "diarization-model", cfg.diarizeModel, "pyannote model name or local path")
	fs.StringVar(&cfg.diarizeDevice, "diarization-device", cfg.diarizeDevice, "Diarization device: auto, cpu, or cuda")
	fs.IntVar(&cfg.numSpeakers, "num-speakers", cfg.numSpeakers, "Exact number of speakers, or 0 to infer")
	fs.IntVar(&cfg.minSpeakers, "min-speakers", cfg.minSpeakers, "Minimum number of speakers, or 0 to infer")
	fs.IntVar(&cfg.maxSpeakers, "max-speakers", cfg.maxSpeakers, "Maximum number of speakers, or 0 to infer")
	fs.BoolVar(&cfg.keepTemp, "keep-temp", cfg.keepTemp, "Keep temporary working files")
	fs.BoolVar(&cfg.verbose, "verbose", cfg.verbose, "Print commands before running them")

	if err := fs.Parse(args); err != nil {
		printUsage(os.Stderr)
		return err
	}

	if fs.NArg() != 2 {
		printUsage(os.Stderr)
		return errors.New("expected <file in> and <file out>")
	}

	inputPath, err := filepath.Abs(fs.Arg(0))
	if err != nil {
		return err
	}
	outputPath, err := filepath.Abs(fs.Arg(1))
	if err != nil {
		return err
	}

	return transcribe(cfg, inputPath, outputPath)
}

func defaultConfig() config {
	return config{
		ffmpegBin:     getenvDefault("FFMPEG_BIN", "ffmpeg"),
		whisperBin:    os.Getenv("WHISPER_BIN"),
		demucsBin:     getenvDefault("DEMUCS_BIN", "demucs"),
		modelPath:     getenvDefault("WHISPER_MODEL", defaultModelPath),
		demucsModel:   getenvDefault("DEMUCS_MODEL", defaultDemucsModel),
		preprocess:    getenvDefault("TRANSCRIBER_PREPROCESS", "auto"),
		format:        getenvDefault("TRANSCRIBER_FORMAT", "text"),
		language:      getenvDefault("WHISPER_LANGUAGE", "auto"),
		maxContext:    0,
		vadModel:      getenvDefault("WHISPER_VAD_MODEL", defaultVADModel),
		diarize:       true,
		diarizerBin:   getenvDefault("DIARIZER_BIN", defaultDiarizer),
		diarizeModel:  getenvDefault("DIARIZATION_MODEL", defaultDiarization),
		diarizeDevice: getenvDefault("DIARIZATION_DEVICE", "auto"),
		threads:       max(1, runtime.NumCPU()/2),
	}
}

func transcribe(cfg config, inputPath, outputPath string) error {
	if err := validateConfig(&cfg); err != nil {
		return err
	}
	if _, err := os.Stat(inputPath); err != nil {
		return fmt.Errorf("input file: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(outputPath), 0o755); err != nil {
		return err
	}

	workDir, err := os.MkdirTemp("", "transcriber-*")
	if err != nil {
		return err
	}
	if cfg.keepTemp {
		fmt.Fprintf(os.Stderr, "keeping temp directory: %s\n", workDir)
	} else {
		defer os.RemoveAll(workDir)
	}

	decodedPath := filepath.Join(workDir, "decoded.wav")
	if err := decodeForProcessing(cfg, inputPath, decodedPath); err != nil {
		return err
	}

	whisperAudioPath := filepath.Join(workDir, "whisper.wav")
	mode, err := preprocessAudio(cfg, decodedPath, whisperAudioPath, workDir)
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "preprocess: %s\n", mode)

	format := selectOutputFormat(cfg.format, outputPath)
	whisperFormat := format
	if cfg.diarize {
		whisperFormat = outputFormat{name: "json-full", whisperArg: "-ojf", extension: ".json"}
	}
	prefix := filepath.Join(workDir, "transcript")
	if err := runWhisper(cfg, whisperAudioPath, prefix, whisperFormat); err != nil {
		return err
	}

	generatedPath := prefix + whisperFormat.extension
	if cfg.diarize {
		diarizationPath := filepath.Join(workDir, "diarization.json")
		if err := runDiarizer(cfg, whisperAudioPath, diarizationPath); err != nil {
			return err
		}
		if err := writeDiarizedTranscript(generatedPath, diarizationPath, outputPath, format); err != nil {
			return fmt.Errorf("write diarized transcript: %w", err)
		}
	} else if err := copyFile(generatedPath, outputPath); err != nil {
		return fmt.Errorf("write transcript: %w", err)
	}

	fmt.Fprintf(os.Stderr, "wrote %s\n", outputPath)
	return nil
}

func validateConfig(cfg *config) error {
	var err error

	cfg.preprocess = strings.ToLower(strings.TrimSpace(cfg.preprocess))
	switch cfg.preprocess {
	case "auto", "demucs", "denoise", "none":
	default:
		return fmt.Errorf("invalid preprocess mode %q", cfg.preprocess)
	}
	cfg.format = strings.ToLower(strings.TrimSpace(cfg.format))
	switch cfg.format {
	case "text", "txt", "json", "srt", "vtt", "auto":
	default:
		return fmt.Errorf("invalid output format %q", cfg.format)
	}
	if cfg.maxContext < -1 {
		return errors.New("max-context must be -1 or greater")
	}
	if cfg.audioStream < 0 {
		return errors.New("audio-stream must be zero or greater")
	}
	if cfg.numSpeakers < 0 || cfg.minSpeakers < 0 || cfg.maxSpeakers < 0 {
		return errors.New("speaker counts must be zero or greater")
	}
	if cfg.numSpeakers > 0 && (cfg.minSpeakers > 0 || cfg.maxSpeakers > 0) {
		return errors.New("num-speakers cannot be combined with min-speakers or max-speakers")
	}
	if cfg.minSpeakers > 0 && cfg.maxSpeakers > 0 && cfg.minSpeakers > cfg.maxSpeakers {
		return errors.New("min-speakers cannot exceed max-speakers")
	}
	cfg.diarizeDevice = strings.ToLower(strings.TrimSpace(cfg.diarizeDevice))
	switch cfg.diarizeDevice {
	case "auto", "cpu", "cuda":
	default:
		return fmt.Errorf("invalid diarization device %q", cfg.diarizeDevice)
	}

	cfg.modelPath, err = expandPath(cfg.modelPath)
	if err != nil {
		return err
	}
	if cfg.vad {
		cfg.vadModel, err = expandPath(cfg.vadModel)
		if err != nil {
			return err
		}
		if _, err := os.Stat(cfg.vadModel); err != nil {
			return fmt.Errorf("VAD model not found at %s; rerun scripts/setup-whispercpp.sh or pass -vad-model", cfg.vadModel)
		}
	}
	if _, err := os.Stat(cfg.modelPath); err != nil {
		return fmt.Errorf("whisper model not found at %s; run scripts/setup-whispercpp.sh or pass -model", cfg.modelPath)
	}

	cfg.ffmpegBin, err = exec.LookPath(cfg.ffmpegBin)
	if err != nil {
		return errors.New("ffmpeg not found; install ffmpeg or set FFMPEG_BIN")
	}

	if cfg.whisperBin == "" {
		cfg.whisperBin = findFirstExecutable("whisper-cli", "main", "whisper")
	}
	if cfg.whisperBin == "" {
		return errors.New("whisper.cpp CLI not found; install whisper.cpp or set WHISPER_BIN")
	}
	cfg.whisperBin, err = exec.LookPath(cfg.whisperBin)
	if err != nil {
		return fmt.Errorf("whisper binary not found: %w", err)
	}

	if cfg.preprocess == "demucs" {
		cfg.demucsBin, err = exec.LookPath(cfg.demucsBin)
		if err != nil {
			return errors.New("demucs requested but not found; install demucs or use -preprocess denoise")
		}
	}
	if cfg.diarize {
		cfg.diarizerBin, err = exec.LookPath(cfg.diarizerBin)
		if err != nil {
			return errors.New("speaker diarization requested but transcriber-diarize was not found; run scripts/setup-diarization.sh or pass -diarizer")
		}
	}

	return nil
}

func decodeForProcessing(cfg config, inputPath, outputPath string) error {
	args := []string{
		"-hide_banner",
		"-nostdin",
		"-y",
		"-i", inputPath,
		"-vn",
		"-map", fmt.Sprintf("0:a:%d", cfg.audioStream),
		"-ac", "2",
		"-ar", "44100",
		"-c:a", "pcm_s16le",
		outputPath,
	}
	return runCommand(cfg, cfg.ffmpegBin, args...)
}

func preprocessAudio(cfg config, inputPath, outputPath, workDir string) (string, error) {
	switch cfg.preprocess {
	case "none":
		return "none", normalizeForWhisper(cfg, inputPath, outputPath, false)
	case "denoise":
		return "denoise", normalizeForWhisper(cfg, inputPath, outputPath, true)
	case "demucs":
		return "demucs", demucsThenNormalize(cfg, inputPath, outputPath, workDir)
	case "auto":
		if _, err := exec.LookPath(cfg.demucsBin); err == nil {
			return "demucs", demucsThenNormalize(cfg, inputPath, outputPath, workDir)
		}
		fmt.Fprintln(os.Stderr, "warning: Demucs not found; continuing with normalization only")
		return "normalize", normalizeForWhisper(cfg, inputPath, outputPath, false)
	default:
		return "", fmt.Errorf("invalid preprocess mode %q", cfg.preprocess)
	}
}

func demucsThenNormalize(cfg config, inputPath, outputPath, workDir string) error {
	demucsOut := filepath.Join(workDir, "demucs")
	args := []string{
		"-n", cfg.demucsModel,
		"--two-stems", "vocals",
		"-o", demucsOut,
		inputPath,
	}
	if err := runCommand(cfg, cfg.demucsBin, args...); err != nil {
		return err
	}

	trackName := strings.TrimSuffix(filepath.Base(inputPath), filepath.Ext(inputPath))
	vocalsPath := filepath.Join(demucsOut, cfg.demucsModel, trackName, "vocals.wav")
	if _, err := os.Stat(vocalsPath); err != nil {
		return fmt.Errorf("demucs vocals stem not found at %s", vocalsPath)
	}

	return normalizeForWhisper(cfg, vocalsPath, outputPath, false)
}

func normalizeForWhisper(cfg config, inputPath, outputPath string, denoise bool) error {
	filter := "loudnorm=I=-16:TP=-1.5:LRA=11"
	if denoise {
		filter = "highpass=f=80,lowpass=f=7800,afftdn=nf=-25," + filter
	}
	args := []string{
		"-hide_banner",
		"-nostdin",
		"-y",
		"-i", inputPath,
		"-vn",
		"-af", filter,
		"-ac", "1",
		"-ar", "16000",
		"-c:a", "pcm_s16le",
		outputPath,
	}
	return runCommand(cfg, cfg.ffmpegBin, args...)
}

func runWhisper(cfg config, audioPath, outputPrefix string, format outputFormat) error {
	return runCommand(cfg, cfg.whisperBin, whisperArgs(cfg, audioPath, outputPrefix, format)...)
}

func whisperArgs(cfg config, audioPath, outputPrefix string, format outputFormat) []string {
	args := []string{
		"-m", cfg.modelPath,
		"-f", audioPath,
		"-of", outputPrefix,
		format.whisperArg,
		"-t", fmt.Sprint(cfg.threads),
		"-mc", fmt.Sprint(cfg.maxContext),
	}
	if cfg.language != "" {
		args = append(args, "-l", cfg.language)
	}
	if cfg.prompt != "" {
		args = append(args, "--prompt", cfg.prompt)
	}
	if cfg.vad {
		args = append(args, "--vad", "--vad-model", cfg.vadModel)
	}
	return args
}

func runDiarizer(cfg config, audioPath, outputPath string) error {
	args := []string{
		"--audio", audioPath,
		"--output", outputPath,
		"--model", cfg.diarizeModel,
		"--device", cfg.diarizeDevice,
	}
	if cfg.numSpeakers > 0 {
		args = append(args, "--num-speakers", fmt.Sprint(cfg.numSpeakers))
	}
	if cfg.minSpeakers > 0 {
		args = append(args, "--min-speakers", fmt.Sprint(cfg.minSpeakers))
	}
	if cfg.maxSpeakers > 0 {
		args = append(args, "--max-speakers", fmt.Sprint(cfg.maxSpeakers))
	}
	return runCommand(cfg, cfg.diarizerBin, args...)
}

func inferOutputFormat(outputPath string) outputFormat {
	switch strings.ToLower(filepath.Ext(outputPath)) {
	case ".srt":
		return outputFormat{name: "srt", whisperArg: "-osrt", extension: ".srt"}
	case ".vtt":
		return outputFormat{name: "vtt", whisperArg: "-ovtt", extension: ".vtt"}
	case ".json":
		return outputFormat{name: "json", whisperArg: "-ojf", extension: ".json"}
	default:
		return outputFormat{name: "txt", whisperArg: "-otxt", extension: ".txt"}
	}
}

func selectOutputFormat(name, outputPath string) outputFormat {
	switch name {
	case "json":
		return outputFormat{name: "json", whisperArg: "-ojf", extension: ".json"}
	case "srt":
		return outputFormat{name: "srt", whisperArg: "-osrt", extension: ".srt"}
	case "vtt":
		return outputFormat{name: "vtt", whisperArg: "-ovtt", extension: ".vtt"}
	case "auto":
		return inferOutputFormat(outputPath)
	default:
		return outputFormat{name: "txt", whisperArg: "-otxt", extension: ".txt"}
	}
}

func runCommand(cfg config, name string, args ...string) error {
	if cfg.verbose {
		fmt.Fprintf(os.Stderr, "+ %s %s\n", name, shellQuote(args))
	}
	cmd := exec.Command(name, args...)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	tmp := dst + ".tmp"
	out, err := os.OpenFile(tmp, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	if err := out.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, dst)
}

func doctor() error {
	cfg := defaultConfig()

	fmt.Printf("transcriber %s\n", version)
	checkExecutable("ffmpeg", cfg.ffmpegBin, true)
	checkWhisper()
	checkExecutable("demucs", cfg.demucsBin, false)
	checkExecutable("speaker diarizer", cfg.diarizerBin, false)

	modelPath, _ := expandPath(cfg.modelPath)
	if _, err := os.Stat(modelPath); err == nil {
		fmt.Printf("ok   whisper model: %s\n", modelPath)
	} else {
		fmt.Printf("miss whisper model: %s\n", modelPath)
	}

	vadModel, _ := expandPath(cfg.vadModel)
	if _, err := os.Stat(vadModel); err == nil {
		fmt.Printf("ok   VAD model: %s\n", vadModel)
	} else {
		fmt.Printf("opt  VAD model: not found at %s\n", vadModel)
	}

	return nil
}

func checkWhisper() {
	candidates := []string{}
	if env := os.Getenv("WHISPER_BIN"); env != "" {
		candidates = append(candidates, env)
	}
	candidates = append(candidates, "whisper-cli", "main", "whisper")
	for _, candidate := range candidates {
		if path, err := exec.LookPath(candidate); err == nil {
			fmt.Printf("ok   whisper.cpp CLI: %s\n", path)
			return
		}
	}
	fmt.Println("miss whisper.cpp CLI: install whisper.cpp or set WHISPER_BIN")
}

func checkExecutable(label, bin string, required bool) {
	if path, err := exec.LookPath(bin); err == nil {
		fmt.Printf("ok   %s: %s\n", label, path)
	} else if required {
		fmt.Printf("miss %s: install it or set %s_BIN\n", label, strings.ToUpper(label))
	} else {
		fmt.Printf("opt  %s: not found\n", label)
	}
}

func findFirstExecutable(candidates ...string) string {
	for _, candidate := range candidates {
		if path, err := exec.LookPath(candidate); err == nil {
			return path
		}
	}
	return ""
}

func getenvDefault(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func expandPath(path string) (string, error) {
	if path == "" || path[0] != '~' {
		return path, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	if path == "~" {
		return home, nil
	}
	if strings.HasPrefix(path, "~/") {
		return filepath.Join(home, path[2:]), nil
	}
	return "", fmt.Errorf("cannot expand path %q", path)
}

func shellQuote(args []string) string {
	quoted := make([]string, len(args))
	for i, arg := range args {
		if strings.ContainsAny(arg, " \t\n'\"") {
			quoted[i] = "'" + strings.ReplaceAll(arg, "'", "'\\''") + "'"
		} else {
			quoted[i] = arg
		}
	}
	return strings.Join(quoted, " ")
}

func printUsage(w io.Writer) {
	fmt.Fprintln(w, `Usage:
  transcriber [flags] <file in> <file out>
  transcriber doctor

Examples:
  transcriber meeting.mp3 meeting.txt
  transcriber -format json interview.mp4 interview.json
  transcriber -format srt lecture.wav lecture.srt
  transcriber -diarize=false lecture.wav lecture.txt

Flags:
  -model PATH          whisper.cpp ggml model path
  -preprocess MODE    auto (Demucs when available), demucs, denoise, none
  -format FORMAT      text (default), json, srt, vtt, or auto by extension
  -language CODE      language code, or auto
  -prompt TEXT        names or specialized vocabulary to prime Whisper
  -threads N          CPU threads for whisper.cpp
  -max-context N      prior text tokens between windows; 0 disables (default)
  -audio-stream N     zero-based input audio stream (default 0)
  -vad                enable Silero voice activity detection
  -vad-model PATH     whisper.cpp Silero VAD model
  -diarize BOOL       detect and label speakers (default true)
  -diarizer PATH      transcriber-diarize executable
  -diarization-model MODEL
                      pyannote model name or local path
  -diarization-device DEVICE
                      auto, cpu, or cuda
  -num-speakers N     exact speaker count, or 0 to infer
  -min-speakers N     minimum speaker count, or 0 to infer
  -max-speakers N     maximum speaker count, or 0 to infer
  -keep-temp          keep intermediate audio
  -verbose            print external commands

Environment:
  WHISPER_MODEL, WHISPER_BIN, WHISPER_VAD_MODEL, FFMPEG_BIN,
  DEMUCS_BIN, DEMUCS_MODEL, DIARIZER_BIN, DIARIZATION_MODEL,
  DIARIZATION_DEVICE, TRANSCRIBER_FORMAT, HF_TOKEN`)
}
