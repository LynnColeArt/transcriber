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
	version            = "0.1.0"
	defaultModelPath   = "~/.local/share/whisper.cpp/models/ggml-large-v3.bin"
	defaultDemucsModel = "htdemucs"
)

type config struct {
	ffmpegBin   string
	whisperBin  string
	demucsBin   string
	modelPath   string
	demucsModel string
	preprocess  string
	language    string
	threads     int
	keepTemp    bool
	verbose     bool
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
	fs.StringVar(&cfg.language, "language", cfg.language, "Whisper language code, or auto")
	fs.IntVar(&cfg.threads, "threads", cfg.threads, "Whisper CPU threads")
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
		ffmpegBin:   getenvDefault("FFMPEG_BIN", "ffmpeg"),
		whisperBin:  os.Getenv("WHISPER_BIN"),
		demucsBin:   getenvDefault("DEMUCS_BIN", "demucs"),
		modelPath:   getenvDefault("WHISPER_MODEL", defaultModelPath),
		demucsModel: getenvDefault("DEMUCS_MODEL", defaultDemucsModel),
		preprocess:  getenvDefault("TRANSCRIBER_PREPROCESS", "auto"),
		language:    getenvDefault("WHISPER_LANGUAGE", "auto"),
		threads:     max(1, runtime.NumCPU()/2),
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

	format := inferOutputFormat(outputPath)
	prefix := filepath.Join(workDir, "transcript")
	if err := runWhisper(cfg, whisperAudioPath, prefix, format); err != nil {
		return err
	}

	generatedPath := prefix + format.extension
	if err := copyFile(generatedPath, outputPath); err != nil {
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

	cfg.modelPath, err = expandPath(cfg.modelPath)
	if err != nil {
		return err
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

	return nil
}

func decodeForProcessing(cfg config, inputPath, outputPath string) error {
	args := []string{
		"-hide_banner",
		"-nostdin",
		"-y",
		"-i", inputPath,
		"-vn",
		"-map", "0:a:0",
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
		demucsPath, err := exec.LookPath(cfg.demucsBin)
		if err == nil {
			cfg.demucsBin = demucsPath
			if err := demucsThenNormalize(cfg, inputPath, outputPath, workDir); err == nil {
				return "demucs", nil
			} else {
				fmt.Fprintf(os.Stderr, "demucs failed, falling back to ffmpeg denoise: %v\n", err)
			}
		} else {
			fmt.Fprintln(os.Stderr, "demucs not found, using ffmpeg denoise")
		}
		return "denoise", normalizeForWhisper(cfg, inputPath, outputPath, true)
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

	return normalizeForWhisper(cfg, vocalsPath, outputPath, true)
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
	args := []string{
		"-m", cfg.modelPath,
		"-f", audioPath,
		"-of", outputPrefix,
		format.whisperArg,
		"-t", fmt.Sprint(cfg.threads),
	}
	if cfg.language != "" && cfg.language != "auto" {
		args = append(args, "-l", cfg.language)
	}
	return runCommand(cfg, cfg.whisperBin, args...)
}

func inferOutputFormat(outputPath string) outputFormat {
	switch strings.ToLower(filepath.Ext(outputPath)) {
	case ".srt":
		return outputFormat{name: "srt", whisperArg: "-osrt", extension: ".srt"}
	case ".vtt":
		return outputFormat{name: "vtt", whisperArg: "-ovtt", extension: ".vtt"}
	case ".json":
		return outputFormat{name: "json", whisperArg: "-oj", extension: ".json"}
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

	modelPath, _ := expandPath(cfg.modelPath)
	if _, err := os.Stat(modelPath); err == nil {
		fmt.Printf("ok   whisper model: %s\n", modelPath)
	} else {
		fmt.Printf("miss whisper model: %s\n", modelPath)
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
  transcriber -preprocess demucs interview.mp4 interview.srt
  transcriber -language en lecture.wav lecture.json

Flags:
  -model PATH          whisper.cpp ggml model path
  -preprocess MODE    auto, demucs, denoise, none
  -language CODE      language code, or auto
  -threads N          CPU threads for whisper.cpp
  -keep-temp          keep intermediate audio
  -verbose            print external commands

Environment:
  WHISPER_MODEL, WHISPER_BIN, FFMPEG_BIN, DEMUCS_BIN, DEMUCS_MODEL`)
}
