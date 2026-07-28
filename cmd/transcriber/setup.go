package main

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
)

func setupModels(args []string) error {
	if len(args) != 0 {
		return errors.New("setup does not accept arguments")
	}

	script, err := findSetupScript()
	if err != nil {
		return err
	}
	fmt.Println("Setting up transcription models and local helpers...")
	cmd := exec.Command(script)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("model setup failed: %w", err)
	}
	return nil
}

func findSetupScript() (string, error) {
	if configured := os.Getenv("TRANSCRIBER_SETUP"); configured != "" {
		path, err := expandPath(configured)
		if err != nil {
			return "", err
		}
		if executable, err := exec.LookPath(path); err == nil {
			return executable, nil
		}
		return "", fmt.Errorf("TRANSCRIBER_SETUP does not point to an executable: %s", path)
	}

	candidates := make([]string, 0, 2)
	if executable, err := os.Executable(); err == nil {
		candidates = append(candidates, filepath.Clean(filepath.Join(filepath.Dir(executable), "..", "libexec", "transcriber", "setup-models.sh")))
	}
	if workingDirectory, err := os.Getwd(); err == nil {
		candidates = append(candidates, filepath.Join(workingDirectory, "scripts", "setup-models.sh"))
	}
	for _, candidate := range candidates {
		if info, err := os.Stat(candidate); err == nil && !info.IsDir() && info.Mode().Perm()&0o111 != 0 {
			return candidate, nil
		}
	}

	return "", errors.New("model setup program not found; rerun make install or set TRANSCRIBER_SETUP")
}
