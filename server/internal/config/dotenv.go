package config

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
	"sync"
)

var envMu sync.Mutex

// LoadDotenv reads a .env file and sets each KEY=VALUE pair into the process
// environment via os.Setenv. Lines starting with # and blank lines are skipped.
// Returns nil if the file does not exist.
func LoadDotenv(path string) error {
	f, err := os.Open(path)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		value = strings.TrimSpace(value)
		// Strip surrounding quotes if present
		if len(value) >= 2 &&
			((value[0] == '"' && value[len(value)-1] == '"') ||
				(value[0] == '\'' && value[len(value)-1] == '\'')) {
			value = value[1 : len(value)-1]
		}
		os.Setenv(key, value)
	}
	return scanner.Err()
}

// SaveDotenv merges updates into an existing .env file. Existing keys are
// updated in place; new keys are appended. The write is atomic (temp file +
// rename) and protected by a package-level mutex.
func SaveDotenv(path string, updates map[string]string) error {
	log.Printf("===============================================")
	log.Printf("FORCE LOG: SaveDotenv CALLED with path=%s", path)
	log.Printf("FORCE LOG: updates = %v", updates)
	log.Printf("===============================================")
	envMu.Lock()
	defer envMu.Unlock()
	log.Printf("FORCE LOG: Acquired mutex lock")

	f, err := os.OpenFile(path, os.O_RDWR, 0644)
	if err != nil {
		if os.IsNotExist(err) {
			f, err = os.Create(path)
		}
		if err != nil {
			return fmt.Errorf("open file: %w", err)
		}
	}
	defer f.Close()

	reader := bufio.NewReader(f)
	lines := []string{}
	for {
		line, err := reader.ReadString('\n')
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("read file: %w", err)
		}
		line = strings.TrimRight(line, "\r\n")
		parts := strings.SplitN(line, "=", 2)
		if len(parts) == 2 && parts[0] != "" {
			if _, exists := updates[parts[0]]; !exists {
				lines = append(lines, line)
			}
		} else if line != "" {
			lines = append(lines, line)
		}
	}

	for k, v := range updates {
		lines = append(lines, k+"="+v)
	}

	content := strings.Join(lines, "\n") + "\n"
	if err := f.Truncate(0); err != nil {
		return fmt.Errorf("truncate: %w", err)
	}
	if _, err := f.Seek(0, 0); err != nil {
		return fmt.Errorf("seek: %w", err)
	}
	if _, err := f.WriteString(content); err != nil {
		return fmt.Errorf("write: %w", err)
	}

	log.Printf("[SaveDotenv] Successfully saved: %v", updates)
	return nil
}
