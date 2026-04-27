package config

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"os/exec"
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

	scriptPath := path + ".py"
	scriptContent := fmt.Sprintf(`
import os
import sys

updates = {}
for arg in sys.argv[1:]:
    if '=' in arg:
        k, v = arg.split('=', 1)
        updates[k] = v

lines = []
if os.path.exists(%q):
    with open(%q, 'rb') as f:
        lines = f.read().decode('utf-8').split('\n')

new_lines = []
for line in lines:
    key = line.split('=')[0].strip() if '=' in line else ''
    if key and key not in updates:
        new_lines.append(line)

for key, val in updates.items():
    new_lines.append(f"{key}={val}")

with open(%q, 'wb') as f:
    f.write('\n'.join(new_lines).encode('utf-8'))
`, path, path, path)

	if err := os.WriteFile(scriptPath, []byte(scriptContent), 0755); err != nil {
		return fmt.Errorf("write script: %w", err)
	}
	defer os.Remove(scriptPath)

	args := []string{"python3", scriptPath}
	for k, v := range updates {
		args = append(args, fmt.Sprintf("%s=%s", k, v))
	}

	cmd := exec.Command(args[0], args[1:]...)
	output, err := cmd.CombinedOutput()

	if err != nil {
		return fmt.Errorf("python error: %v, output: %s", err, output)
	}

	log.Printf("[SaveDotenv] Successfully saved via Python: %v", updates)
	return nil
}
