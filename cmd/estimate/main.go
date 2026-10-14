// Command estimate reads a JSON array of strings on stdin and prints a JSON array with the
// trim.ClassEstimator token estimate of each. tools/calibrate.py uses it so the calibration measures
// exactly the estimator the proxy ships.
package main

import (
	"encoding/json"
	"flag"
	"log"
	"os"

	"github.com/Ahmadkashif/llm-context-window-trim/trim"
)

func main() {
	run := flag.Int("letter-run", 3, "ASCII letters per token")
	flag.Parse()
	var in []string
	if err := json.NewDecoder(os.Stdin).Decode(&in); err != nil {
		log.Fatal(err)
	}
	e := trim.ClassEstimator{LetterRun: *run}
	out := make([]int, len(in))
	for i, s := range in {
		out[i] = e.Text(s)
	}
	if err := json.NewEncoder(os.Stdout).Encode(out); err != nil {
		log.Fatal(err)
	}
}
