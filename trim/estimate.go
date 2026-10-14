package trim

import (
	"unicode"
	"unicode/utf8"
)

// Estimator returns an upper-bound estimate of how many tokens a piece of text costs.
type Estimator interface {
	Text(s string) int
}

// ClassEstimator is a tokenizer-free, deliberately conservative token estimate.
//
// A plain "bytes / N" rule is not safe: BPE tokenizers in the Qwen/Llama family split numbers into
// single digits, so a digit-heavy prompt can cost one token per byte. ClassEstimator counts by
// character class instead:
//
//   - each ASCII digit costs one token;
//   - each ASCII punctuation or symbol character costs one token;
//   - a run of ASCII letters costs ceil(len / LetterRun) tokens;
//   - a run of two or more whitespace characters costs one token; a single space is free before an
//     ASCII letter (it is merged into the word) and costs one token before anything else (digits and
//     symbols are split off on their own);
//   - a non-ASCII rune costs 1 token if it is 2 bytes in UTF-8 (Latin extensions, Cyrillic, Arabic),
//     1.5 if it is 3 bytes (CJK, Devanagari) and 2 if it is 4 bytes (emoji).
//
// It is measured against the served model's real tokenizer by tools/calibrate.py; see
// evidence/calibration.txt for the margin.
type ClassEstimator struct {
	// LetterRun is how many ASCII letters one token is assumed to cover. Default 3.
	LetterRun int
}

// Text implements Estimator.
func (e ClassEstimator) Text(s string) int {
	run := e.LetterRun
	if run <= 0 {
		run = 3
	}
	// half counts half-tokens so that 3-byte runes can cost 1.5.
	half, letters, spaces := 0, 0, 0
	flushLetters := func() {
		if letters > 0 {
			half += 2 * ((letters + run - 1) / run)
			letters = 0
		}
	}
	// flushSpaces settles a whitespace run; nextIsLetter says whether a letter follows it.
	flushSpaces := func(nextIsLetter bool) {
		if spaces > 1 || (spaces == 1 && !nextIsLetter) {
			half += 2
		}
		spaces = 0
	}
	for i := 0; i < len(s); {
		c := s[i]
		if c < utf8.RuneSelf {
			i++
			switch {
			case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z':
				flushSpaces(true)
				letters++
			case c == ' ' || c == '\t' || c == '\n' || c == '\r':
				flushLetters()
				spaces++
			default: // digit, punctuation, symbol, control
				flushLetters()
				flushSpaces(false)
				half += 2
			}
			continue
		}
		r, size := utf8.DecodeRuneInString(s[i:])
		i += size
		flushLetters()
		if unicode.IsSpace(r) {
			spaces++
			continue
		}
		flushSpaces(false)
		switch {
		case size <= 2:
			half += 2
		case size == 3:
			half += 3
		default:
			half += 4
		}
	}
	flushLetters()
	flushSpaces(false)
	return (half + 1) / 2
}
